/*
 *                             The MIT License
 *
 * Wavefront Alignment Algorithms
 * Copyright (c) 2017 by Santiago Marco-Sola  <santiagomsola@gmail.com>
 *
 * This file is part of Wavefront Alignment Algorithms.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * PROJECT: Wavefront Alignment Algorithms
 * AUTHOR(S): Santiago Marco-Sola <santiagomsola@gmail.com>
 * DESCRIPTION: WaveFront alignment module for computing wavefronts (gap-affine)
 */

#include "utils/commons.h"
#include "system/mm_allocator.h"
#include "wavefront_compute.h"
#include "wavefront_backtrace_offload.h"
#include "wavefront_extend_kernels_avx.h" //getting print_m512i here
#include <stdio.h>

#ifdef WFA_PARALLEL
#include <omp.h>
#endif

#include <immintrin.h>

#if __AVX2__ &&  __BYTE_ORDER == __LITTLE_ENDIAN

  #if __AVX512CD__ && __AVX512VL__
  extern void avx512_wavefront_next_iter1(__m512i* ins1_o, __m512i* ins1_e, __m512i* out_i1, __m512i* del1_o, __m512i* del1_e, __m512i* out_d1);
  extern void avx512_wavefront_next_iter2(__m512i* misms, __m512i* out_i1, __m512i* out_d1, __m512i* max);
  extern void avx512_wavefront_next_iter3(__m512i* max, __m512i* out_m,  __m512i* ks, const int32_t* textlength, const int32_t* patternlength);
  #endif
#endif

/*
 * Compute Kernels
 */
void wavefront_compute_affine_idm(
    wavefront_aligner_t* const wf_aligner,
    const wavefront_set_t* const wavefront_set,
    const int lo,
    const int hi) {
  // Parameters
  wavefront_sequences_t* const sequences = &wf_aligner->sequences;
  const int pattern_length = sequences->pattern_length;
  const int text_length = sequences->text_length;
  // In Offsets [No reassignment of values]
  //const removed
  const wf_offset_t* const m_misms = wavefront_set->in_mwavefront_misms->offsets;
  const wf_offset_t* const m_open1 = wavefront_set->in_mwavefront_open1->offsets;
  const wf_offset_t* const i1_ext = wavefront_set->in_i1wavefront_ext->offsets;
  const wf_offset_t* const d1_ext = wavefront_set->in_d1wavefront_ext->offsets;
  // Out Offsets [Reassignment of Values]
  wf_offset_t* const out_m = wavefront_set->out_mwavefront->offsets;
  wf_offset_t* const out_i1 = wavefront_set->out_i1wavefront->offsets;
  wf_offset_t* const out_d1 = wavefront_set->out_d1wavefront->offsets;
  // Compute-Next kernel loop
  //self note: So far from what I can tell, what I can technically do is to get the MAX() of different DP Matrices from different k's (diagonals)
  // the code uses Int32 which is 4 bytes each, 4 bytes = 32 bits, 512 / 32 = 16 different integers, but have to remember its SIGNED INT
  // Does that mean I can compute for the next wavefront through 16 different diagonals? idk.
  //going to need a few offsets[] arrays to hold each in and out

  int k;
  //int32_t one = 1;
  //int32_t zero = 0;
  //PRAGMA_LOOP_VECTORIZE
  //AXV512 testing
  
  
  int k_min = lo;
  int k_max = hi;
  const int elems_per_reg = 16;
  int num_of_diagonals = k_max - k_min + 1;
  int loop_peeling_iters = num_of_diagonals % elems_per_reg;
  
  
  //normal loop until there's enough elements for SIMD operations at AVX512 level
  for (k=k_min;k<=k_min+loop_peeling_iters; ++k) {
    // Update I1
    const wf_offset_t ins1_o = m_open1[k-1]; //self note: wf_offset_t is equivalent to signed int32
    const wf_offset_t ins1_e = i1_ext[k-1];
    const wf_offset_t ins1 = MAX(ins1_o,ins1_e) + 1; //self note: maybe store these values into mask and apply the cmp operation?
    out_i1[k] = ins1;                                // the instruction will be vpcmpgtd k, zmm, zmm
    // Update D1
    const wf_offset_t del1_o = m_open1[k+1];
    const wf_offset_t del1_e = d1_ext[k+1];
    const wf_offset_t del1 = MAX(del1_o,del1_e);
    out_d1[k] = del1;
    // Update M
    const wf_offset_t misms = m_misms[k] + 1;
    wf_offset_t max = MAX(del1,MAX(misms,ins1));
    //m512i checking
    __m512i offset;
    /*
      offset = _mm512_loadu_si512((__m512*)&ins1);
      print_m512i(offset);
      offset = _mm512_loadu_si512((__m512*)&del1);
      print_m512i(offset);
      offset = _mm512_loadu_si512((__m512*)&misms);
      print_m512i(offset);
      offset = _mm512_loadu_si512((__m512*)&max);
      print_m512i(offset);*/

    // Adjust offset out of boundaries !(h>tlen,v>plen) (here to allow vectorization)
    const wf_unsigned_offset_t h = WAVEFRONT_H(k,max); // Make unsigned to avoid checking negative
    const wf_unsigned_offset_t v = WAVEFRONT_V(k,max); // Make unsigned to avoid checking negative
    if (h > text_length) max = WAVEFRONT_OFFSET_NULL;
    if (v > pattern_length) max = WAVEFRONT_OFFSET_NULL;
    printf("MAX FINAL VECTOR: \n");
    offset = _mm512_loadu_si512((__m512*)&max);
    print_m512i(offset);
    out_m[k] = max;
  }

  //AVX512 NEXT-kernel loop
  if(num_of_diagonals < elems_per_reg) return;

  k_min += loop_peeling_iters;//skip the already handled elements
  
  //will be used for the max - k section of V-vector
  __m512i ks = _mm512_set_epi32 (
      k_min+15,k_min+14,k_min+13,k_min+12,k_min+11,k_min+10,k_min+9,k_min+8,
      k_min+7,k_min+6,k_min+5,k_min+4,k_min+3,k_min+2,k_min+1,k_min); 

  for (k=k_min; k <= k_max; k+=elems_per_reg) {

      //const wf_offset_t ins1;
      //pass an mem_address you dumdum
      //const int* patLen = pattern_length;
      //const int* textLen = text_length;
      __m512i ins1_o = _mm512_loadu_si512((__m512i*)&m_open1[k-1]); //self note: wf_offset_t is equivalent to signed int32
      __m512i ins1_e = _mm512_loadu_si512((__m512i*)&i1_ext[k-1]);

      //const wf_offset_t del1;
      __m512i del1_o = _mm512_loadu_si512((__m512i*)&m_open1[k+1]);
      __m512i del1_e =_mm512_loadu_si512((__m512i*)&d1_ext[k+1]);

      __m512i misms = _mm512_loadu_si512((__m512i*)&m_misms[k]); //+1 operation to all operands will occur inside the assembly

      __m512i max; //gonna receive a vector at this point

      //checker
      //Offsets before iterations
      __m512i offset;
      offset = _mm512_loadu_si512((__m512*)&ins1_o);
      printf("ins1_o");
      print_m512i(offset);
      offset = _mm512_loadu_si512((__m512*)&ins1_e);
      printf("ins1_e");
      print_m512i(offset);
      offset = _mm512_loadu_si512((__m512*)&del1_o);
      printf("del1_o");
      print_m512i(offset);
      offset = _mm512_loadu_si512((__m512*)&del1_e);
      printf("del1_e");
      print_m512i(offset);
      offset = _mm512_loadu_si512((__m512*)&misms);
      printf("misms");
      print_m512i(offset);

      //out_i1[k];
      //out_d1[k];
      //out_m[k];
      //int length of pattern and text may cause errors

      avx512_wavefront_next_iter1(&ins1_o, &ins1_e, (__m512i*)&out_i1[k], &del1_o, &del1_e, (__m512i*)&out_d1[k]);
      //
      printf("Offsets after Outputted Iter1 \n");
      offset = _mm512_loadu_si512((__m512*)&out_i1[k]);
      print_m512i(offset);
      offset = _mm512_loadu_si512((__m512*)&out_d1[k]);
      print_m512i(offset);

      avx512_wavefront_next_iter2(&misms, (__m512i*)&out_i1[k], (__m512i*)&out_d1[k], &max);
      //
      printf("Offsets after Outputted Iter2: \n");
      offset = _mm512_loadu_si512((__m512*)&misms); //consistent as of 09/02/2025
      print_m512i(offset);
      
      offset = _mm512_loadu_si512((__m512*)&out_i1[k]); //consistent as of 09/02/2025
      print_m512i(offset);
      
      offset = _mm512_loadu_si512((__m512*)&out_d1[k]);
      print_m512i(offset);

      offset = _mm512_loadu_si512((__m512*)&max);
      print_m512i(offset);
      
      //pattern checks
      avx512_wavefront_next_iter3(&max, (__m512i*)&out_m[k], &ks, &text_length, &pattern_length);
      printf("Offsets after Outputted Iter3: \n");
      /*
      offset = _mm512_loadu_si512((__m512*)&misms); //consistent as of 09/02/2025
      print_m512i(offset);
      
      offset = _mm512_loadu_si512((__m512*)&out_i1[k]); //consistent as of 09/02/2025
      print_m512i(offset);
      
      offset = _mm512_loadu_si512((__m512*)&out_d1[k]);
      print_m512i(offset);
      */
      printf("Final out_m[k] vector: \n");
      offset = _mm512_loadu_si512((__m512*)&out_m[k]);
      print_m512i(offset);
  }


}
/*
 * Compute Kernel (Piggyback)
 */
void wavefront_compute_affine_idm_piggyback(
    wavefront_aligner_t* const wf_aligner,
    const wavefront_set_t* const wavefront_set,
    const int lo,
    const int hi) {
  // Parameters
  wavefront_sequences_t* const sequences = &wf_aligner->sequences;
  const int pattern_length = sequences->pattern_length;
  const int text_length = sequences->text_length;
  // In Offsets
  const wf_offset_t* const m_misms = wavefront_set->in_mwavefront_misms->offsets;
  const wf_offset_t* const m_open1 = wavefront_set->in_mwavefront_open1->offsets;
  const wf_offset_t* const i1_ext  = wavefront_set->in_i1wavefront_ext->offsets;
  const wf_offset_t* const d1_ext  = wavefront_set->in_d1wavefront_ext->offsets;
  // Out Offsets
  wf_offset_t* const out_m  = wavefront_set->out_mwavefront->offsets;
  wf_offset_t* const out_i1 = wavefront_set->out_i1wavefront->offsets;
  wf_offset_t* const out_d1 = wavefront_set->out_d1wavefront->offsets;
  // In BT-pcigar
  const pcigar_t* const m_misms_bt_pcigar = wavefront_set->in_mwavefront_misms->bt_pcigar;
  const pcigar_t* const m_open1_bt_pcigar = wavefront_set->in_mwavefront_open1->bt_pcigar;
  const pcigar_t* const i1_ext_bt_pcigar  = wavefront_set->in_i1wavefront_ext->bt_pcigar;
  const pcigar_t* const d1_ext_bt_pcigar  = wavefront_set->in_d1wavefront_ext->bt_pcigar;
  // In BT-prev
  const bt_block_idx_t* const m_misms_bt_prev = wavefront_set->in_mwavefront_misms->bt_prev;
  const bt_block_idx_t* const m_open1_bt_prev = wavefront_set->in_mwavefront_open1->bt_prev;
  const bt_block_idx_t* const i1_ext_bt_prev  = wavefront_set->in_i1wavefront_ext->bt_prev;
  const bt_block_idx_t* const d1_ext_bt_prev  = wavefront_set->in_d1wavefront_ext->bt_prev;
  // Out BT-pcigar
  pcigar_t* const out_m_bt_pcigar   = wavefront_set->out_mwavefront->bt_pcigar;
  pcigar_t* const out_i1_bt_pcigar  = wavefront_set->out_i1wavefront->bt_pcigar;
  pcigar_t* const out_d1_bt_pcigar  = wavefront_set->out_d1wavefront->bt_pcigar;
  // Out BT-prev
  bt_block_idx_t* const out_m_bt_prev  = wavefront_set->out_mwavefront->bt_prev;
  bt_block_idx_t* const out_i1_bt_prev = wavefront_set->out_i1wavefront->bt_prev;
  bt_block_idx_t* const out_d1_bt_prev = wavefront_set->out_d1wavefront->bt_prev;
  // Compute-Next kernel loop
  int k;
  PRAGMA_LOOP_VECTORIZE // Ifs predicated by the compiler
  for (k=lo;k<=hi;++k) {
    // Update I1
    const wf_offset_t ins1_o = m_open1[k-1];
    const wf_offset_t ins1_e = i1_ext[k-1];
    wf_offset_t ins1;
    pcigar_t ins1_pcigar;
    bt_block_idx_t ins1_block_idx;
    if (ins1_e >= ins1_o) {
      ins1 = ins1_e;
      ins1_pcigar = i1_ext_bt_pcigar[k-1];
      ins1_block_idx = i1_ext_bt_prev[k-1];
    } else {
      ins1 = ins1_o;
      ins1_pcigar = m_open1_bt_pcigar[k-1];
      ins1_block_idx = m_open1_bt_prev[k-1];
    }
    out_i1_bt_pcigar[k] = PCIGAR_PUSH_BACK_INS(ins1_pcigar);
    out_i1_bt_prev[k] = ins1_block_idx;
    out_i1[k] = ++ins1;
    // Update D1
    const wf_offset_t del1_o = m_open1[k+1];
    const wf_offset_t del1_e = d1_ext[k+1];
    wf_offset_t del1;
    pcigar_t del1_pcigar;
    bt_block_idx_t del1_block_idx;
    if (del1_e >= del1_o) {
      del1 = del1_e;
      del1_pcigar = d1_ext_bt_pcigar[k+1];
      del1_block_idx = d1_ext_bt_prev[k+1];
    } else {
      del1 = del1_o;
      del1_pcigar = m_open1_bt_pcigar[k+1];
      del1_block_idx = m_open1_bt_prev[k+1];
    }
    out_d1_bt_pcigar[k] = PCIGAR_PUSH_BACK_DEL(del1_pcigar);
    out_d1_bt_prev[k] = del1_block_idx;
    out_d1[k] = del1;
    // Update M
    const wf_offset_t misms = m_misms[k] + 1;
    wf_offset_t max = MAX(del1,MAX(misms,ins1));
    if (max == ins1) {
      out_m_bt_pcigar[k] = out_i1_bt_pcigar[k];
      out_m_bt_prev[k] = out_i1_bt_prev[k];
    }
    if (max == del1) {
      out_m_bt_pcigar[k] = out_d1_bt_pcigar[k];
      out_m_bt_prev[k] = out_d1_bt_prev[k];
    }
    if (max == misms) {
      out_m_bt_pcigar[k] = m_misms_bt_pcigar[k];
      out_m_bt_prev[k] = m_misms_bt_prev[k];
    }
    // Coming from I/D -> X is fake to represent gap-close
    // Coming from M -> X is real to represent mismatch
    out_m_bt_pcigar[k] = PCIGAR_PUSH_BACK_MISMS(out_m_bt_pcigar[k]);
    // Adjust offset out of boundaries !(h>tlen,v>plen) (here to allow vectorization)
    const wf_unsigned_offset_t h = WAVEFRONT_H(k,max); // Make unsigned to avoid checking negative
    const wf_unsigned_offset_t v = WAVEFRONT_V(k,max); // Make unsigned to avoid checking negative
    if (h > text_length) max = WAVEFRONT_OFFSET_NULL;
    if (v > pattern_length) max = WAVEFRONT_OFFSET_NULL;
    out_m[k] = max;
  }
}
/*
 * Compute Wavefronts (gap-affine)
 */
void wavefront_compute_affine_dispatcher(
    wavefront_aligner_t* const wf_aligner,
    wavefront_set_t* const wavefront_set,
    const int lo,
    const int hi) {
  // Parameters
  const bool bt_piggyback = wf_aligner->wf_components.bt_piggyback;
  const int num_threads = wavefront_compute_num_threads(wf_aligner,lo,hi);
  // Multithreading dispatcher
  //fprintf(stderr, "Compute Affine Dispatcher has been called");
  if (num_threads == 1) {
    // Compute next wavefront
    if (bt_piggyback) {
      wavefront_compute_affine_idm_piggyback(wf_aligner,wavefront_set,lo,hi);
    } else {
      wavefront_compute_affine_idm(wf_aligner,wavefront_set,lo,hi);
    }
  } else {
#ifdef WFA_PARALLEL
    // Compute next wavefront in parallel
    #pragma omp parallel num_threads(num_threads)
    {
      int t_lo, t_hi;
      const int thread_id = omp_get_thread_num();
      const int thread_num = omp_get_num_threads();
      wavefront_compute_thread_limits(thread_id,thread_num,lo,hi,&t_lo,&t_hi);
      if (bt_piggyback) {
        wavefront_compute_affine_idm_piggyback(wf_aligner,wavefront_set,t_lo,t_hi);
      } else {
        wavefront_compute_affine_idm(wf_aligner,wavefront_set,t_lo,t_hi);
      }
    }
#endif
  }
}
void wavefront_compute_affine(
    wavefront_aligner_t* const wf_aligner,
    const int score) {
  // Select wavefronts
  wavefront_set_t wavefront_set;
  wavefront_compute_fetch_input(wf_aligner,&wavefront_set,score);
  // Check null wavefronts
  //Self notes: this only occurs if the lo >= hi of the wavefront during trim ends
  // still means that the operation I need to focus on will be compute_affine_dispatcher
  if (wavefront_set.in_mwavefront_misms->null &&
      wavefront_set.in_mwavefront_open1->null &&
      wavefront_set.in_i1wavefront_ext->null &&
      wavefront_set.in_d1wavefront_ext->null) {
    wf_aligner->align_status.num_null_steps++; // Increment null-steps
    wavefront_compute_allocate_output_null(wf_aligner,score); // Null s-wavefront
    return;
  }
  wf_aligner->align_status.num_null_steps = 0;
  // Set limits
  int hi, lo;
  wavefront_compute_limits_input(wf_aligner,&wavefront_set,&lo,&hi);
  // Allocate wavefronts
  wavefront_compute_allocate_output(wf_aligner,&wavefront_set,score,lo,hi);
  // Init wavefront ends
  wavefront_compute_init_ends(wf_aligner,&wavefront_set,lo,hi);
  // Compute wavefronts
  wavefront_compute_affine_dispatcher(wf_aligner,&wavefront_set,lo,hi);
  // Offload backtrace (if necessary)
  if (wf_aligner->wf_components.bt_piggyback) {
    wavefront_backtrace_offload_affine(wf_aligner,&wavefront_set,lo,hi);
  }
  // Process wavefront ends
  wavefront_compute_process_ends(wf_aligner,&wavefront_set,score);
}


