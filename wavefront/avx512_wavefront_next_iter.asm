section .data

section .text
global avx512_wavefront_next_iter1
global avx512_wavefront_next_iter2
global avx512_wavefront_next_iter3
default rel

;this function assumes that the loop peeling occurs properly
;params will be (ins1_o, ins1_e, out_i1, del1_o, del1_e, out_d1)
;parameter registers: RDI, RSI, RDX, RCX, R8, R9
avx512_wavefront_next_iter1:
    ;loading in values

    ;setup
    mov eax, 1
    ;MAX() + 1
    vmovdqu32 zmm0, [rdi]; dest, src [int32_t]
    vmovdqu32 zmm1, [rsi]
    vpmaxsd zmm1, zmm1, zmm0; destructive
    vpbroadcastd zmm2, eax
    vpaddd zmm3, zmm1, zmm2 ;save to zmm3 
    ;vpxorq zmm2, zmm2, zmm2; clear zmm2
    ;xor eax, eax; clear eax (no longer going to be used)
    vmovdqu32 [rdx], zmm3 ;assignemnt to out_i1

    ;MAX() + 0
    vmovdqu32 zmm0, [rcx] ;destructive
    vmovdqu32 zmm1, [r8]
    vpmaxsd zmm4, zmm1, zmm0 
    
    vmovdqu32 [r9], zmm4 ;assignment to out_d1
    ret
;params will be (misms, out_i1, out_d1, max)
avx512_wavefront_next_iter2: 
;final MAX(del1, MAX(misms, ins1))
    ;additional setup
    mov eax, 1
    vpbroadcastd zmm5, eax
    ;final values
    ;zmm0 = misms
    ;zmm1 = out_i1
    ;zmm2 = out_d1
    vmovdqu32 zmm0, [rdi]
    vpaddd zmm0, zmm0, zmm5 ;add +1 to every misms according to the algorithm
    vmovdqu32 zmm1, [rsi]
    vmovdqu32 zmm2, [rdx]
    ;MAX(del1,MAX(misms,ins1))
    vpmaxsd zmm3, zmm1, zmm0
    vpmaxsd zmm4, zmm2, zmm3 ;completed max

    vmovdqu32 [rcx], zmm4 ;return to max

    ret

;(max, out_m, ks, text length, pattern length)
;note cmpd and cmpud cause very interesting differences
avx512_wavefront_next_iter3:
    ;check for null offset on max which is equivalent to H
    mov eax, [rcx] ;text length
    vpbroadcastd zmm1, eax ;propagate text length
    vmovdqu32 zmm0, [rdi] ; h = max| (offset)
    vpcmpud k1, zmm0, zmm1, 0x06 ;h > text length -- the k1 mask will be used to overwrite the original MAX
    
    ;check for null offset on V
    mov eax, [r8]
    vmovdqu32 zmm2, [rdx] ;ks
    vpsubd zmm3, zmm0, zmm2; V = max - k's | ((offset)-(k)) #note: errors occur at vpbroadcastd line
    vpbroadcastd zmm1, eax
    vpcmpud k3, zmm3, zmm1, 0x06 ;V > pattern length

    ;use vmovdqu32 with mask
    mov eax, 0xC0000000; nullvalue entry
    vpbroadcastd zmm5, eax ;broadcast null value
    vmovdqu32 zmm0 {k1}, zmm5
    vmovdqu32 zmm0 {k2}, zmm5

    vmovdqu32 [rsi], zmm0 ;assign output to out_m
    
    ret