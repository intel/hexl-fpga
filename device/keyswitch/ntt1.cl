// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "channels.h"

#define NTT_INS(NAME, INS_ID, key_modulus_start, key_modulus_end)             \
    __single_task void ntt##NAME(                                             \
        __global HOST_MEM uint64_t* restrict twiddle_factors,                 \
        unsigned int key_modulus_size) {                                      \
        uint54_t local_roots[FPGA_NTT_SIZE / VEC *                            \
                             (key_modulus_end - key_modulus_start)][VEC];     \
        _ntt_internal(ch_ntt_modulus[INS_ID], ch_ntt_key_modulus_idx[INS_ID], \
                      ch_ntt_elements[(INS_ID)*2],                            \
                      ch_ntt_elements[(INS_ID)*2 + 1], twiddle_factors,       \
                      key_modulus_size, local_roots, key_modulus_start,       \
                      key_modulus_end, 4, INS_ID);                            \
    }                                                                         \
    __single_task __autorun void ntt_backward##NAME() {                       \
        _ntt_backward(ch_ntt_elements_in[INS_ID],                             \
                      ch_ntt_elements[(INS_ID)*2]);                           \
    }                                                                         \
    __single_task __autorun void ntt_forward##NAME() {                        \
        _ntt_forward(ch_ntt_elements_out[INS_ID],                             \
                     ch_ntt_elements[(INS_ID)*2 + 1]);                        \
    }

#define NTT1_INS(ENGINE_ID) \
    NTT_INS(ENGINE_ID, ENGINE_ID, ENGINE_ID, ENGINE_ID + 1)

#if MAX_RNS_MODULUS_SIZE > 0
NTT1_INS(0)
#endif

#if MAX_RNS_MODULUS_SIZE > 1
NTT1_INS(1)
#endif

#if MAX_RNS_MODULUS_SIZE > 2
NTT1_INS(2)
#endif

#if MAX_RNS_MODULUS_SIZE > 3
NTT1_INS(3)
#endif

#if MAX_RNS_MODULUS_SIZE > 4
NTT1_INS(4)
#endif

#if MAX_RNS_MODULUS_SIZE > 5
NTT1_INS(5)
#endif

// the last one should be careful
#if MAX_RNS_MODULUS_SIZE > 6
NTT1_INS(6)
#endif
