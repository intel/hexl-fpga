// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "channels.h"

#define NTT_INS(COREID, NAME, INS_ID)                                \
    __single_task __autorun void ntt##COREID##NAME() {               \
        _ntt_internal(ch_ntt_modulus[COREID][INS_ID],                \
                      ch_ntt_elements[COREID][(INS_ID)*2],           \
                      ch_ntt_elements[COREID][(INS_ID)*2 + 1],       \
                      ch_twiddle_factor[COREID][INS_ID], 4, INS_ID); \
    }

#define NTT2_INS(COREID, NAME, INS_ID)                               \
    __single_task __autorun void ntt##COREID##NAME() {               \
        _ntt_internal(ch_ntt_modulus[COREID][INS_ID],                \
                      ch_ntt_elements[COREID][(INS_ID)*2],           \
                      ch_ntt_elements[COREID][(INS_ID)*2 + 1],       \
                      ch_twiddle_factor[COREID][INS_ID], 4, INS_ID); \
    }                                                                \
    __single_task __autorun void ntt_push##COREID##NAME() {          \
        _ntt_backward(ch_ntt_elements_in[COREID][INS_ID],            \
                      ch_ntt_elements[COREID][(INS_ID)*2]);          \
    }                                                                \
    __single_task __autorun void ntt_pop##COREID##NAME() {           \
        _ntt_forward(ch_ntt_elements_out[COREID][INS_ID],            \
                     ch_ntt_elements[COREID][(INS_ID)*2 + 1]);       \
    }

#define NTT1_INS(COREID, ENGINE_ID) NTT_INS(COREID, ENGINE_ID, ENGINE_ID)

__single_task __autorun void ntt1_backward() {
    unsigned data_index = 0;
    ntt_elements elements[CORES][MAX_RNS_MODULUS_SIZE];
    while (true) {
#pragma unroll
        for (int core = 0; core < CORES; core++) {
#pragma unroll
            for (int engine = 0; engine < MAX_RNS_MODULUS_SIZE; engine++) {
#pragma unroll
                for (int i = 0; i < VEC * 2 - 1; i++) {
                    elements[core][engine].data[i] =
                        elements[core][engine].data[i + 1];
                }
                elements[core][engine].data[VEC * 2 - 1] =
                    read_channel_intel(ch_ntt_elements_in[core][engine]);
                if (data_index == (VEC * 2 - 1)) {
                    write_channel_intel(ch_ntt_elements[core][engine * 2],
                                        elements[core][engine]);
                }
            }
        }
        data_index = (data_index + 1) % (VEC * 2);
    }
}

__single_task __autorun void ntt2_forward() {
    int data_index = 0;
    ntt_elements elements[CORES][MAX_RNS_MODULUS_SIZE];
    while (true) {
        if (data_index == 0) {
#pragma unroll
            for (int core = 0; core < CORES; core++) {
#pragma unroll
                for (int engine = 0; engine < MAX_RNS_MODULUS_SIZE; engine++) {
                    elements[core][engine] = read_channel_intel(
                        ch_ntt_elements[core][engine * 2 + 1]);
                }
            }
        }

#pragma unroll
        for (int core = 0; core < CORES; core++) {
#pragma unroll
            for (int engine = 0; engine < MAX_RNS_MODULUS_SIZE; engine++) {
                uint64_t data = elements[core][engine].data[0];
#pragma unroll
                for (int i = 0; i < VEC * 2 - 1; i++) {
                    elements[core][engine].data[i] =
                        elements[core][engine].data[i + 1];
                }
                write_channel_intel(ch_ntt_elements_out[core][engine], data);
            }
        }
        data_index = (data_index + 1) % (VEC * 2);
    }
}

#if MAX_RNS_MODULUS_SIZE > 0
NTT1_INS(0, 0)
#endif

#if MAX_RNS_MODULUS_SIZE > 1
NTT1_INS(0, 1)
#endif

#if MAX_RNS_MODULUS_SIZE > 2
NTT1_INS(0, 2)
#endif

#if MAX_RNS_MODULUS_SIZE > 3
NTT1_INS(0, 3)
#endif

#if MAX_RNS_MODULUS_SIZE > 4
NTT1_INS(0, 4)
#endif

#if MAX_RNS_MODULUS_SIZE > 5
NTT1_INS(0, 5)
#endif

// the last one should be careful
#if MAX_RNS_MODULUS_SIZE > 6
NTT1_INS(0, 6)
#endif

#if CORES > 1
#if MAX_RNS_MODULUS_SIZE > 0
NTT1_INS(1, 0)
#endif

#if MAX_RNS_MODULUS_SIZE > 1
NTT1_INS(1, 1)
#endif

#if MAX_RNS_MODULUS_SIZE > 2
NTT1_INS(1, 2)
#endif

#if MAX_RNS_MODULUS_SIZE > 3
NTT1_INS(1, 3)
#endif

#if MAX_RNS_MODULUS_SIZE > 4
NTT1_INS(1, 4)
#endif

#if MAX_RNS_MODULUS_SIZE > 5
NTT1_INS(1, 5)
#endif

// the last one should be careful
#if MAX_RNS_MODULUS_SIZE > 6
NTT1_INS(1, 6)
#endif
#endif
