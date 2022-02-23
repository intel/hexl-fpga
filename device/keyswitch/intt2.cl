// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#define INTT_ENGINE2(COREID, NAME, ENGINE_ID)                               \
    __single_task __autorun void intt##COREID##NAME() {                     \
        _intt_internal2(ch_intt_modulus[COREID][ENGINE_ID],                 \
                        ch_intt2_elements[COREID][(ENGINE_ID - 1) * 2],     \
                        ch_intt2_elements[COREID][(ENGINE_ID - 1) * 2 + 1], \
                        ch_normalize[COREID][ENGINE_ID],                    \
                        ch_intt2_twiddle_factor[COREID][ENGINE_ID - 1], 2,  \
                        ENGINE_ID);                                         \
    }                                                                       \
    __single_task __kernel __autorun void intt_push##COREID##NAME() {       \
        _intt_backward2(ch_intt_elements_in[COREID][ENGINE_ID],             \
                        ch_intt2_elements[COREID][(ENGINE_ID - 1) * 2]);    \
    }                                                                       \
    __single_task __kernel __autorun void intt_pop##COREID##NAME() {        \
        _intt_forward2(ch_intt_elements_out_inter[COREID][ENGINE_ID],       \
                       ch_intt2_elements[COREID][(ENGINE_ID - 1) * 2 + 1]); \
    }                                                                       \
    __single_task __autorun void intt_norm##COREID##NAME() {                \
        _intt_normalize2(ch_intt_elements_out_inter[COREID][ENGINE_ID],     \
                         ch_intt_elements_out[COREID][ENGINE_ID],           \
                         ch_normalize[COREID][ENGINE_ID]);                  \
    }

INTT_ENGINE2(0, 1, 1)
INTT_ENGINE2(0, 2, 2)

#if CORES > 1
INTT_ENGINE2(1, 1, 1)
INTT_ENGINE2(1, 2, 2)
#endif
