// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "channels.h"

#define INTT_ENGINE(COREID, NAME, ENGINE_ID)                           \
    __single_task __autorun void intt##COREID##NAME() {                \
        _intt_internal(ch_intt_modulus[COREID][ENGINE_ID],             \
                       ch_intt_elements[COREID][ENGINE_ID * 2],        \
                       ch_intt_elements[COREID][ENGINE_ID * 2 + 1],    \
                       ch_normalize[COREID][ENGINE_ID],                \
                       ch_intt1_twiddle_factor[COREID][ENGINE_ID], 2,  \
                       ENGINE_ID);                                     \
    }                                                                  \
    __single_task __kernel __autorun void intt_push##COREID##NAME() {  \
        _intt_backward(ch_intt_elements_in[COREID][ENGINE_ID],         \
                       ch_intt_elements[COREID][ENGINE_ID * 2]);       \
    }                                                                  \
    __single_task __kernel __autorun void intt_pop##COREID##NAME() {   \
        _intt_forward(ch_intt_elements_out_inter[COREID][ENGINE_ID],   \
                      ch_intt_elements[COREID][ENGINE_ID * 2 + 1]);    \
    }                                                                  \
    __single_task __autorun void intt_norm##COREID##NAME() {           \
        _intt_normalize(ch_intt_elements_out_inter[COREID][ENGINE_ID], \
                        ch_intt_elements_out[COREID][ENGINE_ID],       \
                        ch_normalize[COREID][ENGINE_ID]);              \
    }

INTT_ENGINE(0, 0, 0)

#if CORES > 1
INTT_ENGINE(1, 0, 0)
#endif
