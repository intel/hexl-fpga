// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "channels.h"

#define INTT_ENGINE(NAME, ENGINE_ID, key_modulus_start, key_modulus_end)       \
    __single_task void intt##NAME(                                             \
        __global HOST_MEM uint64_t* restrict twiddle_factors, invn_t inv_n,    \
        unsigned int key_modulus_size) {                                       \
        uint54_t local_roots[FPGA_NTT_SIZE / VEC *                             \
                             (key_modulus_end - (key_modulus_start))][VEC];    \
        _intt_internal(ch_intt_modulus[ENGINE_ID],                             \
                       ch_intt_elements[ENGINE_ID * 2],                        \
                       ch_intt_elements[ENGINE_ID * 2 + 1],                    \
                       ch_normalize[ENGINE_ID], twiddle_factors, inv_n,        \
                       local_roots, key_modulus_start, key_modulus_end, 2, 0); \
    }                                                                          \
    __single_task __kernel __autorun void intt_backward##NAME() {              \
        _intt_backward(ch_intt_elements_in[ENGINE_ID],                         \
                       ch_intt_elements[ENGINE_ID * 2]);                       \
    }                                                                          \
    __single_task __kernel __autorun void intt_forward##NAME() {               \
        _intt_forward(ch_intt_elements_out_inter[ENGINE_ID],                   \
                      ch_intt_elements[ENGINE_ID * 2 + 1]);                    \
    }                                                                          \
    __single_task __autorun void intt_normalize##NAME() {                      \
        _intt_normalize(ch_intt_elements_out_inter[ENGINE_ID],                 \
                        ch_intt_elements_out[ENGINE_ID],                       \
                        ch_normalize[ENGINE_ID]);                              \
    }

INTT_ENGINE(1, 0, 0, MAX_KEY_MODULUS_SIZE - 1)
