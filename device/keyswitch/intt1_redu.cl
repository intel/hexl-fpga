// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "channels.h"

__single_task __autorun void intt1_redu() {
#pragma disable_loop_pipelining
    while (true) {
        ulong4 moduli[CORES][MAX_RNS_MODULUS_SIZE];
#pragma unroll
        for (int core = 0; core < CORES; core++) {
#pragma unroll
            for (int i = 0; i < MAX_RNS_MODULUS_SIZE; i++) {
                moduli[core][i] =
                    read_channel_intel(ch_intt_redu_params[core][i]);
                write_channel_intel(ch_ntt_modulus[core][i], moduli[core][i]);
            }
        }
        unsigned coeff_count = GET_COEFF_COUNT(moduli[0][0].s0);
        for (unsigned j = 0; j < coeff_count; j++) {
#pragma unroll
            for (int core = 0; core < CORES; core++) {
#pragma unroll
                for (int i = 0; i < MAX_RNS_MODULUS_SIZE; i++) {
                    uint64_t val =
                        read_channel_intel(ch_intt_elements_out_rep[core][i]);
                    uint64_t val_redu = BarrettReduce64(
                        val, moduli[core][i].s0 & MODULUS_BIT_MASK,
                        moduli[core][i].s1);
                    write_channel_intel(ch_ntt_elements_in[core][i], val_redu);
                }
            }
        }
    }
}
