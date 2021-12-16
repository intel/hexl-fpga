// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "channels.h"

__kernel void load(__global HOST_MEM uint64_t* restrict t_target_iter_ptr,
                   moduli_t moduli, uint64_t coeff_count,
                   uint64_t decomp_modulus_size, uint64_t num_batch,
                   unsigned rmem) {
    unsigned n = 0;
    unsigned decomp_index = 0;
    for (unsigned int i = 0; i < coeff_count * num_batch * decomp_modulus_size;
         i++) {
        if (n == 0) {
            ulong4 cur_moduli = moduli.data[decomp_index];
            cur_moduli.s2 = decomp_index;
            write_channel_intel(ch_intt_modulus[0], cur_moduli);
            STEP(decomp_index, decomp_modulus_size);
        }
        write_channel_intel(ch_intt_elements_in[0],
                            rmem ? t_target_iter_ptr[i] : 999);
        STEP(n, coeff_count);
    }
}
