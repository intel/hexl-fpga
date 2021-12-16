// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "channels.h"

__single_task void store(__global HOST_MEM ulong2* restrict result,
                         uint64_t num_batch, uint64_t coeff_count,
                         uint64_t decomp_modulus_size, moduli_t moduli,
                         unsigned rmem, unsigned wmem) {
    unsigned j = 0;
    unsigned k = 0;
    uint64_t modulus;

#pragma ivdep array(result)
    for (unsigned i = 0; i < num_batch * coeff_count * decomp_modulus_size;
         i++) {
        if (j == 0) {
            modulus = moduli.data[k].s0;
            STEP(k, decomp_modulus_size);
        }
        uint64_t res1 = read_channel_intel(ch_result[0]);
        uint64_t res2 = read_channel_intel(ch_result[1]);

#ifdef SUM_RESULT
        ulong2 input = rmem ? result[i] : 999;
        res1 += input.s0;
        res2 += input.s1;
        res1 = res1 >= modulus ? res1 - modulus : res1;
        res2 = res2 >= modulus ? res2 - modulus : res2;
#endif

        ulong2 output;
        output.s0 = res1;
        output.s1 = res2;
        if (wmem) {
            result[i] = output;
        }
        STEP(j, coeff_count);
    }
}
