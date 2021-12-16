// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "channels.h"

void _ms(moduli_t moduli, uint64_t coeff_count, uint64_t decomp_modulus_size,
         uint64_t key_modulus_size, int key_component) {
    unsigned j = 0;
    unsigned i = 0;
    uint64_t modulus;
    uint64_t arg2;
    uint64_t rk;
    int InputModFactor = 8;

    while (true) {
        if (j == 0) {
            ulong4 moduli_cur = moduli.data[i];
            modulus = moduli_cur.s0;
            arg2 = moduli_cur.s2;
            rk = moduli_cur.s3;
        }
        uint64_t t_ith_poly;
        switch (i) {
        case 5:
            t_ith_poly =
                read_channel_intel(ch_t_poly_prod_iter[5][key_component]);
            break;
        case 4:
            t_ith_poly =
                read_channel_intel(ch_t_poly_prod_iter[4][key_component]);
            break;
        case 3:
            t_ith_poly =
                read_channel_intel(ch_t_poly_prod_iter[3][key_component]);
            break;
        case 2:
            t_ith_poly =
                read_channel_intel(ch_t_poly_prod_iter[2][key_component]);
            break;
        case 1:
            t_ith_poly =
                read_channel_intel(ch_t_poly_prod_iter[1][key_component]);
            break;
        default:
            t_ith_poly =
                read_channel_intel(ch_t_poly_prod_iter[0][key_component]);
            break;
        }

        uint64_t twice_modulus = 2 * modulus;
        uint64_t four_times_modulus = 4 * modulus;
        uint64_t qi_lazy = modulus << 2;

        uint64_t data = read_channel_intel(
            ch_ntt_elements_out[MAX_RNS_MODULUS_SIZE + key_component]);
        uint64_t in = t_ith_poly + qi_lazy - data;
        uint64_t arg1_val = ReduceMod(InputModFactor, in, modulus,
                                      &twice_modulus, &four_times_modulus);
        write_channel_intel(ch_result[key_component],
                            MultiplyUIntMod(arg1_val, arg2, modulus, rk));
        STEP(j, coeff_count);
        if (j == 0) {
            STEP(i, decomp_modulus_size);
        }
    }
}

__single_task void ms1(moduli_t moduli, uint64_t coeff_count,
                       uint64_t decomp_modulus_size,
                       uint64_t key_modulus_size) {
    _ms(moduli, coeff_count, decomp_modulus_size, key_modulus_size, 0);
}

__single_task void ms2(moduli_t moduli, uint64_t coeff_count,
                       uint64_t decomp_modulus_size,
                       uint64_t key_modulus_size) {
    _ms(moduli, coeff_count, decomp_modulus_size, key_modulus_size, 1);
}
