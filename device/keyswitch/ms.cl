// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "channels.h"

void _ms(int COREID, uint64_t key_modulus_size, int key_component) {
    const int InputModFactor = 8;

#pragma disable_loop_pipelining
    while (true) {
        ulong4 moduli = read_channel_intel(ch_ms_params[COREID][key_component]);
        uint64_t decomp_modulus_index = moduli.s1;
        uint64_t modulus = moduli.s0 & MODULUS_BIT_MASK;
        uint64_t arg2 = moduli.s2;
        uint64_t modulus_r = moduli.s3;
        unsigned char modulus_k = gen_modulus_k(modulus);
        unsigned coeff_count = GET_COEFF_COUNT(moduli.s0);

        for (unsigned j = 0; j < coeff_count; j++) {
            uint64_t t_ith_poly;
            switch (decomp_modulus_index) {
            case 5:
                t_ith_poly = read_channel_intel(
                    ch_t_poly_prod_iter[COREID][5][key_component]);
                break;
            case 4:
                t_ith_poly = read_channel_intel(
                    ch_t_poly_prod_iter[COREID][4][key_component]);
                break;
            case 3:
                t_ith_poly = read_channel_intel(
                    ch_t_poly_prod_iter[COREID][3][key_component]);
                break;
            case 2:
                t_ith_poly = read_channel_intel(
                    ch_t_poly_prod_iter[COREID][2][key_component]);
                break;
            case 1:
                t_ith_poly = read_channel_intel(
                    ch_t_poly_prod_iter[COREID][1][key_component]);
                break;
            default:
                t_ith_poly = read_channel_intel(
                    ch_t_poly_prod_iter[COREID][0][key_component]);
                break;
            }

            uint64_t twice_modulus = 2 * modulus;
            uint64_t four_times_modulus = 4 * modulus;
            uint64_t qi_lazy = modulus << 2;

            uint64_t data = read_channel_intel(
                ch_ntt_elements_out[COREID]
                                   [MAX_RNS_MODULUS_SIZE + key_component]);
            uint64_t in = t_ith_poly + qi_lazy - data;
            uint64_t arg1_val = ReduceMod(InputModFactor, in, modulus,
                                          &twice_modulus, &four_times_modulus);
            write_channel_intel(
                ch_result[COREID][key_component],
                MultiplyUIntMod(arg1_val, arg2, modulus, modulus_r, modulus_k));
        }
    }
}

__single_task __autorun void ms00() { _ms(0, MAX_KEY_MODULUS_SIZE, 0); }

__single_task __autorun void ms01() { _ms(0, MAX_KEY_MODULUS_SIZE, 1); }

#if CORES > 1
__single_task __autorun void ms10() { _ms(1, MAX_KEY_MODULUS_SIZE, 0); }

__single_task __autorun void ms11() { _ms(1, MAX_KEY_MODULUS_SIZE, 1); }
#endif
