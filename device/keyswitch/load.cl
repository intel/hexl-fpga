// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "channels.h"

__single_task void load(__global HOST_MEM uint64_t* restrict t_target_iter_ptr,
                        moduli_t moduli, uint64_t coeff_count,
                        uint64_t decomp_modulus_size, uint64_t num_batch,
                        __global DEVICE_MEM uint256_t* restrict k_switch_keys1,
                        __global DEVICE_MEM uint256_t* restrict k_switch_keys2,
                        __global DEVICE_MEM uint256_t* restrict k_switch_keys3,
                        __global HOST_MEM uint64_t* restrict twiddle_factors,
                        invn_t inv_n, unsigned reload_twiddle_factors,
                        unsigned rmem) {
    unsigned i = 0;
    unsigned coeff_count_int = coeff_count;

    // forward the twiddle factors
    if (reload_twiddle_factors) {
        twiddle_factors_t tfs = {true, coeff_count_int, twiddle_factors};
        write_channel_intel(ch_twiddle_factors, tfs);
    }

    unsigned ptr_index[CORES];
    unsigned num_batch_per_core = (num_batch - 1) / CORES + 1;
    unsigned max_ptr = num_batch * decomp_modulus_size * coeff_count_int;

#pragma unroll
    for (int i = 0; i < CORES; i++) {
        ptr_index[i] =
            num_batch_per_core * decomp_modulus_size * coeff_count_int * i;
    }

    unsigned decomp_index = 0;

#pragma unroll
    for (int i = 0; i < MAX_KEY_MODULUS_SIZE; i++) {
        moduli.data[i].s0 =
            moduli.data[i].s0 | ((coeff_count >> 10) << MAX_MODULUS_BITS);
    }

#pragma disable_loop_pipelining
    for (unsigned j = 0; j < decomp_modulus_size * num_batch_per_core; j++) {
        keyswitch_params params = {decomp_modulus_size * coeff_count_int,
                                   k_switch_keys1, k_switch_keys2,
                                   k_switch_keys3};
        // only broadcast only once
        if (decomp_index == 0) {
            write_channel_intel(
                ch_ntt2_decomp_size,
                decomp_modulus_size * coeff_count_int / VEC - 1);
            write_channel_intel(
                ch_intt1_decomp_size,
                decomp_modulus_size * coeff_count_int / VEC - 1);
            write_channel_intel(ch_keyswitch_params, params);
        }
#pragma unroll
        for (int COREID = 0; COREID < CORES; COREID++) {
#pragma unroll
            for (int engid = 0; engid < MAX_RNS_MODULUS_SIZE; engid++) {
                write_channel_intel(ch_intt_redu_params[COREID][engid],
                                    moduli.data[engid]);
            }

            ulong4 ms_params = moduli.data[decomp_index];
            ms_params.s1 = decomp_index;

#pragma unroll
            for (int engid = 0; engid < MAX_KEY_COMPONENT_SIZE; engid++) {
                write_channel_intel(ch_ms_params[COREID][engid], ms_params);
            }

            ulong4 intt2_redu_params = moduli.data[decomp_index];
            intt2_redu_params.s2 =
                ((moduli.data[MAX_KEY_MODULUS_SIZE - 1].s0 & MODULUS_BIT_MASK)
                 << 4) |
                decomp_index;

#pragma unroll
            for (int engid = 0; engid < MAX_KEY_COMPONENT_SIZE; engid++) {
                write_channel_intel(ch_intt2_redu_params[COREID][engid],
                                    intt2_redu_params);
            }

            ulong4 dyadmult_params;
#pragma unroll
            for (int engid = 0; engid < MAX_RNS_MODULUS_SIZE; engid++) {
                dyadmult_params = moduli.data[engid];
                dyadmult_params.s1 = (decomp_modulus_size << 4) | decomp_index;
                dyadmult_params.s2 = inv_n.data[engid].s0;
                write_channel_intel(ch_dyadmult_params[COREID][engid],
                                    dyadmult_params);
            }

            ulong4 cur_moduli = moduli.data[decomp_index];
            cur_moduli.s2 = inv_n.data[decomp_index].s0;
            write_channel_intel(ch_intt_modulus[COREID][0], cur_moduli);
        }
        STEP(decomp_index, decomp_modulus_size);

        for (unsigned n = 0; n < coeff_count_int; n++) {
#pragma unroll
            for (int COREID = 0; COREID < CORES; COREID++) {
                write_channel_intel(ch_intt_elements_in[COREID][0],
                                    ptr_index[COREID] < max_ptr
                                        ? t_target_iter_ptr[ptr_index[COREID]]
                                        : 0);
                ptr_index[COREID]++;
            }
        }
    }
}
