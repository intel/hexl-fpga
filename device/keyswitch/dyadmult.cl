// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "channels.h"

__single_task __autorun void broadcast_keys() {
#pragma disable_loop_pipelining
    while (true) {
        keyswitch_params params = read_channel_intel(ch_keyswitch_params);

        __global uint256_t* restrict k_switch_keys1 = params.k_switch_keys1;
        __global uint256_t* restrict k_switch_keys2 = params.k_switch_keys2;
        __global uint256_t* restrict k_switch_keys3 = params.k_switch_keys3;

        for (int i = 0; i < params.size; i++) {
            uint256_t keys1 = k_switch_keys1[i];
            uint256_t keys2 = k_switch_keys2[i];
            uint256_t keys3 = k_switch_keys3[i];

            ulong keys[MAX_RNS_MODULUS_SIZE * 2];
            int j = 0;
            keys[j++] = keys1 & BIT_MASK_52;
            keys[j++] = (keys1 >> 52) & BIT_MASK_52;
            keys[j++] = (keys1 >> (52 * 2)) & BIT_MASK_52;
            keys[j++] = (keys1 >> (52 * 3)) & BIT_MASK_52;
            keys[j++] = ((keys1 >> (52 * 4)) & BIT_MASK_52) |
                        ((keys2 & BIT_MASK_4) << 48);

            keys[j++] = (keys2 >> 4) & BIT_MASK_52;
            keys[j++] = (keys2 >> (4 + 52)) & BIT_MASK_52;
            keys[j++] = (keys2 >> (4 + 52 * 2)) & BIT_MASK_52;
            keys[j++] = (keys2 >> (4 + 52 * 3)) & BIT_MASK_52;
            keys[j++] = ((keys2 >> (4 + 52 * 4)) & BIT_MASK_52) |
                        ((keys3 & BIT_MASK_8) << 44);

            keys[j++] = (keys3 >> 8) & BIT_MASK_52;
            keys[j++] = (keys3 >> (8 + 52)) & BIT_MASK_52;
            keys[j++] = (keys3 >> (8 + 52 * 2)) & BIT_MASK_52;
            keys[j++] = (keys3 >> (8 + 52 * 3)) & BIT_MASK_52;

#pragma unroll
            for (int ins = 0; ins < MAX_RNS_MODULUS_SIZE; ins++) {
                ulong2 key;
                key.s0 = keys[ins * 2];
                key.s1 = keys[ins * 2 + 1];
                ASSERT(key.s0 < MAX_KEY, "key > MAX_KEY\n");
                ASSERT(key.s1 < MAX_KEY, "key > MAX_KEY\n");
#pragma unroll
                for (int core = 0; core < CORES; core++) {
                    write_channel_intel(ch_dyadmult_keys[core][ins], key);
                }
            }
        }
    }
}

void _dyadmult(int COREID, unsigned key_modulus_size,
               unsigned key_component_count, int ntt_ins) {
    uint64_t t_poly_lazy[MAX_COFF_COUNT][MAX_KEY_COMPONENT_SIZE];

#pragma disable_loop_pipelining
    while (true) {
        ulong4 curr_moduli =
            read_channel_intel(ch_dyadmult_params[COREID][ntt_ins]);
        uint64_t decomp_modulus_index = curr_moduli.s1 & 0xf;
        uint64_t decomp_modulus_size = curr_moduli.s1 >> 4;

        if (ntt_ins == (key_modulus_size - 1) && decomp_modulus_index == 0) {
#pragma unroll
            for (int key_component = 0; key_component < MAX_KEY_COMPONENT_SIZE;
                 key_component++) {
                write_channel_intel(ch_intt_modulus[COREID][1 + key_component],
                                    curr_moduli);
            }
        }
        unsigned coeff_count = GET_COEFF_COUNT(curr_moduli.s0);

#pragma ivdep
        for (unsigned j = 0; j < coeff_count; j++) {
            uint64_t val =
                read_channel_intel(ch_ntt_elements_out[COREID][ntt_ins]);
            ulong2 keys = read_channel_intel(ch_dyadmult_keys[COREID][ntt_ins]);

#pragma unroll
            for (unsigned k = 0; k < key_component_count; ++k) {
                ASSERT(((ulong*)&keys)[k] < MAX_MODULUS, "y >= modulus\n");
                uint64_t prod = MultiplyUIntMod(
                    val, ((ulong*)&keys)[k], curr_moduli.s0 & MODULUS_BIT_MASK,
                    curr_moduli.s3);

                uint64_t prev = t_poly_lazy[j][k];
                prev = decomp_modulus_index == 0 ? 0 : prev;

                ASSERT(prod < (curr_moduli.s0 & MODULUS_BIT_MASK),
                       "x >= modulus, engine_id = %d\n", ntt_ins);
                uint64_t sum =
                    AddUIntMod(prod, prev, curr_moduli.s0 & MODULUS_BIT_MASK);
                t_poly_lazy[j][k] = sum;

                // save in the last iteration
                // only the decomp engines and the last engine are valid
                bool valid_engine = ntt_ins < decomp_modulus_size ||
                                    ntt_ins == (key_modulus_size - 1);
                if (decomp_modulus_index == (decomp_modulus_size - 1) &&
                    valid_engine) {
                    // 0 - n, 7n - 8n
                    // n - 2n, 8n - 9n
                    // 2n - 3n, 9n - 10n
                    // ...
                    // 6n - 7n, 13n - 14n
                    // overall 14n
                    if (ntt_ins < (MAX_RNS_MODULUS_SIZE - 1)) {
                        write_channel_intel(
                            ch_t_poly_prod_iter[COREID][ntt_ins][k], sum);
                    } else {
                        write_channel_intel(ch_t_poly_prod_iter_last[COREID][k],
                                            sum);
                    }
                }
            }
        }
    }
}

#define DYADMULT_INS(COREID, INS_ID)                                    \
    __single_task __autorun void dyadmult##COREID##INS_ID() {           \
        _dyadmult(COREID, MAX_KEY_MODULUS_SIZE, MAX_KEY_COMPONENT_SIZE, \
                  INS_ID);                                              \
    }

#if MAX_RNS_MODULUS_SIZE > 0
DYADMULT_INS(0, 0)
#endif

#if MAX_RNS_MODULUS_SIZE > 1
DYADMULT_INS(0, 1)
#endif

#if MAX_RNS_MODULUS_SIZE > 2
DYADMULT_INS(0, 2)
#endif

#if MAX_RNS_MODULUS_SIZE > 3
DYADMULT_INS(0, 3)
#endif

#if MAX_RNS_MODULUS_SIZE > 4
DYADMULT_INS(0, 4)
#endif

#if MAX_RNS_MODULUS_SIZE > 5
DYADMULT_INS(0, 5)
#endif

#if MAX_RNS_MODULUS_SIZE > 6
DYADMULT_INS(0, 6)
#endif

#if CORES > 1
#if MAX_RNS_MODULUS_SIZE > 0
DYADMULT_INS(1, 0)
#endif

#if MAX_RNS_MODULUS_SIZE > 1
DYADMULT_INS(1, 1)
#endif

#if MAX_RNS_MODULUS_SIZE > 2
DYADMULT_INS(1, 2)
#endif

#if MAX_RNS_MODULUS_SIZE > 3
DYADMULT_INS(1, 3)
#endif

#if MAX_RNS_MODULUS_SIZE > 4
DYADMULT_INS(1, 4)
#endif

#if MAX_RNS_MODULUS_SIZE > 5
DYADMULT_INS(1, 5)
#endif

#if MAX_RNS_MODULUS_SIZE > 6
DYADMULT_INS(1, 6)
#endif
#endif
