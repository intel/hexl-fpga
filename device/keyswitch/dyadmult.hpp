
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "../mod_ops.hpp"
#include "pipes_keyswitch.hpp"
#include "../utils/unroller.hpp"
#include "../utils/kernel_assert.hpp"

// info Dyadic Multiplier kernel core definitions

// info Key broadcaster
template <class tt_kernelNameClass, class tt_ch_keyswitch_params,
          class tt_ch_dyadmult_keys, unsigned int TOTAL_NUM_CORES,
          unsigned int tp_MAX_RNS_MODULUS_SIZE>
void broadcast_keys(sycl::queue& q,
                    sycl::buffer<uint256_t>& buff_k_switch_keys1,
                    sycl::buffer<uint256_t>& buff_k_switch_keys2,
                    sycl::buffer<uint256_t>& buff_k_switch_keys3,
                    int batch_size) {
    auto qSubLambda = [&](sycl::handler& h) {
        sycl::accessor k_switch_keys1(buff_k_switch_keys1, h, sycl::read_only);
        sycl::accessor k_switch_keys2(buff_k_switch_keys2, h, sycl::read_only);
        sycl::accessor k_switch_keys3(buff_k_switch_keys3, h, sycl::read_only);
        const unsigned int KEYS_LEN = tp_MAX_RNS_MODULUS_SIZE * 2;
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            for (int i = 0; i < batch_size; i++) {
                unsigned params_size = tt_ch_keyswitch_params::read();
                for (int i = 0; i < params_size; i++) {
                    uint256_t keys1 = k_switch_keys1[i];
                    uint256_t keys2 = k_switch_keys2[i];
                    uint256_t keys3 = k_switch_keys3[i];
                    ulong keys[KEYS_LEN];
                    int j = 0;
                    keys[j++] = (keys1 & BIT_MASK_52).to_uint64();
                    keys[j++] = ((keys1 >> 52) & BIT_MASK_52).to_uint64();
                    keys[j++] = ((keys1 >> (52 * 2)) & BIT_MASK_52).to_uint64();
                    keys[j++] = ((keys1 >> (52 * 3)) & BIT_MASK_52).to_uint64();
                    keys[j++] = (((keys1 >> (52 * 4)) & BIT_MASK_52) |
                                 ((keys2 & BIT_MASK_4) << 48))
                                    .to_uint64();

                    keys[j++] = ((keys2 >> 4) & BIT_MASK_52).to_uint64();
                    keys[j++] = ((keys2 >> (4 + 52)) & BIT_MASK_52).to_uint64();
                    keys[j++] =
                        ((keys2 >> (4 + 52 * 2)) & BIT_MASK_52).to_uint64();
                    keys[j++] =
                        ((keys2 >> (4 + 52 * 3)) & BIT_MASK_52).to_uint64();
                    keys[j++] = (((keys2 >> (4 + 52 * 4)) & BIT_MASK_52) |
                                 ((keys3 & BIT_MASK_8) << 44))
                                    .to_uint64();

                    keys[j++] = ((keys3 >> 8) & BIT_MASK_52).to_uint64();
                    keys[j++] = ((keys3 >> (8 + 52)) & BIT_MASK_52).to_uint64();
                    keys[j++] =
                        ((keys3 >> (8 + 52 * 2)) & BIT_MASK_52).to_uint64();
                    keys[j++] =
                        ((keys3 >> (8 + 52 * 3)) & BIT_MASK_52).to_uint64();

                    Unroller<0, tp_MAX_RNS_MODULUS_SIZE>::Step([&](auto ins) {
                        sycl::ulong2 key;
                        key.s0() = keys[ins * 2];
                        key.s1() = keys[ins * 2 + 1];
                        ASSERT(key.s0() < MAX_KEY, "key > MAX_KEY\n");
                        ASSERT(key.s1() < MAX_KEY, "key > MAX_KEY\n");
                        Unroller<0, TOTAL_NUM_CORES>::Step([&](auto core) {
                            using outPipe =
                                typename tt_ch_dyadmult_keys::template PipeAt<
                                    core, ins>;
                            outPipe::write(key);
                        });
                    });
                }
            }
            //}
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}

// info core dyadicmult kernel definition
template <class tt_kernelNameClass, unsigned int tp_COREID, int tp_ntt_ins,
          unsigned tp_key_modulus_size, unsigned tp_key_component_count,
          class tt_ch_dyadmult_params, class tt_ch_ntt_elements_out,
          class tt_ch_dyadmult_keys, class tt_ch_intt_modulus,
          class tt_ch_t_poly_prod_iter, unsigned int TOTAL_NUM_CORES,
          unsigned int tp_MAX_RNS_MODULUS_SIZE>
void _dyadmult(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            uint64_t t_poly_lazy[MAX_COFF_COUNT][MAX_KEY_COMPONENT_SIZE];

            [[intel::disable_loop_pipelining]] while (true) {
                using temp_pipe0 =
                    typename tt_ch_dyadmult_params::template PipeAt<tp_COREID,
                                                                    tp_ntt_ins>;
                sycl::ulong4 curr_moduli = temp_pipe0::read();
                uint64_t decomp_modulus_index = curr_moduli.s1() & 0xf;
                uint64_t decomp_modulus_size = curr_moduli.s1() >> 4;

                if (tp_ntt_ins == (tp_key_modulus_size - 1) &&
                    decomp_modulus_index == 0) {
                    Unroller<0, MAX_KEY_COMPONENT_SIZE>::Step(
                        [&](auto key_component) {
                            using pipe =
                                typename tt_ch_intt_modulus::template PipeAt<
                                    tp_COREID, 1 + key_component>;
                            pipe::write(curr_moduli);
                        });
                }
                unsigned coeff_count = GET_COEFF_COUNT(curr_moduli.s0());
                [[intel::ivdep]] for (unsigned j = 0; j < coeff_count; j++) {
                    using temp_valpipe =
                        typename tt_ch_ntt_elements_out::template PipeAt<
                            tp_COREID, tp_ntt_ins>;
                    uint64_t val = temp_valpipe::read();
                    using temp_keyspipe =
                        typename tt_ch_dyadmult_keys::template PipeAt<
                            tp_COREID, tp_ntt_ins>;
                    sycl::ulong2 keys = temp_keyspipe::read();
                    Unroller<0, tp_key_component_count>::Step([&](auto k) {
                        ASSERT(((ulong*)&keys)[k] < MAX_MODULUS,
                               "y >= modulus\n");
                        uint64_t prod =
                            MultiplyUIntMod(val, ((ulong*)&keys)[k],
                                            curr_moduli.s0() & MODULUS_BIT_MASK,
                                            curr_moduli.s3());

                        uint64_t prev = t_poly_lazy[j][k];
                        prev = decomp_modulus_index == 0 ? 0 : prev;

                        ASSERT(prod < (curr_moduli.s0() & MODULUS_BIT_MASK),
                               "x >= modulus, engine_id = %d\n", tp_ntt_ins);
                        uint64_t sum = AddUIntMod(
                            prod, prev, curr_moduli.s0() & MODULUS_BIT_MASK);
                        t_poly_lazy[j][k] = sum;

                        // save in the last iteration
                        // only the decomp engines and the last engine are valid
                        bool valid_engine =
                            tp_ntt_ins < decomp_modulus_size ||
                            tp_ntt_ins == (tp_key_modulus_size - 1);
                        if (decomp_modulus_index == (decomp_modulus_size - 1) &&
                            valid_engine) {
                            // 0 - n, 7n - 8n
                            // n - 2n, 8n - 9n
                            // 2n - 3n, 9n - 10n
                            // ...
                            // 6n - 7n, 13n - 14n
                            // overall 14n
                            using pipe_wr1 = typename tt_ch_t_poly_prod_iter::
                                template PipeAt<tp_COREID, tp_ntt_ins, k>;
                            pipe_wr1::write(sum);
                        }
                    });
                }
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}

// info core dyadicmult kernel definition
template <class tt_kernelNameClass, unsigned int tp_COREID,
          int tp_ntt_ins,  // last instance specialization
          unsigned tp_key_modulus_size, unsigned tp_key_component_count,
          class tt_ch_dyadmult_params, class tt_ch_ntt_elements_out,
          class tt_ch_dyadmult_keys, class tt_ch_intt_modulus,
          class tt_ch_t_poly_prod_iter_last, unsigned int TOTAL_NUM_CORES,
          unsigned int tp_MAX_RNS_MODULUS_SIZE>
void _dyadmult_last_stage(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            uint64_t t_poly_lazy[MAX_COFF_COUNT][MAX_KEY_COMPONENT_SIZE];

            [[intel::disable_loop_pipelining]] while (true) {
                using temp_pipe0 =
                    typename tt_ch_dyadmult_params::template PipeAt<tp_COREID,
                                                                    tp_ntt_ins>;
                sycl::ulong4 curr_moduli = temp_pipe0::read();
                uint64_t decomp_modulus_index = curr_moduli.s1() & 0xf;
                uint64_t decomp_modulus_size = curr_moduli.s1() >> 4;

                if (tp_ntt_ins == (tp_key_modulus_size - 1) &&
                    decomp_modulus_index == 0) {
                    Unroller<0, MAX_KEY_COMPONENT_SIZE>::Step(
                        [&](auto key_component) {
                            using pipe =
                                typename tt_ch_intt_modulus::template PipeAt<
                                    tp_COREID, 1 + key_component>;
                            pipe::write(curr_moduli);
                        });
                }
                unsigned coeff_count = GET_COEFF_COUNT(curr_moduli.s0());
                [[intel::ivdep]] for (unsigned j = 0; j < coeff_count; j++) {
                    using temp_valpipe =
                        typename tt_ch_ntt_elements_out::template PipeAt<
                            tp_COREID, tp_ntt_ins>;
                    uint64_t val = temp_valpipe::read();
                    using temp_keyspipe =
                        typename tt_ch_dyadmult_keys::template PipeAt<
                            tp_COREID, tp_ntt_ins>;
                    sycl::ulong2 keys = temp_keyspipe::read();
                    Unroller<0, tp_key_component_count>::Step([&](auto k) {
                        ASSERT(((ulong*)&keys)[k] < MAX_MODULUS,
                               "y >= modulus\n");
                        uint64_t prod =
                            MultiplyUIntMod(val, ((ulong*)&keys)[k],
                                            curr_moduli.s0() & MODULUS_BIT_MASK,
                                            curr_moduli.s3());

                        uint64_t prev = t_poly_lazy[j][k];
                        prev = decomp_modulus_index == 0 ? 0 : prev;

                        ASSERT(prod < (curr_moduli.s0() & MODULUS_BIT_MASK),
                               "x >= modulus, engine_id = %d\n", tp_ntt_ins);
                        uint64_t sum = AddUIntMod(
                            prod, prev, curr_moduli.s0() & MODULUS_BIT_MASK);
                        t_poly_lazy[j][k] = sum;

                        bool valid_engine =
                            tp_ntt_ins < decomp_modulus_size ||
                            tp_ntt_ins == (tp_key_modulus_size - 1);
                        if (decomp_modulus_index == (decomp_modulus_size - 1) &&
                            valid_engine) {
                            // 0 - n, 7n - 8n
                            // n - 2n, 8n - 9n
                            // 2n - 3n, 9n - 10n
                            // ...
                            // 6n - 7n, 13n - 14n
                            // overall 14n
                            using pipe_wr_last =
                                typename tt_ch_t_poly_prod_iter_last::
                                    template PipeAt<tp_COREID, k>;
                            pipe_wr_last::write(sum);
                        }
                    });
                }
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}
