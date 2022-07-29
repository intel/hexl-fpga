
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "pipes_keyswitch.hpp"

template <class tt_kernelNameClass, int tp_COREID, int tp_key_component,
          uint64_t tp_key_modulus_size, class tt_ch_ms_params,
          class tt_ch_t_poly_prod_iter, class tt_ch_ntt_elements_out,
          class tt_ch_result>
void _ms(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            const int InputModFactor = 8;
            [[intel::disable_loop_pipelining]] while (true) {
                using temp_pipe =
                    typename tt_ch_ms_params::template PipeAt<tp_COREID,
                                                              tp_key_component>;
                sycl::ulong4 moduli = temp_pipe::read();
                uint64_t decomp_modulus_index = moduli.s1();
                uint64_t modulus = moduli.s0() & MODULUS_BIT_MASK;
                uint64_t arg2 = moduli.s2();
                uint64_t rk = moduli.s3();
                unsigned coeff_count = GET_COEFF_COUNT(moduli.s0());
                for (unsigned j = 0; j < coeff_count; j++) {
                    uint64_t t_ith_poly;
                    switch (decomp_modulus_index) {
                    case 5: {
                        using pipe_read_5 =
                            typename tt_ch_t_poly_prod_iter::template PipeAt<
                                tp_COREID, 5, tp_key_component>;
                        t_ith_poly = pipe_read_5::read();
                    } break;
                    case 4: {
                        using pipe_read_4 =
                            typename tt_ch_t_poly_prod_iter::template PipeAt<
                                tp_COREID, 4, tp_key_component>;
                        t_ith_poly = pipe_read_4::read();
                    } break;
                    case 3: {
                        using pipe_read_3 =
                            typename tt_ch_t_poly_prod_iter::template PipeAt<
                                tp_COREID, 3, tp_key_component>;
                        t_ith_poly = pipe_read_3::read();
                    } break;
                    case 2: {
                        using pipe_read_2 =
                            typename tt_ch_t_poly_prod_iter::template PipeAt<
                                tp_COREID, 2, tp_key_component>;
                        t_ith_poly = pipe_read_2::read();
                    } break;
                    case 1: {
                        using pipe_read_1 =
                            typename tt_ch_t_poly_prod_iter::template PipeAt<
                                tp_COREID, 1, tp_key_component>;
                        t_ith_poly = pipe_read_1::read();
                    } break;
                    default: {
                        using pipe_read_0 =
                            typename tt_ch_t_poly_prod_iter::template PipeAt<
                                tp_COREID, 0, tp_key_component>;
                        t_ith_poly = pipe_read_0::read();
                    } break;
                    }

                    uint64_t twice_modulus = 2 * modulus;
                    uint64_t four_times_modulus = 4 * modulus;
                    uint64_t qi_lazy = modulus << 2;
                    using temp_pipe2 =
                        typename tt_ch_ntt_elements_out::template PipeAt<
                            tp_COREID, MAX_RNS_MODULUS_SIZE + tp_key_component>;
                    uint64_t data = temp_pipe2::read();
                    uint64_t in = t_ith_poly + qi_lazy - data;
                    uint64_t arg1_val =
                        ReduceMod(InputModFactor, in, modulus, &twice_modulus,
                                  &four_times_modulus);
                    using temp_pipe3 = typename tt_ch_result::template PipeAt<
                        tp_COREID, tp_key_component>;
                    uint64_t modRst =
                        MultiplyUIntMod(arg1_val, arg2, modulus, rk);
                    temp_pipe3::write(modRst);
                }
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}
