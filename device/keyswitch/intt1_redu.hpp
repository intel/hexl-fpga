// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "../../common/types.hpp"
template <class tt_kernelNameClass, class tt_ch_intt_redu_params,
          class tt_ch_ntt_modulus, class tt_ch_intt_elements_out_rep,
          class tt_ch_ntt_elements_in>
void intt1_redu(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            [[intel::disable_loop_pipelining]] while (true) {
                sycl::ulong4 moduli[NUM_CORES][MAX_RNS_MODULUS_SIZE];
                Unroller<0, NUM_CORES>::Step([&](auto core) {
                    //
                    Unroller<0, MAX_RNS_MODULUS_SIZE>::Step([&](auto i) {
                        using pipe =
                            typename tt_ch_intt_redu_params::template PipeAt<
                                core, i>;
                        moduli[core][i] = pipe::read();
                        using pipe2 =
                            typename tt_ch_ntt_modulus::template PipeAt<core,
                                                                        i>;
                        pipe2::write(moduli[core][i]);
                    });
                });

                unsigned coeff_count = GET_COEFF_COUNT(moduli[0][0].s0());
                for (unsigned j = 0; j < coeff_count; j++) {
                    Unroller<0, NUM_CORES>::Step([&](auto core) {
                        Unroller<0, MAX_RNS_MODULUS_SIZE>::Step([&](auto i) {
                            using pipe = typename tt_ch_intt_elements_out_rep::
                                template PipeAt<core, i>;
                            uint64_t val = pipe::read();
                            uint64_t val_redu = BarrettReduce64(
                                val, moduli[core][i].s0() & MODULUS_BIT_MASK,
                                moduli[core][i].s1());
                            using pipe2 =
                                typename tt_ch_ntt_elements_in::template PipeAt<
                                    core, i>;
                            pipe2::write(val_redu);
                        });
                    });
                }
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}
