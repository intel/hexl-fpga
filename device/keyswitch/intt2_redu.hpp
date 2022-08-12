
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "../utils/kernel_assert.hpp"

template <class tt_kernelNameClass, int COREID, int key_component,
          uint64_t key_modulus_size, class tt_ch_intt2_redu_params,
          class tt_ch_intt_elements_out, class tt_ch_ntt_modulus,
          class tt_ch_ntt_elements_in>
void _intt2_redu(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            uint64_t elements[MAX_COFF_COUNT];

            [[intel::disable_loop_pipelining]] while (true) {
                using temp_pipe =
                    typename tt_ch_intt2_redu_params::template PipeAt<
                        COREID, key_component>;
                sycl::ulong4 moduli = temp_pipe::read();
                uint64_t decomp_modulus_index = moduli.s2() & 0xf;
                uint64_t qk = moduli.s2() >> 4;
                uint64_t qk_half = qk >> 1;
                uint64_t qi = moduli.s0() & MODULUS_BIT_MASK;
                using temp_pipe2 = typename tt_ch_ntt_modulus::template PipeAt<
                    COREID, MAX_RNS_MODULUS_SIZE + key_component>;
                temp_pipe2::write(moduli);
                uint64_t barrett_factor = moduli.s1();
                uint64_t fix =
                    qi - BarrettReduce64(qk_half, qi, barrett_factor);
                unsigned coeff_count = GET_COEFF_COUNT(moduli.s0());

                for (int j = 0; j < coeff_count; j++) {
                    uint64_t val;
                    if (decomp_modulus_index == 0) {
                        using temp_pipe3 =
                            typename tt_ch_intt_elements_out::template PipeAt<
                                COREID, 1 + key_component>;
                        val = temp_pipe3::read();
                        ASSERT(val < qk, "x >= modulus\n");
                        val = AddUIntMod(val, qk_half, qk);
                        elements[j] = val;
                    } else {
                        val = elements[j];
                    }
                    // TO BE CONFIRMED: add the fix before the barrett reduce
                    val += fix;
                    uint64_t val_redu =
                        BarrettReduce64(val, qi, barrett_factor);
                    using temp_pipe4 =
                        typename tt_ch_ntt_elements_in::template PipeAt<
                            COREID, MAX_RNS_MODULUS_SIZE + key_component>;
                    temp_pipe4::write(val_redu);
                }
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}
