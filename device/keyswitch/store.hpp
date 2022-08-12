
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "pipes_keyswitch.hpp"
template <unsigned int iid>
class _store_kernelNameClass;
template <class tt_kernelNameClass = _store_kernelNameClass<0>>
sycl::event store(sycl::queue& q, sycl::event* inDepsEv,
                  sycl::buffer<sycl::ulong2>& buffer_result_in,
                  uint64_t num_batch, uint64_t coeff_count,
                  uint64_t decomp_modulus_size, moduli_t moduli, unsigned rmem,
                  unsigned wmem) {
    auto qSubLambda = [&](sycl::handler& h) {
        if (inDepsEv) {
            for (size_t evn = 0; evn < num_batch; evn++) {
                inDepsEv[evn].wait();
            }
        }
        sycl::accessor dp_resulta(buffer_result_in, h, sycl::write_only,
                                  sycl::no_init);

        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            unsigned j = 0;
            unsigned k = 0;
            uint64_t modulus;
            unsigned num_batch_per_core = (num_batch - 1) / NUM_CORES + 1;
            unsigned max_ptr = num_batch * decomp_modulus_size * coeff_count;
            sycl::device_ptr<sycl::ulong2> dp_result(dp_resulta);
            [[intel::ivdep]] for (unsigned i = 0;
                                  i < num_batch_per_core * coeff_count *
                                          decomp_modulus_size;
                                  i++) {
                if (j == 0) {
                    modulus = moduli.data[k].s0();
                    STEP(k, decomp_modulus_size);
                }

                Unroller<0, NUM_CORES>::Step([&](auto core) {
                    unsigned core_ptr = num_batch_per_core *
                                        decomp_modulus_size * coeff_count *
                                        core;
                    unsigned ptr = core_ptr + i;
                    using temp_pipe1 =
                        typename ch_result::template PipeAt<core, 0>;
                    uint64_t res1 = temp_pipe1::read();
                    using temp_pipe2 =
                        typename ch_result::template PipeAt<core, 1>;
                    uint64_t res2 = temp_pipe2::read();

#ifdef SUM_RESULT
                    sycl::ulong2 input =
                        (ptr < max_ptr && rmem) ? dp_result[ptr] : 0;
                    res1 += input.s0();
                    res2 += input.s1();
                    res1 = res1 >= modulus ? res1 - modulus : res1;
                    res2 = res2 >= modulus ? res2 - modulus : res2;
#endif
                    sycl::ulong2 output;
                    output.s0() = res1;
                    output.s1() = res2;
                    if (wmem && ptr < max_ptr) {
                        dp_result[ptr] = output;
                    }
                });
                STEP(j, coeff_count);
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    auto event = q.submit(qSubLambda);
    return event;
}
