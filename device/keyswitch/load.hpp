
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __LOAD_HPP__
#define __LOAD_HPP__

#include "pipes_keyswitch.hpp"
#include "../utils/pipe_array.hpp"
#include "../utils/unroller.hpp"

template <int id = 22786>
class load_kernelNameClass;
template <class tt_kernelNameClass = load_kernelNameClass<>>
sycl::event load(sycl::queue& q, sycl::event* inDepsEv,
                 sycl::buffer<uint64_t>& buff_t_target_iter_ptr,
                 moduli_t moduli_in, uint64_t coeff_count,
                 uint64_t decomp_modulus_size, uint64_t num_batch, invn_t inv_n,
                 unsigned rmem) {
    auto qSubLambda = [&](sycl::handler& h) {
        if (inDepsEv) {
            for (size_t evn = 0; evn < num_batch; evn++) {
                inDepsEv[evn].wait();
            }
        }
        sycl::accessor dp_t_target_iter_ptr(buff_t_target_iter_ptr, h,
                                            sycl::read_only);
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            moduli_t moduli = moduli_in;
            unsigned i = 0;
            sycl::device_ptr<uint64_t> t_target_iter_ptr(dp_t_target_iter_ptr);
            unsigned ptr_index[NUM_CORES];
            unsigned num_batch_per_core = (num_batch - 1) / NUM_CORES + 1;
            unsigned max_ptr = num_batch * decomp_modulus_size * coeff_count;
#pragma unroll
            for (int i = 0; i < NUM_CORES; i++) {
                ptr_index[i] =
                    num_batch_per_core * decomp_modulus_size * coeff_count * i;
            }
            unsigned decomp_index = 0;
#pragma unroll
            for (int i = 0; i < MAX_KEY_MODULUS_SIZE; i++) {
                moduli.data[i].s0() = moduli.data[i].s0() |
                                      ((coeff_count >> 10) << MAX_MODULUS_BITS);
            }

            [[intel::disable_loop_pipelining]] for (unsigned j = 0;
                                                    j < decomp_modulus_size *
                                                            num_batch_per_core;
                                                    j++) {
                if (decomp_index == 0) {
                    ch_ntt2_decomp_size::write(decomp_modulus_size *
                                               coeff_count);
                    ch_intt1_decomp_size::write(decomp_modulus_size *
                                                coeff_count);
                    ch_keyswitch_params::write(decomp_modulus_size *
                                               coeff_count);
                }

                Unroller<0, NUM_CORES>::Step([&](auto COREID) {
                    Unroller<0, MAX_RNS_MODULUS_SIZE>::Step([&](auto engid) {
                        using temp_pipe =
                            typename ch_intt_redu_params::template PipeAt<
                                COREID, engid>;
                        temp_pipe::write(moduli.data[engid]);
                    });

                    sycl::ulong4 ms_params = moduli.data[decomp_index];
                    ms_params.s1() = decomp_index;

                    Unroller<0, MAX_KEY_COMPONENT_SIZE>::Step([&](auto engid) {
                        using temp_pipe =
                            typename ch_ms_params::template PipeAt<COREID,
                                                                   engid>;
                        temp_pipe::write(ms_params);
                    });

                    sycl::ulong4 intt2_redu_params = moduli.data[decomp_index];
                    intt2_redu_params.s2() =
                        ((moduli.data[MAX_KEY_MODULUS_SIZE - 1].s0() &
                          MODULUS_BIT_MASK)
                         << 4) |
                        decomp_index;

                    Unroller<0, MAX_KEY_COMPONENT_SIZE>::Step([&](auto engid) {
                        using temp_pipe =
                            typename ch_intt2_redu_params::template PipeAt<
                                COREID, engid>;
                        temp_pipe::write(intt2_redu_params);
                    });

                    sycl::ulong4 dyadmult_params;
                    Unroller<0, MAX_RNS_MODULUS_SIZE>::Step([&](auto engid) {
                        dyadmult_params = moduli.data[engid];
                        dyadmult_params.s1() =
                            (decomp_modulus_size << 4) | decomp_index;
                        dyadmult_params.s2() = inv_n.data[engid].s0();
                        using temp_pipe =
                            typename ch_dyadmult_params::template PipeAt<COREID,
                                                                         engid>;
                        temp_pipe::write(dyadmult_params);
                    });

                    sycl::ulong4 cur_moduli = moduli.data[decomp_index];
                    cur_moduli.s2() = inv_n.data[decomp_index].s0();
                    using temp_pipe =
                        typename ch_intt_modulus::template PipeAt<COREID, 0>;

                    temp_pipe::write(cur_moduli);
                });
                STEP(decomp_index, decomp_modulus_size);

                for (int n = 0; n < coeff_count; n++) {
                    Unroller<0, NUM_CORES>::Step([&](auto COREID) {
                        using temp_pipe =
                            typename ch_intt_elements_in::template PipeAt<
                                COREID, 0>;
                        uint64_t toWrite;
                        if (ptr_index[COREID] < max_ptr) {
                            toWrite = t_target_iter_ptr[ptr_index[COREID]];
                        } else
                            toWrite = 0;
                        temp_pipe::write(toWrite);
                        ptr_index[COREID]++;
                    });
                }
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    auto event = q.submit(qSubLambda);
    return event;
}

#endif
