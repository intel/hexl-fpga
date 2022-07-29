
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "pipes_keyswitch.hpp"
#include "ntt_core.hpp"
#include "../utils/pipe_array.hpp"
#include "../utils/unroller.hpp"

template <class tt_kernelNameClass, class tt_ch_ntt_elements_in,
          class tt_ch_ntt_elements, int TOTAL_NUM_CORES>
void ntt1_backward(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            unsigned data_index = 0;
            WideVec_t elements[TOTAL_NUM_CORES][MAX_RNS_MODULUS_SIZE];
            while (true) {
                Unroller<0, TOTAL_NUM_CORES>::Step([&](auto core) {
                    Unroller<0, MAX_RNS_MODULUS_SIZE>::Step([&](auto engine) {
#pragma unroll
                        for (int i = 0; i < VEC * 2 - 1; i++) {
                            elements[core][engine].data[i] =
                                elements[core][engine].data[i + 1];
                        }
                        using readPipe1 =
                            typename tt_ch_ntt_elements_in::template PipeAt<
                                core, engine>;
                        elements[core][engine].data[VEC * 2 - 1] =
                            readPipe1::read();
                        if (data_index == (VEC * 2 - 1)) {
                            using writePipe1 =
                                typename tt_ch_ntt_elements::template PipeAt<
                                    core, engine * 2>;
                            writePipe1::write(elements[core][engine]);
                        }
                    });
                });
                data_index = (data_index + 1) % (VEC * 2);
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}

template <class tt_kernelNameClass, class tt_ch_ntt_elements,
          class tt_ch_ntt_elements_out, int TOTAL_NUM_CORES>
void ntt2_forward(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            int data_index = 0;
            WideVec_t elements[TOTAL_NUM_CORES][MAX_RNS_MODULUS_SIZE];
            while (true) {
                if (data_index == 0) {
                    Unroller<0, TOTAL_NUM_CORES>::Step([&](auto core) {
                        Unroller<0, MAX_RNS_MODULUS_SIZE>::Step(
                            [&](auto engine) {
                                using readPipe1 = typename tt_ch_ntt_elements::
                                    template PipeAt<core, engine * 2 + 1>;
                                elements[core][engine] = readPipe1::read();
                            });
                    });
                }
                Unroller<0, TOTAL_NUM_CORES>::Step([&](auto core) {
                    Unroller<0, MAX_RNS_MODULUS_SIZE>::Step([&](auto engine) {
                        uint64_t data = elements[core][engine].data[0];
#pragma unroll
                        for (int i = 0; i < VEC * 2 - 1; i++) {
                            elements[core][engine].data[i] =
                                elements[core][engine].data[i + 1];
                        }
                        using writePipe1 =
                            typename tt_ch_ntt_elements_out::template PipeAt<
                                core, engine>;
                        writePipe1::write(data);
                    });
                });
                data_index = (data_index + 1) % (VEC * 2);
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}

template <class tt_kernelNameClass

          >
void inttStreamingKernel(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]]{};
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}

// INFO::  ntt1.cl NTT_INS macro replacement will be used to generate ntts for
// multiple cores

template <unsigned int iid, unsigned int coreid, unsigned int ins_id>
class ntt_internal_kernelNameClass;

template <class tt_ch_ntt_modulus, class tt_ch_ntt_elements,
          class tt_ch_twiddle_factor, unsigned int TOTAL_NUM_CORES,
          unsigned int TOTAL_INS>
void NTT_INS_generator(sycl::queue& q) {
    Unroller<0, TOTAL_NUM_CORES>::Step([&](auto coreNum) {
        Unroller<0, TOTAL_INS>::Step([&](auto iid) {
            using pipeMod =
                typename tt_ch_ntt_modulus::template PipeAt<coreNum, iid>;
            using pipeElemsIn =
                typename tt_ch_ntt_elements::template PipeAt<coreNum, iid * 2>;
            using pipeElemsOut =
                typename tt_ch_ntt_elements::template PipeAt<coreNum,
                                                             iid * 2 + 1>;
            using pipeTwiddles =
                typename tt_ch_twiddle_factor::template PipeAt<coreNum, iid>;

            _ntt_internal<ntt_internal_kernelNameClass<0, coreNum, iid>,
                          pipeMod, pipeElemsIn, pipeElemsOut, pipeTwiddles, iid,
                          4>(q);
        });
    });
}
