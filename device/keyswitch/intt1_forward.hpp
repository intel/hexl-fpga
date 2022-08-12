// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

template <class tt_kernelNameClass, class tt_ch_intt_elements_out,
          class tt_ch_intt_elements_out_rep, int COREID>
void _intt_broadcast(sycl::queue& q) {
    //
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            while (true) {
                using temp_pipe =
                    typename tt_ch_intt_elements_out::template PipeAt<COREID,
                                                                      0>;
                uint64_t data = temp_pipe::read();
                Unroller<0, MAX_RNS_MODULUS_SIZE>::Step([&](auto ins) {
                    using temp_pipe1 =
                        typename tt_ch_intt_elements_out_rep::template PipeAt<
                            COREID, ins>;
                    temp_pipe1::write(data);
                });
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}
