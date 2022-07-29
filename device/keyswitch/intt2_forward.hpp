
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "../../common/types.hpp"
template <class tt_kernelNameClass, int COREID, int key_component,
          class tt_ch_t_poly_prod_iter_last, class tt_ch_intt_elements_in>
void _intt2_forward(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            while (true) {
                using temp_pipe =
                    typename tt_ch_t_poly_prod_iter_last::template PipeAt<
                        COREID, key_component>;
                uint64_t data = temp_pipe::read();
                using temp_pipe2 =
                    typename tt_ch_intt_elements_in::template PipeAt<
                        COREID, 1 + key_component>;
                temp_pipe2::write(data);
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}
