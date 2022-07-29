
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "intt_core.hpp"
#include "../utils/unroller.hpp"
template <int tp_numCores, int tp_startEngineID>
class _intt_backward_kernelNameClass;

template <int tp_numCores, int tp_startEngineID>
class _intt_internal_kernelNameClass;
template <int tp_numCores, int tp_startEngineID>
class _intt_forward_kernelNameClass;

template <int tp_numCores, int tp_startEngineID>
class _intt_normalize_kernelNameClass;

template <int tp_numCores, int tp_startEngineID, class tt_ch_intt_modulus,
          class tt_ch_intt_elements, class tt_ch_normalize,
          class tt_ch_intt1_twiddle_factor, class tt_ch_intt_elements_in,
          class tt_ch_intt_elements_out_inter, class tt_ch_intt_elements_out>

void _intt_1(sycl::queue& q) {
    //
    Unroller<0, tp_numCores>::Step([&](auto coreNum) {
        using tt_ch_intt_modulus_pipe =
            typename tt_ch_intt_modulus::template PipeAt<coreNum,
                                                         tp_startEngineID>;
        using tt_ch_intt_elements_in_pipeIn =
            typename tt_ch_intt_elements::template PipeAt<coreNum,
                                                          tp_startEngineID * 2>;
        using tt_ch_intt_elements_in_pipeOut =
            typename tt_ch_intt_elements::template PipeAt<
                coreNum, tp_startEngineID * 2 + 1>;
        using tt_ch_normalize_pipe =
            typename tt_ch_normalize::template PipeAt<coreNum,
                                                      tp_startEngineID>;
        using tt_ch_intt1_twiddle_factor_pipe =
            typename tt_ch_intt1_twiddle_factor::template PipeAt<
                coreNum, tp_startEngineID>;

        _intt_internal<
            _intt_internal_kernelNameClass<coreNum, tp_startEngineID>,
            tt_ch_intt_modulus_pipe, tt_ch_intt_elements_in_pipeIn,
            tt_ch_intt_elements_in_pipeOut, tt_ch_normalize_pipe,
            tt_ch_intt1_twiddle_factor_pipe, 2, tp_startEngineID>(q);

        // INFO: Serial to Parallel Conversion
        // Generate handle to individual pipes for instance id and core number
        using tt_ch_intt_elements_in_pipe =
            typename tt_ch_intt_elements_in::template PipeAt<coreNum,
                                                             tp_startEngineID>;
        _intt_backward<
            _intt_backward_kernelNameClass<coreNum, tp_startEngineID>,
            tt_ch_intt_elements_in_pipe, tt_ch_intt_elements_in_pipeIn>(q);

        // Generate handle to individual pipes for instance id and core number
        using ch_intt_elements_out_inter_pipe =
            typename ch_intt_elements_out_inter::template PipeAt<
                coreNum, tp_startEngineID>;
        _intt_forward<_intt_forward_kernelNameClass<coreNum, tp_startEngineID>,
                      ch_intt_elements_out_inter_pipe,
                      tt_ch_intt_elements_in_pipeOut>(q);

        // INFO:
        // Generate handle to individual pipes for instance id and core number
        using tt_ch_intt_elements_out_pipe =
            typename tt_ch_intt_elements_out::template PipeAt<coreNum,
                                                              tp_startEngineID>;

        _intt_normalize<
            _intt_normalize_kernelNameClass<coreNum, tp_startEngineID>,
            ch_intt_elements_out_inter_pipe, tt_ch_intt_elements_out_pipe,
            tt_ch_normalize_pipe>(q);
    });
    //
}
