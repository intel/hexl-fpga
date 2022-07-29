// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pipes_keyswitch.hpp"
#include "ntt_core.hpp"
template <unsigned int iid, unsigned int coreid, unsigned int ins_id>
class _ntt2_itnernal_kernelNameClass;
template <unsigned int iid, unsigned int coreid, unsigned int ins_id>
class _ntt2_backward_kernelNameClass;
template <unsigned int iid, unsigned int coreid, unsigned int ins_id>
class _ntt2_forward_kernelNameClass;

template <class tt_ch_ntt_modulus, class tt_ch_ntt_elements,
          class tt_ch_twiddle_factor, class tt_ch_ntt_elements_in,
          class tt_ch_ntt_elements_out, unsigned int TOTAL_NUM_CORES,
          unsigned int INS_ID_START, unsigned int INDS_ID_END>
void NTT2_INS_generator(sycl::queue& q) {
    Unroller<0, TOTAL_NUM_CORES>::Step([&](auto coreNum) {
        Unroller<INS_ID_START, INDS_ID_END + 1>::Step([&](auto iid) {
            using pipeMod =
                typename tt_ch_ntt_modulus::template PipeAt<coreNum, iid>;  // 1
            using pipeElemsIn =
                typename tt_ch_ntt_elements::template PipeAt<coreNum,
                                                             iid * 2>;  // 2
            using pipeElemsOut =
                typename tt_ch_ntt_elements::template PipeAt<coreNum,
                                                             iid * 2 + 1>;  // 3
            using pipeTwiddles =
                typename tt_ch_twiddle_factor::template PipeAt<coreNum,
                                                               iid>;  // 4
            using pipeElemsIn2 =
                typename tt_ch_ntt_elements_in::template PipeAt<coreNum,
                                                                iid>;  // 5
            using pipeElemsOut2 =
                typename tt_ch_ntt_elements_out::template PipeAt<coreNum,
                                                                 iid>;  // 6

            _ntt_internal<_ntt2_itnernal_kernelNameClass<0, coreNum, iid>,
                          pipeMod, pipeElemsIn, pipeElemsOut, pipeTwiddles, iid,
                          4>(q);

            _ntt_backward<_ntt2_backward_kernelNameClass<0, coreNum, iid>,
                          pipeElemsIn2, pipeElemsIn>(q);
            _ntt_forward<_ntt2_forward_kernelNameClass<0, coreNum, iid>,
                         pipeElemsOut2, pipeElemsOut>(q);
        });
    });
}
