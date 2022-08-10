
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef VEC
#define VEC_LOG 3
#define VEC (1 << VEC_LOG)
#endif

using NTTElements_t = uint64_t;
template <unsigned int tp_width = 2, unsigned int tp_vec = VEC>
struct WideVector_t {
    static const int DATA_LEN = VEC * tp_width;
    NTTElements_t data[DATA_LEN];
};

using TwiddleFactor2_t = WideVector_t<1, 2>;
using TwiddleFactor_t = WideVector_t<1>;
using WideVec_t = WideVector_t<2, VEC>;
using WideVecINTT2_t = WideVector_t<2, VEC_INTT2>;
