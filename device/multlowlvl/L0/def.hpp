// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <CL/sycl.hpp>

namespace L0 {
template <unsigned int VEC>
struct WideVector_t {
  uint64_t data[VEC * 2];
};
template <unsigned int VEC>
struct TwiddleFactor_t {
  uint64_t data[VEC];
};
}  // namespace L0
