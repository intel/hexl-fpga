// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "number_theory.hpp"
#include "utils.hpp"

namespace L0 {
template <class kernelNameClass, class pipe_store>
event store(sycl::queue &q, sycl::buffer<uint64_t> &data) {
  unsigned size = data.size();
  event e = q.submit([&](handler &h) {
    accessor acc(data, h, sycl::write_only, sycl::no_init);
    h.single_task<kernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      for (unsigned i = 0; i < size; i++) {
        acc[i] = pipe_store::read();
      }
    });
  });
  return e;
}

template <class kernelNameClass, class pipe_store>
event store2(sycl::queue &q, sycl::buffer<uint64_t> &data1,
             sycl::buffer<uint64_t> &data2) {
  unsigned size = data1.size();
  assert(data1.size() == data2.size());
  event e = q.submit([&](handler &h) {
    accessor acc1(data1, h, sycl::write_only, sycl::no_init);
    accessor acc2(data2, h, sycl::write_only, sycl::no_init);
    h.single_task<kernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      for (unsigned i = 0; i < size; i++) {
        ulong2 tmp = pipe_store::read();
        acc1[i] = tmp.s0();
        acc2[i] = tmp.s1();
      }
    });
  });
  return e;
}
}  // namespace L0
