// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "number-theory.hpp"
#include "utils.hpp"

namespace L0 {
template <class kernelNameClass, class pipe_store>
event store(sycl::queue &q, sycl::buffer<uint64_t> &data, unsigned size) {
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

template <class kernelNameClass, class pipe_store, class pipe_offset,
          uint coeff_count>
sycl::event store(sycl::queue &q, sycl::buffer<uint64_t> &data, unsigned size,
                  int flag = 0xff) {
  event e = q.submit([&](handler &h) {
    accessor acc(data, h, sycl::write_only, sycl::no_init);
    h.single_task<kernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      uint read_size = 0;
      while (read_size < size) {
        uint offset = BIT(flag, 0) ? pipe_offset::read() : 0;
        for (uint i = 0; i < coeff_count; i++) {
          uint64_t data = BIT(flag, 1) ? pipe_store::read() : 0;
          if (BIT(flag, 2)) {
            acc[offset * coeff_count + i] = data;
          }
        }
        read_size += coeff_count;
      }
    });
  });
  return e;
}
}  // namespace L0
