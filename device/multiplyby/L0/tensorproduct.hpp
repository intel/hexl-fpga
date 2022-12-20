// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "number-theory.hpp"
#include "utils.hpp"

namespace L0 {
template <class pipe_ready, int N, class pipe_output>
sycl::event TensorProduct(sycl::queue &q, sycl::buffer<ulong> &a0,
                          sycl::buffer<ulong> &a1, sycl::buffer<ulong> &b0,
                          sycl::buffer<ulong> &b1, sycl::buffer<ulong> &c01,
                          sycl::buffer<ulong> &c2, sycl::buffer<ulong4> &primes,
                          unsigned num_primes, int offset1, int offset2,
                          int offset3, int offset4, sycl::event &depend_event,
                          int flag) {
  unsigned c1_offset = num_primes * N;
  event e = q.submit([&](handler &h) {
    // h.depends_on(depend_event);
    accessor acc_primes(primes, h, read_only);
#if 0
    accessor acc_a0(a0, h, read_only);
    accessor acc_a1(a1, h, read_only);
    accessor acc_b0(b0, h, read_only);
    accessor acc_b1(b1, h, read_only);

    accessor acc_c0(c01, h, write_only, sycl::no_init);
    accessor acc_c1(c01, h, write_only, sycl::no_init);
    accessor acc_c2(c2, h, write_only, sycl::no_init);
#endif
    h.single_task<class TensorProductV2>([=]() [[intel::kernel_args_restrict]] {
      // wait on the blocking pipe_read until notified by the producer
      uint64_t *data_ptr = (uint64_t *)pipe_ready::read();

      // use atomic_fence to ensure memory ordering
      atomic_fence(memory_order::seq_cst, memory_scope::device);

      ulong a[4];
      ulong4 prime;
      auto acc_a0_ushort = reinterpret_cast<ushort *>(data_ptr + offset1);
      auto acc_a1_ushort = reinterpret_cast<ushort *>(data_ptr + offset2);
      auto acc_b0_ushort = reinterpret_cast<ushort *>(data_ptr + offset3);
      auto acc_b1_ushort = reinterpret_cast<ushort *>(data_ptr + offset4);

      for (int j = 0; j < N * 4 * num_primes; j++) {
        int i = j / (N * 4);
        if (j % (N * 4) == 0) prime = acc_primes[i];
        int k = j % 4;
        if (k == 0) {
          a[0] = 0;
          a[1] = 0;
          a[2] = 0;
          a[3] = 0;
        }
        ulong b[4];
        b[0] = BIT(flag, 0) ? acc_a0_ushort[j] : 0;
        b[1] = BIT(flag, 1) ? acc_a1_ushort[j] : 0;
        b[2] = BIT(flag, 2) ? acc_b0_ushort[j] : 0;
        b[3] = BIT(flag, 3) ? acc_b1_ushort[j] : 0;

        a[0] >>= 16;
        a[0] |= (b[0] << 48);
        a[1] >>= 16;
        a[1] |= (b[1] << 48);
        a[2] >>= 16;
        a[2] |= (b[2] << 48);
        a[3] >>= 16;
        a[3] |= (b[3] << 48);

        int offset = j / 4;
        if (k == 3) {
          ulong c[3];
          c[0] =
              MultiplyUIntMod(a[0], a[2], prime.s0(), prime.s1(), prime.s2());
          auto part1 =
              MultiplyUIntMod(a[0], a[3], prime.s0(), prime.s1(), prime.s2());
          auto part2 =
              MultiplyUIntMod(a[1], a[2], prime.s0(), prime.s1(), prime.s2());
          c[1] = MOD_ONCE(part1 + part2, prime.s0());
          c[2] =
              MultiplyUIntMod(a[1], a[3], prime.s0(), prime.s1(), prime.s2());
          pipe_output::write({c[0], c[1], c[2], 0});
        }
      }
    });
  });
  return e;
}

template <class KernelNameClass, class pipe_output, class pipe_data_ptr_c01,
          class pipe_completed>
event StoreTensorProduct(sycl::queue &q, sycl::buffer<uint64_t> &data1,
                         sycl::buffer<uint64_t> &data2,
                         sycl::buffer<uint64_t> &data3, unsigned size,
                         int flag) {
  event e = q.submit([&](handler &h) {
    accessor acc1(data1, h, sycl::write_only, sycl::no_init);
    accessor acc2(data2, h, sycl::write_only, sycl::no_init);
    accessor acc3(data3, h, sycl::write_only, sycl::no_init);
    h.single_task<KernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      for (unsigned i = 0; i < size; i++) {
        auto tmp = pipe_output::read();
        if (BIT(flag, 0)) {
          acc1[i] = tmp.s0();
        }
        if (BIT(flag, 1)) {
          acc2[size + i] = tmp.s1();
        }
        if (BIT(flag, 2)) {
          acc3[i] = tmp.s2();
        }
      }
      uint64_t *c2_ptr = acc3.get_pointer().get();
      uint64_t *c0_ptr = acc1.get_pointer().get();
      // use atomic_fence to ensure memory ordering
      atomic_fence(memory_order::seq_cst, memory_scope::device);
      pipe_completed::write((uint64_t)c2_ptr);
      pipe_data_ptr_c01::write((uint64_t)c0_ptr);
    });
  });
  return e;
}
}  // namespace L0
