// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "number-theory.hpp"
#include "utils.hpp"

namespace L0 {
template <int N>
sycl::event TensorProduct(sycl::queue &q, sycl::buffer<ulong> &a0,
                          sycl::buffer<ulong> &a1, sycl::buffer<ulong> &b0,
                          sycl::buffer<ulong> &b1, sycl::buffer<ulong> &c0,
                          sycl::buffer<ulong> &c1, sycl::buffer<ulong> &c2,
                          sycl::buffer<ulong4> &primes, unsigned num_primes,
                          int offset1, int offset2, int offset3, int offset4,
                          sycl::event &depend_event, int flag) {
  event e = q.submit([&](handler &h) {
    h.depends_on(depend_event);
    accessor acc_primes(primes, h, read_only);
    accessor acc_a0(a0, h, read_only);
    accessor acc_a1(a1, h, read_only);
    accessor acc_b0(b0, h, read_only);
    accessor acc_b1(b1, h, read_only);
    accessor acc_c0(c0, h, write_only, sycl::no_init);
    accessor acc_c1(c1, h, write_only, sycl::no_init);
    accessor acc_c2(c2, h, write_only, sycl::no_init);
    h.single_task<class TensorProductV2>([=]() [[intel::kernel_args_restrict]] {
      ulong a[4];
      ulong4 prime;
      auto acc_a0_ushort =
          reinterpret_cast<ushort *>(acc_a0.get_pointer().get() + offset1);
      auto acc_a1_ushort =
          reinterpret_cast<ushort *>(acc_a1.get_pointer().get() + offset2);
      auto acc_b0_ushort =
          reinterpret_cast<ushort *>(acc_b0.get_pointer().get() + offset3);
      auto acc_b1_ushort =
          reinterpret_cast<ushort *>(acc_b1.get_pointer().get() + offset4);

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
          if (BIT(flag, 4)) acc_c0[offset] = c[0];
          if (BIT(flag, 5)) acc_c1[offset] = c[1];
          if (BIT(flag, 6)) acc_c2[offset] = c[2];
        }
      }
    });
  });
  return e;
}
}  // namespace L0
