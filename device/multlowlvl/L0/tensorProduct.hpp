// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "number_theory.hpp"
#include "utils.hpp"

namespace L0 {
template <int N, class PipeInput1, class PipeInput2, class PipeOutput0,
          class PipeOutput12>
event TensorProduct(sycl::queue &q, sycl::buffer<ulong4> &primes) {
  int num_primes = primes.size();
  auto p_cache_buf = new sycl::buffer<ulong2>(primes.size() * N);
  p_cache_buf->set_write_back(false);
  event e = q.submit([&](handler &h) {
    accessor primes_acc(primes, h, read_only);
    accessor cache_acc(*p_cache_buf, h, read_write);
    h.single_task<class TensorProduct2>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < num_primes * 2; i++) {
        int j = i >= num_primes ? i - num_primes : i;
        auto prime = primes_acc[j];
        PRINTF("Process Prime %d/%d\n", i, num_primes * 2);
        for (int n = 0; n < N; n++) {
          ulong a = PipeInput1::read();
          ulong c = PipeInput2::read();
          // cache the first polynomial to DDR
          if (i < num_primes) {
            // the size of cache is num_primes * N
            cache_acc[j * N + n] = {a, c};
            // a*c
            ulong c0 =
                MultiplyUIntMod(a, c, prime.s0(), prime.s1(), prime.s2());
            // output_c0_acc[j * N + n] = c0;
            PipeOutput0::write(c0);
          } else {
            ulong b = a;
            ulong d = c;
            ulong2 ac = cache_acc[j * N + n];
            a = ac.s0();
            c = ac.s1();

            // a*d+b*c
            ulong c1 =
                MultiplyUIntMod(a, d, prime.s0(), prime.s1(), prime.s2()) +
                MultiplyUIntMod(b, c, prime.s0(), prime.s1(), prime.s2());
            c1 = MOD_ONCE(c1, prime.s0());
            // b*d
            ulong c2 =
                MultiplyUIntMod(b, d, prime.s0(), prime.s1(), prime.s2());
            // output_c1_acc[j * N + n] = c1;
            // output_c2_acc[j * N + n] = c2;
            PipeOutput12::write({c1, c2});
          }
        }
      }
    });
  });
  return e;
}
}  // namespace L0
