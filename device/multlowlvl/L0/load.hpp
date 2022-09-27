// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "number_theory.hpp"
#include "utils.hpp"

namespace L0 {
template <class kernelNameClass, class pipe_load, class pipe_prime_index,
          unsigned coeff_count>
event load(sycl::queue &q, sycl::event &depends, sycl::buffer<uint64_t> &data,
           sycl::buffer<uint8_t> &prime_index_set_buf) {
  int prime_size = prime_index_set_buf.size();
  event e = q.submit([&](handler &h) {
    h.depends_on(depends);
    accessor data_acc(data, h, read_only);
    accessor prime_index_set_acc(prime_index_set_buf, h, read_only);
    h.single_task<kernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      for (int p = 0; p < prime_size; p++) {
        uint8_t prime_index = prime_index_set_acc[p];
        pipe_prime_index::write(prime_index);
        for (unsigned i = 0; i < coeff_count; i++) {
          pipe_load::write(data_acc[p * coeff_count + i]);
        }
      }
    });
  });
  return e;
}

/*
template <class kernelNameClass, class pipe_load1, class pipe_prime_index1,
          class pipe_load2, class pipe_prime_index2, unsigned coeff_count>
event load2(sycl::queue &q, sycl::event &depends, sycl::buffer<uint64_t> &data1,
            sycl::buffer<uint64_t> &data2,
            sycl::buffer<uint8_t> &prime_index_set_buf1,
            sycl::buffer<uint8_t> &prime_index_set_buf2) {
  int prime_size1 = prime_index_set_buf1.size();
  int prime_size2 = prime_index_set_buf2.size();
  int max_prime_size = std::max(prime_size1, prime_size2);
  event e = q.submit([&](handler &h) {
    h.depends_on(depends);
    accessor data_acc1(data1, h, read_only);
    accessor data_acc2(data2, h, read_only);
    accessor prime_index_set_acc1(prime_index_set_buf1, h, read_only);
    accessor prime_index_set_acc2(prime_index_set_buf2, h, read_only);
    h.single_task<kernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      for (int p = 0; p < max_prime_size; p++) {
        if (p < prime_size1) {
          uint8_t prime_index1 = prime_index_set_acc1[p];
          pipe_prime_index1::write(prime_index1);
        }
        if (p < prime_size2) {
          uint8_t prime_index2 = prime_index_set_acc2[p];
          pipe_prime_index2::write(prime_index2);
        }

        for (unsigned i = 0; i < coeff_count; i++) {
          if (p < prime_size1) {
            pipe_load1::write(data_acc1[p * coeff_count + i]);
          }
          if (p < prime_size2) {
            pipe_load2::write(data_acc2[p * coeff_count + i]);
          }
        }
      }
    });
  });
  return e;
}
*/

template <class kernelNameClass, class pipe_prime_index>
event LoadPrimeIndexGeneric(sycl::queue &q,
                            sycl::uchar2 prime_index_start_end) {
  event e = q.submit([&](handler &h) {
    h.single_task<kernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      for (int prime_index = prime_index_start_end[0];
           prime_index <= prime_index_start_end[1]; prime_index++) {
        pipe_prime_index::write(prime_index);
      }
    });
  });
  return e;
}

template <class kernelNameClass, class pipe_prime_index, int times>
event LoadPrimeIndexGeneric2(sycl::queue &q,
                             sycl::buffer<uint8_t> &primes_index) {
  event e = q.submit([&](handler &h) {
    accessor primes_index_acc(primes_index, h, read_only);
    h.single_task<kernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      for (int t = 0; t < times; t++) {
        for (int i = 0; i < primes_index_acc.size(); i++) {
          pipe_prime_index::write(primes_index_acc[i]);
        }
      }
    });
  });
  return e;
}

template <class kernelNameClass, class pipe_load>
event generic_load(sycl::queue &q, sycl::buffer<uint64_t> &data) {
  unsigned size = data.size();
  event e = q.submit([&](handler &h) {
    accessor acc(data, h, read_only);
    h.single_task<kernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      for (unsigned i = 0; i < size; i++) {
        pipe_load::write(acc[i]);
      }
    });
  });
  return e;
}
}  // namespace L0
