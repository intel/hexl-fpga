// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "number-theory.hpp"
#include "utils.hpp"

namespace L0 {
template <class kernelNameClass, class pipe_load, class pipe_prime_index,
          unsigned coeff_count>
event load(sycl::queue &q, sycl::buffer<uint64_t> &data,
           sycl::buffer<uint8_t> &prime_index_set_buf, unsigned prime_size,
           int flag) {
  event e = q.submit([&](handler &h) {
    accessor data_acc(data, h, read_only);
    accessor prime_index_set_acc(prime_index_set_buf, h, read_only);
    h.single_task<kernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      auto data_ptr = data_acc.get_pointer().get();
      for (int p = 0; p < prime_size; p++) {
        using PipelinedLSU = ext::intel::lsu<>;
        uint8_t prime_index =
            PipelinedLSU::load(prime_index_set_acc.get_pointer() + p);
        pipe_prime_index::write(prime_index);
        for (unsigned i = 0; i < coeff_count; i++) {
          pipe_load::write(BIT(flag, 0) ? data_ptr[i] : 0);
        }
        data_ptr += coeff_count;
      }
    });
  });
  return e;
}

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
    int primes_index_acc_size = primes_index_acc.size();
    h.single_task<kernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      for (int t = 0; t < times; t++) {
        for (int i = 0; i < primes_index_acc_size; i++) {
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

template <class kernelNameClass, class pipe_load, class pipe_prime_index,
          int coeff_count>
event generic_load_with_prime_index(sycl::queue &q,
                                    sycl::buffer<uint64_t> &data,
                                    unsigned size) {
  int num_primes = size / coeff_count;
  event e = q.submit([&](handler &h) {
    accessor acc(data, h, read_only);
    h.single_task<kernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      for (int j = 0; j < num_primes; j++) {
        pipe_prime_index::write(j);
        for (int i = 0; i < coeff_count; i++) {
          pipe_load::write(acc[j * coeff_count + i]);
        }
      }
    });
  });
  return e;
}

template <class kernelNameClass, class pipe_load, class pipe_prime_index,
          class pipe_ready, int coeff_count>
event generic_load_with_prime_index(sycl::queue &q,
                                    sycl::buffer<uint64_t> &data, unsigned size,
                                    int flag, sycl::event depend) {
  int num_primes = size / coeff_count;
  event e = q.submit([&](handler &h) {
    // h.depends_on(depend);
    accessor acc(data, h, read_only);
    h.single_task<kernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      // wait on the blocking pipe_read until notified by the producer
      uint64_t data_ptr_int = pipe_ready::read();
      uint64_t *data_ptr = (uint64_t *)data_ptr_int;
      // auto data_ptr = acc.get_pointer().get();

      // use atomic_fence to ensure memory ordering
      atomic_fence(memory_order::seq_cst, memory_scope::device);
      for (int j = 0; j < num_primes; j++) {
        pipe_prime_index::write(j);
        for (int i = 0; i < coeff_count; i++) {
          pipe_load::write(BIT(flag, 0) ? data_ptr[i] : 0);
        }
        data_ptr += coeff_count;
      }
    });
  });
  return e;
}

template <class kernelNameClass, class pipe_prime, class T, int times>
event LoadPrimesGeneric(sycl::queue &q, sycl::buffer<T> &primes) {
  event e = q.submit([&](handler &h) {
    accessor primes_acc(primes, h, read_only);
    int primes_acc_size = primes_acc.size();
    h.single_task<kernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      for (int t = 0; t < times; t++) {
        for (int i = 0; i < primes_acc_size; i++) {
          pipe_prime::write(primes_acc[i]);
        }
      }
    });
  });
  return e;
}
}  // namespace L0
