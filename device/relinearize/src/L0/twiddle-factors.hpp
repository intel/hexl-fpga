// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "number-theory.hpp"
#include "utils.hpp"

// enable generate the twiddle factors on the fly to reduce
// the memory bandwidth requirement
#define TF_ON_THE_FLY 1

namespace L0 {
template <class kernelNameClass, class pipe_prime_index,
          class pipe_prime_index_next, class pipe_twiddle_factor, unsigned VEC,
          unsigned coeff_count>
event TwiddleFactor(sycl::queue &q, const std::vector<uint64_t> &tf_set,
                    int flag) {
  // the input vector should not be freed. Keep it with static or never freed
  // memory before application exit. Otherwise the accessor will get random
  // values
  auto buf_tf = new buffer<uint64_t>(tf_set.size(),
                                     {sycl::property::buffer::mem_channel{1}});
  event copy_e = q.submit([&](handler &h) {
    // copy pt
    auto tf_accessor =
        buf_tf->template get_access<access::mode::discard_write>(h);
    h.copy(tf_set.data(), tf_accessor);
  });

  copy_e.wait();

  event e = q.submit([&](handler &h) {
    accessor acc_tf(*buf_tf, h, read_only);
    h.single_task<kernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      using PrefetchingLSU =
          ext::intel::lsu<ext::intel::prefetch<true>,
                          ext::intel::statically_coalesce<false>>;
      [[intel::disable_loop_pipelining]] while (true) {
        auto tmp = pipe_prime_index::read();
        pipe_prime_index_next::write(tmp);
        auto prime_index = tmp & 0x3f;
        // VEC*2, 2 is due to the y berret
        unsigned tf_base = prime_index * (coeff_count / VEC + VEC * 2);
        PRINTF("TwiddleFactor_iNTT: tf_base = %d, prime_index = %d\n", tf_base,
               prime_index);

        // twiddle factors are reodered, be careful the last group
        // N/2 ~ N
        // N/4 ~ N/2
        // N/8 ~ N/4
        // .....
        // 16 ~ 32
        // 8 ~ 16
        // 4 ~ 8
        // 2 ~ 4
        // 1
        // 0
        // The last group: 8,9,10,11,12,13,14,15,4,5,6,7,2,3,1,0
        // w^N/2, w^N/4, w^N/8, w^N/16 at index 1,2,4,8 -> 14,12,8,0
        // other groups still follow the NTT pattern

        // get the last group
        L0::TwiddleFactor_t<VEC> last_tf;
        uint64_t allw[VEC];
        for (unsigned j = 0; j < VEC; j++) {
          last_tf.data[j] =
              PrefetchingLSU::load(acc_tf.get_pointer() + tf_base + j * 2);
          allw[j] =
              PrefetchingLSU::load(acc_tf.get_pointer() + tf_base + j * 2 + 1);
        }

        ulong2 m_w[VEC];
        int offset[] = {1,  3,  2,  7,  6,  5,  4,  15, 14, 13, 12,
                        11, 10, 9,  8,  31, 30, 29, 28, 27, 26, 25,
                        24, 23, 22, 21, 20, 19, 18, 17, 16};

#pragma unroll
        for (unsigned j = 0; j < VEC - 1; j++) {
          m_w[j + 1] = {last_tf.data[VEC - 1 - offset[j]],
                        allw[VEC - 1 - offset[j]]};
        }

        uint64_t prime = last_tf.data[VEC - 1];

        L0::TwiddleFactor_t<VEC> tf;
        ulong first;
        for (unsigned i = 0; i < coeff_count; i++) {
          unsigned j = i % VEC;
#pragma unroll
          for (int k = 0; k < VEC - 1; k++) {
            tf.data[k] = tf.data[k + 1];
          }

#if TF_ON_THE_FLY
          if (j == 0) {
            tf.data[VEC - 1] = first =
                BIT(flag, 2) ? PrefetchingLSU::load(acc_tf.get_pointer() +
                                                    tf_base + VEC * 2 + i / VEC)
                             : 0;
          } else {
            tf.data[VEC - 1] = mulmod(first, m_w[j], prime);
          }
#else
          tf.data[VEC - 1] = acc_tf[tf_base++];
#endif
          if (j == (VEC - 1)) {
            if (BIT(flag, 1)) {
              if (i == (coeff_count - 1)) {
                pipe_twiddle_factor::write(last_tf);
              } else {
                pipe_twiddle_factor::write(tf);
              }
            }
          }
        }
      }
    });
  }  // namespace L0
  );
  return e;
}  // namespace L0

// twiddle factor factory for the ntt kernels
template <class kernelNameClass, class pipe_prime_index,
          class pipe_prime_index_next, class pipe_twiddle_factor, unsigned VEC,
          unsigned coeff_count>
event TwiddleFactor_NTT(sycl::queue &q, const std::vector<uint64_t> &tf_set,
                        int flag) {
  // the input vector should not be freed. Keep it with static or never freed
  // memory before application exit. Otherwise the accessor will get random
  // values
  auto buf_tf = new buffer<uint64_t>(tf_set.size(),
                                     {sycl::property::buffer::mem_channel{2}});
  event copy_e = q.submit([&](handler &h) {
    // copy pt
    auto tf_accessor =
        buf_tf->template get_access<access::mode::discard_write>(h);
    h.copy(tf_set.data(), tf_accessor);
  });

  copy_e.wait();

  event e = q.submit([&](handler &h) {
    accessor acc_tf(*buf_tf, h, read_only);
    h.single_task<kernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      using PrefetchingLSU =
          ext::intel::lsu<ext::intel::prefetch<true>,
                          ext::intel::statically_coalesce<false>>;
      [[intel::disable_loop_pipelining]] while (true) {
        unsigned char tmp = pipe_prime_index::read();
        pipe_prime_index_next::write(tmp);
        unsigned char prime_index = tmp & 0x3f;
        int tf_base = prime_index * (VEC * 2 + coeff_count / VEC);

        // Twiddle factor pattern
        // 0, 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, 1/16, 9/16, 5/16, 13/16, 3/16,
        // 11/16, 7/16, 15/16, 1/32, 17/32, 9/32, 25/32, 5/32, 21/32, 13/32,
        // 29/32, 3/32, 19/32, 11/32, 27/32, 7/32, 23/32, 15/32, 31/32, 1/64,
        // 33/64, 17/64, 49/64, 9/64, 41/64, 25/64, 57/64, 5/64, 37/64, 21/64,
        // 53/64, 13/64, 45/64, 29/64, 61/64, 3/64, 35/64, 19/64, 51/64, 11/64,
        // 43/64, 27/64, 59/64, 7/64, 39/64, 23/64, 55/64, 15/64, 47/64, 31/64,
        // 63/64
        ulong2 m_w[VEC];
        for (int i = 0; i < VEC * 2; i++) {
          ((ulong *)m_w)[i] =
              PrefetchingLSU::load(acc_tf.get_pointer() + tf_base + i);
        }

        // get the prime from the first place
        uint64_t prime = m_w[0].s0();
        PRINTF("TwiddleFactor_NTT: prime_index = %d, prime = %ld\n",
               prime_index, prime);

        L0::TwiddleFactor_t<VEC> tf;
        ulong first;
        for (unsigned i = 0; i < coeff_count; i++) {
          unsigned j = i % VEC;
#pragma unroll
          for (int k = 0; k < VEC - 1; k++) {
            tf.data[k] = tf.data[k + 1];
          }

#if TF_ON_THE_FLY
          if (j == 0) {
            tf.data[VEC - 1] = first =
                BIT(flag, 2) ? PrefetchingLSU::load(acc_tf.get_pointer() +
                                                    tf_base + VEC * 2 + i / VEC)
                             : 0;
          } else {
            tf.data[VEC - 1] = mulmod(first, m_w[j], prime);
          }
#else
          tf.data[VEC - 1] = acc_tf[tf_base++];
#endif
          if (j == (VEC - 1)) {
            if (BIT(flag, 1)) {
              pipe_twiddle_factor::write(tf);
            }
          }
        }
      }
    });
  }  // namespace L0
  );
  return e;
}
}  // namespace L0
