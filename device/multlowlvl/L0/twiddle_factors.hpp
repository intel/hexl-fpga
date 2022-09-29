// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "def.hpp"
#include "number_theory.hpp"
#include "utils.hpp"

// enable generate the twiddle factors on the fly to reduce
// the memory bandwidth requirement
#define TF_ON_THE_FLY 1

namespace L0 {
template <class kernelNameClass, class pipe_prime_index,
          class pipe_prime_index_next, class pipe_twiddle_factor, unsigned VEC,
          unsigned coeff_count>
event TwiddleFactor(sycl::queue &q, const std::vector<uint64_t> &tf_set) {
  // the input vector should not be freed. Keep it with static or never freed
  // memory before application exit. Otherwise the accessor will get random
  // values
  auto buf_tf = new buffer<uint64_t>(tf_set.size());
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
      [[intel::disable_loop_pipelining]] while (true) {
        auto tmp = pipe_prime_index::read();
        pipe_prime_index_next::write(tmp);
        auto prime_index = tmp & 0x3f;
        // VEC*2, 2 is due to the y berret
        unsigned tf_base = prime_index * (coeff_count / VEC + VEC * 2);
        PRINTF("tf_base = %d, prime_index = %d\n", tf_base, prime_index);

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
          last_tf.data[j] = acc_tf[tf_base++];
          allw[j] = acc_tf[tf_base++];
        }

        // 4 if VEC 16, 3 if VEC 8
        ulong2 m_w[4];
        m_w[0] = {last_tf.data[14 % VEC], allw[14 % VEC]};
        m_w[1] = {last_tf.data[12 % VEC], allw[12 % VEC]};
        m_w[2] = {last_tf.data[8 % VEC], allw[8 % VEC]};
        // The last one only exists when VEC is 16
        m_w[3] = {last_tf.data[0], allw[0]};

        uint64_t prime = last_tf.data[VEC - 1];

        for (unsigned i = 0; i < coeff_count / VEC; i++) {
          L0::TwiddleFactor_t<VEC> tf;
#if TF_ON_THE_FLY
          uint64_t base, first, second;

          // the first element is always 1, this place is occupied by the
          // prime
          tf.data[0] = first = base = acc_tf[tf_base++];
          tf.data[1] = mulmod(first, m_w[0], prime);           //*w^N/2
          tf.data[2] = second = mulmod(first, m_w[1], prime);  //*w^N/4
          tf.data[3] = mulmod(second, m_w[0], prime);          //*w^N/2

          // the second group
          tf.data[4] = first = mulmod(base, m_w[2], prime);    //*w^N/8
          tf.data[5] = mulmod(first, m_w[0], prime);           //*w^N/2
          tf.data[6] = second = mulmod(first, m_w[1], prime);  //*w^N/4
          tf.data[7] = mulmod(second, m_w[0], prime);          //*w^N/2
          if (VEC >= 16) {
            // the third group
            tf.data[8] = first = mulmod(base, m_w[3], prime);     //*w^N/16
            tf.data[9] = mulmod(first, m_w[0], prime);            //*w^N/2
            tf.data[10] = second = mulmod(first, m_w[1], prime);  //*w^N/4
            tf.data[11] = mulmod(second, m_w[0], prime);          //*w^N/2

            // the forth group
            tf.data[12] = first = mulmod(tf.data[8], m_w[2], prime);  //*w^N/8
            tf.data[13] = mulmod(first, m_w[0], prime);               //*w^N/2
            tf.data[14] = second = mulmod(first, m_w[1], prime);      //*w^N/4
            tf.data[15] = mulmod(second, m_w[0], prime);              //*w^N/2
          }
#else
#pragma unroll
              for (unsigned j = 0; j < VEC; j++) {
                tf.data[j] = acc_tf[tf_base + i * VEC + j];
              }
#endif

          if (i == (coeff_count / VEC - 1)) {
            pipe_twiddle_factor::write(last_tf);
          } else {
            pipe_twiddle_factor::write(tf);
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
event TwiddleFactor_NTT(sycl::queue &q, const std::vector<uint64_t> &tf_set) {
  // the input vector should not be freed. Keep it with static or never freed
  // memory before application exit. Otherwise the accessor will get random
  // values
  auto buf_tf = new buffer<uint64_t>(tf_set.size());
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
      [[intel::disable_loop_pipelining]] while (true) {
        unsigned char tmp = pipe_prime_index::read();
        pipe_prime_index_next::write(tmp);
        unsigned char prime_index = tmp & 0x3f;
        unsigned tf_base = prime_index * (8 + coeff_count / VEC);

        // VEC 16: w^N/2, w^N/4, w^N/8, w^N/16 at index 1,2,4,8
        // VEC 8:  w^N/2, w^N/4, w^N/8 at index 1,2,4
        ulong2 m_w[4];
        for (int i = 0; i < 8; i++) {
          ((ulong *)m_w)[i] = acc_tf[tf_base++];
        }

        // store the prime in the first place as we don't use this always 1
        uint64_t prime = acc_tf[tf_base++];
        PRINTF("TwiddleFactor_NTT: prime_index = %d, prime = %ld\n",
               prime_index, prime);

        for (unsigned i = 0; i < coeff_count / VEC; i++) {
          L0::TwiddleFactor_t<VEC> tf;
#if TF_ON_THE_FLY
          uint64_t base, first, second;

          // the first element is always 1, this place is occupied by the
          // prime
          tf.data[0] = first = base = i == 0 ? 1 : acc_tf[tf_base++];
          tf.data[1] = mulmod(first, m_w[0], prime);           //*w^N/2
          tf.data[2] = second = mulmod(first, m_w[1], prime);  //*w^N/4
          tf.data[3] = mulmod(second, m_w[0], prime);          //*w^N/2

          // the second group
          tf.data[4] = first = mulmod(base, m_w[2], prime);    //*w^N/8
          tf.data[5] = mulmod(first, m_w[0], prime);           //*w^N/2
          tf.data[6] = second = mulmod(first, m_w[1], prime);  //*w^N/4
          tf.data[7] = mulmod(second, m_w[0], prime);          //*w^N/2
          if (VEC >= 16) {
            // the third group
            tf.data[8] = first = mulmod(base, m_w[3], prime);     //*w^N/16
            tf.data[9] = mulmod(first, m_w[0], prime);            //*w^N/2
            tf.data[10] = second = mulmod(first, m_w[1], prime);  //*w^N/4
            tf.data[11] = mulmod(second, m_w[0], prime);          //*w^N/2

            // the forth group
            tf.data[12] = first = mulmod(tf.data[8], m_w[2], prime);  //*w^N/8
            tf.data[13] = mulmod(first, m_w[0], prime);               //*w^N/2
            tf.data[14] = second = mulmod(first, m_w[1], prime);      //*w^N/4
            tf.data[15] = mulmod(second, m_w[0], prime);              //*w^N/2
          }
#else
#pragma unroll
          for (unsigned j = 0; j < VEC; j++) {
            tf.data[j] = acc_tf[tf_base + i * VEC + j];
          }
#endif
          pipe_twiddle_factor::write(tf);
        }
      }
    });
  }  // namespace L0
  );
  return e;
}
}  // namespace L0
