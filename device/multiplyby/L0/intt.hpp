// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "number-theory.hpp"
#include "utils.hpp"

namespace L0 {
namespace INTT {
template <class tt_kernelNameClass, class tt_ch_intt_elements_out_inter,
          class tt_ch_intt_elements_out, class tt_ch_normalize,
          unsigned coeff_count>
event norm(sycl::queue &q) {
  auto e = q.submit([&](handler &h) {
    h.single_task<tt_kernelNameClass>([=]() {
      while (true) {
        // s0: modulus
        // s1: the r value of the modulus
        // s2: N^-1 mod modulus
        // s3: the k value of the modulus
        ulong4 moduli = tt_ch_normalize::read();
        for (unsigned i = 0; i < coeff_count; i++) {
          uint64_t data = tt_ch_intt_elements_out_inter::read();
          data = MultiplyUIntMod(data, moduli.s2(), moduli.s0(), moduli.s1(),
                                 moduli.s3());
          tt_ch_intt_elements_out::write(data);
        }
      }
    });
  });
  return e;
}

template <unsigned VEC, int cur_t>
void reorder(uint64_t (&curX)[VEC * 2], uint64_t (&curX_rep)[VEC * 2]) {
#pragma unroll
  for (int n = 0; n < VEC; n++) {
    const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
    const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
    curX_rep[n] = curX[Xn];
    curX_rep[VEC + n] = curX[Xnt];
  }
}

template <unsigned VEC, int cur_t>
void reorder_back(uint64_t (&curX)[VEC * 2], uint64_t (&curX_rep)[VEC * 2]) {
#pragma unroll
  for (int n = 0; n < VEC; n++) {
    const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
    const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
    curX_rep[Xn] = curX[n];
    curX_rep[Xnt] = curX[VEC + n];
  }
}

template <class tt_kernelNameClass, class tt_ch_prime_index,
          class tt_ch_intt_elements_in, class tt_ch_intt_elements_out,
          class tt_ch_normalize, class tt_ch_twiddle_factor,
          unsigned coeff_count, unsigned VEC>
event intt(sycl::queue &q, const std::vector<sycl::ulong4> &configs, int flag) {
  buffer<sycl::ulong4> *buf_configs = new buffer(configs);
  auto e = q.submit([&](handler &h) {
    accessor acc_configs(*buf_configs, h, read_only);
    h.single_task<tt_kernelNameClass>([=]() {
      unsigned long X[coeff_count / VEC / 2][VEC];
      unsigned long X2[coeff_count / VEC / 2][VEC];

      [[intel::disable_loop_pipelining]] while (true) {
        auto tmp = tt_ch_prime_index::read();
        unsigned char prime_index = tmp & 0x3f;
        ulong4 modulus = acc_configs[prime_index];
        // s0: modulus
        // s1: the rk value of the modulus
        // s2: N^-1 mod modulus
        ulong prime = modulus.s0();
        ulong prime_r = modulus.s1();
        ulong prime_k = modulus.s3();
        // PRINTF("Get INTT modulus = %lu, r = %lu\n", prime, prime_r);
        unsigned fpga_ntt_size = coeff_count;
        unsigned long twice_mod = prime << 1;
        unsigned t = 1;
        unsigned logt = 0;
        unsigned int g_elements_index = 0;

        unsigned roots_acc = 0;
        tt_ch_normalize::write(modulus);

        int last_tf_index = -1;
        L0::TwiddleFactor_t<VEC> tf;

        [[intel::disable_loop_pipelining]] for (unsigned m =
                                                    (fpga_ntt_size >> 1);
                                                m >= 1; m >>= 1) {
          bool b_first_stage = t == 1;
          unsigned VEC_LOG = get_vec_log(VEC);
          unsigned rw_x_groups_log =
              get_ntt_log(fpga_ntt_size) - 1 - VEC_LOG - logt + VEC_LOG;
          unsigned rw_x_groups = 1 << rw_x_groups_log;
          unsigned rw_x_group_size_log = logt - VEC_LOG;
          unsigned rw_x_group_size = 1 << rw_x_group_size_log;
          unsigned Xm_group_log = rw_x_group_size_log - 1;
          [[intel::ivdep(X2)]] for (unsigned k = 0; k < fpga_ntt_size / 2 / VEC;
                                    k++) {
            [[intel::fpga_register]] unsigned long curX[VEC * 2];
            [[intel::fpga_register]] unsigned long curX_rep[VEC * 2];

            unsigned i0 = (k * VEC + 0) >> logt;  // i is the index of groups
            unsigned j0 =
                (k * VEC + 0) & (t - 1);  // j is the position of a group
            unsigned j10 = i0 << (logt + 1);

            bool b_rev = ((k >> rw_x_group_size_log) & 1);
            if (t < VEC) b_rev = 0;

            if (b_first_stage) {
              WideVector_t<VEC> elements = tt_ch_intt_elements_in::read();
#pragma unroll
              for (int n = 0; n < VEC * 2; n++) {
                curX[n] = elements.data[n];
                ASSERT(elements.data[n] < prime, "iNTT: %lu > %lu\n",
                       elements.data[n], prime);
              }
            }

            unsigned long localX[VEC];
            unsigned long localX2[VEC];

            // store from the high end
            unsigned rw_x_group_index =
                rw_x_groups - 1 - (k >> rw_x_group_size_log);
            unsigned rw_pos = (rw_x_group_index << rw_x_group_size_log) +
                              (k & (rw_x_group_size - 1));
            if (t < VEC) {
              rw_pos = fpga_ntt_size / 2 / VEC - 1 - k;
            }
            unsigned Xm_group_index = k >> Xm_group_log;
            bool b_X = !(Xm_group_index & 1);
            if (t <= VEC) {
              b_X = true;
            }

#pragma unroll
            for (unsigned n = 0; n < VEC; n++) {
              localX[n] = X[k][n] & BIT_MASK(MAX_PRIME_BITS);
              localX2[n] = X2[rw_pos][n] & BIT_MASK(MAX_PRIME_BITS);

              if (!b_first_stage) {
                curX[n] = b_X ? localX[n] : localX2[n];
                curX[n + VEC] = (!b_X) ? localX[n] : localX2[n];
              }
            }

            if (t == 1) {
              reorder<VEC, 1>(curX, curX_rep);
            } else if (t == 2 && VEC >= 4) {
              reorder<VEC, 2>(curX, curX_rep);
            } else if (t == 4 && VEC >= 8) {
              reorder<VEC, 4>(curX, curX_rep);
            } else if (t == 8 && VEC >= 16) {
              reorder<VEC, 8>(curX, curX_rep);
            } else if (t == 16 && VEC >= 32) {
              reorder<VEC, 16>(curX, curX_rep);
            } else {
#pragma unroll
              for (int n = 0; n < VEC; n++) {
                curX_rep[n] = curX[n];
                curX_rep[VEC + n] = curX[VEC + n];
              }
            }
            unsigned shift_left_elements = (roots_acc + i0) % VEC;
            unsigned ivec = (k * VEC + VEC - 1) >> logt;
            unsigned long cur_roots[VEC];
            unsigned long cur_precons[VEC];

            int tf_index = (roots_acc + i0) / VEC;
            if (tf_index != last_tf_index) {
              if (BIT(flag, 0)) {
                tf = tt_ch_twiddle_factor::read();
              }
            }
            last_tf_index = tf_index;

#pragma unroll
            for (int n = 0; n < VEC; n++) {
              cur_roots[n] = tf.data[n];
            }

            // typedef unsigned int __attribute__((__ap_int(VEC * 64)))
            // uint_vec_t;
            typedef ac_int<VEC * 64, false> uint_vec_t;
            *(uint_vec_t *)cur_roots =
                (*(uint_vec_t *)cur_roots) >> (shift_left_elements * 64);

            unsigned select_num =
                (roots_acc + ivec) % VEC - (roots_acc + i0) % VEC + 1;

            unsigned long reorder_roots[VEC];

#pragma unroll
            for (int n = 0; n < VEC; n++) {
              reorder_roots[n] = cur_roots[n];
            }
            if (select_num == 1 && VEC >= 2) {
              // distribute to 00000000
#pragma unroll
              for (int n = 1; n < VEC; n++) {
                reorder_roots[n] = cur_roots[0];
              }
            } else if (select_num == 2 && VEC >= 4) {
              // distribute to 00001111
#pragma unroll
              for (int n = 0; n < VEC / 2; n++) {
                reorder_roots[n] = cur_roots[0];
                reorder_roots[n + VEC / 2] = cur_roots[1];
              }
            } else if (select_num == 4 && VEC >= 8) {
              // distribute to 00112233
#pragma unroll
              for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int n = 0; n < VEC / 4; n++) {
                  reorder_roots[i * VEC / 4 + n] = cur_roots[i];
                }
              }
            } else if (select_num == 8 && VEC >= 16) {
#pragma unroll
              for (int i = 0; i < 8; i++) {
#pragma unroll
                for (int n = 0; n < VEC / 8; n++) {
                  reorder_roots[i * VEC / 8 + n] = cur_roots[i];
                }
              }
            } else if (select_num == 16 && VEC >= 32) {
#pragma unroll
              for (int i = 0; i < 16; i++) {
#pragma unroll
                for (int n = 0; n < VEC / 16; n++) {
                  reorder_roots[i * VEC / 16 + n] = cur_roots[i];
                }
              }
            }

            WideVector_t<VEC> elements;
#pragma unroll
            for (int n = 0; n < VEC; n++) {
              unsigned long W_op = reorder_roots[n];

              // Butterfly
              unsigned long x_j1 = curX_rep[n];
              unsigned long x_j2 = curX_rep[VEC + n];

              // X', Y' = X + Y (mod q), W(X - Y) (mod q).
              ASSERT(x_j1 < prime, "x >= modulus\n");
              ASSERT(W_op < prime, "y >= modulus\n");

              curX[n] = AddUIntMod(x_j1, x_j2, prime);
              curX[VEC + n] = MultiplyUIntMod(SubUIntMod(x_j1, x_j2, prime),
                                              W_op, prime, prime_r, prime_k);

              elements.data[n * 2] = curX[n];
              elements.data[n * 2 + 1] = curX[VEC + n];
            }
            if (m == 1) {
              tt_ch_intt_elements_out::write(elements);
            }

            // reoder back
            if (t == 1) {
              reorder_back<VEC, 1>(curX, curX_rep);
            } else if (t == 2 && VEC >= 4) {
              reorder_back<VEC, 2>(curX, curX_rep);
            } else if (t == 4 && VEC >= 8) {
              reorder_back<VEC, 4>(curX, curX_rep);
            } else if (t == 8 && VEC >= 16) {
              reorder_back<VEC, 8>(curX, curX_rep);
            } else if (t == 16 && VEC >= 32) {
              reorder_back<VEC, 16>(curX, curX_rep);
            } else {
#pragma unroll
              for (int n = 0; n < VEC; n++) {
                curX_rep[n] = curX[n];
                curX_rep[VEC + n] = curX[VEC + n];
              }
            }

#pragma unroll
            for (int n = 0; n < VEC; n++) {
              X[k][n] = (b_rev ? curX_rep[n + VEC] : curX_rep[n]) &
                        BIT_MASK(MAX_PRIME_BITS);
              X2[rw_pos][n] = (b_rev ? curX_rep[n] : curX_rep[n + VEC]) &
                              BIT_MASK(MAX_PRIME_BITS);
            }
          }

          roots_acc += m;
          t <<= 1;
          logt++;
        }
      }
    });
  });
  return e;
}  // namespace INTT
}  // namespace INTT
}  // namespace L0
