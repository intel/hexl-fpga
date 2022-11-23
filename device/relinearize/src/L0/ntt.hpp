// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "number-theory.hpp"
#include "utils.hpp"

namespace L0 {
// ntt read kernel - read 1 element per cycle, and output a VEC element to the
// ntt kernel
// input should be interleaved
template <class tt_kernelNameClass, class tt_ch_ntt_elements_in,
          class tt_ch_ntt_elements, unsigned int VEC>
event read(sycl::queue &q) {
  auto e = q.submit([&](handler &h) {
    h.single_task<tt_kernelNameClass>([=]() {
      int data_index = 0;
      WideVector_t<VEC> elements;
      while (true) {
#pragma unroll
        for (int i = 0; i < VEC * 2 - 1; i++) {
          elements.data[i] = elements.data[i + 1];
        }
        elements.data[VEC * 2 - 1] = tt_ch_ntt_elements_in::read();
        if (data_index == (VEC * 2 - 1)) {
          tt_ch_ntt_elements::write(elements);
        }
        data_index = (data_index + 1) % (VEC * 2);
      }
    });
  });
  return e;
}

// ntt write kernel - read a VEC elements and write output to other kernel with
// 1 element per cycle
template <class tt_kernelNameClass, class tt_ch_ntt_elements,
          class tt_ch_ntt_elements_out, unsigned int VEC>
event write(sycl::queue &q) {
  auto e = q.submit([&](handler &h) {
    h.single_task<tt_kernelNameClass>([=]() {
      int data_index = 0;
      L0::WideVector_t<VEC> elements;
      while (true) {
        if (data_index == 0) {
          elements = tt_ch_ntt_elements::read();
        }
        uint64_t data = elements.data[0];
#pragma unroll
        for (int i = 0; i < VEC * 2 - 1; i++) {
          elements.data[i] = elements.data[i + 1];
        }
        tt_ch_ntt_elements_out::write(data);
        data_index = (data_index + 1) % (VEC * 2);
      }
    });
  });
  return e;
}
namespace NTT {
// -------------------- ntt core kernel -----------------------------------
template <class tt_kernelNameClass, class tt_ch_prime_index,
          class tt_ch_ntt_elements_in, class tt_ch_ntt_elements_out,
          class tt_ch_twiddle_factor, unsigned int VEC,
          unsigned MAX_COEFF_COUNT>
event ntt(sycl::queue &q, const std::vector<ulong4> &config, int flag) {
  buffer<sycl::ulong4> *buf_config = new buffer(config);
  auto e = q.submit([&](handler &h) {
    accessor acc_config(*buf_config, h, read_only);
    h.single_task<tt_kernelNameClass>([=]() {
      unsigned long X[MAX_COEFF_COUNT / VEC / 2][VEC];
      unsigned long X2[MAX_COEFF_COUNT / VEC / 2][VEC];

      [[intel::disable_loop_pipelining]] while (true) {
        unsigned char tmp = tt_ch_prime_index::read();
        unsigned char prime_index = tmp & 0x3f;
        using PipelinedLSU = ext::intel::lsu<>;
        ulong4 cur_moduli =
            PipelinedLSU::load(acc_config.get_pointer() + prime_index);
        uint64_t modulus = cur_moduli.s0();
        uint64_t modulus_r = cur_moduli.s1();
        uint64_t modulus_k = cur_moduli.s3();
        unsigned fpga_ntt_size = MAX_COEFF_COUNT;
        ASSERT(VEC >= 2 && VEC <= 32, "Not supported VEC\n");
        PRINTF("NTT: fpga_ntt_size = %llu, modulus = %llu\n", fpga_ntt_size,
               modulus);
        unsigned long coeff_mod = modulus;
        unsigned long twice_mod = modulus << 1;
        unsigned long t = (fpga_ntt_size >> 1);
        unsigned int t_log = get_ntt_log(fpga_ntt_size) - 1;
        unsigned int roots_off = 0;
        int last_tf_index = -1;
        unsigned VEC_LOG = get_vec_log(VEC);

        L0::TwiddleFactor_t<VEC> tf;
        [[intel::disable_loop_pipelining]] for (unsigned int m = 1, mlog = 0;
                                                m < fpga_ntt_size;
                                                m <<= 1, mlog++) {
          unsigned rw_x_groups = m;
          unsigned rw_x_group_size = (fpga_ntt_size / 2 / VEC) >> mlog;
          unsigned rw_x_group_size_log =
              get_ntt_log(fpga_ntt_size) - 1 - VEC_LOG - mlog;
          unsigned Xm_group_log = rw_x_group_size_log;
          [[intel::ivdep(X2)]] for (unsigned int k = 0;
                                    k < fpga_ntt_size / 2 / VEC; k++) {
            [[intel::fpga_register]] unsigned long curX[VEC * 2];
            [[intel::fpga_register]] unsigned long curX_rep[VEC * 2];

            unsigned i0 = (k * VEC + 0) >> t_log;  // i is the index of groups
            unsigned j0 =
                (k * VEC + 0) & (t - 1);  // j is the position of a group

            bool b_rev = j0 >= (t / 2);
            if (t <= VEC) b_rev = 0;

            if (m == 1) {
              WideVector_t<VEC> elements = tt_ch_ntt_elements_in::read();

#pragma unroll
              for (int n = 0; n < VEC; n++) {
                curX[n] = elements.data[n * 2];
                ASSERT(elements.data[n * 2] < modulus, "NTT: %lu > %lu\n",
                       elements.data[n * 2], modulus);
                curX[n + VEC] = elements.data[n * 2 + 1];
                ASSERT(elements.data[n * 2 + 1] < modulus, "NTT: %lu > %lu\n",
                       elements.data[n * 2 + 1], modulus);
              }
            }
            unsigned long localX[VEC];
            unsigned long localX2[VEC];
            // store from the high end
            unsigned rw_x_group_index =
                rw_x_groups - 1 - (k >> rw_x_group_size_log);
            unsigned rw_pos = (rw_x_group_index << rw_x_group_size_log) +
                              (k & (rw_x_group_size - 1));
            if (t <= VEC) {
              rw_pos = fpga_ntt_size / 2 / VEC - 1 - k;
            }
            unsigned Xm_group_index = k >> Xm_group_log;
            bool b_X = !(Xm_group_index & 1);
            if (t < VEC) {
              b_X = true;
            }

#pragma unroll
            for (int n = 0; n < VEC; n++) {
              localX[n] = X[k][n] & BIT_MASK(MAX_PRIME_BITS);
              localX2[n] = X2[rw_pos][n] & BIT_MASK(MAX_PRIME_BITS);

              if (m != 1) {
                curX[n] = b_X ? localX[n] : localX2[n];
                curX[n + VEC] = (!b_X) ? localX[n] : localX2[n];
              }
            }

            if (t == 1) {
#pragma unroll
              for (int n = 0; n < VEC; n++) {
                const int cur_t = 1;
                const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                curX_rep[n] = curX[Xn];
                curX_rep[VEC + n] = curX[Xnt];
              }
            } else if (t == 2 && VEC >= 4) {
#pragma unroll
              for (int n = 0; n < VEC; n++) {
                const int cur_t = 2;
                const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                curX_rep[n] = curX[Xn];
                curX_rep[VEC + n] = curX[Xnt];
              }
            } else if (t == 4 && VEC >= 8) {
#pragma unroll
              for (int n = 0; n < VEC; n++) {
                const int cur_t = 4;
                const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                curX_rep[n] = curX[Xn];
                curX_rep[VEC + n] = curX[Xnt];
              }
            } else if (t == 8 && VEC >= 16) {
#pragma unroll
              for (int n = 0; n < VEC; n++) {
                const int cur_t = 8;
                const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                curX_rep[n] = curX[Xn];
                curX_rep[VEC + n] = curX[Xnt];
              }
            } else if (t == 16 && VEC >= 32) {
#pragma unroll
              for (int n = 0; n < VEC; n++) {
                const int cur_t = 16;
                const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                curX_rep[n] = curX[Xn];
                curX_rep[VEC + n] = curX[Xnt];
              }
            } else {
#pragma unroll
              for (int n = 0; n < VEC; n++) {
                curX_rep[n] = curX[n];
                curX_rep[VEC + n] = curX[VEC + n];
              }
            }

            unsigned ivec = (k * VEC + VEC - 1) >> t_log;
            unsigned roots_start = roots_off + m + i0;
            unsigned roots_end = roots_off + m + ivec;

            unsigned shift_left_elements = (roots_start) % VEC;
            unsigned long cur_roots[VEC];

            int tf_index = roots_start / VEC;
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
            // opencl line solution: typedef unsigned int
            // __attribute__((ap_int(VEC * 64))) uint_vec_t;
            typedef ac_int<VEC * 64, false> uint_vec_t;
            *(uint_vec_t *)cur_roots =
                (*(uint_vec_t *)cur_roots) >> (shift_left_elements * 64);

            unsigned select_num = roots_end % VEC - roots_start % VEC + 1;
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
              // distribute to 01234567
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

            L0::WideVector_t<VEC> elements;
#pragma unroll
            for (int n = 0; n < VEC; n++) {
              const unsigned long W_op = reorder_roots[n];
              unsigned long tx = curX_rep[n];
              unsigned long a = curX_rep[VEC + n];

              ASSERT(W_op < coeff_mod,
                     "y (%ld) >= modulus (%ld) at m = %d, k = %d, n = %d\n",
                     W_op, coeff_mod, m, k, n);
              uint64_t W_x_Y =
                  MultiplyUIntMod(a, W_op, coeff_mod, modulus_r, modulus_k);
              ASSERT(tx < coeff_mod,
                     "x (%ld) >= modulus (%ld) at m = %d, k = %d, n = %d\n", tx,
                     coeff_mod, m, k, n);
              curX[n] = AddUIntMod(tx, W_x_Y, coeff_mod);
              curX[VEC + n] = SubUIntMod(tx, W_x_Y, coeff_mod);
              elements.data[n * 2] = curX[n];
              elements.data[n * 2 + 1] = curX[VEC + n];
            }

            if (m == (fpga_ntt_size / 2)) {
              tt_ch_ntt_elements_out::write(elements);
            }
            // reoder back
            if (t == 1) {
#pragma unroll
              for (int n = 0; n < VEC; n++) {
                const int cur_t = 1;
                const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                curX_rep[Xn] = curX[n];
                curX_rep[Xnt] = curX[VEC + n];
              }
            } else if (t == 2 && VEC >= 4) {
#pragma unroll
              for (int n = 0; n < VEC; n++) {
                const int cur_t = 2;
                const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                curX_rep[Xn] = curX[n];
                curX_rep[Xnt] = curX[VEC + n];
              }
            } else if (t == 4 && VEC >= 8) {
#pragma unroll
              for (int n = 0; n < VEC; n++) {
                const int cur_t = 4;
                const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                curX_rep[Xn] = curX[n];
                curX_rep[Xnt] = curX[VEC + n];
              }
            } else if (t == 8 && VEC >= 16) {
#pragma unroll
              for (int n = 0; n < VEC; n++) {
                const int cur_t = 8;
                const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                curX_rep[Xn] = curX[n];
                curX_rep[Xnt] = curX[VEC + n];
              }
            } else if (t == 16 && VEC >= 32) {
#pragma unroll
              for (int n = 0; n < VEC; n++) {
                const int cur_t = 16;
                const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                curX_rep[Xn] = curX[n];
                curX_rep[Xnt] = curX[VEC + n];
              }
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
          t >>= 1;
          t_log -= 1;
        }
      }
    });
  });
  return e;
}  // namespace NTT
}  // namespace NTT
}  // namespace L0
