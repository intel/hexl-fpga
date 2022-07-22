// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __INTT2_CORE_HPP__
#define __INTT2_CORE_HPP__

#include "pipes_keyswitch.hpp"
#include "../device_types.hpp"
#include "../mod_ops.hpp"
#include "../utils/kernel_assert.hpp"

template <class tt_kernelNameClass, class tt_ch_intt_elements_in,
          class tt_ch_intt_elements>
void _intt_backward2(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            int data_index = 0;
            WideVecINTT2_t elements;
            while (true) {
#pragma unroll
                for (int i = 0; i < VEC_INTT2 * 2 - 1; i++) {
                    elements.data[i] = elements.data[i + 1];
                }
                elements.data[VEC_INTT2 * 2 - 1] =
                    tt_ch_intt_elements_in::read();
                if (data_index == (VEC_INTT2 * 2 - 1)) {
                    tt_ch_intt_elements::write(elements);
                }
                data_index = (data_index + 1) % (VEC_INTT2 * 2);
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}

template <class tt_kernelNameClass, class tt_ch_intt_elements_out,
          class tt_ch_intt_elements>
void _intt_forward2(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            int data_index = 0;
            WideVecINTT2_t elements;
            while (true) {
                if (data_index == 0) {
                    elements = tt_ch_intt_elements::read();
                }
                uint64_t data = elements.data[0];
#pragma unroll
                for (int i = 0; i < VEC_INTT2 * 2 - 1; i++) {
                    elements.data[i] = elements.data[i + 1];
                }
                tt_ch_intt_elements_out::write(data);
                data_index = (data_index + 1) % (VEC_INTT2 * 2);
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}

template <class tt_kernelNameClass, uint64_t output_mod_factor,
          class tt_ch_intt_modulus, class tt_ch_intt_elements_in,
          class tt_ch_intt_elements_out, class tt_ch_normalize,
          class tt_ch_twiddle_factor>
void _intt_internal2(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            unsigned long X[MAX_COFF_COUNT / VEC_INTT2 / 2][VEC_INTT2];
            unsigned long X2[MAX_COFF_COUNT / VEC_INTT2 / 2][VEC_INTT2];

            [[intel::disable_loop_pipelining]] while (true) {
                sycl::ulong4 modulus = tt_ch_intt_modulus::read();
                ulong prime = modulus.s0() & MODULUS_BIT_MASK;
                ulong prime_k = modulus.s3();
                unsigned fpga_ntt_size = GET_COEFF_COUNT(modulus.s0());
                unsigned long twice_mod = prime << 1;
                unsigned t = 1;
                unsigned logt = 0;
                unsigned int g_elements_index = 0;
                unsigned roots_acc = 0;
                int last_tf_index = -1;
                TwiddleFactor2_t tf;

                tt_ch_normalize::write(modulus);

                // Normalize the Transform by N
                // Stages Loop
                [[intel::disable_loop_pipelining]] for (unsigned m =
                                                            (fpga_ntt_size >>
                                                             1);
                                                        m >= 1; m >>= 1) {
                    bool b_first_stage = t == 1;

                    unsigned rw_x_groups_log = get_ntt_log(fpga_ntt_size) - 1 -
                                               VEC_LOG_INTT2 - logt +
                                               VEC_LOG_INTT2;
                    unsigned rw_x_groups = 1 << rw_x_groups_log;
                    unsigned rw_x_group_size_log = logt - VEC_LOG_INTT2;
                    unsigned rw_x_group_size = 1 << rw_x_group_size_log;
                    unsigned Xm_group_log = rw_x_group_size_log - 1;

                    // Flights
                    [[intel::ivdep(X)]] [[intel::ivdep(
                        X2)]] for (unsigned k = 0;
                                   k < fpga_ntt_size / 2 / VEC_INTT2; k++) {
                        [[intel::fpga_register]] unsigned long
                            curX[VEC_INTT2 * 2];
                        [[intel::fpga_register]] unsigned long
                            curX_rep[VEC_INTT2 * 2];
                        unsigned i0 = (k * VEC_INTT2 + 0) >>
                                      logt;  // i is the index of groups
                        unsigned j0 = (k * VEC_INTT2 + 0) &
                                      (t - 1);  // j is the position of a group
                        unsigned j10 = i0 << (logt + 1);

                        bool b_rev = ((k >> rw_x_group_size_log) & 1);
                        if (t < VEC_INTT2) b_rev = 0;

                        if (b_first_stage) {
                            WideVecINTT2_t elements =
                                tt_ch_intt_elements_in::read();
#pragma unroll
                            for (int n = 0; n < VEC_INTT2 * 2; n++) {
                                curX[n] = elements.data[n];
                            }
                        }

                        unsigned long localX[VEC_INTT2];
                        unsigned long localX2[VEC_INTT2];

                        // store from the high end
                        unsigned rw_x_group_index =
                            rw_x_groups - 1 - (k >> rw_x_group_size_log);
                        unsigned rw_pos =
                            (rw_x_group_index << rw_x_group_size_log) +
                            (k & (rw_x_group_size - 1));
                        if (t < VEC_INTT2) {
                            rw_pos = fpga_ntt_size / 2 / VEC_INTT2 - 1 - k;
                        }
                        unsigned Xm_group_index = k >> Xm_group_log;
                        bool b_X = !(Xm_group_index & 1);
                        if (t <= VEC_INTT2) {
                            b_X = true;
                        }

#pragma unroll
                        for (unsigned n = 0; n < VEC_INTT2; n++) {
                            unsigned i = (k * VEC_INTT2 + n) >>
                                         logt;  // i is the index of groups
                            unsigned j =
                                (k * VEC_INTT2 + n) &
                                (t - 1);  // j is the position of a group
                            unsigned j1 = i * 2 * t;

                            localX[n] = X[k][n];
                            localX2[n] = X2[rw_pos][n];

                            if (b_first_stage) {
                                // curX[n] = elements_in.data[n];
                                // curX[n + VEC_INTT2] = elements_in.data[n +
                                // VEC_INTT2];
                            } else {
                                curX[n] = b_X ? localX[n] : localX2[n];
                                curX[n + VEC_INTT2] =
                                    (!b_X) ? localX[n] : localX2[n];
                            }
                        }

                        if (t == 1) {
#pragma unroll
                            for (int n = 0; n < VEC_INTT2; n++) {
                                const int cur_t = 1;
                                const int Xn =
                                    n / cur_t * (2 * cur_t) + n % cur_t;
                                const int Xnt =
                                    Xn +
                                    ((cur_t < VEC_INTT2) ? cur_t : VEC_INTT2);
                                curX_rep[n] = curX[Xn];
                                curX_rep[VEC_INTT2 + n] = curX[Xnt];
                            }
#if VEC_INTT2 >= 4
                        } else if (t == 2) {
#pragma unroll
                            for (int n = 0; n < VEC_INTT2; n++) {
                                const int cur_t = 2;
                                const int Xn =
                                    n / cur_t * (2 * cur_t) + n % cur_t;
                                const int Xnt =
                                    Xn +
                                    ((cur_t < VEC_INTT2) ? cur_t : VEC_INTT2);
                                curX_rep[n] = curX[Xn];
                                curX_rep[VEC_INTT2 + n] = curX[Xnt];
                            }
#endif
#if VEC_INTT2 >= 8
                        } else if (t == 4) {
#pragma unroll
                            for (int n = 0; n < VEC_INTT2; n++) {
                                const int cur_t = 4;
                                const int Xn =
                                    n / cur_t * (2 * cur_t) + n % cur_t;
                                const int Xnt =
                                    Xn +
                                    ((cur_t < VEC_INTT2) ? cur_t : VEC_INTT2);
                                curX_rep[n] = curX[Xn];
                                curX_rep[VEC_INTT2 + n] = curX[Xnt];
                            }
#endif
#if VEC_INTT2 >= 16
                        } else if (t == 8) {
#pragma unroll
                            for (int n = 0; n < VEC_INTT2; n++) {
                                const int cur_t = 8;
                                const int Xn =
                                    n / cur_t * (2 * cur_t) + n % cur_t;
                                const int Xnt =
                                    Xn +
                                    ((cur_t < VEC_INTT2) ? cur_t : VEC_INTT2);
                                curX_rep[n] = curX[Xn];
                                curX_rep[VEC_INTT2 + n] = curX[Xnt];
                            }
#endif
#if VEC_INTT2 >= 32
                        } else if (t == 16) {
#pragma unroll
                            for (int n = 0; n < VEC_INTT2; n++) {
                                const int cur_t = 16;
                                const int Xn =
                                    n / cur_t * (2 * cur_t) + n % cur_t;
                                const int Xnt =
                                    Xn +
                                    ((cur_t < VEC_INTT2) ? cur_t : VEC_INTT2);
                                curX_rep[n] = curX[Xn];
                                curX_rep[VEC_INTT2 + n] = curX[Xnt];
                            }
#endif
                        } else {
#pragma unroll
                            for (int n = 0; n < VEC_INTT2; n++) {
                                curX_rep[n] = curX[n];
                                curX_rep[VEC_INTT2 + n] = curX[VEC_INTT2 + n];
                            }
                        }
                        unsigned shift_left_elements =
                            (roots_acc + i0) % VEC_INTT2;
                        unsigned ivec = (k * VEC_INTT2 + VEC_INTT2 - 1) >> logt;
                        unsigned long cur_roots[VEC_INTT2];
                        unsigned long cur_precons[VEC_INTT2];

                        int tf_index = (roots_acc + i0) / VEC_INTT2;
                        if (tf_index != last_tf_index) {
                            tf = tt_ch_twiddle_factor::read();
                        }
                        last_tf_index = tf_index;

#pragma unroll
                        for (int n = 0; n < VEC_INTT2; n++) {
                            cur_roots[n] = tf.data[n];
                        }

                        typedef ac_int<VEC_INTT2 * 64, false> uint_vec_t;
                        *(uint_vec_t*)cur_roots = (*(uint_vec_t*)cur_roots) >>
                                                  (shift_left_elements * 64);

                        unsigned select_num = (roots_acc + ivec) % VEC_INTT2 -
                                              (roots_acc + i0) % VEC_INTT2 + 1;

                        unsigned long reorder_roots[VEC_INTT2];

#pragma unroll
                        for (int n = 0; n < VEC_INTT2; n++) {
                            reorder_roots[n] = cur_roots[n];
                        }

                        if (select_num == 1) {
#pragma unroll
                            for (int n = 1; n < VEC_INTT2; n++) {
                                reorder_roots[n] = cur_roots[0];
                            }
                        } else if (select_num == 2) {
#pragma unroll
                            for (int n = 0; n < VEC_INTT2 / 2; n++) {
                                reorder_roots[n] = cur_roots[0];
                                reorder_roots[n + VEC_INTT2 / 2] = cur_roots[1];
                            }
                        } else if (select_num == 4) {
#pragma unroll
                            for (int n = 0; n < 4; n++) {
                                reorder_roots[n * 2] = cur_roots[n];
                                reorder_roots[n * 2 + 1] = cur_roots[n];
                            }
                        }

                        WideVecINTT2_t elements;
#pragma unroll
                        for (int n = 0; n < VEC_INTT2; n++) {
                            unsigned i = (k * VEC_INTT2 + n) >>
                                         logt;  // i is the index of groups
                            unsigned j =
                                (k * VEC_INTT2 + n) &
                                (t - 1);  // j is the position of a group
                            unsigned j1 = i * 2 * t + j;
                            unsigned j2 = j1 + t;

                            unsigned long W_op = reorder_roots[n];
                            unsigned long tx = 0;
                            unsigned long ty = 0;

                            // Butterfly
                            unsigned long x_j1 = curX_rep[n];
                            unsigned long x_j2 = curX_rep[VEC_INTT2 + n];

                            // X', Y' = X + Y (mod q), W(X - Y) (mod q).
                            // ASSERT(x_j1 < prime, "x >= modulus, engine_id =
                            // %d\n", engine_id);
                            curX[n] = AddUIntMod(x_j1, x_j2, prime);
                            ASSERT(W_op < MAX_MODULUS, "y >= modulus\n");
                            curX[VEC_INTT2 + n] =
                                MultiplyUIntMod(SubUIntMod(x_j1, x_j2, prime),
                                                W_op, prime, prime_k);

                            elements.data[n * 2] = curX[n];
                            elements.data[n * 2 + 1] = curX[VEC_INTT2 + n];
                        }
                        if (m == 1) {
                            tt_ch_intt_elements_out::write(elements);
                        }

                        // reoder back
                        if (t == 1) {
#pragma unroll
                            for (int n = 0; n < VEC_INTT2; n++) {
                                const int cur_t = 1;
                                const int Xn =
                                    n / cur_t * (2 * cur_t) + n % cur_t;
                                const int Xnt =
                                    Xn +
                                    ((cur_t < VEC_INTT2) ? cur_t : VEC_INTT2);
                                curX_rep[Xn] = curX[n];
                                curX_rep[Xnt] = curX[VEC_INTT2 + n];
                            }
#if VEC_INTT2 >= 4
                        } else if (t == 2) {
#pragma unroll
                            for (int n = 0; n < VEC_INTT2; n++) {
                                const int cur_t = 2;
                                const int Xn =
                                    n / cur_t * (2 * cur_t) + n % cur_t;
                                const int Xnt =
                                    Xn +
                                    ((cur_t < VEC_INTT2) ? cur_t : VEC_INTT2);
                                curX_rep[Xn] = curX[n];
                                curX_rep[Xnt] = curX[VEC_INTT2 + n];
                            }
#endif
#if VEC_INTT2 >= 8
                        } else if (t == 4) {
#pragma unroll
                            for (int n = 0; n < VEC_INTT2; n++) {
                                const int cur_t = 4;
                                const int Xn =
                                    n / cur_t * (2 * cur_t) + n % cur_t;
                                const int Xnt =
                                    Xn +
                                    ((cur_t < VEC_INTT2) ? cur_t : VEC_INTT2);
                                curX_rep[Xn] = curX[n];
                                curX_rep[Xnt] = curX[VEC_INTT2 + n];
                            }
#endif
#if VEC_INTT2 >= 16
                        } else if (t == 8) {
#pragma unroll
                            for (int n = 0; n < VEC_INTT2; n++) {
                                const int cur_t = 8;
                                const int Xn =
                                    n / cur_t * (2 * cur_t) + n % cur_t;
                                const int Xnt =
                                    Xn +
                                    ((cur_t < VEC_INTT2) ? cur_t : VEC_INTT2);
                                curX_rep[Xn] = curX[n];
                                curX_rep[Xnt] = curX[VEC_INTT2 + n];
                            }
#endif
#if VEC_INTT2 >= 32
                        } else if (t == 16) {
#pragma unroll
                            for (int n = 0; n < VEC_INTT2; n++) {
                                const int cur_t = 16;
                                const int Xn =
                                    n / cur_t * (2 * cur_t) + n % cur_t;
                                const int Xnt =
                                    Xn +
                                    ((cur_t < VEC_INTT2) ? cur_t : VEC_INTT2);
                                curX_rep[Xn] = curX[n];
                                curX_rep[Xnt] = curX[VEC_INTT2 + n];
                            }
#endif
                        } else {
#pragma unroll
                            for (int n = 0; n < VEC_INTT2; n++) {
                                curX_rep[n] = curX[n];
                                curX_rep[VEC_INTT2 + n] = curX[VEC_INTT2 + n];
                            }
                        }

#pragma unroll
                        for (int n = 0; n < VEC_INTT2; n++) {
                            X[k][n] =
                                b_rev ? curX_rep[n + VEC_INTT2] : curX_rep[n];
                            X2[rw_pos][n] =
                                b_rev ? curX_rep[n] : curX_rep[n + VEC_INTT2];
                        }
                    }

                    roots_acc += m;
                    t <<= 1;
                    logt++;
                }
            }
            return;
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}

template <class tt_kernelNameClass, class tt_ch_intt_elements_out_inter,
          class tt_ch_intt_elements_out, class tt_ch_normalize>
void _intt_normalize2(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            while (true) {
                sycl::ulong4 moduli = tt_ch_normalize::read();
                unsigned coeff_count = GET_COEFF_COUNT(moduli.s0());

                for (unsigned i = 0; i < coeff_count; i++) {
                    uint64_t data = tt_ch_intt_elements_out_inter::read();
                    ASSERT((moduli.s0 & MODULUS_BIT_MASK) < MAX_MODULUS,
                           "y >= modulus\n");
                    data = MultiplyUIntMod(data, moduli.s2(),
                                           moduli.s0() & MODULUS_BIT_MASK,
                                           moduli.s3());
                    tt_ch_intt_elements_out::write(data);
                }
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}

#endif
