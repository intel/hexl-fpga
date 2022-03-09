// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common.h"

void _ntt_backward(channel uint64_t ch_ntt_elements_in,
                   channel ntt_elements ch_ntt_elements) {
    int data_index = 0;
    ntt_elements elements;
    while (true) {
#pragma unroll
        for (int i = 0; i < VEC * 2 - 1; i++) {
            elements.data[i] = elements.data[i + 1];
        }
        elements.data[VEC * 2 - 1] = read_channel_intel(ch_ntt_elements_in);
        if (data_index == (VEC * 2 - 1)) {
            write_channel_intel(ch_ntt_elements, elements);
        }
        data_index = (data_index + 1) % (VEC * 2);
    }
}

void _ntt_forward(channel uint64_t ch_ntt_elements_out,
                  channel ntt_elements ch_ntt_elements) {
    int data_index = 0;
    ntt_elements elements;
    while (true) {
        if (data_index == 0) {
            elements = read_channel_intel(ch_ntt_elements);
        }
        uint64_t data = elements.data[0];
#pragma unroll
        for (int i = 0; i < VEC * 2 - 1; i++) {
            elements.data[i] = elements.data[i + 1];
        }
        write_channel_intel(ch_ntt_elements_out, data);
        data_index = (data_index + 1) % (VEC * 2);
    }
}

void _ntt_internal(channel ulong4 ch_ntt_modulus,
                   channel ntt_elements ch_ntt_elements_in,
                   channel ntt_elements ch_ntt_elements_out,
                   channel twiddle_factor ch_twiddle_factor,
                   uint64_t output_mod_factor, int engine_id) {
    unsigned long X[MAX_COFF_COUNT / VEC / 2][VEC];
    unsigned long X2[MAX_COFF_COUNT / VEC / 2][VEC];

#pragma disable_loop_pipelining
    while (true) {
        ulong4 cur_moduli = read_channel_intel(ch_ntt_modulus);
        uint64_t modulus = cur_moduli.s0 & MODULUS_BIT_MASK;
        uint64_t modulus_k = cur_moduli.s3;
        unsigned fpga_ntt_size = GET_COEFF_COUNT(cur_moduli.s0);

        unsigned long coeff_mod = modulus;
        unsigned long twice_mod = modulus << 1;

        unsigned long t = (fpga_ntt_size >> 1);
        unsigned int t_log = get_ntt_log(fpga_ntt_size) - 1;

        unsigned int roots_off = 0;

        int last_tf_index = -1;
        twiddle_factor tf;

#pragma disable_loop_pipelining
        for (unsigned int m = 1, mlog = 0; m < fpga_ntt_size; m <<= 1, mlog++) {
            unsigned rw_x_groups = m;
            unsigned rw_x_group_size = (fpga_ntt_size / 2 / VEC) >> mlog;
            unsigned rw_x_group_size_log =
                get_ntt_log(fpga_ntt_size) - 1 - VEC_LOG - mlog;
            unsigned Xm_group_log = rw_x_group_size_log;

#pragma ivdep array(X)
#pragma ivdep array(X2)
            for (unsigned int k = 0; k < fpga_ntt_size / 2 / VEC; k++) {
                unsigned long curX[VEC * 2] __attribute__((register));
                unsigned long curX_rep[VEC * 2] __attribute__((register));

                unsigned i0 =
                    (k * VEC + 0) >> t_log;  // i is the index of groups
                unsigned j0 =
                    (k * VEC + 0) & (t - 1);  // j is the position of a group

                bool b_rev = j0 >= (t / 2);
                if (t <= VEC) b_rev = 0;

                if (m == 1) {
                    ntt_elements elements =
                        read_channel_intel(ch_ntt_elements_in);
#pragma unroll
                    for (int n = 0; n < VEC; n++) {
                        curX[n] = elements.data[n * 2];
                        curX[n + VEC] = elements.data[n * 2 + 1];
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
                    localX[n] = X[k][n];
                    localX2[n] = X2[rw_pos][n];

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
#if VEC >= 4
                } else if (t == 2) {
#pragma unroll
                    for (int n = 0; n < VEC; n++) {
                        const int cur_t = 2;
                        const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                        const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                        curX_rep[n] = curX[Xn];
                        curX_rep[VEC + n] = curX[Xnt];
                    }
#endif
#if VEC >= 8
                } else if (t == 4) {
#pragma unroll
                    for (int n = 0; n < VEC; n++) {
                        const int cur_t = 4;
                        const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                        const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                        curX_rep[n] = curX[Xn];
                        curX_rep[VEC + n] = curX[Xnt];
                    }
#endif
#if VEC >= 16
                } else if (t == 8) {
#pragma unroll
                    for (int n = 0; n < VEC; n++) {
                        const int cur_t = 8;
                        const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                        const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                        curX_rep[n] = curX[Xn];
                        curX_rep[VEC + n] = curX[Xnt];
                    }
#endif
#if VEC >= 32
                } else if (t == 16) {
#pragma unroll
                    for (int n = 0; n < VEC; n++) {
                        const int cur_t = 16;
                        const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                        const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                        curX_rep[n] = curX[Xn];
                        curX_rep[VEC + n] = curX[Xnt];
                    }
#endif
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

                tf = read_channel_intel(ch_twiddle_factor);

#pragma unroll
                for (int n = 0; n < VEC; n++) {
                    cur_roots[n] = tf.data[n];
                }

                typedef unsigned int __attribute__((__ap_int(VEC * 64)))
                uint_vec_t;
                *(uint_vec_t*)cur_roots =
                    (*(uint_vec_t*)cur_roots) >> (shift_left_elements * 64);

                unsigned select_num = roots_end % VEC - roots_start % VEC + 1;

                unsigned long reorder_roots[VEC];

#pragma unroll
                for (int n = 0; n < VEC; n++) {
                    reorder_roots[n] = cur_roots[n];
                }

                if (select_num == 1) {
#pragma unroll
                    for (int n = 1; n < VEC; n++) {
                        reorder_roots[n] = cur_roots[0];
                    }
                } else if (select_num == 2) {
#pragma unroll
                    for (int n = 0; n < VEC / 2; n++) {
                        reorder_roots[n] = cur_roots[0];
                        reorder_roots[n + VEC / 2] = cur_roots[1];
                    }
                } else if (select_num == 4) {
#pragma unroll
                    for (int n = 0; n < 4; n++) {
                        reorder_roots[n * 2] = cur_roots[n];
                        reorder_roots[n * 2 + 1] = cur_roots[n];
                    }
                }

                ntt_elements elements;
#pragma unroll
                for (int n = 0; n < VEC; n++) {
                    unsigned i =
                        (k * VEC + n) >> t_log;  // i is the index of groups
                    unsigned j = (k * VEC + n) &
                                 (t - 1);  // j is the position of a group
                    unsigned j1 = i * 2 * t;

                    const unsigned long W_op = reorder_roots[n];

                    const int Xn = n / t * (2 * t) + n % t;
                    unsigned long tx = curX_rep[n];
                    unsigned long a = curX_rep[VEC + n];

                    ASSERT(W_op < MAX_MODULUS, "y >= modulus, ntt_ins = %d\n",
                           engine_id);
                    uint64_t W_x_Y =
                        MultiplyUIntMod(a, W_op, coeff_mod, modulus_k);
                    ASSERT(tx < coeff_mod, "x >= modulus, engine_id = %d\n",
                           engine_id);
                    curX[n] = AddUIntMod(tx, W_x_Y, coeff_mod);
                    curX[VEC + n] = SubUIntMod(tx, W_x_Y, coeff_mod);
                    elements.data[n * 2] = curX[n];
                    elements.data[n * 2 + 1] = curX[VEC + n];
                }

                if (m == (fpga_ntt_size / 2)) {
                    write_channel_intel(ch_ntt_elements_out, elements);
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
#if VEC >= 4
                } else if (t == 2) {
#pragma unroll
                    for (int n = 0; n < VEC; n++) {
                        const int cur_t = 2;
                        const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                        const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                        curX_rep[Xn] = curX[n];
                        curX_rep[Xnt] = curX[VEC + n];
                    }
#endif
#if VEC >= 8
                } else if (t == 4) {
#pragma unroll
                    for (int n = 0; n < VEC; n++) {
                        const int cur_t = 4;
                        const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                        const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                        curX_rep[Xn] = curX[n];
                        curX_rep[Xnt] = curX[VEC + n];
                    }
#endif
#if VEC >= 16
                } else if (t == 8) {
#pragma unroll
                    for (int n = 0; n < VEC; n++) {
                        const int cur_t = 8;
                        const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                        const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                        curX_rep[Xn] = curX[n];
                        curX_rep[Xnt] = curX[VEC + n];
                    }
#endif
#if VEC >= 32
                } else if (t == 16) {
#pragma unroll
                    for (int n = 0; n < VEC; n++) {
                        const int cur_t = 16;
                        const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                        const int Xnt = Xn + ((cur_t < VEC) ? cur_t : VEC);
                        curX_rep[Xn] = curX[n];
                        curX_rep[Xnt] = curX[VEC + n];
                    }
#endif
                } else {
#pragma unroll
                    for (int n = 0; n < VEC; n++) {
                        curX_rep[n] = curX[n];
                        curX_rep[VEC + n] = curX[VEC + n];
                    }
                }

#pragma unroll
                for (int n = 0; n < VEC; n++) {
                    X[k][n] = b_rev ? curX_rep[n + VEC] : curX_rep[n];
                    X2[rw_pos][n] = b_rev ? curX_rep[n] : curX_rep[n + VEC];
                }
            }
            t >>= 1;
            t_log -= 1;
        }
    }
}
