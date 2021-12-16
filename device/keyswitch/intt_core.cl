// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "channels.h"
#include "common.h"

void _intt_backward(channel uint64_t ch_intt_elements_in,
                    channel intt_elements ch_intt_elements) {
    int data_index = 0;
    intt_elements elements;
    while (true) {
#pragma unroll
        for (int i = 0; i < VEC * 2 - 1; i++) {
            elements.data[i] = elements.data[i + 1];
        }
        elements.data[VEC * 2 - 1] = read_channel_intel(ch_intt_elements_in);
        if (data_index == (VEC * 2 - 1)) {
            write_channel_intel(ch_intt_elements, elements);
        }
        data_index = (data_index + 1) % (VEC * 2);
    }
}

void _intt_forward(channel uint64_t ch_intt_elements_out,
                   channel intt_elements ch_intt_elements) {
    int data_index = 0;
    intt_elements elements;
    while (true) {
        if (data_index == 0) {
            elements = read_channel_intel(ch_intt_elements);
        }
        uint64_t data = elements.data[0];
#pragma unroll
        for (int i = 0; i < VEC * 2 - 1; i++) {
            elements.data[i] = elements.data[i + 1];
        }
        write_channel_intel(ch_intt_elements_out, data);
        data_index = (data_index + 1) % (VEC * 2);
    }
}

void _intt_internal(channel ulong4 ch_intt_modulus,
                    channel intt_elements ch_intt_elements_in,
                    channel intt_elements ch_intt_elements_out,
                    channel ulong4 ch_normalize,
                    __global HOST_MEM uint64_t* restrict twiddle_factors,
                    invn_t g_inv_n, uint54_t local_roots[][VEC],
                    const unsigned int key_modulus_start,
                    const unsigned int key_modulus_end,
                    uint64_t output_mod_factor, unsigned int engine_id) {
    int n = (int)FPGA_NTT_SIZE;

    unsigned long X[FPGA_NTT_SIZE / VEC / 2][VEC];
    unsigned long X2[FPGA_NTT_SIZE / VEC / 2][VEC];

    for (int k = key_modulus_start; k < key_modulus_end; k++) {
        unsigned int offset = FPGA_NTT_SIZE * k * 4;
        for (int i = 0; i < FPGA_NTT_SIZE / VEC; i++) {
#pragma unroll
            for (int j = 0; j < VEC; j++) {
                local_roots[FPGA_NTT_SIZE / VEC * (k - key_modulus_start) +
                            i][j] =
                    __pipelined_load(twiddle_factors + offset + i * VEC + j);
            }
        }
    }

    while (true) {
        ulong4 modulus = read_channel_intel(ch_intt_modulus);
        ulong prime = modulus.s0;
        ulong prime_k = modulus.s3;
        unsigned int key_modulus_idx = modulus.s2;
        unsigned long twice_mod = prime << 1;
        unsigned t = 1;
        unsigned logt = 0;
        unsigned int g_elements_index = 0;

        unsigned roots_acc =
            FPGA_NTT_SIZE * (key_modulus_idx - key_modulus_start);

        ulong4 inv_n_u4 = g_inv_n.data[key_modulus_idx];
        modulus.s1 = inv_n_u4.s0;
        write_channel_intel(ch_normalize, modulus);

        // Normalize the Transform by N
        // Stages
        for (unsigned m = (n >> 1); m >= 1; m >>= 1) {
            bool b_first_stage = t == 1;

            unsigned rw_x_groups_log =
                FPGA_NTT_SIZE_LOG - 1 - VEC_LOG - logt + VEC_LOG;
            unsigned rw_x_groups = 1 << rw_x_groups_log;
            unsigned rw_x_group_size_log = logt - VEC_LOG;
            unsigned rw_x_group_size = 1 << rw_x_group_size_log;
            unsigned Xm_group_log = rw_x_group_size_log - 1;

// Flights
#pragma ivdep array(X)
#pragma ivdep array(X2)
            for (unsigned k = 0; k < FPGA_NTT_SIZE / 2 / VEC; k++) {
                unsigned long curX[VEC * 2] __attribute__((register));
                unsigned long curX_rep[VEC * 2] __attribute__((register));

                unsigned i0 =
                    (k * VEC + 0) >> logt;  // i is the index of groups
                unsigned j0 =
                    (k * VEC + 0) & (t - 1);  // j is the position of a group
                unsigned j10 = i0 << (logt + 1);

                bool b_rev = ((k >> rw_x_group_size_log) & 1);
                if (t < VEC) b_rev = 0;

                if (b_first_stage) {
                    intt_elements elements =
                        read_channel_intel(ch_intt_elements_in);
#pragma unroll
                    for (int n = 0; n < VEC * 2; n++) {
                        curX[n] = elements.data[n];
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
                    rw_pos = FPGA_NTT_SIZE / 2 / VEC - 1 - k;
                }
                unsigned Xm_group_index = k >> Xm_group_log;
                bool b_X = !(Xm_group_index & 1);
                if (t <= VEC) {
                    b_X = true;
                }

#pragma unroll
                for (unsigned n = 0; n < VEC; n++) {
                    unsigned i =
                        (k * VEC + n) >> logt;  // i is the index of groups
                    unsigned j = (k * VEC + n) &
                                 (t - 1);  // j is the position of a group
                    unsigned j1 = i * 2 * t;

                    localX[n] = X[k][n];
                    localX2[n] = X2[rw_pos][n];

                    if (b_first_stage) {
                        // curX[n] = elements_in.data[n];
                        // curX[n + VEC] = elements_in.data[n + VEC];
                    } else {
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
                unsigned shift_left_elements = (roots_acc + i0) % VEC;
                unsigned ivec = (k * VEC + VEC - 1) >> logt;
                unsigned long cur_roots[VEC];
                unsigned long cur_precons[VEC];

#pragma unroll
                for (int n = 0; n < VEC; n++) {
                    cur_roots[n] = local_roots[(roots_acc + i0) / VEC][n];
                }

                typedef unsigned int __attribute__((__ap_int(VEC * 64)))
                uint_vec_t;
                *(uint_vec_t*)cur_roots =
                    (*(uint_vec_t*)cur_roots) >> (shift_left_elements * 64);

                unsigned select_num =
                    (roots_acc + ivec) % VEC - (roots_acc + i0) % VEC + 1;

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

                intt_elements elements;
#pragma unroll
                for (int n = 0; n < VEC; n++) {
                    unsigned i =
                        (k * VEC + n) >> logt;  // i is the index of groups
                    unsigned j = (k * VEC + n) &
                                 (t - 1);  // j is the position of a group
                    unsigned j1 = i * 2 * t + j;
                    unsigned j2 = j1 + t;

                    unsigned long W_op = reorder_roots[n];
                    unsigned long tx = 0;
                    unsigned long ty = 0;

                    // Butterfly
                    unsigned long x_j1 = curX_rep[n];
                    unsigned long x_j2 = curX_rep[VEC + n];

                    // X', Y' = X + Y (mod q), W(X - Y) (mod q).
                    curX[n] = AddUIntMod(x_j1, x_j2, prime);
                    curX[VEC + n] = MultiplyUIntMod(
                        SubUIntMod(x_j1, x_j2, prime), W_op, prime, prime_k);

                    elements.data[n * 2] = curX[n];
                    elements.data[n * 2 + 1] = curX[VEC + n];
                }
                if (m == 1) {
                    write_channel_intel(ch_intt_elements_out, elements);
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

            roots_acc += m;
            t <<= 1;
            logt++;
        }
    }
    return;
}

void _intt_normalize(channel uint64_t ch_intt_elements_out_inter,
                     channel uint64_t ch_intt_elements_out,
                     channel ulong4 ch_normalize) {
    unsigned i = 0;
    ulong4 moduli;

    while (true) {
        if (i == 0) {
            moduli = read_channel_intel(ch_normalize);
        }
        uint64_t data = read_channel_intel(ch_intt_elements_out_inter);
        data = MultiplyUIntMod(data, moduli.s1, moduli.s0, moduli.s3);
        write_channel_intel(ch_intt_elements_out, data);
        STEP(i, FPGA_NTT_SIZE);
    }
}
