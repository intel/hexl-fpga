// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mod_ops.h"

#pragma OPENCL EXTENSION cl_intel_channels : enable

#define TASK __attribute__((max_global_work_dim(0)))

#define DATA_PATH 8

typedef struct {
    ubitwidth_t moduli;
    ubitwidth_t len;
    ubitwidth_t barr_lo;
} moduli_info_t;

typedef struct {
    ubitwidth_t n;
    ubitwidth_t moduli;
    ubitwidth_t twice_moduli;
    ubitwidth_t len;
    ubitwidth_t barr_lo;
} moduli_unit_t;

typedef struct {
    struct {
        ubitwidth_t x0;
        ubitwidth_t x1;
        ubitwidth_t y0;
        ubitwidth_t y1;
    } op[2];
} operands_t;

typedef struct {
    __global ubitwidth_t* restrict operands_in_ddr;
    __global moduli_info_t* moduli_info;
    ubitwidth_t n;
    ubitwidth_t n_moduli;
    ubitwidth_t n_batch;
} operands_fetcher_info;

typedef struct {
    operands_fetcher_info data_info;
    __global ubitwidth_t* restrict results_ddr;
    int tag;
} input_t;

typedef struct {
    __global ubitwidth_t* restrict results_ddr;
    ubitwidth_t n;
    ubitwidth_t n_moduli;
    ubitwidth_t n_batch;
    int tag;
} output_t;

channel input_t input_channel __attribute__((depth(16)));
channel output_t output_channel __attribute__((depth(16)));
channel operands_fetcher_info operands_fetcher_info_channel
    __attribute__((depth(16)));
channel ulong2 output_results_channel0 __attribute__((depth(16384 << 2)));
channel ulong2 output_results_channel1 __attribute__((depth(16384 << 2)));
channel ulong2 output_results_channel2 __attribute__((depth(16384 << 2)));
channel operands_t operands_in_channel __attribute__((depth(16384 >> 1)));
channel moduli_unit_t modulus_info_channel __attribute__((depth(16)));

TASK kernel void input_fifo(__global __attribute((buffer_location("host")))
                            ubitwidth_t* restrict operand1_in_svm,
                            __global __attribute((buffer_location("host")))
                            ubitwidth_t* restrict operand2_in_svm,
                            ubitwidth_t n,
                            __global __attribute((buffer_location("host")))
                            moduli_info_t* restrict moduli_info,
                            ubitwidth_t n_moduli, int tag,
                            __global __attribute((buffer_location("device")))
                            ubitwidth_t* restrict operands_in_ddr,
                            __global __attribute((buffer_location("device")))
                            ubitwidth_t* restrict results_ddr,
                            ubitwidth_t n_batch) {
    ubitwidth_t ddr_offset = 0;
    ubitwidth_t nn = n >> 1;

#pragma ivdep
    for (ubitwidth_t batch = 0; batch < n_batch; batch++) {
        ubitwidth_t batch_offset = batch * 2 * n_moduli * n;
#pragma ivdep
#pragma ii 1
        for (ubitwidth_t m = 0; m < n_moduli; m++) {
            ubitwidth_t poly0_offset = batch_offset + m * n;
            ubitwidth_t poly1_offset = batch_offset + (m + n_moduli) * n;
#pragma ivdep
            for (ubitwidth_t i = 0; i < nn; i++) {
                ubitwidth_t p0 = poly0_offset + i * 2;
                ubitwidth_t p1 = poly1_offset + i * 2;

                ubitwidth_t data[8];
#pragma unroll
                for (ubitwidth_t j = 0; j < 2; j++) {
                    data[4 * j + 0] = operand1_in_svm[p0 + j];
                    data[4 * j + 1] = operand1_in_svm[p1 + j];
                    data[4 * j + 2] = operand2_in_svm[p0 + j];
                    data[4 * j + 3] = operand2_in_svm[p1 + j];
                }
#pragma unroll
                for (ubitwidth_t j = 0; j < DATA_PATH; j++) {
                    operands_in_ddr[ddr_offset] = data[j];
                    ddr_offset++;
                }
            }
        }
    }
    mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);

    operands_fetcher_info data_info;
    data_info.operands_in_ddr =
        (__global ubitwidth_t * restrict) operands_in_ddr;
    data_info.moduli_info = (__global moduli_info_t * restrict) moduli_info;
    data_info.n = n;
    data_info.n_moduli = n_moduli;
    data_info.n_batch = n_batch;

    input_t input_info;
    input_info.data_info = data_info;
    input_info.results_ddr = (__global ubitwidth_t * restrict) results_ddr;
    input_info.tag = tag;

    write_channel_intel(input_channel, input_info);
}

TASK kernel void output_nb_fifo(
    __global __attribute((buffer_location("host")))
    ubitwidth_t* restrict results_svm,
    __global __attribute((buffer_location("host"))) int* restrict tag,
    __global
    __attribute((buffer_location("host"))) int* restrict output_valid) {
    bool valid = 0;
    output_t output_info = read_channel_nb_intel(output_channel, &valid);

    if (valid) {
        ubitwidth_t nn = output_info.n >> 3;
        for (ubitwidth_t batch = 0; batch < output_info.n_batch; batch++) {
            ubitwidth_t batch_poly_offset = batch * output_info.n_moduli;
            for (ubitwidth_t m = 0; m < output_info.n_moduli; m++) {
                ubitwidth_t poly0_offset =
                    (batch_poly_offset * 3 + m) * output_info.n;
                ubitwidth_t poly1_offset =
                    (batch_poly_offset * 3 + m + output_info.n_moduli) *
                    output_info.n;
                ubitwidth_t poly2_offset =
                    (batch_poly_offset * 3 + m + 2 * output_info.n_moduli) *
                    output_info.n;

#pragma ivdep
                for (ubitwidth_t i = 0; i < nn; i++) {
                    ubitwidth_t p0 = poly0_offset + i * DATA_PATH;
                    ubitwidth_t p1 = poly1_offset + i * DATA_PATH;
                    ubitwidth_t p2 = poly2_offset + i * DATA_PATH;

                    ubitwidth_t results0[DATA_PATH];
                    ubitwidth_t results1[DATA_PATH];
                    ubitwidth_t results2[DATA_PATH];

#pragma ivdep
#pragma unroll
                    for (ubitwidth_t j = 0; j < DATA_PATH; j++) {
                        results0[j] = output_info.results_ddr[p0 + j];
                        results1[j] = output_info.results_ddr[p1 + j];
                        results2[j] = output_info.results_ddr[p2 + j];
                    }

#pragma ivdep
#pragma unroll
                    for (ubitwidth_t j = 0; j < DATA_PATH; j++) {
                        results_svm[p0 + j] = results0[j];
                    }
#pragma ivdep
#pragma unroll
                    for (ubitwidth_t j = 0; j < DATA_PATH; j++) {
                        results_svm[p1 + j] = results1[j];
                    }
#pragma ivdep
#pragma unroll
                    for (ubitwidth_t j = 0; j < DATA_PATH; j++) {
                        results_svm[p2 + j] = results2[j];
                    }
                }
            }
        }
        *tag = output_info.tag;
    }

    *output_valid = valid;
}

__attribute__((autorun)) TASK kernel void dyadic_multiply_eu() {
    moduli_unit_t modulus = read_channel_intel(modulus_info_channel);
    ubitwidth_t nn = modulus.n >> 1;
    ubitwidth_t moduli = modulus.moduli;
    ubitwidth_t twice_moduli = modulus.twice_moduli;
    ubitwidth_t len = modulus.len;
    ubitwidth_t barr_lo = modulus.barr_lo;

    for (ubitwidth_t i = 0; i < nn; i++) {
        ulong2 results0, results1, results2;

        operands_t operands = read_channel_intel(operands_in_channel);

#pragma unroll
        for (ubitwidth_t j = 0; j < 2; j++) {
            ubitwidth_t m0 = MultMod(operands.op[j].x0, operands.op[j].y1,
                                     moduli, twice_moduli, len, barr_lo);
            ubitwidth_t m1 = MultMod(operands.op[j].x1, operands.op[j].y0,
                                     moduli, twice_moduli, len, barr_lo);

            results1[j] = AddMod(m0, m1, moduli);
            results0[j] = MultMod(operands.op[j].x0, operands.op[j].y0, moduli,
                                  twice_moduli, len, barr_lo);
            results2[j] = MultMod(operands.op[j].x1, operands.op[j].y1, moduli,
                                  twice_moduli, len, barr_lo);
        }
        write_channel_intel(output_results_channel0, results0);
        write_channel_intel(output_results_channel1, results1);
        write_channel_intel(output_results_channel2, results2);
    }
}

__attribute__((autorun)) TASK kernel void operands_fetcher() {
    operands_fetcher_info info =
        read_channel_intel(operands_fetcher_info_channel);

    ubitwidth_t ddr_offset = 0;
    ubitwidth_t nn = info.n >> 1;
    for (unsigned int batch = 0; batch < info.n_batch; batch++) {
        unsigned int batch_mod_offset = batch * info.n_moduli;
        for (unsigned int m = 0; m < info.n_moduli; m++) {
            unsigned int m_offset = batch_mod_offset + m;

            moduli_unit_t modulus_info;
            modulus_info.n = info.n;
            modulus_info.moduli = info.moduli_info[m_offset].moduli;
            modulus_info.twice_moduli = info.moduli_info[m_offset].moduli << 1;
            modulus_info.len = info.moduli_info[m_offset].len;
            modulus_info.barr_lo = info.moduli_info[m_offset].barr_lo;

            write_channel_intel(modulus_info_channel, modulus_info);
        }

#pragma ivdep
#pragma ii 1
        for (ubitwidth_t m = 0; m < info.n_moduli; m++) {
            for (ubitwidth_t i = 0; i < nn; i++) {
                operands_t operands;
#pragma unroll
                for (ubitwidth_t j = 0; j < 2; j++) {
                    operands.op[j].x0 = info.operands_in_ddr[ddr_offset++];
                    operands.op[j].x1 = info.operands_in_ddr[ddr_offset++];
                    operands.op[j].y0 = info.operands_in_ddr[ddr_offset++];
                    operands.op[j].y1 = info.operands_in_ddr[ddr_offset++];
                }
                write_channel_intel(operands_in_channel, operands);
            }
        }
    }
}

__attribute__((autorun)) TASK kernel void dyadic_multiply() {
    while (1) {
        input_t input_info = read_channel_intel(input_channel);
        operands_fetcher_info info = input_info.data_info;
        write_channel_intel(operands_fetcher_info_channel, info);

        output_t output_info;
        output_info.tag = input_info.tag;
        output_info.n = info.n;
        output_info.n_batch = info.n_batch;
        output_info.n_moduli = info.n_moduli;
        output_info.results_ddr = input_info.results_ddr;

        ubitwidth_t nn = info.n >> 3;

        for (ubitwidth_t batch = 0; batch < info.n_batch; batch++) {
            ubitwidth_t batch_poly_offset = batch * info.n_moduli;
            for (ubitwidth_t m = 0; m < info.n_moduli; m++) {
                ubitwidth_t result_poly0_offset =
                    (batch_poly_offset * 3 + m) * info.n;
                ubitwidth_t result_poly1_offset =
                    (batch_poly_offset * 3 + m + info.n_moduli) * info.n;
                ubitwidth_t result_poly2_offset =
                    (batch_poly_offset * 3 + m + 2 * info.n_moduli) * info.n;

#pragma ivdep
                for (ubitwidth_t i = 0; i < nn; i++) {
                    ubitwidth_t p0 = result_poly0_offset + i * DATA_PATH;
                    ubitwidth_t p1 = result_poly1_offset + i * DATA_PATH;
                    ubitwidth_t p2 = result_poly2_offset + i * DATA_PATH;

                    ulong8 results0, results1, results2;
                    for (ubitwidth_t k = 0; k < 4; k++) {
                        ulong2 results0_int =
                            read_channel_intel(output_results_channel0);
                        ulong2 results1_int =
                            read_channel_intel(output_results_channel1);
                        ulong2 results2_int =
                            read_channel_intel(output_results_channel2);
                        results0[2 * k] = results0_int[0];
                        results1[2 * k] = results1_int[0];
                        results2[2 * k] = results2_int[0];
                        results0[2 * k + 1] = results0_int[1];
                        results1[2 * k + 1] = results1_int[1];
                        results2[2 * k + 1] = results2_int[1];
                    }
#pragma unroll
                    for (ubitwidth_t j = 0; j < DATA_PATH; j++) {
                        output_info.results_ddr[p0 + j] = results0[j];
                    }
#pragma unroll
                    for (ubitwidth_t j = 0; j < DATA_PATH; j++) {
                        output_info.results_ddr[p1 + j] = results1[j];
                    }
#pragma unroll
                    for (ubitwidth_t j = 0; j < DATA_PATH; j++) {
                        output_info.results_ddr[p2 + j] = results2[j];
                    }
                }
            }
        }
        mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_CHANNEL_MEM_FENCE);
        write_channel_intel(output_channel, output_info);
    }
}
