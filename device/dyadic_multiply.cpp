// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <vector>
#include "mod_ops.hpp"
#include "./utils/pipe_def_marcos.hpp"
#include "../common/types.hpp"
#define DATA_PATH 8
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
    ubitwidth_t* operands_in_ddr;
    moduli_info_t* moduli_info;
    ubitwidth_t n;
    ubitwidth_t n_moduli;
    ubitwidth_t n_batch;
} operands_fetcher_info;

typedef struct {
    operands_fetcher_info data_info;
    ubitwidth_t* results_ddr;
    int tag;
} input_t;

typedef struct {
    ubitwidth_t* results_ddr;
    ubitwidth_t n;
    ubitwidth_t n_moduli;
    ubitwidth_t n_batch;
    int tag;
} output_t;

__extension__ typedef unsigned __int128 fpga_uint128_t;

static void generate_moduli_info_t(std::vector<moduli_info_t>& moduli_info_vec,
                                   uint64_t* moduli, uint64_t n_moduli,
                                   uint64_t n_batch) {
    for (uint64_t batch = 0; batch < n_batch; batch++) {
        for (uint64_t i = 0; i < n_moduli; i++) {
            uint64_t modulus = moduli[batch * n_moduli + i];
            uint64_t len = uint64_t(floorl(std::log2l(modulus)) - 1);
            fpga_uint128_t n = fpga_uint128_t(1) << (len + 64);
            uint64_t barr_lo = uint64_t(n / modulus);
            moduli_info_vec.push_back((moduli_info_t){modulus, len, barr_lo});
        }
    }
}

// pipe arrays definitaion using macro.
defLongPipe(input_pipe, input_t, 16) defLongPipe(output_pipe, output_t, 16);
defLongPipe(operands_fetcher_info_pipe, operands_fetcher_info, 16);
defLongPipe(output_results_pipe0, sycl::ulong2, (16384 << 2));
defLongPipe(output_results_pipe1, sycl::ulong2, (16384 << 2));
defLongPipe(output_results_pipe2, sycl::ulong2, (16384 << 2));
defLongPipe(operands_in_pipe, operands_t, (16384 >> 1));
defLongPipe(modulus_info_pipe, moduli_unit_t, 16);

class input_fifo_kern_usm;
void input_fifo_kernel(ubitwidth_t* operand1_host, ubitwidth_t* operand2_host,
                       ubitwidth_t n, moduli_info_t* moduli_info_host,
                       ubitwidth_t n_moduli, int tag,
                       ubitwidth_t* operands_in_ddr_dev,
                       ubitwidth_t* results_ddr_dev, ubitwidth_t n_batch) {
    sycl::host_ptr<ubitwidth_t> operand1_in_svm(operand1_host);
    sycl::host_ptr<ubitwidth_t> operand2_in_svm(operand2_host);
    sycl::host_ptr<moduli_info_t> moduli_info(moduli_info_host);

    sycl::device_ptr<ubitwidth_t> operands_in_ddr(operands_in_ddr_dev);
    sycl::device_ptr<ubitwidth_t> results_ddr(results_ddr_dev);

    ubitwidth_t ddr_offset = 0;
    ubitwidth_t nn = n >> 1;

    [[intel::ivdep]] for (ubitwidth_t batch = 0; batch < n_batch; batch++) {
        ubitwidth_t batch_offset = batch * 2 * n_moduli * n;

        [[intel::ivdep]] [[intel::initiation_interval(
            1)]] for (ubitwidth_t m = 0; m < n_moduli; m++) {
            ubitwidth_t poly0_offset = batch_offset + m * n;
            ubitwidth_t poly1_offset = batch_offset + (m + n_moduli) * n;

            [[intel::ivdep]] for (ubitwidth_t i = 0; i < nn; i++) {
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

    sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);

    operands_fetcher_info data_info;
    data_info.operands_in_ddr = (ubitwidth_t*)operands_in_ddr;
    data_info.moduli_info = (moduli_info_t*)moduli_info;
    data_info.n = n;
    data_info.n_moduli = n_moduli;
    data_info.n_batch = n_batch;

    input_t input_info;
    input_info.data_info = data_info;
    input_info.results_ddr = (ubitwidth_t*)results_ddr;
    input_info.tag = tag;

    input_pipe::write(input_info);
}

class output_nb_fifo_kern_usm;
void output_nb_fifo_kernel(ubitwidth_t* results_host, int* tag_host,
                           int* output_valid_host) {
    sycl::host_ptr<ubitwidth_t> results_svm(results_host);
    sycl::host_ptr<int> tag(tag_host);
    sycl::host_ptr<int> output_valid(output_valid_host);

    bool valid = 0;
    output_t output_info;
    while (!valid) {
        output_info = output_pipe::read(valid);
    }

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

                [[intel::ivdep]] for (ubitwidth_t i = 0; i < nn; i++) {
                    ubitwidth_t p0 = poly0_offset + i * DATA_PATH;
                    ubitwidth_t p1 = poly1_offset + i * DATA_PATH;
                    ubitwidth_t p2 = poly2_offset + i * DATA_PATH;

                    ubitwidth_t results0[DATA_PATH];
                    ubitwidth_t results1[DATA_PATH];
                    ubitwidth_t results2[DATA_PATH];

#pragma unroll
                    [[intel::ivdep]] for (ubitwidth_t j = 0; j < DATA_PATH;
                                          j++) {
                        results0[j] = output_info.results_ddr[p0 + j];
                        results1[j] = output_info.results_ddr[p1 + j];
                        results2[j] = output_info.results_ddr[p2 + j];
                    }

#pragma unroll
                    [[intel::ivdep]] for (ubitwidth_t j = 0; j < DATA_PATH;
                                          j++) {
                        results_svm[p0 + j] = results0[j];
                    }

#pragma unroll
                    [[intel::ivdep]] for (ubitwidth_t j = 0; j < DATA_PATH;
                                          j++) {
                        results_svm[p1 + j] = results1[j];
                    }

#pragma unroll
                    [[intel::ivdep]] for (ubitwidth_t j = 0; j < DATA_PATH;
                                          j++) {
                        results_svm[p2 + j] = results2[j];
                    }
                }
            }
        }

        *tag = output_info.tag;
    }

    *output_valid = valid;
}

class dyadic_multiply_ey_kern;
void dyadic_multiply_eu_kernel() {
    while (1) {
        moduli_unit_t modulus = modulus_info_pipe::read();
        ubitwidth_t nn = modulus.n >> 1;
        ubitwidth_t moduli = modulus.moduli;
        ubitwidth_t twice_moduli = modulus.twice_moduli;
        ubitwidth_t len = modulus.len;
        ubitwidth_t barr_lo = modulus.barr_lo;

        for (ubitwidth_t i = 0; i < nn; i++) {
            sycl::ulong2 results0, results1, results2;

            operands_t operands = operands_in_pipe::read();

#pragma unroll
            for (ubitwidth_t j = 0; j < 2; j++) {
                ubitwidth_t m0 = MultMod(operands.op[j].x0, operands.op[j].y1,
                                         moduli, twice_moduli, len, barr_lo);
                ubitwidth_t m1 = MultMod(operands.op[j].x1, operands.op[j].y0,
                                         moduli, twice_moduli, len, barr_lo);

                results1[j] = AddMod(m0, m1, moduli);
                results0[j] = MultMod(operands.op[j].x0, operands.op[j].y0,
                                      moduli, twice_moduli, len, barr_lo);
                results2[j] = MultMod(operands.op[j].x1, operands.op[j].y1,
                                      moduli, twice_moduli, len, barr_lo);
            }

            output_results_pipe0::write(results0);
            output_results_pipe1::write(results1);
            output_results_pipe2::write(results2);
        }
    }
}

class operands_fetcher_kern;
void operands_fetcher_kernel() {
    while (1) {
        operands_fetcher_info info = operands_fetcher_info_pipe::read();
        ubitwidth_t ddr_offset = 0;
        ubitwidth_t nn = info.n >> 1;

        for (unsigned int batch = 0; batch < info.n_batch; batch++) {
            unsigned int batch_mod_offset = batch * info.n_moduli;
            for (unsigned int m = 0; m < info.n_moduli; m++) {
                unsigned int m_offset = batch_mod_offset + m;

                moduli_unit_t modulus_info;
                modulus_info.n = info.n;
                modulus_info.moduli = info.moduli_info[m_offset].moduli;
                modulus_info.twice_moduli = info.moduli_info[m_offset].moduli
                                            << 1;
                modulus_info.len = info.moduli_info[m_offset].len;
                modulus_info.barr_lo = info.moduli_info[m_offset].barr_lo;

                modulus_info_pipe::write(modulus_info);
            }

            [[intel::ivdep]] [[intel::initiation_interval(
                1)]] for (ubitwidth_t m = 0; m < info.n_moduli; m++) {
                for (ubitwidth_t i = 0; i < nn; i++) {
                    operands_t operands;

#pragma unroll
                    for (ubitwidth_t j = 0; j < 2; j++) {
                        operands.op[j].x0 = info.operands_in_ddr[ddr_offset++];
                        operands.op[j].x1 = info.operands_in_ddr[ddr_offset++];
                        operands.op[j].y0 = info.operands_in_ddr[ddr_offset++];
                        operands.op[j].y1 = info.operands_in_ddr[ddr_offset++];
                    }
                    operands_in_pipe::write(operands);
                }
            }
        }
    }
}

class dyadic_multiply_kern;
void dyadic_multiply_kernel() {
    while (1) {
        input_t input_info = input_pipe::read();
        operands_fetcher_info info = input_info.data_info;
        operands_fetcher_info_pipe::write(info);

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

                [[intel::ivdep]] for (ubitwidth_t i = 0; i < nn; i++) {
                    ubitwidth_t p0 = result_poly0_offset + i * DATA_PATH;
                    ubitwidth_t p1 = result_poly1_offset + i * DATA_PATH;
                    ubitwidth_t p2 = result_poly2_offset + i * DATA_PATH;

                    sycl::ulong8 results0, results1, results2;
                    for (ubitwidth_t k = 0; k < 4; k++) {
                        sycl::ulong2 results0_int =
                            output_results_pipe0::read();
                        sycl::ulong2 results1_int =
                            output_results_pipe1::read();
                        sycl::ulong2 results2_int =
                            output_results_pipe2::read();
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

        sycl::atomic_fence(sycl::memory_order::seq_cst,
                           sycl::memory_scope::device);

        output_pipe::write(output_info);
    }
}

extern "C" {

// the interface for dyadic multiply, aligned with hexl-fpga dyadic_multiply.cl
// file.

sycl::event input_fifo_usm(sycl::queue& q, ubitwidth_t* operand1_in_svm,
                           ubitwidth_t* operand2_in_svm, ubitwidth_t n,
                           moduli_info_t* moduli_info, ubitwidth_t n_moduli,
                           int tag, ubitwidth_t* operands_in_ddr,
                           ubitwidth_t* results_ddr, ubitwidth_t n_batch) {
    sycl::event e = q.submit([&](sycl::handler& h) {
        h.single_task<input_fifo_kern_usm>([=
        ]() [[intel::kernel_args_restrict]] {
            input_fifo_kernel(operand1_in_svm, operand2_in_svm, n, moduli_info,
                              n_moduli, tag, operands_in_ddr, results_ddr,
                              n_batch);
        });
    });

    return e;
}

sycl::event output_nb_fifo_usm(sycl::queue& q, ubitwidth_t* results_in_svm,
                               int* tag, int* output_valid) {
    sycl::event e = q.submit([&](sycl::handler& h) {
        h.single_task<output_nb_fifo_kern_usm>([=
        ]() [[intel::kernel_args_restrict]] {
            output_nb_fifo_kernel(results_in_svm, tag, output_valid);
        });
    });

    return e;
}

void dyadic_multiply_eu(sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.single_task<dyadic_multiply_ey_kern>(
            [=]() { dyadic_multiply_eu_kernel(); });
    });
}

void operands_fetcher(sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.single_task<operands_fetcher_kern>(
            [=]() { operands_fetcher_kernel(); });
    });
}

void dyadic_multiply(sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.single_task<dyadic_multiply_kern>(
            [=]() { dyadic_multiply_kernel(); });
    });
}

void submit_autorun_kernels(sycl::queue& q) {
    // submit all autorun kernels
    dyadic_multiply(q);
    operands_fetcher(q);
    dyadic_multiply_eu(q);
}

}  // end of extern "C"
