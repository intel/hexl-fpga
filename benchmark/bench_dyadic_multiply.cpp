// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <vector>

#include "hexl-fpga.h"

class dyadic_multiply : public benchmark::Fixture {
public:
    void setup_dyadic_multiply_io(uint64_t n_dyadic_multiply,
                                  uint64_t num_moduli, uint64_t coeff_count);
    void bench_dyadic_multiply(std::vector<uint64_t>& out,
                               uint64_t n_dyadic_multiply, uint64_t num_moduli,
                               uint64_t coeff_count);

private:
    std::vector<uint64_t> moduli;
    std::vector<uint64_t> op1;
    std::vector<uint64_t> op2;
};

void dyadic_multiply::setup_dyadic_multiply_io(uint64_t n_dyadic_multiply,
                                               uint64_t num_moduli,
                                               uint64_t coeff_count) {
    for (uint64_t b = 0; b < n_dyadic_multiply; b++) {
        for (uint64_t m = 0; m < num_moduli; m++) {
            moduli.push_back((b + m + 1) * 10);
        }

        for (uint64_t m = 0; m < num_moduli; m++) {
            for (uint64_t i = 0; i < coeff_count; i++) {
                op1.push_back(b + i + 1 + m * coeff_count);
                op2.push_back(b + i + 2 + m * coeff_count);
            }
        }

        for (uint64_t m = 0; m < num_moduli; m++) {
            for (uint64_t i = 0; i < coeff_count; i++) {
                op1.push_back(b + i + 11 + m * coeff_count);
                op2.push_back(b + i + 22 + m * coeff_count);
            }
        }
    }
}

void dyadic_multiply::bench_dyadic_multiply(std::vector<uint64_t>& out,
                                            uint64_t n_dyadic_multiply,
                                            uint64_t num_moduli,
                                            uint64_t coeff_count) {
    intel::hexl::set_worksize_DyadicMultiply(n_dyadic_multiply);
    for (uint64_t b = 0; b < n_dyadic_multiply; b++) {
        uint64_t* pout = &out[0] + b * num_moduli * coeff_count * 3;
        uint64_t* pop1 = &op1[0] + b * num_moduli * coeff_count * 2;
        uint64_t* pop2 = &op2[0] + b * num_moduli * coeff_count * 2;
        uint64_t* pmoduli = &moduli[0] + b * num_moduli;
        intel::hexl::DyadicMultiply(pout, pop1, pop2, coeff_count, pmoduli,
                                    num_moduli);
    }
    intel::hexl::DyadicMultiplyCompleted();
}

BENCHMARK_F(dyadic_multiply, dyadic_multiply_p16384_m7_b1_4096)
(benchmark::State& state) {
    uint64_t coeff_count = 16384 / 2;
    uint64_t num_moduli = 7;
    uint64_t n_dyadic_multiply = 4096;

    setup_dyadic_multiply_io(n_dyadic_multiply, num_moduli, coeff_count);

    std::vector<uint64_t> out(n_dyadic_multiply * 3 * num_moduli * coeff_count,
                              0);

    for (auto st : state) {
        bench_dyadic_multiply(out, n_dyadic_multiply, num_moduli, coeff_count);
    }
}
