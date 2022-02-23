// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "hexl-fpga.h"

class inv_ntt : public benchmark::Fixture {
public:
    void load_ntt_data(size_t work_size);
    void fpga_inv_ntt_test(size_t work_size);

private:
    uint64_t poly_degree_;
    uint64_t coeff_modulus_;
    std::vector<uint64_t> inv_roots_;
    std::vector<uint64_t> inv_precons_;
    std::vector<uint64_t> input_;
    std::vector<uint64_t> results_;
    uint64_t inv_n_;
    uint64_t inv_n_w_;
};

void inv_ntt::load_ntt_data(size_t work_size) {
    std::random_device rd;
    std::mt19937 gen(rd());

    poly_degree_ = 16384;
    coeff_modulus_ = 136314881;

    std::uniform_int_distribution<uint64_t> distrib(0, coeff_modulus_ - 1);

    for (unsigned j = 0; j < work_size * poly_degree_; j++) {
        input_.emplace_back(distrib(gen));
    }

    for (unsigned j = 0; j < poly_degree_; j++) {
        inv_roots_.emplace_back(distrib(gen));
        inv_precons_.emplace_back(distrib(gen));
        results_.emplace_back(0);
    }

    inv_n_ = distrib(gen);
    inv_n_w_ = distrib(gen);
}

void inv_ntt::fpga_inv_ntt_test(size_t work_size) {
    intel::hexl::_set_worksize_INTT(work_size);
    for (unsigned int j = 0; j < work_size; j++) {
        intel::hexl::_INTT(input_.data() + j * poly_degree_, inv_roots_.data(),
                           inv_precons_.data(), coeff_modulus_, inv_n_,
                           inv_n_w_, poly_degree_);
    }
    intel::hexl::_INTTCompleted();
}

BENCHMARK_F(inv_ntt, fpga_inv_ntt_p16384_ws4096)
(benchmark::State& state) {
    size_t work_size = 4096;

    load_ntt_data(work_size);

    for (auto st : state) {
        fpga_inv_ntt_test(work_size);
    }
}
