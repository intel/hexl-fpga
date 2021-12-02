// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "hexl-fpga.h"

class hexl_fpga : public ::testing::Test {
public:
    void ntt_test();

    void TestBody() override{};

private:
    void load_ntt_data();

    std::vector<uint64_t> poly_degree_;
    std::vector<uint64_t> coeff_modulus_;
    std::vector<std::vector<uint64_t>> roots_;
    std::vector<std::vector<uint64_t>> precons_;
    std::vector<std::vector<uint64_t>> input_;
    std::vector<std::vector<uint64_t>> expected_;
};

void hexl_fpga::load_ntt_data() {
    std::random_device rd;
    std::mt19937 gen(rd());
    uint64_t n = 16384;
    uint64_t prime = 136314881;

    for (unsigned i = 0; i < 10; i++) {
        poly_degree_.push_back(n);
        coeff_modulus_.push_back(prime);

        std::uniform_int_distribution<uint64_t> distrib(0, prime - 1);

        std::vector<uint64_t> root, precon, input, out;

        for (unsigned j = 0; j < n; j++) {
            root.push_back(distrib(gen));
            precon.push_back(distrib(gen));
            input.push_back(distrib(gen));
            out.push_back(distrib(gen) / 2);
        }

        roots_.push_back(root);
        precons_.push_back(precon);
        input_.push_back(input);
        expected_.push_back(out);
    }

    return;
}

void hexl_fpga::ntt_test() {
    load_ntt_data();

    for (unsigned i = 0; i < input_.size(); i++) {
        std::vector<uint64_t> results = input_[i];

        intel::hexl::set_worksize_NTT(1);
        intel::hexl::NTT(results.data(), roots_[i].data(), precons_[i].data(),
                         coeff_modulus_[i], poly_degree_[i]);
        intel::hexl::NTTCompleted();
        intel::hexl::set_worksize_INTT(1);
        intel::hexl::INTT(results.data(), roots_[i].data(), precons_[i].data(),
                          coeff_modulus_[i], 1, 1, poly_degree_[i]);
        intel::hexl::INTTCompleted();
    }

    return;
}

TEST_F(hexl_fpga, p16384_ntt) {
    hexl_fpga he_fpga_api;
    he_fpga_api.ntt_test();
}
