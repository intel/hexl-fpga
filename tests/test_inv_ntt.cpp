// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "hexl-fpga.h"
#include "test_utils/ntt.hpp"
static const uint64_t ntt_degree = 16384;
static const uint64_t ntt_prime = 136314881;
namespace hetest {
namespace utils {
enum StimulusType {
    RANDOM = 0,
    RAMP,
    ALL_ONES,
    ALL_ZEROS,
    IMPULSE,
    ALL_MAX_VALUES,
    ALL_MIN_VALUES,
};
template <class T>
static void genStimulusForNTT(std::vector<T>& inVec, T prime,
                              StimulusType stimulusType) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> distrib(0, prime - 1);
    if (stimulusType == RANDOM) {
        for (size_t i = 0; i < inVec.size(); ++i) {
            inVec[i] = distrib(gen);
        }
    } else if (stimulusType == RAMP) {
        for (size_t i = 0; i < inVec.size(); i++) {
            inVec[i] = i;
        }
    } else if (stimulusType == ALL_ZEROS) {
        for (size_t i = 0; i < inVec.size(); i++) {
            inVec[i] = 0;
        }
    } else if (stimulusType == ALL_MAX_VALUES) {
        for (size_t i = 0; i < inVec.size(); i++) {
            inVec[i] = std::numeric_limits<T>::max();
        }
    } else if (stimulusType == ALL_MIN_VALUES) {
        for (size_t i = 0; i < inVec.size(); i++) {
            inVec[i] = std::numeric_limits<T>::min();
        }
    } else if (stimulusType == ALL_ONES) {
        for (size_t i = 0; i < inVec.size(); i++) {
            inVec[i] = 1;
        }
    } else if (stimulusType == IMPULSE) {
        for (size_t i = 0; i < inVec.size(); i++) {
            if (i == 0)
                inVec[i] = 1;
            else
                inVec[i] = 0;
        }
    }
}

}  // namespace utils
}  // namespace hetest
using StimulusType = hetest::utils::StimulusType;

class inv_ntt_test : public ::testing::Test {
public:
    void run_inv_ntt_test(StimulusType stimulusType, uint64_t iterations,
                          uint64_t bitsForPrime);
    void TestBody() override {}

private:
    void load_inv_ntt_data(StimulusType stimulusType, uint64_t iterations,
                           uint64_t bitsForPrime);
    std::vector<std::vector<uint64_t>> input_;
    std::vector<uint64_t> primes_;
};

void inv_ntt_test::load_inv_ntt_data(StimulusType stimulusType,
                                     uint64_t iterations,
                                     uint64_t bitsForPrime) {
    std::random_device rd;
    std::mt19937 gen(rd());
    this->primes_ =
        hetest::utils::GeneratePrimes(iterations, bitsForPrime, ntt_degree);
    for (unsigned i = 0; i < iterations; i++) {
        std::vector<uint64_t> input;
        input.resize(ntt_degree);
        hetest::utils::genStimulusForNTT<uint64_t>(input, primes_[i],
                                                   stimulusType);
        input_.push_back(input);
    }
}
void inv_ntt_test::run_inv_ntt_test(StimulusType stimulusType,
                                    uint64_t iterations,
                                    uint64_t bitsForPrime) {
    load_inv_ntt_data(stimulusType, iterations, bitsForPrime);

    for (unsigned i = 0; i < iterations; i++) {
        hetest::utils::NTT::NTTImpl ntt(ntt_degree, this->primes_[i]);

        const uint64_t* inv_root_of_unity_powers =
            ntt.GetInvRootOfUnityPowersPtr();
        uint64_t inv_ntt_degree =
            hetest::utils::InverseUIntMod(ntt_degree, this->primes_[i]);
        uint64_t W_op = inv_root_of_unity_powers[ntt_degree - 1];
        uint64_t inv_n_w = hetest::utils::MultiplyUIntMod(inv_ntt_degree, W_op,
                                                          this->primes_[i]);

        std::vector<uint64_t> results = input_[i];
        std::vector<uint64_t> inINTT = input_[i];
        std::vector<uint64_t> outINTT = input_[i];
        ntt.ComputeInverse(outINTT.data(), results.data(), 1, 1);
        intel::hexl::_set_worksize_INTT(1);
        intel::hexl::_INTT(results.data(), ntt.GetInvRootOfUnityPowersPtr(),
                           ntt.GetPrecon64InvRootOfUnityPowersPtr(),
                           this->primes_[i], inv_ntt_degree, inv_n_w,
                           ntt_degree);
        intel::hexl::_INTTCompleted();
        ASSERT_EQ(results, outINTT);
    }
}

TEST_F(inv_ntt_test, p16384_INV_INTT_iRAND_iters4_pbits18) {
    inv_ntt_test he_fpga_api;
    he_fpga_api.run_inv_ntt_test(StimulusType::RANDOM, 4, 20);
}

TEST_F(inv_ntt_test, p16384_INV_INTT_iRAMP_iters4_pbits55) {
    inv_ntt_test he_fpga_api;
    he_fpga_api.run_inv_ntt_test(StimulusType::RAMP, 4, 55);
}

TEST_F(inv_ntt_test, p16384_INV_INTT_iALL_ZEROS_iters4_pbits55) {
    inv_ntt_test he_fpga_api;
    he_fpga_api.run_inv_ntt_test(StimulusType::ALL_ZEROS, 4, 55);
}

TEST_F(inv_ntt_test, p16384_INV_INTT_iALL_ONES_iters4_pbits55) {
    inv_ntt_test he_fpga_api;
    he_fpga_api.run_inv_ntt_test(StimulusType::ALL_ONES, 4, 55);
}

TEST_F(inv_ntt_test, p16384_INV_INTT_iALL_MAX_POS_iters4_pbits55) {
    inv_ntt_test he_fpga_api;
    he_fpga_api.run_inv_ntt_test(StimulusType::ALL_MAX_VALUES, 4, 55);
}

TEST_F(inv_ntt_test, p16384_INV_INTT_iALL_MAX_POS_iters4_pbits62) {
    inv_ntt_test he_fpga_api;
    he_fpga_api.run_inv_ntt_test(StimulusType::ALL_MAX_VALUES, 4, 62);
}

TEST_F(inv_ntt_test, p16384_INV_INTT_iIMPULSE_iters4_pbits62) {
    inv_ntt_test he_fpga_api;
    he_fpga_api.run_inv_ntt_test(StimulusType::IMPULSE, 4, 62);
}
TEST_F(inv_ntt_test, p16384_INV_INTT_iRAND_iters4_pbits55) {
    inv_ntt_test he_fpga_api;
    he_fpga_api.run_inv_ntt_test(StimulusType::RANDOM, 4, 55);
}

TEST_F(inv_ntt_test, p16384_INV_INTT_iRAND_iters4_pbits32) {
    inv_ntt_test he_fpga_api;
    he_fpga_api.run_inv_ntt_test(StimulusType::RANDOM, 4, 32);
}
TEST_F(inv_ntt_test, p16384_INV_INTT_iRAND_iters400_pbits55) {
    inv_ntt_test he_fpga_api;
    he_fpga_api.run_inv_ntt_test(StimulusType::RANDOM, 100, 32);
}

TEST_F(inv_ntt_test, p16384_INV_INTT_iRAND_iters410_pbits64) {
    inv_ntt_test he_fpga_api;
    he_fpga_api.run_inv_ntt_test(StimulusType::RANDOM, 10, 62);
}
