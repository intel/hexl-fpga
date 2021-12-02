// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"
#include "hexl-fpga.h"

class dyadic_multiply_test : public ::testing::Test {
public:
    void test_dyadic_multiply(uint64_t num_dyadic_multiply, uint64_t num_moduli,
                              uint64_t coeff_count, bool death = false);
    void test_matrix_dyadic_multiply(uint64_t n_rows, uint64_t n_columns,
                                     uint64_t num_moduli, uint64_t coeff_count);

    void TestBody() override{};

private:
    void setup_dyadic_io(uint64_t num_dyadic_multiply, uint64_t num_moduli,
                         uint64_t coeff_count);
    uint64_t input_size;
    uint64_t coeff_count;
    uint64_t num_moduli;
    uint64_t num_dyadic_multiply;
    std::vector<uint64_t> moduli;
    std::vector<uint64_t> op1;
    std::vector<uint64_t> op2;
    std::vector<uint64_t> exp_out;
};

void dyadic_multiply_test::setup_dyadic_io(uint64_t num_dyadic_multiply,
                                           uint64_t num_moduli,
                                           uint64_t coeff_count) {
    for (uint64_t b = 0; b < num_dyadic_multiply; b++) {
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

        for (uint64_t m = 0; m < num_moduli; m++) {
            uint64_t poly0_offset =
                b * num_moduli * coeff_count * 2 + m * coeff_count;
            uint64_t m_offset = b * num_moduli + m;
            for (uint64_t i = 0; i < coeff_count; i++) {
                uint64_t a = op1[poly0_offset + i] * op2[poly0_offset + i];
                exp_out.push_back(a % moduli[m_offset]);
            }
        }
        for (uint64_t m = 0; m < num_moduli; m++) {
            uint64_t poly0_offset =
                b * num_moduli * coeff_count * 2 + m * coeff_count;
            uint64_t poly1_offset = b * num_moduli * coeff_count * 2 +
                                    (m + num_moduli) * coeff_count;
            uint64_t m_offset = b * num_moduli + m;
            for (uint64_t i = 0; i < coeff_count; i++) {
                uint64_t x = op1[poly0_offset + i] * op2[poly1_offset + i];
                uint64_t y = op1[poly1_offset + i] * op2[poly0_offset + i];
                exp_out.push_back((x + y) % moduli[m_offset]);
            }
        }
        for (uint64_t m = 0; m < num_moduli; m++) {
            uint64_t poly1_offset = b * num_moduli * coeff_count * 2 +
                                    (m + num_moduli) * coeff_count;
            uint64_t m_offset = b * num_moduli + m;
            for (uint64_t i = 0; i < coeff_count; i++) {
                uint64_t x = op1[poly1_offset + i] * op2[poly1_offset + i];
                exp_out.push_back(x % moduli[m_offset]);
            }
        }
    }
}

void dyadic_multiply_test::test_dyadic_multiply(uint64_t num_dyadic_multiply,
                                                uint64_t num_moduli,
                                                uint64_t coeff_count,
                                                bool death) {
    setup_dyadic_io(num_dyadic_multiply, num_moduli, coeff_count);

    std::vector<uint64_t> out(
        num_dyadic_multiply * 3 * num_moduli * coeff_count, 0);

    intel::hexl::set_worksize_DyadicMultiply(num_dyadic_multiply);
    for (uint64_t n = 0; n < num_dyadic_multiply; n++) {
        uint64_t* pout = &out[0] + n * num_moduli * coeff_count * 3;
        uint64_t* pop1 = &op1[0] + n * num_moduli * coeff_count * 2;
        uint64_t* pop2 = &op2[0] + n * num_moduli * coeff_count * 2;
        uint64_t* pmoduli = &moduli[0] + n * num_moduli;
        intel::hexl::DyadicMultiply(pout, pop1, pop2, coeff_count, pmoduli,
                                    num_moduli);
    }
    intel::hexl::DyadicMultiplyCompleted();
    if (!death) {
        ASSERT_EQ(out, exp_out);
    }
}

void dyadic_multiply_test::test_matrix_dyadic_multiply(uint64_t n_rows,
                                                       uint64_t n_columns,
                                                       uint64_t num_moduli,
                                                       uint64_t coeff_count) {
    setup_dyadic_io(n_columns, num_moduli, coeff_count);

    std::vector<std::vector<uint64_t>> out;
    for (uint64_t r = 0; r < n_rows; r++) {
        std::vector<uint64_t> out_r(n_columns * 3 * num_moduli * coeff_count,
                                    0);
        out.emplace_back(out_r);
    }

    uint64_t block = 8;
    intel::hexl::set_worksize_DyadicMultiply(n_rows * n_columns);

    for (uint64_t r = 0; r < n_rows; r++) {
        for (uint64_t c = 0; c < (n_columns + block - 1) / block; c++) {
            for (uint64_t b = 0; b < block; b++) {
                uint64_t* pout =
                    &out[r][0] + (c * block + b) * num_moduli * coeff_count * 3;
                uint64_t* pop1 =
                    &op1[0] + (c * block + b) * num_moduli * coeff_count * 2;
                uint64_t* pop2 =
                    &op2[0] + (c * block + b) * num_moduli * coeff_count * 2;
                uint64_t* pmoduli = &moduli[0] + (c * block + b) * num_moduli;
                intel::hexl::DyadicMultiply(pout, pop1, pop2, coeff_count,
                                            pmoduli, num_moduli);
            }
        }
    }
    intel::hexl::DyadicMultiplyCompleted();

    for (uint64_t r = 0; r < n_rows; r++) {
        ASSERT_EQ(out[r], exp_out);
    }
}

TEST_F(dyadic_multiply_test, p512_m1_b1_16) {
    uint64_t coeff_count = 512 / 2;
    uint64_t num_moduli = 1;
    uint64_t num_dyadic_multiply = 16;

    dyadic_multiply_test mult;
    mult.test_dyadic_multiply(num_dyadic_multiply, num_moduli, coeff_count);
}

TEST_F(dyadic_multiply_test, p1024_m1_b1_16) {
    uint64_t coeff_count = 1024 / 2;
    uint64_t num_moduli = 1;
    uint64_t num_dyadic_multiply = 16;

    dyadic_multiply_test mult;
    mult.test_dyadic_multiply(num_dyadic_multiply, num_moduli, coeff_count);
}

TEST_F(dyadic_multiply_test, p2048_m1_b1_16) {
    uint64_t coeff_count = 2048 / 2;
    uint64_t num_moduli = 1;
    uint64_t num_dyadic_multiply = 16;

    dyadic_multiply_test mult;
    mult.test_dyadic_multiply(num_dyadic_multiply, num_moduli, coeff_count);
}

TEST_F(dyadic_multiply_test, p4096_m2_b1_16) {
    uint64_t coeff_count = 4096 / 2;
    uint64_t num_moduli = 2;
    uint64_t num_dyadic_multiply = 16;

    dyadic_multiply_test mult;
    mult.test_dyadic_multiply(num_dyadic_multiply, num_moduli, coeff_count);
}

TEST_F(dyadic_multiply_test, p8192_m4_b1_16) {
    uint64_t coeff_count = 8192 / 2;
    uint64_t num_moduli = 4;
    uint64_t num_dyadic_multiply = 16;

    dyadic_multiply_test mult;
    mult.test_dyadic_multiply(num_dyadic_multiply, num_moduli, coeff_count);
}

TEST_F(dyadic_multiply_test, p16384_m7_b1_16) {
    uint64_t coeff_count = 16384 / 2;
    uint64_t num_moduli = 7;
    uint64_t num_dyadic_multiply = 16;

    dyadic_multiply_test mult;
    mult.test_dyadic_multiply(num_dyadic_multiply, num_moduli, coeff_count);
}

TEST_F(dyadic_multiply_test, matrix_p16384_m7_b1_16x8) {
    uint64_t coeff_count = 16384 / 2;
    uint64_t num_moduli = 7;
    uint64_t n_rows = 16;
    uint64_t n_columns = 8;

    dyadic_multiply_test mult;
    mult.test_matrix_dyadic_multiply(n_rows, n_columns, num_moduli,
                                     coeff_count);
}

TEST_F(dyadic_multiply_test, matrix_p16384_m7_b1_256x8) {
    uint64_t coeff_count = 16384 / 2;
    uint64_t num_moduli = 7;
    uint64_t n_rows = 256;
    uint64_t n_columns = 8;

    dyadic_multiply_test mult;
    mult.test_matrix_dyadic_multiply(n_rows, n_columns, num_moduli,
                                     coeff_count);
}

TEST_F(dyadic_multiply_test, matrix_p16384_m7_b1_16x16) {
    uint64_t coeff_count = 16384 / 2;
    uint64_t num_moduli = 7;
    uint64_t n_rows = 16;
    uint64_t n_columns = 16;

    dyadic_multiply_test mult;
    mult.test_matrix_dyadic_multiply(n_rows, n_columns, num_moduli,
                                     coeff_count);
}

TEST_F(dyadic_multiply_test, p32768_m14_b1_2) {
    uint64_t coeff_count = 32768 / 2;
    uint64_t num_moduli = 14;
    uint64_t num_dyadic_multiply = 2;

    dyadic_multiply_test mult;
    mult.test_dyadic_multiply(num_dyadic_multiply, num_moduli, coeff_count);
}

TEST_F(dyadic_multiply_test, matrix_p32768_m14_b1_8x8) {
    uint64_t coeff_count = 32768 / 2;
    uint64_t num_moduli = 14;
    uint64_t n_rows = 8;
    uint64_t n_columns = 8;

    dyadic_multiply_test mult;
    mult.test_matrix_dyadic_multiply(n_rows, n_columns, num_moduli,
                                     coeff_count);
}

TEST_F(dyadic_multiply_test, matrix_p32768_m14_b1_16x8) {
    uint64_t coeff_count = 32768 / 2;
    uint64_t num_moduli = 14;
    uint64_t n_rows = 16;
    uint64_t n_columns = 8;

    dyadic_multiply_test mult;
    mult.test_matrix_dyadic_multiply(n_rows, n_columns, num_moduli,
                                     coeff_count);
}

TEST_F(dyadic_multiply_test, set_worksize_crash) {
#ifdef FPGA_DEBUG
    EXPECT_DEATH(intel::hexl::set_worksize_DyadicMultiply(0), "Assertion");
#endif
}

TEST_F(dyadic_multiply_test, m0_death) {
#ifdef FPGA_DEBUG
    uint64_t coeff_count = 8192;
    uint64_t num_moduli = 0;
    uint64_t num_dyadic_multiply = 16;

    dyadic_multiply_test mult;
    EXPECT_DEATH(mult.test_dyadic_multiply(num_dyadic_multiply, num_moduli,
                                           coeff_count, true),
                 "Assertion");
#endif
}

TEST_F(dyadic_multiply_test, coeff_count0_death) {
#ifdef FPGA_DEBUG
    uint64_t coeff_count = 0;
    uint64_t num_moduli = 5;
    uint64_t num_dyadic_multiply = 16;

    dyadic_multiply_test mult;
    EXPECT_DEATH(mult.test_dyadic_multiply(num_dyadic_multiply, num_moduli,
                                           coeff_count, true),
                 "Assertion");
#endif
}
