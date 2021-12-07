// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#ifndef __EXAMPLE_DYADIC_MULTIPLY_H__
#define __EXAMPLE_DYADIC_MULTIPLY_H__

#include <cstdint>
#include <vector>

/// @brief
/// Class example_dyadic_multiply
///
/// @function setup_dyadic_multiply
/// @param[in] n_dyadic_multiply number of multiplications
/// @param[in] num_moduli number of moduli
/// @param[in] coeff_count number of polynomial coefficients
///
/// @function execute_dyadic_multiply
/// @param[out] out vector of results
/// @param[in] n_dyadic_multiply number of multiplications
/// @param[in] num_moduli number of modulis
/// @param[in] coeff_count number of polynomial coefficients
///
/// @function is_equal verifies the equality of two vectors
///
/// @var moduli vector of moduli
/// @var op1 vector of first operand for the multiplication
/// @var op2 vector of second operand for the multiplication
/// @var exp_out vector of ground truth values for the multiplication
///
class example_dyadic_multiply {
public:
    void setup_dyadic_multiply(uint64_t n_dyadic_multiply, uint64_t num_moduli,
                               uint64_t coeff_count);
    void execute_dyadic_multiply(std::vector<uint64_t>& out,
                                 uint64_t n_dyadic_multiply,
                                 uint64_t num_moduli, uint64_t coeff_count);
    std::vector<uint64_t>& exp_output() { return exp_out; }

private:
    std::vector<uint64_t> moduli;
    std::vector<uint64_t> op1;
    std::vector<uint64_t> op2;
    std::vector<uint64_t> exp_out;
};
#endif
