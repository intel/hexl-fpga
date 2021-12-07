// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "example_dyadic_multiply.h"

#include "hexl-fpga.h"

/// @brief
/// @function example_dyadic_multiply::setup_dyadic_multiply
/// Initializes the multiplication two operands and the vector of moduli
/// Compute and stores the expected multiplication result for future validation.

void example_dyadic_multiply::setup_dyadic_multiply(uint64_t n_dyadic_multiply,
                                                    uint64_t num_moduli,
                                                    uint64_t coeff_count) {
    for (uint64_t b = 0; b < n_dyadic_multiply; b++) {
        // Compute the moduli
        for (uint64_t m = 0; m < num_moduli; m++) {
            moduli.push_back((b + m + 1) * 10);
        }

        // Prepare the operands
        for (uint64_t m = 0; m < num_moduli; m++) {
            for (uint64_t i = 0; i < coeff_count; i++) {
                op1.push_back(b + i + 1 + m * coeff_count);
                op2.push_back(b + i + 2 + m * coeff_count);
            }
        }

        // Prepare the operands
        for (uint64_t m = 0; m < num_moduli; m++) {
            for (uint64_t i = 0; i < coeff_count; i++) {
                op1.push_back(b + i + 11 + m * coeff_count);
                op2.push_back(b + i + 22 + m * coeff_count);
            }
        }
        // Compute the expected output and store it for future validation
        for (uint64_t m = 0; m < num_moduli; m++) {
            uint64_t poly0_offset =
                b * num_moduli * coeff_count * 2 + m * coeff_count;
            uint64_t m_offset = b * num_moduli + m;
            for (uint64_t i = 0; i < coeff_count; i++) {
                uint64_t a = op1[poly0_offset + i] * op2[poly0_offset + i];
                exp_out.push_back(a % moduli[m_offset]);
            }
        }
        // Compute the expected output and store it for future validation
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
        // Compute the expected output and store it for future validation
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

/// @brief
/// @function example_dyadic_multiply::execute_dyadic_multiply
/// sets the work size for the multiplication
/// calls n_dyadic_multiply times the multiplication function
/// calls DyadicMultiplyCompleted after completion of all multiplications
///
void example_dyadic_multiply::execute_dyadic_multiply(
    std::vector<uint64_t>& out, uint64_t n_dyadic_multiply, uint64_t num_moduli,
    uint64_t coeff_count) {
    // Set the worksize for the multiplication
    intel::hexl::set_worksize_DyadicMultiply(n_dyadic_multiply);
    for (uint64_t b = 0; b < n_dyadic_multiply; b++) {
        // Set the input and output parameters
        uint64_t* pout = &out[0] + b * num_moduli * coeff_count * 3;
        uint64_t* pop1 = &op1[0] + b * num_moduli * coeff_count * 2;
        uint64_t* pop2 = &op2[0] + b * num_moduli * coeff_count * 2;
        uint64_t* pmoduli = &moduli[0] + b * num_moduli;
        // Call the multiplication function. This will be executed on the FPGA.
        intel::hexl::DyadicMultiply(pout, pop1, pop2, coeff_count, pmoduli,
                                    num_moduli);
    }
    // Synchronize at this point until all dyadic multiplications complete
    intel::hexl::DyadicMultiplyCompleted();
}
