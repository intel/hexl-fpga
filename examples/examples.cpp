// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <vector>

#include "example_dyadic_multiply.h"
#include "hexl-fpga.h"

void check_equal(const std::vector<uint64_t>& x,
                 const std::vector<uint64_t>& y) {
    if (x == y) {
        std::cout << "Correct multiplication: both vectors are equal"
                  << std::endl;
    } else {
        std::cout << "Error in the multiplication:both vectors are not equal"
                  << std::endl;
    }
}

void run_example_dyadic_multiply() {
    std::cout << "Running example_dyadic_multiply..." << std::endl;

    // Set the parameter values for dyadic multiplication
    uint64_t coeff_count = 16384 / 2;
    uint64_t num_moduli = 6;
    uint64_t n_dyadic_multiply = 40;

    // Instantiate the class dyadic_multiply
    example_dyadic_multiply example;

    // Create the two operands for the multiplication.
    // Compute the expected output (exp_out) to allow validation of the FPGA
    // computation
    example.setup_dyadic_multiply(n_dyadic_multiply, num_moduli, coeff_count);

    // Create a vector to store the result
    std::vector<uint64_t> out(n_dyadic_multiply * 3 * num_moduli * coeff_count,
                              0);
    // Call the multiplication function (This will run on the FPGA)
    example.execute_dyadic_multiply(out, n_dyadic_multiply, num_moduli,
                                    coeff_count);

    // Validate the result
    check_equal(out, example.exp_output());

    std::cout << "Done running example_dyadic_multiply..." << std::endl;
}

int main(int argc, char** argv) {
    // Setup the FPGA device
    // call once at the beginning of the application
    intel::hexl::acquire_FPGA_resources();

    run_example_dyadic_multiply();

    // Cleanup FPGA context
    // call once at the end of the application
    intel::hexl::release_FPGA_resources();

    return 0;
}
