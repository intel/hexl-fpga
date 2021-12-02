// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __DYADIC_MULTIPLY_INT_H__
#define __DYADIC_MULTIPLY_INT_H__

#include <cstdint>

namespace intel {
namespace hexl {
namespace fpga {
/// @brief
/// @function set_worksize_DyadicMultiply_int
/// Sets the worksize for the multiplication
/// @param[in] n work size
///
void set_worksize_DyadicMultiply_int(uint64_t n);
/// @brief
/// @function DyadicMultiply_int
/// Internal implementation of the DyadicMultiply function call
/// @param[out] results stores the output of the multiplication
/// @param[in] operand1 vector of polynomial coefficients
/// @param[in] operand2 vector of polynomial coefficients
/// @param[in] n polynomial size
/// @param[in] moduli vector of coefficient modulus
/// @param[in] n_moduli number of modulus in the vector of modulus
///
void DyadicMultiply_int(uint64_t* results, const uint64_t* operand1,
                        const uint64_t* operand2, uint64_t n,
                        const uint64_t* moduli, uint64_t n_moduli);
/// @brief
/// @function DyadicMultiplyCompleted_int
/// Internal implementation of the DyadicMultiplyCompleted function.
/// Called after completion of the multiplication operation
///
bool DyadicMultiplyCompleted_int();

}  // namespace fpga
}  // namespace hexl
}  // namespace intel

#endif
