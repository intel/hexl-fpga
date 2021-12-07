// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __DYADIC_MULTIPLY_H__
#define __DYADIC_MULTIPLY_H__

#include <cstdint>

namespace intel {
namespace hexl {
namespace fpga {
/// @brief
/// function set_worksize_DyadicMultiply
/// @param[in] ws work size
///
void set_worksize_DyadicMultiply(uint64_t ws);
/// @brief
/// function DyadicMultiply
/// Implements the multiplication of two ciphertexts
/// @param[out] results stores the result of the multiplication
/// @param[in] operand1 vector of polynomial coefficients
/// @param[in] operand2 vector of polynomial coefficients
/// @param[in] n polynomial size
/// @param[in] moduli vector of modulus
/// @param[in] n_moduli number of modulus in the vector of modulus
///
void DyadicMultiply(uint64_t* results, const uint64_t* operand1,
                    const uint64_t* operand2, uint64_t n,
                    const uint64_t* moduli, uint64_t n_moduli);
/// @brief
/// @function DyadicMultiplyCompleted
/// Executed after the multiplication to wrap up the operation
///
bool DyadicMultiplyCompleted();

}  // namespace fpga
}  // namespace hexl
}  // namespace intel

#endif
