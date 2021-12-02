// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __NTT_INT_H__
#define __NTT_INT_H__

#include <cstdint>

namespace intel {
namespace hexl {
namespace fpga {

/// @brief
/// @function set_worksize_NTT_int
/// Sets the work size for NTT. Internal implementation.
/// @param[in] ws stores the worksize of the NTT
///
void set_worksize_NTT_int(uint64_t n);

/// @brief
/// @function NTT_int
/// Calls the Number Theorectic Transform. Internal implementation.
/// @param[in] coeff_poly vector of polynomial coefficients
/// @param[out] coeff_poly vector of polynomial coefficients
/// @param[in] root_of_unity_powers vector of twiddle factors
/// @param[in] precon_root_of_unity_power vector of twiddle factors for the
/// constant
/// @param[in] coeff_modulus stores the coefficient modulus
/// @param[in] n stores the polynomial size
///
void NTT_int(uint64_t* coeff_poly, const uint64_t* root_of_unity_powers,
             const uint64_t* precon_root_of_unity_powers,
             uint64_t coeff_modulus, uint64_t n);

/// @brief
/// @function NTTCompleted_int
/// Called after completion of the Number Theoretic Transform. Internal
/// implementation.
///
bool NTTCompleted_int();

}  // namespace fpga
}  // namespace hexl
}  // namespace intel

#endif
