// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __NTT_H__
#define __NTT_H__

#include <cstdint>

namespace intel {
namespace hexl {
namespace fpga {
/// @brief
/// @function set_worksize_NTT
/// Sets the work size for NTT
/// @param[in] ws stores the worksize of the NTT
///
void set_worksize_NTT(uint64_t ws);
/// @brief
/// @function NTT
/// Calls the Number Theorectic Transform
/// @param[in] coeff_poly vector of polynomial coefficients
/// @param[out] coeff_poly vector of polynomial coefficients
/// @param[in] root_of_unity_powers vector of twiddle factors
/// @param[in] precon_root_of_unity_power vector of twiddle factors for the
/// constant
/// @param[in] coeff_modulus stores the coefficient modulus
/// @param[in] n stores the polynomial size
///
void NTT(uint64_t* coeff_poly, const uint64_t* root_of_unity_powers,
         const uint64_t* precon_root_of_unity_powers, uint64_t coeff_modulus,
         uint64_t n);
/// @brief
/// @function NTTCompleted
/// Called after completion of the Number Theoretic Transform
///
bool NTTCompleted();

}  // namespace fpga
}  // namespace hexl
}  // namespace intel

#endif
