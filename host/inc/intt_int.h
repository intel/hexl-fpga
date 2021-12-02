// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __INTT_INT_H__
#define __INTT_INT_H__

#include <cstdint>

namespace intel {
namespace hexl {
namespace fpga {

/// @brief
/// @function set_worksize_INTT_int
/// Internal implementation. Sets the work size of the INTT operation
///
void set_worksize_INTT_int(uint64_t n);

/// @brief
/// @function INTT
/// Calls the Inverse Number Theoretic Transform
/// @param[out] coef_poly vector of polynomial coefficients
/// @param[in] coef_poly vector of polynomial coefficients
/// @param[in] inv_root_of_unity_powers vector of twiddle factors
/// @param[in] precon_inv_root_of_unity_powers vector of twiddle factors for the
/// constant
/// @param[in] coeff_modulus coefficient modulus
/// @param[in] inv_n normalization factor
/// @param[in] inv_n_w normalization factor for the constant
/// @param[in] n polynomial size
///
void INTT_int(uint64_t* coeff_poly, const uint64_t* inv_root_of_unity_powers,
              const uint64_t* precon_inv_root_of_unity_powers,
              uint64_t coeff_modulus, uint64_t inv_n, uint64_t inv_n_w,
              uint64_t n);
/// @brief
/// @function INTTCompleted_int
/// Called after the completion of the INTT operation. Internal implementation.
///

bool INTTCompleted_int();

}  // namespace fpga
}  // namespace hexl
}  // namespace intel

#endif
