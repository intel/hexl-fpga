// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __HEXL_FPGA_H__
#define __HEXL_FPGA_H__

#include <cstdint>

namespace intel {
namespace hexl {
/// @brief
/// Function acquire_FPGA_resources
/// Called without any parameter, reserves the FPGA hardware resources
///
void acquire_FPGA_resources();
/// @brief
/// Function release_FPGA_resources
/// Called without any parameter, releases the FPGA hardware resources once we
/// are done.
///
void release_FPGA_resources();

// DyadicMultiply Section
/// @brief
/// Function set_worksize_DyadicMultiply
/// Reserves software resources for the multiplication
/// @param ws integer storing the worksize
///
void set_worksize_DyadicMultiply(uint64_t ws);
/// @brief
///
/// Function DyadicMultiply
/// Executes ciphertext ciphertext multiplication
/// @param[out] results stores the multiplication results
/// @param[in]  operand1 stores the input ciphertext 1
/// @param[in]  operand2 stores the input ciphertext 2
/// @param[in]  n stores polynomial size
/// @param[in]  moduli stores modulus size
/// @param[in]  n_moduli stores the number of moduli
///
void DyadicMultiply(uint64_t* results, const uint64_t* operand1,
                    const uint64_t* operand2, uint64_t n,
                    const uint64_t* moduli, uint64_t n_moduli);

/// @brief
///
/// Function DyadicMultiplyCompleted
/// Executed after ciphertext ciphertext multiplication to wrap
/// up the task
bool DyadicMultiplyCompleted();

// NTT Section

/// @brief
/// Function set_worksize_NTT
/// Reserves software resources for the Number Theoretic Transform
/// @param ws integer storing the worksize
///
void set_worksize_NTT(uint64_t ws);
/// @brief
/// Function NTT
/// Executes in place the Number Theoretic Transform
/// @param[in] operand input ciphertext. This is also the output result since
/// the transform is in place.
/// @param[out] operand output ciphertext. This is also the input result since
/// the transform is in place.
/// @param[in] root_of_unity_powers vector of twiddle factors
/// @param[in] precon_root_of_unity_powers vector of precomputed inverse twiddle
/// factors
/// @param[in] coeff_modulus stores the modulus
/// @param[in] n stores the size of the Number Theoretic Transform
/////
void NTT(uint64_t* operand, const uint64_t* root_of_unity_powers,
         const uint64_t* precon_root_of_unity_powers, uint64_t coeff_modulus,
         uint64_t n);
/// @brief
/// Function NTTCompleted
/// Executed after the NTT to wrap up the computation
/// No parameters
///
bool NTTCompleted();

// INTT Section
/// @brief
/// Function set_worksize_INTT
/// Reserves software resources for the inverse Number Theoretic Transform
/// @param ws integer storing the worksize
///
void set_worksize_INTT(uint64_t ws);

/// @brief
/// Function INTT
/// Executes in place the inverse Number Theoretic Transform
/// @param[in] operand input ciphertext. This is also the output result since
/// the transform is in place.
/// @param[out] operand output ciphertext. This is also the input result since
/// the transform is in place.
/// @param[in] inv_root_of_unity_powers vector of twiddle factors
/// @param[in] precon_inv_root_of_unity_powers vector of precomputed inverse
/// twiddle factors
/// @param[in] coeff_modulus stores the modulus
/// @param[in] inv_n  stores the normalization factor for the inverse transform.
/// Inverse of the polynomial size ( 1/n)
/// @param[in] inv_n_w  stores the  normalization factor for the constant.
/// @param[in] n stores the size of the Number Theoretic Transform
///
void INTT(uint64_t* operand, const uint64_t* inv_root_of_unity_powers,
          const uint64_t* precon_inv_root_of_unity_powers,
          uint64_t coeff_modulus, uint64_t inv_n, uint64_t inv_n_w, uint64_t n);

/// @brief
/// Function INTTCompleted
/// Executed after the INTT to wrap up the computation
/// No parameters
///
bool INTTCompleted();

}  // namespace hexl
}  // namespace intel

#endif
