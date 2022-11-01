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

// KeySwitch Section
/// @brief
/// Function set_worksize_KeySwitch
/// Reserves software resources for the KeySwitch
/// @param ws integer storing the worksize
///
void set_worksize_KeySwitch(uint64_t ws);
/// @brief
///
/// Function KeySwitch
/// Executes KeySwitch operation
/// @param[out] results stores the keyswitch results
/// @param[in]  t_target_iter_ptr stores the input ciphertext data
/// @param[in]  n stores polynomial size
/// @param[in]  decomp_modulus_size stores modulus size
/// @param[in]  key_modulus_size stores key modulus size
/// @param[in]  rns_modulus_size stores the rns modulus size
/// @param[in]  key_component_size stores the key component size
/// @param[in]  moduli stores the moduli
/// @param[in]  k_switch_keys stores the keys for keyswitch operation
/// @param[in]  modswitch_factors stores the factors for modular switch
/// @param[in]  twiddle_factors stores the twiddle factors
///
void KeySwitch(uint64_t* result, const uint64_t* t_target_iter_ptr, uint64_t n,
               uint64_t decomp_modulus_size, uint64_t key_modulus_size,
               uint64_t rns_modulus_size, uint64_t key_component_count,
               const uint64_t* moduli, const uint64_t** k_switch_keys,
               const uint64_t* modswitch_factors,
               const uint64_t* twiddle_factors = nullptr);

/// @brief
///
/// Function KeySwitchCompleted
/// Executed after KeySwitch to sync up the outstanding KeySwitch tasks
bool KeySwitchCompleted();


// MultLowLvl Section
void set_worksize_MultLowLvl(uint64_t ws);

void MultLowLvl(uint64_t* a0, uint64_t* a1, uint64_t a_primes_size, uint8_t* a_primes_index,
                uint64_t* b0, uint64_t* b1, uint64_t b_primes_size, uint8_t* b_primes_index,
                uint64_t plainText, uint64_t coeff_count, 
                uint64_t* c0, uint64_t* c1, uint64_t* c2, uint64_t c_primes_size, 
                uint8_t* output_primes_index);

bool MultLowLvlCompleted();

////////////////////////////////////////////////////////////////////////////////////////
//
// WARNING: The following NTT and INTT related APIs are deprecated since
// version 1.1. //
//
////////////////////////////////////////////////////////////////////////////////////////
// NTT Section

/// @brief
/// Function _set_worksize_NTT [[deprecated]]
/// Reserves software resources for the Number Theoretic Transform
/// @param ws integer storing the worksize
///
[[deprecated]] void _set_worksize_NTT(uint64_t ws);
/// @brief
/// Function _NTT [[deprecated]]
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
///
[[deprecated]] void _NTT(uint64_t* operand,
                         const uint64_t* root_of_unity_powers,
                         const uint64_t* precon_root_of_unity_powers,
                         uint64_t coeff_modulus, uint64_t n);
/// @brief
/// Function _NTTCompleted [[deprecated]]
/// Executed after the NTT to wrap up the computation
/// No parameters
///
[[deprecated]] bool _NTTCompleted();

// INTT Section
/// @brief
/// Function _set_worksize_INTT [[deprecated]]
/// Reserves software resources for the inverse Number Theoretic Transform
/// @param ws integer storing the worksize
///
[[deprecated]] void _set_worksize_INTT(uint64_t ws);

/// @brief
/// Function _INTT [[deprecated]]
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
[[deprecated]] void _INTT(uint64_t* operand,
                          const uint64_t* inv_root_of_unity_powers,
                          const uint64_t* precon_inv_root_of_unity_powers,
                          uint64_t coeff_modulus, uint64_t inv_n,
                          uint64_t inv_n_w, uint64_t n);

/// @brief
/// Function _INTTCompleted [[deprecated]]
/// Executed after the INTT to wrap up the computation
/// No parameters
///
[[deprecated]] bool _INTTCompleted();

}  // namespace hexl
}  // namespace intel

#endif
