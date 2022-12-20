#pragma once
#include <algorithm>
#include <iomanip>
#include <CL/sycl.hpp>
#include <hexl-fpga.h>

/// @brief PreComputeBringToSet
/// @param all_primes
/// @param pi_primes_index
/// @param qj_prime_index
/// @param pi_reorder_primes_index
/// @param scale_param_set
/// @param P
/// @param Q
/// @param I
/// @param plainText
void PreComputeBringToSet(const std::vector<uint64_t>& all_primes,
                          const std::vector<uint8_t>& pi_primes_index,
                          const std::vector<uint8_t>& qj_prime_index,
                          std::vector<uint8_t>& pi_reorder_primes_index,
                          std::vector<sycl::ulong2>& scale_param_set, size_t& P,
                          size_t& Q, size_t& I, uint64_t plainText);

/// @brief PreComputeBreakIntoDigits
/// @param context
/// @param primes_index
/// @param num_digits_primes
/// @param packed_precomuted_params
void PreComputeBreakIntoDigits(
    const FpgaHEContext& context, const std::vector<uint8_t>& primes_index,
    std::vector<unsigned>& num_digits_primes,
    std::vector<sycl::ulong2>& packed_precomuted_params);

/// @brief PreComputeTensorProduct
/// @param context
/// @param primes_index
/// @param params
void PreComputeTensorProduct(const FpgaHEContext& context,
                             const std::vector<uint8_t>& primes_index,
                             std::vector<ulong4>& params);

/// @brief PreComputeKeySwitchDigits
/// @param context
/// @param primes_index
/// @param params
void PreComputeKeySwitchDigits(const FpgaHEContext& context,
                               const std::vector<uint8_t>& primes_index,
                               std::vector<ulong4>& params);