// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __KEYSWITCH_H__
#define __KEYSWITCH_H__

#include <cstdint>

namespace intel {
namespace hexl {
namespace fpga {
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

}  // namespace fpga
}  // namespace hexl
}  // namespace intel

#endif
