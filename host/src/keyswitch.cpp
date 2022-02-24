// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "fpga.h"
#include "fpga_assert.h"
#include "hexl-fpga.h"
#include "keyswitch_int.h"

namespace intel {
namespace hexl {
namespace fpga {

void KeySwitch(uint64_t* result, const uint64_t* t_target_iter_ptr, uint64_t n,
               uint64_t decomp_modulus_size, uint64_t key_modulus_size,
               uint64_t rns_modulus_size, uint64_t key_component_count,
               const uint64_t* moduli, const uint64_t** k_switch_keys,
               const uint64_t* modswitch_factors,
               const uint64_t* twiddle_factors) {
    FPGA_ASSERT(result, "requires result != nullptr");
    FPGA_ASSERT(t_target_iter_ptr, "requires t_target_iter_ptr != nullptr");
    FPGA_ASSERT((n == 16384) || (n == 8192) || (n == 4096) || (n == 2048) ||
                    (n == 1024),
                "requires n = 16384/8192/4096/2048/1024");
    FPGA_ASSERT(decomp_modulus_size > 0, "requires decomp_modulus_size > 0");
    FPGA_ASSERT(key_modulus_size <= 7, "requires key_modulus_size <= 7");
    FPGA_ASSERT(rns_modulus_size > 0, "requires rns_modulus_size > 0");
    FPGA_ASSERT(key_component_count == 2, "requires key_component_count = 2");
    FPGA_ASSERT(moduli, "requires moduli != nullptr");
    FPGA_ASSERT(k_switch_keys, "requires k_switch_keys != nullptr");
    FPGA_ASSERT(modswitch_factors, "requires modswitch_factors != nullptr");

    KeySwitch_int(result, t_target_iter_ptr, n, decomp_modulus_size,
                  key_modulus_size, rns_modulus_size, key_component_count,
                  moduli, k_switch_keys, modswitch_factors, twiddle_factors);
}

bool KeySwitchCompleted() { return KeySwitchCompleted_int(); }

void set_worksize_KeySwitch(uint64_t n) {
    FPGA_ASSERT(
        n > 0,
        "n must be positive integer. n==1 indicates synchronous execution. n>1 "
        "indidates n keyswitch(s) run asynchronously.");
    set_worksize_KeySwitch_int(n);
}

}  // namespace fpga
}  // namespace hexl
}  // namespace intel
