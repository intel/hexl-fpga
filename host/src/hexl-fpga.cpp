// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl-fpga.h"

#include <cstdint>

#include "dyadic_multiply.h"
#include "fpga_context.h"
#include "intt.h"
#include "keyswitch.h"
#include "ntt.h"
#include "multiplyby.h"

namespace intel {
namespace hexl {

// FPGA_CONTEXT
void acquire_FPGA_resources() { intel::hexl::fpga::acquire_FPGA_resources(); }

void release_FPGA_resources() { intel::hexl::fpga::release_FPGA_resources(); }

// DyadicMultiply Section
void DyadicMultiply(uint64_t* results, const uint64_t* operand1,
                    const uint64_t* operand2, uint64_t n,
                    const uint64_t* moduli, uint64_t n_moduli) {
    intel::hexl::fpga::DyadicMultiply(results, operand1, operand2, n, moduli,
                                      n_moduli);
}

void set_worksize_DyadicMultiply(uint64_t ws) {
    intel::hexl::fpga::set_worksize_DyadicMultiply(ws);
}

bool DyadicMultiplyCompleted() {
    return intel::hexl::fpga::DyadicMultiplyCompleted();
}

// KeySwitch Section
void KeySwitch(uint64_t* result, const uint64_t* t_target_iter_ptr, uint64_t n,
               uint64_t decomp_modulus_size, uint64_t key_modulus_size,
               uint64_t rns_modulus_size, uint64_t key_component_count,
               const uint64_t* moduli, const uint64_t** k_switch_keys,
               const uint64_t* modswitch_factors,
               const uint64_t* twiddle_factors) {
    intel::hexl::fpga::KeySwitch(
        result, t_target_iter_ptr, n, decomp_modulus_size, key_modulus_size,
        rns_modulus_size, key_component_count, moduli, k_switch_keys,
        modswitch_factors, twiddle_factors);
}

void set_worksize_KeySwitch(uint64_t ws) {
    intel::hexl::fpga::set_worksize_KeySwitch(ws);
}

bool KeySwitchCompleted() { return intel::hexl::fpga::KeySwitchCompleted(); }

// MultiplyBy Section
void MultiplyBy(const MultiplyByContext& context,
                const std::vector<uint64_t>& operand1,
                const std::vector<uint8_t>& operand1_primes_index,
                const std::vector<uint64_t>& operand2,
                const std::vector<uint8_t>& operand2_primes_index,
                std::vector<uint64_t>& result,
                const std::vector<uint8_t>& result_primes_index) {
    intel::hexl::fpga::MultiplyBy(context, operand1, operand1_primes_index,
                                  operand2, operand2_primes_index, result,
                                  result_primes_index);
}

void set_worksize_MultiplyBy(uint64_t ws) {
    intel::hexl::fpga::set_worksize_MultiplyBy(ws);
}

bool MultiplyByCompleted() { return intel::hexl::fpga::MultiplyByCompleted(); }
////////////////////////////////////////////////////////////////////////////////////////
//
// WARNING: The following NTT and INTT related APIs are deprecated since
// version 1.1. //
//
////////////////////////////////////////////////////////////////////////////////////////

// NTT Section

void _NTT(uint64_t* coeff_poly, const uint64_t* root_of_unity_powers,
          const uint64_t* precon_root_of_unity_powers, uint64_t coeff_modulus,
          uint64_t n) {
    intel::hexl::fpga::NTT(coeff_poly, root_of_unity_powers,
                           precon_root_of_unity_powers, coeff_modulus, n);
}

void _set_worksize_NTT(uint64_t ws) { intel::hexl::fpga::set_worksize_NTT(ws); }

bool _NTTCompleted() { return intel::hexl::fpga::NTTCompleted(); }

// INTT Section
void _INTT(uint64_t* coeff_poly, const uint64_t* inv_root_of_unity_powers,
           const uint64_t* precon_inv_root_of_unity_powers,
           uint64_t coeff_modulus, uint64_t inv_n, uint64_t inv_n_w,
           uint64_t n) {
    intel::hexl::fpga::INTT(coeff_poly, inv_root_of_unity_powers,
                            precon_inv_root_of_unity_powers, coeff_modulus,
                            inv_n, inv_n_w, n);
}

void _set_worksize_INTT(uint64_t ws) {
    intel::hexl::fpga::set_worksize_INTT(ws);
}

bool _INTTCompleted() { return intel::hexl::fpga::INTTCompleted(); }

}  // namespace hexl
}  // namespace intel
