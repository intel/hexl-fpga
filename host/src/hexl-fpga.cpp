// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl-fpga.h"

#include <cstdint>

#include "dyadic_multiply.h"
#include "fpga_context.h"
#include "intt.h"
#include "ntt.h"

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

// NTT Section
void NTT(uint64_t* coeff_poly, const uint64_t* root_of_unity_powers,
         const uint64_t* precon_root_of_unity_powers, uint64_t coeff_modulus,
         uint64_t n) {
    intel::hexl::fpga::NTT(coeff_poly, root_of_unity_powers,
                           precon_root_of_unity_powers, coeff_modulus, n);
}

void set_worksize_NTT(uint64_t ws) { intel::hexl::fpga::set_worksize_NTT(ws); }

bool NTTCompleted() { return intel::hexl::fpga::NTTCompleted(); }

// INTT Section
void INTT(uint64_t* coeff_poly, const uint64_t* inv_root_of_unity_powers,
          const uint64_t* precon_inv_root_of_unity_powers,
          uint64_t coeff_modulus, uint64_t inv_n, uint64_t inv_n_w,
          uint64_t n) {
    intel::hexl::fpga::INTT(coeff_poly, inv_root_of_unity_powers,
                            precon_inv_root_of_unity_powers, coeff_modulus,
                            inv_n, inv_n_w, n);
}

void set_worksize_INTT(uint64_t ws) {
    intel::hexl::fpga::set_worksize_INTT(ws);
}

bool INTTCompleted() { return intel::hexl::fpga::INTTCompleted(); }

}  // namespace hexl
}  // namespace intel
