// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "fpga.h"
#include "fpga_assert.h"
#include "hexl-fpga.h"
#include "ntt_int.h"

namespace intel {
namespace hexl {
namespace fpga {

void NTT(uint64_t* coeff_poly, const uint64_t* root_of_unity_powers,
         const uint64_t* precon_root_of_unity_powers, uint64_t coeff_modulus,
         uint64_t n) {
    FPGA_ASSERT(coeff_poly, "requires coeff_poly != nullptr");
    FPGA_ASSERT(root_of_unity_powers,
                "requires root_of_unity_powers != nullptr");
    FPGA_ASSERT(precon_root_of_unity_powers,
                "requires precon_root_of_unity_powers != nullptr");
    FPGA_ASSERT(coeff_modulus > 0, "coeff_modulus must be positive integer");
    FPGA_ASSERT(n == 16384, "requires n = 16384");

    NTT_int(coeff_poly, root_of_unity_powers, precon_root_of_unity_powers,
            coeff_modulus, n);
}

void set_worksize_NTT(uint64_t n) {
    FPGA_ASSERT(
        n > 0,
        "n must be positive integer. n==1 indicates synchronous execution.");
    set_worksize_NTT_int(n);
}
bool NTTCompleted() { return NTTCompleted_int(); }

}  // namespace fpga
}  // namespace hexl
}  // namespace intel
