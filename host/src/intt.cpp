// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "fpga.h"
#include "fpga_assert.h"
#include "hexl-fpga.h"
#include "intt_int.h"

namespace intel {
namespace hexl {
namespace fpga {

void INTT(uint64_t* coeff_poly, const uint64_t* inv_root_of_unity_powers,
          const uint64_t* precon_inv_root_of_unity_powers,
          uint64_t coeff_modulus, uint64_t inv_n, uint64_t inv_n_w,
          uint64_t n) {
    FPGA_ASSERT(coeff_poly, "requires coeff_poly != nullptr");
    FPGA_ASSERT(inv_root_of_unity_powers,
                "requires inv_root_of_unity_powers != nullptr");
    FPGA_ASSERT(precon_inv_root_of_unity_powers,
                "requires inv_precon_root_of_unity_powers != nullptr");
    FPGA_ASSERT(coeff_modulus > 0, "coeff_modulus must be positive integer");
    FPGA_ASSERT(n == 16384, "requires n = 16384");

    INTT_int(coeff_poly, inv_root_of_unity_powers,
             precon_inv_root_of_unity_powers, coeff_modulus, inv_n, inv_n_w, n);
}

void set_worksize_INTT(uint64_t n) {
    FPGA_ASSERT(
        n > 0,
        "n must be positive integer. n==1 indicates synchronous execution.");
    set_worksize_INTT_int(n);
}
bool INTTCompleted() { return INTTCompleted_int(); }

}  // namespace fpga
}  // namespace hexl
}  // namespace intel
