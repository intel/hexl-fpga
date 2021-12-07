// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "dyadic_multiply_int.h"
#include "fpga.h"
#include "fpga_assert.h"
#include "hexl-fpga.h"

namespace intel {
namespace hexl {
namespace fpga {

void DyadicMultiply(uint64_t* results, const uint64_t* operand1,
                    const uint64_t* operand2, uint64_t n,
                    const uint64_t* moduli, uint64_t n_moduli) {
    FPGA_ASSERT(results, "requires results != nullptr");
    FPGA_ASSERT(operand1, "requires operand1 != nullptr");
    FPGA_ASSERT(operand2, "requires operand2 != nullptr");
    FPGA_ASSERT(n > 0, "n must be positive integer");
    FPGA_ASSERT(moduli, "requires moduli != nullptr");
    FPGA_ASSERT(n_moduli > 0, "n_moduli must be positive integer");

    DyadicMultiply_int(results, operand1, operand2, n, moduli, n_moduli);
}

bool DyadicMultiplyCompleted() { return DyadicMultiplyCompleted_int(); }

void set_worksize_DyadicMultiply(uint64_t n) {
    FPGA_ASSERT(
        n > 0,
        "n must be positive integer. n==1 indicates synchronous execution. n>1 "
        "indidates n DyadicMultiply(s) run asynchronously.");
    set_worksize_DyadicMultiply_int(n);
}

}  // namespace fpga
}  // namespace hexl
}  // namespace intel
