// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "fpga.h"
#include "fpga_assert.h"
#include "hexl-fpga.h"
#include "multiplyby_int.h"

namespace intel {
namespace hexl {
namespace fpga {

void MultiplyBy(const MultiplyByContext& context,
                const std::vector<uint64_t>& operand1,
                const std::vector<uint8_t>& operand1_primes_index,
                const std::vector<uint64_t>& operand2,
                const std::vector<uint8_t>& operand2_primes_index,
                std::vector<uint64_t>& result,
                const std::vector<uint8_t>& result_primes_index) {
    // TODO: check parameters
    MultiplyBy_int(context, operand1, operand1_primes_index, operand2,
                   operand2_primes_index, result, result_primes_index);
}

bool MultiplyByCompleted() { return MultiplyByCompleted_int(); }

void set_worksize_MultiplyBy(uint64_t n) {
    FPGA_ASSERT(
        n > 0,
        "n must be positive integer. n==1 indicates synchronous execution. n>1 "
        "indidates n MultiplyBy(s) run asynchronously.");
    set_worksize_MultiplyBy_int(n);
}

}  // namespace fpga
}  // namespace hexl
}  // namespace intel
