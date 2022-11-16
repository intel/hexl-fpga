// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "fpga.h"
#include "fpga_assert.h"
#include "hexl-fpga.h"
#include "multlowlvl_int.h"

namespace intel {
namespace hexl {
namespace fpga {


void MultLowLvl(uint64_t* a0, uint64_t* a1, uint64_t a_primes_len, uint8_t* a_primes_index,
                uint64_t* b0, uint64_t* b1, uint64_t b_primes_len, uint8_t* b_primes_index,
                uint64_t plainText, uint64_t coeff_count, 
                uint64_t* c0, uint64_t* c1, uint64_t* c2, 
                uint64_t c_primes_len, uint8_t* output_primes_index,
                uint64_t all_primes_len, uint64_t* all_primes) {
    
    FPGA_ASSERT(a0, "requires a0 != nullptr");
    FPGA_ASSERT(a1, "requires a1 != nullptr");
    FPGA_ASSERT(a_primes_index, "requires a_primes_index != nullptr");
    FPGA_ASSERT(b0, "requires b0 != nullptr");
    FPGA_ASSERT(b1, "requires b1 != nullptr");
    FPGA_ASSERT(b_primes_index, "requires b_primes_index != nullptr");
    FPGA_ASSERT((coeff_count == 65536), "requires coeff_count = 65536");
    FPGA_ASSERT((plainText == 65537), "requires plainText = 65537");
    FPGA_ASSERT((a_primes_len == b_primes_len), "requires a_primes_len = b_primes_len");
    FPGA_ASSERT(c0, "requires c0 != nullptr");
    FPGA_ASSERT(c1, "requires c1 != nullptr");
    FPGA_ASSERT(c2, "requires c2 != nullptr");
    FPGA_ASSERT(output_primes_index, "requires output_primes_index != nulltptr");
    FPGA_ASSERT(all_primes, "requires all_primes != nulltptr");

    MultLowLvl_int(a0, a1, a_primes_len, a_primes_index,
                   b0, b1, b_primes_len, b_primes_index,
                   plainText, coeff_count,
                   c0, c1, c2, c_primes_len, output_primes_index, 
                   all_primes_len, all_primes);
    
} 

bool MultLowLvlCompleted() {return MultLowLvlCompleted_int();}

void set_worksize_MultLowLvl(uint64_t n) {

    set_worksize_MultLowLvl_int(n);
}


} // namespace fpga
} // namespace hexl
} // namespace intel