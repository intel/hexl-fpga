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

void MultLowLvl(std::vector<uint64_t> &a0, std::vector<uint64_t> &a1,
                std::vector<uint8_t> &a_primes_index, std::vector<uint64_t> &b0,
                std::vector<uint64_t> &b1, std::vector<uint8_t> &b_primes_index,
                uint64_t plainText, std::vector<uint64_t> &c0,
                std::vector<uint64_t> &c1, std::vector<uint64_t> &c2,
                std::vector<uint8_t> &output_primes_index) {
    
    MultLowLvl_int(a0, a1, a_primes_index, b0, b1, b_primes_index, plainText, c0, c1, 
                   c2, output_primes_index);

}

bool MultLowLvlCompleted() {return KeySwitchCompleted_int();}

void set_worksize_MultLowLvl(uint64_t n) {

    set_worksize_MultLowLvl_int(n);
}


} // namespace fpga
} // namespace hexl
} // namespace intel