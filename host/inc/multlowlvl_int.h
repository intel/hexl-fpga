// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __MULTLOWLVL_INT_H__
#define __MULTLOWLVL_INT_H__

#include <vector>
#include <cstdint>

namespace intel {
namespace hexl {
namespace fpga {

void set_worksize_MultLowLvl_int(uint64_t ws);

// TODU: Change to pass pointers as parameters to align with KeySwitch.
// Check with Yan.
void MultLowLvl_int(std::vector<uint64_t> &a0, std::vector<uint64_t> &a1,
                    std::vector<uint8_t> &a_primes_index, std::vector<uint64_t> &b0,
                    std::vector<uint64_t> &b1, std::vector<uint8_t> &b_primes_index,
                    uint64_t plainText, std::vector<uint64_t> &c0,
                    std::vector<uint64_t> &c1, std::vector<uint64_t> &c2,
                    std::vector<uint8_t> &output_primes_index);

bool MultLowLvlCompleted_int();


} // namespace fpga
} // namespace hexl
} // namespace intel


#endif