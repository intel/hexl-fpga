// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __MULTLOWLVL_H__
#define __MULTLOWLVL_H__

#include <cstdint>
#include <vector>

namespace intel {
namespace hexl {
namespace fpga {

void set_worksize_MultLowLvl(uint64_t ws);

void MultLowLvl(std::vector<uint64_t> &a0, std::vector<uint64_t> &a1,
                std::vector<uint8_t> &a_primes_index, std::vector<uint64_t> &b0,
                std::vector<uint64_t> &b1, std::vector<uint8_t> &b_primes_index,
                uint64_t plainText, std::vector<uint64_t> &c0,
                std::vector<uint64_t> &c1, std::vector<uint64_t> &c2,
                std::vector<uint8_t> &output_primes_index);


bool MultLowLvlCompleted();


} // namespace fpga
} // namespace hexl
} // namespace intel


#endif