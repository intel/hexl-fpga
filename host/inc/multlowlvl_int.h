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


void MultLowLvl_int(uint64_t* a0, uint64_t* a1, uint64_t a_primes_size, uint8_t* a_primes_index,
                    uint64_t* b0, uint64_t* b1, uint64_t b_primes_size, uint8_t* b_primes_index,
                    uint64_t plainText, uint64_t coeff_count, 
                    uint64_t* c0, uint64_t* c1, uint64_t* c2, 
                    uint64_t c_primes_size, uint8_t* output_primes_index, uint64_t* primes, uint64_t primes_size);

bool MultLowLvlCompleted_int();


} // namespace fpga
} // namespace hexl
} // namespace intel


#endif