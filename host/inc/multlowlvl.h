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

//
// a0.size() = b1.size() = coeff_count * a_primes_size, a_primes_index.size() = a_primes_size;
// b0.size() = b1.size() = coeff_count * b_primes_size. b_primes_index.size() = b_primes_size;
// c0.size() = c1.size() = c2.size() = coeff_count * c_primes_size. output_primes_index.size() = c_primes_size;

void MultLowLvl(uint64_t* a0, uint64_t* a1, uint64_t a_primes_size, uint8_t* a_primes_index,
                uint64_t* b0, uint64_t* b1, uint64_t b_primes_size, uint8_t* b_primes_index,
                uint64_t plainText, uint64_t coeff_count, 
                uint64_t* c0, uint64_t* c1, uint64_t* c2, uint64_t c_primes_size, uint8_t* output_primes_index);


bool MultLowLvlCompleted();

} // namespace fpga
} // namespace hexl
} // namespace intel


#endif