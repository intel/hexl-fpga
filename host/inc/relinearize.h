// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __RELINEARIZE_H_
#define __RELINEARIZE_H_

#include <cstdint>
#include <vector>

namespace intel {
namespace hexl {
namespace fpga {

void set_worksize_ReLinearize(uint64_t ws);

void ReLinearize(uint64_t* all_primes, size_t all_primes_len, 
                uint64_t* keys1, uint64_t* keys2,
                uint64_t* keys3, uint64_t* keys4, size_t keys_len,
                uint64_t* c2, size_t c2_len,
                uint64_t* pi, size_t pi_len,
                unsigned* num_designed_digits_primes, size_t digits_primes_len,
                size_t num_special_primes, uint8_t* primes_index, size_t primes_index_len,
                uint64_t* output, size_t output_len);

bool ReLinearizeCompleted();

} // namespace fpga
} // namespace hexl
} // namespace intel


#endif

