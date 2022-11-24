// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

#include "fpga.h"
#include "fpga_assert.h"
#include "hexl-fpga.h"
#include "relinearize_int.h"


namespace intel {
namespace hexl {
namespace fpga {

void ReLinearize(uint64_t* all_primes, size_t all_primes_len, 
                uint64_t* keys1, uint64_t* keys2,
                uint64_t* keys3, uint64_t* keys4, size_t keys_len,
                uint64_t* c2, size_t c2_len,
                uint64_t* pi, size_t pi_len,
                unsigned* num_designed_digits_primes, size_t digits_primes_len,
                size_t num_special_primes, uint8_t* primes_index, size_t primes_index_len,
                uint64_t* output, size_t output_len) {
    
    ReLinearize_int(all_primes, all_primes_len, keys1, keys2, keys3, keysa4, keys_len,
                    c2, c2_len, pi, pi_len, 
                    num_designed_digits_primes, digits_primes_len, 
                    num_special_primes, primes_index_len,
                    output, output_len);
    
}

bool ReLinearizeCompleted() {
    return ReLinearizeCompleted_int();
}


void set_worksize_ReLinearize(uint64_t n) {
    set_worksize_ReLinearize_int(n);
}



} // namespace fpga
} // namespace hexl
} // namespace intel