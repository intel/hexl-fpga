// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "relinearize/include/L2/breakintodigits-impl.hpp"
#include "relinearize/include/L2/keyswitchdigits-impl.hpp"

extern "C" {

// Init function required FPGA kernel interfaces.
// Let's try return a static object of INTT/NTT.
// see breakintodigits-impl.hpp Init function.


break_into_digits_intt_t& get_breakintodigits_intt() {

}

breakIntoDigits_ntt_t& get_breakintodigits_ntt() {

}



// Using a struct breakintodigits_method object to get all functions defined in
// src/L1/breakintodigits.cpp file, using this method, we can avoid expose to much
// FPGA kernel APIs.

breakintodisgits_ops_t& get_breakintodigits_ops_IF() {
    
}


}