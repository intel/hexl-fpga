// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "relinearize/include/L2/breakintodigits-impl.hpp"
#include "relinearize/include/L2/keyswitchdigits-impl.hpp"

extern "C" {

// Init function required FPGA kernel interfaces.
// Let's try return a static object of INTT/NTT.
// see breakintodigits-impl.hpp Init function.


// break_into_digits_intt_t& get_breakintodigits_intt() {

// }

// breakIntoDigits_ntt_t& get_breakintodigits_ntt() {

// }


// relinearize operations mainly contains two sub-modules: (1) breakintodigits and 
// (2) keyswitchdigits.


// Using a struct breakintodigits_method object to get all functions defined in
// src/L1/breakintodigits.cpp file, using this method, we can avoid expose to much
// FPGA kernel APIs.

breakintodisgits_ops_t& get_breakintodigits_ops_IF() {
    return L1::BreakIntoDigits::get_breakintodigits_ops();
}

// // keyswitcgdigits only has 1 kernel functions, so we call it directly.
// sycl::event keySwitchDitgits_IF(sycl::queue &q, sycl::buffer<sycl::ulong4> primes,
//     sycl::buffer<sycl::ulong2> &keys1, sycl::buffer<sycl::ulong2> &keys2,
//     sycl::buffer<sycl::ulong2> &keys3, sycl::buffer<sycl::ulong2> &keys4,
//     sycl::buffer<uint64_t> &digit1, sycl::buffer<uint64_t> &digit2,
//     sycl::buffer<uint64_t> &digit3, sycl::buffer<uint64_t> &digit4,
//     sycl::buffer<uint64_t> &c0, sycl::buffer<uint64_t> &c1, unsigned num_digits,
//     unsigned num_primes, sycl::ulong4 digits_offset, sycl::event depend_event,
//     unsigned flag) {
//     return L1:: keySwitchDigits(
//         q, primes, keys1, keys2, keys3, keys4, digit1, digit2, digit3, digit4, c0,
//         c1, num_digits, num_primes, digits_offset, depend_event, flag);
// };

// or we can also using the same method with breakintodigits.

keyswitchdigits_ops_t& get_keyswitchdigits_ops_IF() {
    return L1::get_keyswitchdigits_ops();
}


}