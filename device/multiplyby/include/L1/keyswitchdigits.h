#pragma once
#include "common.h"
#include <CL/sycl.hpp>

extern "C" {
/**
 * @brief keySwitchDigits
 *
 * @param q
 * @param primes
 * @param keys
 * @param digits
 * @param c_input
 * @param c_output
 * @param num_digits
 * @param num_primes
 * @param digits_offset
 * @param depend_event
 * @param flag
 * @return sycl::event
 */
sycl::event KeySwitchDigits(
    sycl::queue &q, sycl::event depend_event, sycl::buffer<ulong4> &primes,
    sycl::buffer<ulong2> &keys, sycl::buffer<uint64_t> &digits,
    sycl::buffer<uint64_t> &c_input, sycl::buffer<uint64_t> &output1,
    sycl::buffer<uint64_t> &output2, unsigned num_digits, unsigned num_primes,
    unsigned num_normal_primes, unsigned num_all_primes, unsigned flag);

sycl::event StoreKeySwitchDigits(sycl::queue &q,
                                 sycl::buffer<uint64_t> &output1,
                                 sycl::buffer<uint64_t> &output2, unsigned size,
                                 unsigned flag);
}
