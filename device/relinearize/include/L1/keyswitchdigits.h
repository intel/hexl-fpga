#pragma once
#include "common.h"
#include <CL/sycl.hpp>

namespace L1 {
/**
 * @brief keySwitchDigits
 *
 * @param q
 * @param primes
 * @param keys
 * @param digit1
 * @param digit2
 * @param digit2
 * @param digit4
 * @param c0
 * @param c1
 * @param num_keys_primes
 * @param num_digits
 * @param num_primes
 * @return sycl::event
 */
sycl::event keySwitchDigits(
    sycl::queue &q, sycl::buffer<sycl::ulong4> primes,
    sycl::buffer<sycl::ulong2> &keys1, sycl::buffer<sycl::ulong2> &keys2,
    sycl::buffer<sycl::ulong2> &keys3, sycl::buffer<sycl::ulong2> &keys4,
    sycl::buffer<uint64_t> &digit1, sycl::buffer<uint64_t> &digit2,
    sycl::buffer<uint64_t> &digit3, sycl::buffer<uint64_t> &digit4,
    sycl::buffer<uint64_t> &c0, sycl::buffer<uint64_t> &c1, unsigned num_digits,
    unsigned num_primes, sycl::ulong4 digits_offset, sycl::event depend_event,
    unsigned flag);
}  // namespace L1
