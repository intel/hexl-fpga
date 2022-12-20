#pragma once
#include <CL/sycl.hpp>
#include <L1/common.h>

extern "C" {
/**
 * @brief load kernel
 *
 * @param q
 * @param c
 * @return sycl::event
 */
sycl::event LoadBreakIntoDigits(sycl::queue &q, sycl::buffer<uint64_t> &c,
                                unsigned size, int flag, sycl::event depend);

/**
 * @brief BreakIntoDigits
 *
 * @param q
 * @param packed_precomuted_params_buf
 * @param num_digit1_primes
 * @param num_digit2_primes
 * @param num_special_primes
 * @return event
 */
sycl::event BreakIntoDigits(
    sycl::queue &q, sycl::buffer<sycl::ulong2> &packed_precomuted_params_buf,
    uint num_digits, uint num_digit1_primes, uint num_digit2_primes,
    uint num_digit3_primes, uint num_digit4_primes, uint num_special_primes,
    uint special_primes_offset, sycl::event depend, int flag);

/**
 * @brief store kernel
 *
 * @param q
 * @param c
 * @return sycl::event
 */
sycl::event StoreBreakIntoDigits(sycl::queue &q, sycl::buffer<uint64_t> &c,
                                 unsigned size, int flag);

void LaunchBreakIntoDigitsINTT(const std::vector<uint64_t> &primes,
                               uint64_t coeff_count, int flag = 0xff);

void LaunchBreakIntoDigitsNTT(const std::vector<uint64_t> &primes,
                              uint64_t coeff_count, int flag = 0xff);
}
