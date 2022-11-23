#pragma once
#include <CL/sycl.hpp>
#include <L1/common.h>

namespace L1 {
namespace BreakIntoDigits {
// relinearlize
#define MAX_NORMAL_PRIMES 18
#define MAX_SPECIAL_PRIMES 9
#define MAX_C2_DROP_SMALL_P 4
#define MAX_C2_DROP_SMALL_P_BANKS 4
#define MAX_C2_DROP_SMALL_Q 23

#define MAX_DIGIT_SIZE_POW_2 4
#define MAX_DIGIT_SIZE 4
#define MAX_DIGITS 4

/**
 * @brief load kernel
 *
 * @param q
 * @param c
 * @return sycl::event
 */
sycl::event load(sycl::queue &q, sycl::buffer<uint64_t> &c, unsigned size);

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
sycl::event kernel(sycl::queue &q,
                   sycl::buffer<sycl::ulong2> &packed_precomuted_params_buf,
                   uint num_digits, uint num_digit1_primes,
                   uint num_digit2_primes, uint num_digit3_primes,
                   uint num_digit4_primes, uint num_special_primes,
                   uint special_primes_offset, int flag);

/**
 * @brief store kernel
 *
 * @param q
 * @param c
 * @return sycl::event
 */
sycl::event store(sycl::queue &q, sycl::buffer<uint64_t> &c, unsigned size,
                  int flag);

void intt(const std::vector<uint64_t> &primes, uint64_t coeff_count,
          int flag = 0xff);

void ntt(const std::vector<uint64_t> &primes, uint64_t coeff_count,
         int flag = 0xff);

}  // namespace BreakIntoDigits
}  // namespace L1
