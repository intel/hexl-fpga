#pragma once
#include <L1/common.h>

extern "C" {
void LaunchBringToSetINTT(const std::vector<uint64_t> &primes,
                          uint64_t coeff_count, int flag = 0xff);
void LaunchBringToSetNTT(const std::vector<uint64_t> &primes,
                         uint64_t coeff_count, int flag = 0xff);
/**
 * @brief Load the input for rescale
 *
 * @param q
 * @param c
 * @param prime_index_set_buf
 * @return sycl::event
 */
sycl::event LoadBringToSet(sycl::queue &q, sycl::buffer<uint64_t> &c,
                           sycl::buffer<uint8_t> &prime_index_set_buf,
                           unsigned prime_size, int flag);

/**
 * @brief Rescale
 *
 * @return sycl::event
 */
sycl::event BringToSet(sycl::queue &q, uint32_t coeff_count,
                       sycl::buffer<sycl::ulong2> &scale_param_set_buf,
                       uint32_t P, uint32_t Q, uint I, uint64_t t);

/**
 * @brief store the rescale output
 *
 */
sycl::event StoreBringToSet(sycl::queue &q, sycl::buffer<uint64_t> &c,
                            unsigned size, int flag);
}
