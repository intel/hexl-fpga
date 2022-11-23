#pragma once
#include <L1/common.h>

namespace L1 {
namespace BringToSet {
#define MAX_RESCALE_P 5
#define MAX_RESCALE_P_BANKS 8
#define MAX_RESCALE_Q 23

#define RESCALE_INTT_ID 1
#define RESCALE_NTT_ID 2

#define RESCALE_INTT_VEC 8
#define RESCALE_NTT_VEC 8

void intt(const std::vector<uint64_t> &primes, uint64_t coeff_count,
          int flag = 0xff);
void ntt(const std::vector<uint64_t> &primes, uint64_t coeff_count,
         int flag = 0xff);
/**
 * @brief Load the input for rescale
 *
 * @param q
 * @param c
 * @param prime_index_set_buf
 * @return sycl::event
 */
sycl::event load(sycl::queue &q, sycl::buffer<uint64_t> &c,
                 sycl::buffer<uint8_t> &prime_index_set_buf,
                 unsigned prime_size);

/**
 * @brief Rescale
 *
 * @return sycl::event
 */
sycl::event kernel(sycl::queue &q, uint32_t coeff_count,
                   sycl::buffer<sycl::ulong2> &scale_param_set_buf, uint32_t P,
                   uint32_t Q, uint I, uint64_t t);

/**
 * @brief store the rescale output
 *
 */
sycl::event store(sycl::queue &q, sycl::buffer<uint64_t> &c, unsigned size);
}  // namespace BringToSet
}  // namespace L1
