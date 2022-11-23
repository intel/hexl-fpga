#pragma once
#include <cinttypes>
#include <vector>
#include <iomanip>

namespace L2 {
namespace helib {
namespace bgv {
namespace TensorProduct {
/**
 * @brief init
 *
 */
void Init(const std::vector<uint64_t> &primes, uint32_t input_mem_channel = 1,
          uint32_t output_mem_channel = 2);
/**
 * @brief wait
 *
 */
void Wait();

/**
 * @brief TensorProduct
 *
 * @param primes_index
 * @param inputs
 * @param c0
 * @param c1
 * @param c2
 */
void TensorProduct(const std::vector<uint8_t> &primes_index,
                   const std::vector<uint64_t> &inputs,
                   std::vector<uint64_t> &c0, std::vector<uint64_t> &c1,
                   std::vector<uint64_t> &c2);
}  // namespace TensorProduct
}  // namespace bgv
}  // namespace helib
}  // namespace L2
