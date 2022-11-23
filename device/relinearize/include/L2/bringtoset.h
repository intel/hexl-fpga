#pragma once
#include <cinttypes>
#include <vector>
#include <iomanip>

namespace L2 {
namespace helib {
namespace bgv {
namespace BringToSet {
/**
 * @brief Perform BringToSet for a, b and output to c
 *
 * @param primes
 * @param a
 * @param a_primes_index
 * @param b
 * @param b_primes_index
 * @param plainText
 * @param c
 * @param output_primes_index
 */
void BringToSet(uint64_t plainText, std::vector<uint64_t> &a,
                std::vector<uint8_t> &a_primes_index, std::vector<uint64_t> &b,
                std::vector<uint8_t> &b_primes_index, std::vector<uint64_t> &c,
                std::vector<uint8_t> &output_primes_index);

/**
 * @brief init
 *
 * @param primes
 */
void init(std::vector<uint64_t> primes, uint32_t input_mem_channel = 1,
          uint32_t output_mem_channel = 2);

/**
 * @brief wait
 *
 */
void wait();
}  // namespace BringToSet
}  // namespace bgv
}  // namespace helib
}  // namespace L2
