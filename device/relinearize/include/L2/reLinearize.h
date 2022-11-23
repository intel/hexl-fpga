#pragma once
#include <cinttypes>
#include <vector>
#include "utils.h"
#include <iomanip>

namespace L2 {
namespace helib {
namespace bgv {
namespace Relinearize {
/**
 * @brief Init
 *
 * @param all_primes
 * @param keys1
 * @param keys2
 * @param keys3
 * @param keys4
 */
void Init(std::vector<uint64_t> &all_primes, const std::vector<uint64_t> &keys1,
          const std::vector<uint64_t> &keys2,
          const std::vector<uint64_t> &keys3,
          const std::vector<uint64_t> &keys4);
/**
 * @brief Relinerize
 *
 * @param input
 * @param pi
 * @param num_designed_digits_primes
 * @param num_special_primes
 * @param primes_index
 * @param c0
 * @param c1
 */
void Relinearize(std::vector<uint64_t> &input, std::vector<uint64_t> &pi,
                 std::vector<unsigned> num_designed_digits_primes,
                 unsigned num_special_primes,
                 const std::vector<uint8_t> &primes_index,
                 std::vector<uint64_t> &output);
/**
 * @brief Process left output
 *
 */
void Wait();
}  // namespace Relinearize
}  // namespace bgv
}  // namespace helib
}  // namespace L2
