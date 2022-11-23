#pragma once
#include <cinttypes>
#include <vector>
#include "utils.h"
#include <iomanip>

namespace L2 {
namespace helib {
namespace bgv {
namespace KeySwitchDigits {
/**
 * @brief Init
 *
 * @param primes
 * @param keys1
 * @param keys2
 * @param keys3
 * @param keys4
 */
void Init(const std::vector<uint64_t> &primes,
          const std::vector<uint64_t> &keys1,
          const std::vector<uint64_t> &keys2,
          const std::vector<uint64_t> &keys3,
          const std::vector<uint64_t> &keys4);

/**
 * @brief KeySwitchDigits
 *
 * @param primes_index
 * @param digit1
 * @param digit2
 * @param digit3
 * @param digit4
 * @param c0
 * @param c1
 */
void KeySwitchDigits(const std::vector<uint8_t> &primes_index,
                     const std::vector<uint64_t> &digits,
                     std::vector<uint64_t> &output);

/**
 * @brief Wait to complete all requests
 *
 */
void Wait();
}  // namespace KeySwitchDigits
}  // namespace bgv
}  // namespace helib
}  // namespace L2
