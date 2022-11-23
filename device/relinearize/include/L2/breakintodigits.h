#pragma once
#include <cinttypes>
#include <vector>
#include "utils.h"
#include <iomanip>

namespace L2 {
namespace helib {
namespace bgv {
namespace BreakIntoDigits {
/**
 * @brief init
 *
 * @param all_primes
 */
void Init(std::vector<uint64_t> &all_primes);
/**
 * @brief Break into digits
 *
 * @param input
 * @param output
 * @param pi
 * @param num_designed_digits_primes
 * @param num_special_primes
 */
void BreakIntoDigits(std::vector<uint64_t> &input,
                     std::vector<uint64_t> &output, std::vector<uint64_t> &pi,
                     std::vector<unsigned> num_designed_digits_primes,
                     unsigned num_special_primes);
/**
 * @brief Process left output
 *
 */
void Wait();
}  // namespace BreakIntoDigits
}  // namespace bgv
}  // namespace helib
}  // namespace L2
