#pragma once
#include <cinttypes>
#include <vector>
#include "utils.h"

namespace L2 {
namespace helib {
namespace bgv {
/**
 * @brief init
 *
 * @param primes
 */
void Init(std::vector<uint64_t> &primes);

/**
 * @brief load1
 *
 * @param input
 * @param primes_index
 */
void Load1(std::vector<uint64_t> &input, std::vector<uint8_t> &primes_index);

/**
 * @brief load for engine 2
 *
 * @param input
 * @param primes_index
 */
void Load2(std::vector<uint64_t> &input, std::vector<uint8_t> &primes_index);

/**
 * @brief TensorProduct
 *
 * @param input1
 * @param input2
 * @param output1
 * @param output2
 * @param output3
 * @param primes all the primes including the small primes, normal primes, and
 * special primes
 * @param primes_index index the current primes
 */
void TensorProduct(std::vector<uint64_t> &primes,
                   std::vector<uint8_t> &primes_index);

/**
 * @brief store the multiLowLvl result
 *
 * @param output1
 * @param output2
 * @param output3
 * @param BATCH
 */
void Store(std::vector<uint64_t> &output1, std::vector<uint64_t> &output2,
           std::vector<uint64_t> &output3, size_t BATCH = 1);

/**
 * @brief MultLowLvl
 *
 * @param a0
 * @param a1
 * @param a_primes_index
 * @param b0
 * @param b1
 * @param b_primes_index
 * @param plainText
 * @param c0
 * @param c1
 * @param c2
 * @param output_primes_index
 */
void MultLowLvl(std::vector<uint64_t> &a0, std::vector<uint64_t> &a1,
                std::vector<uint8_t> &a_primes_index, std::vector<uint64_t> &b0,
                std::vector<uint64_t> &b1, std::vector<uint8_t> &b_primes_index,
                uint64_t plainText, std::vector<uint64_t> &c0,
                std::vector<uint64_t> &c1, std::vector<uint64_t> &c2,
                std::vector<uint8_t> &output_primes_index);
/**
 * @brief drop the small primes for the polynomial x^2
 *
 * @param c2
 * @param coeff_count
 * @param pi
 * @param qj
 * @param plainText
 */
void C2DropSmall(std::vector<uint64_t> &c2, uint32_t coeff_count,
                 std::vector<uint64_t> &pi, std::vector<uint64_t> &qj,
                 uint64_t plainText);

/**
 * @brief Break into digits for the polynomial x^2
 *
 * @param coeff_count
 * @param pi
 * @param output
 * @param num_digit_primes
 * @param num_special_primes
 */
void BreakIntoDigits(uint32_t coeff_count, std::vector<uint64_t> &pi,
                     std::vector<uint64_t> &all_primes,
                     std::vector<uint64_t> &output, unsigned num_digit_primes,
                     unsigned num_special_primes);

/**
 * @brief Launch the KeySwitchDigits NTT kernel
 *
 * @param primes
 */
void LaunchKeySwitchDigitsNTT(const std::vector<uint64_t> &primes);

/**
 * @brief Return c1 = a * wa mod modulus and c2 = b * wb mod modulus
 *
 * @param coeff_count Coeff count of input and output polynomial
 * @param primes the primes of a and b
 * @param primes_index_set the prime index of a and b, zero is the first prime
 * of normal primes, not the small primes
 * @param a input polynomial digit 0
 * @param b input polynomial digit 1. It is possible to have only 1 digit
 * @param wa the keys of digit 0
 * @param wb the keys of digit 1
 * @param c1 output polynomial c1
 * @param c2 output polynomial c2
 */
void KeySwitchDigits(uint32_t coeff_count, const std::vector<uint64_t> &primes,
                     const std::vector<uint8_t> &primes_index,
                     const std::vector<uint64_t> &input,
                     const std::vector<uint64_t> &wa,
                     const std::vector<uint64_t> &wb, std::vector<uint64_t> &c1,
                     std::vector<uint64_t> &c2);

/**
 * @brief drop small for c0
 *
 * @param input
 * @param coeff_count
 * @param pi
 * @param qj
 * @param qj_primes_index
 * @param special_primes
 * @param plainText
 * @param output
 */
void C0DropSmall(std::vector<uint64_t> &input, uint32_t coeff_count,
                 std::vector<uint64_t> &pi, std::vector<uint64_t> &qj,
                 std::vector<uint64_t> &all_primes,
                 std::vector<uint8_t> &qj_primes_index,
                 std::vector<uint64_t> &special_primes, uint64_t plainText,
                 std::vector<uint64_t> &output);

/**
 * @brief LaunchReLinearizeC01NTT
 *
 * @param primes
 */
void LaunchReLinearizeC01NTT(const std::vector<uint64_t> &primes);
}  // namespace bgv
}  // namespace helib
}  // namespace L2
