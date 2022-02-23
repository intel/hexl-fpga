// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __NUMBER_THEORY_H__
#define __NUMBER_THEORY_H__

#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "fpga_assert.h"

namespace intel {
namespace hexl {
namespace fpga {

__extension__ typedef unsigned __int128 uint128_t;

// Returns whether or not num is a power of two
inline bool IsPowerOfTwo(uint64_t num) { return num && !(num & (num - 1)); }

// Returns maximum number of possible significant bits given modulus
inline uint64_t MSB(uint64_t modulus) {
    return static_cast<uint64_t>(std::log2l(modulus));
}

// Returns log2(x) for x a power of 2
inline uint64_t Log2(uint64_t x) {
    FPGA_ASSERT(IsPowerOfTwo(x), "x not a power of 2");
    return MSB(x);
}

// Returns the maximum value that can be represented using bits bits
inline uint64_t MaximumValue(uint64_t bits) {
    FPGA_ASSERT(bits <= 64, "MaximumValue requires bits <= 64");
    if (bits == 64) {
        return (std::numeric_limits<uint64_t>::max)();
    }
    return (1ULL << bits) - 1;
}

// Return x * y as 128-bit integer
// Correctness if x * y < 128 bits
inline uint128_t MultiplyUInt64(uint64_t x, uint64_t y) {
    return uint128_t(x) * uint128_t(y);
}

// Multiplies x * y as 128-bit integer.
// @param prod_hi Stores high 64 bits of product
// @param prod_lo Stores low 64 bits of product
inline void MultiplyUInt64(uint64_t x, uint64_t y, uint64_t* prod_hi,
                           uint64_t* prod_lo) {
    uint128_t prod = MultiplyUInt64(x, y);
    *prod_hi = static_cast<uint64_t>(prod >> 64);
    *prod_lo = static_cast<uint64_t>(prod);
}

// Return the high 128 minus BitShift bits of the 128-bit product x * y
template <int BitShift>
inline uint64_t MultiplyUInt64Hi(uint64_t x, uint64_t y) {
    uint128_t product = MultiplyUInt64(x, y);
    return static_cast<uint64_t>(product >> BitShift);
}

// Returns low 64bit of 128b/64b where x1=high 64b, x0=low 64b
inline uint64_t DivideUInt128UInt64Lo(uint64_t x1, uint64_t x0, uint64_t y) {
    uint128_t n =
        (static_cast<uint128_t>(x1) << 64) | (static_cast<uint128_t>(x0));
    uint128_t q = n / y;

    return static_cast<uint64_t>(q);
}

// Computes (x * y) mod modulus, except that the output is in [0, 2 * modulus]
// @param modulus_precon Pre-computed Barrett reduction factor
template <int BitShift>
inline uint64_t MultiplyUIntModLazy(uint64_t x, uint64_t y_operand,
                                    uint64_t y_barrett_factor,
                                    uint64_t modulus) {
    FPGA_ASSERT(y_operand < modulus, "y_operand must be less than modulus");
    FPGA_ASSERT(modulus <= MaximumValue(BitShift), "Modulus exceeds bound");
    FPGA_ASSERT(x <= MaximumValue(BitShift), "x exceeds bound");

    uint64_t Q = MultiplyUInt64Hi<BitShift>(x, y_barrett_factor);
    return y_operand * x - Q * modulus;
}

// Computes (x * y) mod modulus, except that the output is in [0, 2 * modulus]
template <int BitShift>
inline uint64_t MultiplyUIntModLazy(uint64_t x, uint64_t y, uint64_t modulus) {
    FPGA_ASSERT(BitShift == 64 || BitShift == 52, "Unsupport BitShift");
    FPGA_ASSERT(x <= MaximumValue(BitShift), "x exceeds bound");
    FPGA_ASSERT(y < modulus, "y must be less than modulus");
    FPGA_ASSERT(modulus <= MaximumValue(BitShift), "modulus exceeds bound");
    uint64_t y_hi{0};
    uint64_t y_lo{0};
    if (BitShift == 64) {
        y_hi = y;
        y_lo = 0;
    } else if (BitShift == 52) {
        y_hi = y >> 12;
        y_lo = y << 52;
    }
    uint64_t y_barrett = DivideUInt128UInt64Lo(y_hi, y_lo, modulus);
    return MultiplyUIntModLazy<BitShift>(x, y, y_barrett, modulus);
}

// Adds two unsigned 64-bit integers
// @param operand1 Number to add
// @param operand2 Number to add
// @param result Stores the sum
// @return The carry bit
inline unsigned char AddUInt64(uint64_t operand1, uint64_t operand2,
                               uint64_t* result) {
    *result = operand1 + operand2;
    return static_cast<unsigned char>(*result < operand1);
}

// Returns whether or not the input is prime
bool IsPrime(uint64_t n);

// Generates a list of num_primes primes in the range [2^(bit_size,
// 2^(bit_size+1)]. Ensures each prime q satisfies
// q % (2*ntt_size+1)) == 1
// @param num_primes Number of primes to generate
// @param bit_size Bit size of each prime
// @param ntt_size N such that each prime q satisfies q % (2N) == 1. N must be
// a power of two
std::vector<uint64_t> GeneratePrimes(size_t num_primes, size_t bit_size,
                                     size_t ntt_size = 1);

// returns input mod modulus, computed via Barrett reduction
// @param q_barr floor(2^64 / p)
uint64_t BarrettReduce64(uint64_t input, uint64_t modulus, uint64_t q_barr);

template <int InputModFactor>
uint64_t ReduceMod(uint64_t x, uint64_t modulus,
                   const uint64_t* twice_modulus = nullptr,
                   const uint64_t* four_times_modulus = nullptr) {
    FPGA_ASSERT(InputModFactor == 1 || InputModFactor == 2 ||
                    InputModFactor == 4 || InputModFactor == 8,
                "InputModFactor should be 1, 2, 4, or 8");
    if (InputModFactor == 1) {
        return x;
    }
    if (InputModFactor == 2) {
        if (x >= modulus) {
            x -= modulus;
        }
        return x;
    }
    if (InputModFactor == 4) {
        FPGA_ASSERT(twice_modulus != nullptr,
                    "twice_modulus should not be nullptr");
        if (x >= *twice_modulus) {
            x -= *twice_modulus;
        }
        if (x >= modulus) {
            x -= modulus;
        }
        return x;
    }
    if (InputModFactor == 8) {
        FPGA_ASSERT(twice_modulus != nullptr,
                    "twice_modulus should not be nullptr");
        FPGA_ASSERT(four_times_modulus != nullptr,
                    "four_times_modulus should not be nullptr");

        if (x >= *four_times_modulus) {
            x -= *four_times_modulus;
        }
        if (x >= *twice_modulus) {
            x -= *twice_modulus;
        }
        if (x >= modulus) {
            x -= modulus;
        }
        return x;
    }
    FPGA_ASSERT(false, "Should be unreachable");
    return x;
}

inline uint64_t BarrettReduce128(uint64_t input_hi, uint64_t input_lo,
                                 uint64_t modulus) {
    FPGA_ASSERT(modulus != 0, "modulus == 0")
    uint128_t n = (static_cast<uint128_t>(input_hi) << 64) |
                  (static_cast<uint128_t>(input_lo));

    return static_cast<uint64_t>(n % modulus);
}

// Stores an integer on which modular multiplication can be performed more
// efficiently, at the cost of some precomputation.
class MultiplyFactor {
public:
    MultiplyFactor() = default;

    // Computes and stores the Barrett factor (operand << bit_shift) / modulus
    MultiplyFactor(uint64_t operand, uint64_t bit_shift, uint64_t modulus)
        : m_operand(operand) {
        FPGA_ASSERT(operand <= modulus, "operand must be less than modulus");
        FPGA_ASSERT(bit_shift == 64 || bit_shift == 52, "Unsupport BitShift");
        uint64_t op_hi{0};
        uint64_t op_lo{0};

        if (bit_shift == 64) {
            op_hi = operand;
            op_lo = 0;
        } else if (bit_shift == 52) {
            op_hi = operand >> 12;
            op_lo = operand << 52;
        }
        m_barrett_factor = DivideUInt128UInt64Lo(op_hi, op_lo, modulus);
    }

    inline uint64_t BarrettFactor() const { return m_barrett_factor; }
    inline uint64_t Operand() const { return m_operand; }

private:
    uint64_t m_operand;
    uint64_t m_barrett_factor;
};

// Reverses the bits
uint64_t ReverseBitsUInt(uint64_t x, uint64_t bits);

// Returns a^{-1} mod modulus
uint64_t InverseUIntMod(uint64_t a, uint64_t modulus);

//// Returns (x * y) mod modulus
//// Assumes x, y < modulus
uint64_t MultiplyUIntMod(uint64_t x, uint64_t y, uint64_t modulus);

// Returns (x * y) mod modulus
// @param y_precon floor(2**64 / modulus)
uint64_t MultiplyMod(uint64_t x, uint64_t y, uint64_t y_precon,
                     uint64_t modulus);

// Returns (x + y) mod modulus
// Assumes x, y < modulus
uint64_t AddUIntMod(uint64_t x, uint64_t y, uint64_t modulus);

// Returns (x - y) mod modulus
// Assumes x, y < modulus
uint64_t SubUIntMod(uint64_t x, uint64_t y, uint64_t modulus);

// Returns base^exp mod modulus
uint64_t PowMod(uint64_t base, uint64_t exp, uint64_t modulus);

// Returns true whether root is a degree-th root of unity
// degree must be a power of two.
bool IsPrimitiveRoot(uint64_t root, uint64_t degree, uint64_t modulus);

// Tries to return a primtiive degree-th root of unity
// Returns -1 if no root is found
uint64_t GeneratePrimitiveRoot(uint64_t degree, uint64_t modulus);

// Returns true whether root is a degree-th root of unity
// degree must be a power of two.
uint64_t MinimalPrimitiveRoot(uint64_t degree, uint64_t modulus);

void ComputeRootOfUnityPowers(uint64_t m_q, uint64_t m_degree,
                              uint64_t m_degree_bits, uint64_t m_w,
                              uint64_t* inv_root_of_unity_powers,
                              uint64_t* precon64_inv_root_of_unity_powers,
                              uint64_t* root_of_unity_powers,
                              uint64_t* precon64_root_of_unity_powers);
}  // namespace fpga
}  // namespace hexl
}  // namespace intel

#endif
