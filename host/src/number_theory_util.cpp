// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <random>

#include "number_theory_util.h"

namespace intel {
namespace hexl {
namespace fpga {

uint64_t InverseUIntMod(uint64_t input, uint64_t modulus) {
    uint64_t a = input % modulus;
    FPGA_ASSERT(a != 0, "input does not have a InverseMod");

    if (modulus == 1) {
        return 0;
    }

    int64_t m0 = static_cast<int64_t>(modulus);
    int64_t y = 0;
    int64_t x = 1;
    while (a > 1) {
        // q is quotient
        int64_t q = static_cast<int64_t>(a / modulus);

        int64_t t = static_cast<int64_t>(modulus);
        modulus = a % modulus;
        a = static_cast<uint64_t>(t);

        // Update y and x
        t = y;
        y = x - q * y;
        x = t;
    }

    // Make x positive
    if (x < 0) x += m0;

    return uint64_t(x);
}

uint64_t BarrettReduce64(uint64_t input, uint64_t modulus, uint64_t q_barr) {
    FPGA_ASSERT(modulus != 0, "modulus == 0");
    uint64_t q = MultiplyUInt64Hi<64>(input, q_barr);
    uint64_t q_times_input = input - q * modulus;
    return q_times_input >= modulus ? q_times_input - modulus : q_times_input;
}

uint64_t MultiplyUIntMod(uint64_t x, uint64_t y, uint64_t modulus) {
    FPGA_ASSERT(modulus != 0, "modulus == 0");
    FPGA_ASSERT(x < modulus, "x >= modulus");
    FPGA_ASSERT(y < modulus, "y >= modulus");
    uint64_t prod_hi, prod_lo;
    MultiplyUInt64(x, y, &prod_hi, &prod_lo);

    return BarrettReduce128(prod_hi, prod_lo, modulus);
}

uint64_t MultiplyMod(uint64_t x, uint64_t y, uint64_t y_precon,
                     uint64_t modulus) {
    uint64_t q = MultiplyUInt64Hi<64>(x, y_precon);
    q = x * y - q * modulus;
    return q >= modulus ? q - modulus : q;
}

uint64_t AddUIntMod(uint64_t x, uint64_t y, uint64_t modulus) {
    FPGA_ASSERT(x < modulus, "x >= modulus");
    FPGA_ASSERT(y < modulus, "y >= modulus");
    uint64_t sum = x + y;
    return (sum >= modulus) ? (sum - modulus) : sum;
}

uint64_t SubUIntMod(uint64_t x, uint64_t y, uint64_t modulus) {
    FPGA_ASSERT(x < modulus, "x >= modulus");
    FPGA_ASSERT(y < modulus, "y >= modulus");
    uint64_t diff = (x + modulus) - y;
    return (diff >= modulus) ? (diff - modulus) : diff;
}

// Returns base^exp mod modulus
uint64_t PowMod(uint64_t base, uint64_t exp, uint64_t modulus) {
    base %= modulus;
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) {
            result = MultiplyUIntMod(result, base, modulus);
        }
        base = MultiplyUIntMod(base, base, modulus);
        exp >>= 1;
    }
    return result;
}

// Returns true whether root is a degree-th root of unity
// degree must be a power of two.
bool IsPrimitiveRoot(uint64_t root, uint64_t degree, uint64_t modulus) {
    if (root == 0) {
        return false;
    }
    FPGA_ASSERT(IsPowerOfTwo(degree), "degree is not a power of 2");

    // Check if root^(degree/2) == -1 mod modulus
    return PowMod(root, degree / 2, modulus) == (modulus - 1);
}

// Tries to return a primitive degree-th root of unity
// throw error if no root is found
uint64_t GeneratePrimitiveRoot(uint64_t degree, uint64_t modulus) {
    std::default_random_engine generator;
    std::uniform_int_distribution<uint64_t> distribution(0, modulus - 1);

    // We need to divide modulus-1 by degree to get the size of the quotient
    // group
    uint64_t size_entire_group = modulus - 1;

    // Compute size of quotient group
    uint64_t size_quotient_group = size_entire_group / degree;

    for (int trial = 0; trial < 200; ++trial) {
        uint64_t root = distribution(generator);
        root = PowMod(root, size_quotient_group, modulus);

        if (IsPrimitiveRoot(root, degree, modulus)) {
            return root;
        }
    }
    FPGA_ASSERT(false, "no primitive root found for degree, modulus");
    return 0;
}

// Returns true whether root is a degree-th root of unity
// degree must be a power of two.
uint64_t MinimalPrimitiveRoot(uint64_t degree, uint64_t modulus) {
    FPGA_ASSERT(IsPowerOfTwo(degree), "degere is not a power of 2");

    uint64_t root = GeneratePrimitiveRoot(degree, modulus);

    uint64_t generator_sq = MultiplyUIntMod(root, root, modulus);
    uint64_t current_generator = root;

    uint64_t min_root = root;

    // Check if root^(degree/2) == -1 mod modulus
    for (size_t i = 0; i < degree; ++i) {
        if (current_generator < min_root) {
            min_root = current_generator;
        }
        current_generator =
            MultiplyUIntMod(current_generator, generator_sq, modulus);
    }

    return min_root;
}

uint64_t ReverseBitsUInt(uint64_t x, uint64_t bit_width) {
    if (bit_width == 0) {
        return 0;
    }
    uint64_t rev = 0;
    for (uint64_t i = bit_width; i > 0; i--) {
        rev |= ((x & 1) << (i - 1));
        x >>= 1;
    }
    return rev;
}

// Miller-Rabin primality test
bool IsPrime(uint64_t n) {
    // n < 2^64, so it is enough to test a=2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    // 31, and 37. See
    // https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test#Testing_against_small_sets_of_bases
    static const std::vector<uint64_t> as{2,  3,  5,  7,  11, 13,
                                          17, 19, 23, 29, 31, 37};

    for (const uint64_t a : as) {
        if (n == a) return true;
        if (n % a == 0) return false;
    }

    // Write n == 2**r * d + 1 with d odd.
    uint64_t r = 63;
    while (r > 0) {
        uint64_t two_pow_r = (1ULL << r);
        if ((n - 1) % two_pow_r == 0) {
            break;
        }
        --r;
    }
    FPGA_ASSERT(r != 0, "Error factoring n");
    uint64_t d = (n - 1) / (1ULL << r);

    FPGA_ASSERT(n == (1ULL << r) * d + 1, "Error factoring n");
    FPGA_ASSERT(d % 2 == 1, "d is even");

    for (const uint64_t a : as) {
        uint64_t x = PowMod(a, d, n);
        if ((x == 1) || (x == n - 1)) {
            continue;
        }

        bool prime = false;
        for (uint64_t i = 1; i < r; ++i) {
            x = PowMod(x, 2, n);
            if (x == n - 1) {
                prime = true;
                break;
            }
        }
        if (!prime) {
            return false;
        }
    }
    return true;
}

std::vector<uint64_t> GeneratePrimes(size_t num_primes, size_t bit_size,
                                     size_t ntt_size) {
    FPGA_ASSERT(num_primes > 0, "num_primes == 0");
    FPGA_ASSERT(IsPowerOfTwo(ntt_size), "ntt_size is not a power of two");
    FPGA_ASSERT(Log2(ntt_size) < bit_size,
                "log2(ntt_size) should be less than bit_size");

    uint64_t value = (1ULL << bit_size) + 1;

    std::vector<uint64_t> ret;

    while (value < (1ULL << (bit_size + 1))) {
        if (IsPrime(value)) {
            ret.emplace_back(value);
            if (ret.size() == num_primes) {
                return ret;
            }
        }
        value += 2 * ntt_size;
    }

    FPGA_ASSERT(false, "Failed to find enough primes");
    return ret;
}

}  // namespace fpga
}  // namespace hexl
}  // namespace intel
