// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ntt.hpp"

#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
namespace hetest {
namespace utils {

uint64_t InverseUIntMod(uint64_t input, uint64_t modulus) {
    uint64_t a = input % modulus;
    UTILS_CHECK(a != 0, input << " does not have a InverseMod");

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
    UTILS_CHECK(modulus != 0, "modulus == 0");
    uint64_t q = MultiplyUInt64Hi<64>(input, q_barr);
    uint64_t q_times_input = input - q * modulus;
    return q_times_input >= modulus ? q_times_input - modulus : q_times_input;
}

uint64_t MultiplyUIntMod(uint64_t x, uint64_t y, uint64_t modulus) {
    UTILS_CHECK(modulus != 0, "modulus == 0");
    UTILS_CHECK(x < modulus, "x " << x << " >= modulus " << modulus);
    UTILS_CHECK(y < modulus, "y " << y << " >= modulus " << modulus);
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
    UTILS_CHECK(x < modulus, "x " << x << " >= modulus " << modulus);
    UTILS_CHECK(y < modulus, "y " << y << " >= modulus " << modulus);
    uint64_t sum = x + y;
    return (sum >= modulus) ? (sum - modulus) : sum;
}

uint64_t SubUIntMod(uint64_t x, uint64_t y, uint64_t modulus) {
    UTILS_CHECK(x < modulus, "x " << x << " >= modulus " << modulus);
    UTILS_CHECK(y < modulus, "y " << y << " >= modulus " << modulus);
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
    UTILS_CHECK(IsPowerOfTwo(degree), degree << " not a power of 2");

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
    UTILS_CHECK(false, "no primitive root found for degree "
                           << degree << " modulus " << modulus);
    return 0;
}

// Returns true whether root is a degree-th root of unity
// degree must be a power of two.
uint64_t MinimalPrimitiveRoot(uint64_t degree, uint64_t modulus) {
    UTILS_CHECK(IsPowerOfTwo(degree),
                "Degere " << degree << " is not a power of 2");

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
    UTILS_CHECK(r != 0, "Error factoring n " << n);
    uint64_t d = (n - 1) / (1ULL << r);

    UTILS_CHECK(n == (1ULL << r) * d + 1, "Error factoring n " << n);
    UTILS_CHECK(d % 2 == 1, "d is even");

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
    UTILS_CHECK(num_primes > 0, "num_primes == 0");
    UTILS_CHECK(IsPowerOfTwo(ntt_size),
                "ntt_size " << ntt_size << " is not a power of two");
    UTILS_CHECK(Log2(ntt_size) < bit_size,
                "log2(ntt_size) " << Log2(ntt_size)
                                  << " should be less than bit_size "
                                  << bit_size);

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

    UTILS_CHECK(false, "Failed to find enough primes");
    return ret;
}

}  // namespace utils
}  // namespace hetest

namespace hetest {
namespace utils {

AllocatorStrategyPtr mallocStrategy =
    AllocatorStrategyPtr(new details::MallocStrategy);

NTT::NTTImpl::NTTImpl(uint64_t degree, uint64_t q, uint64_t root_of_unity,
                      std::shared_ptr<AllocatorBase> alloc_ptr)
    : m_degree(degree),
      m_q(q),
      m_w(root_of_unity),
      alloc(alloc_ptr),
      m_precon52_root_of_unity_powers(AlignedAllocator<uint64_t, 64>(alloc)),
      m_precon64_root_of_unity_powers(AlignedAllocator<uint64_t, 64>(alloc)),
      m_root_of_unity_powers(AlignedAllocator<uint64_t, 64>(alloc)),
      m_precon52_inv_root_of_unity_powers(
          AlignedAllocator<uint64_t, 64>(alloc)),
      m_precon64_inv_root_of_unity_powers(
          AlignedAllocator<uint64_t, 64>(alloc)),
      m_inv_root_of_unity_powers(AlignedAllocator<uint64_t, 64>(alloc)) {
    alloc = alloc_ptr;

    UTILS_CHECK(CheckNTTArguments(degree, q), "");
    UTILS_CHECK(
        IsPrimitiveRoot(m_w, 2 * degree, q),
        m_w << " is not a primitive 2*" << degree << "'th root of unity");

    m_degree_bits = Log2(m_degree);
    m_winv = InverseUIntMod(m_w, m_q);
    ComputeRootOfUnityPowers();
}

NTT::NTTImpl::NTTImpl(uint64_t degree, uint64_t q,
                      std::shared_ptr<AllocatorBase> alloc_ptr)
    : NTTImpl(degree, q, MinimalPrimitiveRoot(2 * degree, q), alloc_ptr) {}

NTT::NTTImpl::~NTTImpl() = default;

void NTT::NTTImpl::ComputeRootOfUnityPowers() {
    AlignedVector64<uint64_t> root_of_unity_powers(
        m_degree, 0, AlignedAllocator<uint64_t, 64>(alloc));
    AlignedVector64<uint64_t> inv_root_of_unity_powers(
        m_degree, 0, AlignedAllocator<uint64_t, 64>(alloc));

    // 64-bit preconditioning
    root_of_unity_powers[0] = 1;
    inv_root_of_unity_powers[0] = InverseUIntMod(1, m_q);
    uint64_t idx = 0;
    uint64_t prev_idx = idx;

    for (size_t i = 1; i < m_degree; i++) {
        idx = ReverseBitsUInt(i, m_degree_bits);
        root_of_unity_powers[idx] =
            MultiplyUIntMod(root_of_unity_powers[prev_idx], m_w, m_q);
        inv_root_of_unity_powers[idx] =
            InverseUIntMod(root_of_unity_powers[idx], m_q);

        prev_idx = idx;
    }

    // Reordering inv_root_of_powers
    AlignedVector64<uint64_t> temp(m_degree, 0,
                                   AlignedAllocator<uint64_t, 64>(alloc));
    temp[0] = inv_root_of_unity_powers[0];
    idx = 1;

    for (size_t m = (m_degree >> 1); m > 0; m >>= 1) {
        for (size_t i = 0; i < m; i++) {
            temp[idx] = inv_root_of_unity_powers[m + i];
            idx++;
        }
    }
    inv_root_of_unity_powers = std::move(temp);

    // 64-bit preconditioned root of unity powers
    AlignedVector64<uint64_t> precon64_root_of_unity_powers(
        (AlignedAllocator<uint64_t, 64>(alloc)));
    precon64_root_of_unity_powers.reserve(m_degree);
    for (uint64_t root_of_unity : root_of_unity_powers) {
        MultiplyFactor mf(root_of_unity, 64, m_q);
        precon64_root_of_unity_powers.push_back(mf.BarrettFactor());
    }

    NTT::NTTImpl::GetPrecon64RootOfUnityPowers() =
        std::move(precon64_root_of_unity_powers);

    // 52-bit preconditioned root of unity powers
    // if (has_avx512ifma) {
    if (false) {
        AlignedVector64<uint64_t> precon52_root_of_unity_powers(
            (AlignedAllocator<uint64_t, 64>(alloc)));
        precon52_root_of_unity_powers.reserve(m_degree);
        for (uint64_t root_of_unity : root_of_unity_powers) {
            MultiplyFactor mf(root_of_unity, 52, m_q);
            precon52_root_of_unity_powers.push_back(mf.BarrettFactor());
        }

        NTT::NTTImpl::GetPrecon52RootOfUnityPowers() =
            std::move(precon52_root_of_unity_powers);
    }

    NTT::NTTImpl::GetRootOfUnityPowers() = std::move(root_of_unity_powers);

    // 64-bit preconditioned inverse root of unity powers
    AlignedVector64<uint64_t> precon64_inv_root_of_unity_powers(
        (AlignedAllocator<uint64_t, 64>(alloc)));
    precon64_inv_root_of_unity_powers.reserve(m_degree);
    for (uint64_t inv_root_of_unity : inv_root_of_unity_powers) {
        MultiplyFactor mf(inv_root_of_unity, 64, m_q);
        precon64_inv_root_of_unity_powers.push_back(mf.BarrettFactor());
    }

    NTT::NTTImpl::GetPrecon64InvRootOfUnityPowers() =
        std::move(precon64_inv_root_of_unity_powers);

    // 52-bit preconditioned inverse root of unity powers
    // if (has_avx512ifma) {
    if (false) {
        AlignedVector64<uint64_t> precon52_inv_root_of_unity_powers(
            (AlignedAllocator<uint64_t, 64>(alloc)));
        precon52_inv_root_of_unity_powers.reserve(m_degree);
        for (uint64_t inv_root_of_unity : inv_root_of_unity_powers) {
            MultiplyFactor mf(inv_root_of_unity, 52, m_q);
            precon52_inv_root_of_unity_powers.push_back(mf.BarrettFactor());
        }

        NTT::NTTImpl::GetPrecon52InvRootOfUnityPowers() =
            std::move(precon52_inv_root_of_unity_powers);
    }

    NTT::NTTImpl::GetInvRootOfUnityPowers() =
        std::move(inv_root_of_unity_powers);
}

void NTT::NTTImpl::ComputeForward(uint64_t* result, const uint64_t* operand,
                                  uint64_t input_mod_factor,
                                  uint64_t output_mod_factor) {
    UTILS_CHECK(result != nullptr, "result == nullptr");
    UTILS_CHECK(operand != nullptr, "operand == nullptr");
    UTILS_CHECK_BOUNDS(
        operand, m_degree, m_q * input_mod_factor,
        "value in operand exceeds bound " << m_q * input_mod_factor);

    if (result != operand) {
        std::memcpy(result, operand, m_degree * sizeof(uint64_t));
    }

    const uint64_t* root_of_unity_powers = GetRootOfUnityPowersPtr();
    const uint64_t* precon_root_of_unity_powers =
        GetPrecon64RootOfUnityPowersPtr();

    ForwardTransformToBitReverse64(result, m_degree, m_q, root_of_unity_powers,
                                   precon_root_of_unity_powers,
                                   input_mod_factor, output_mod_factor);
}

void NTT::NTTImpl::ComputeInverse(uint64_t* result, const uint64_t* operand,
                                  uint64_t input_mod_factor,
                                  uint64_t output_mod_factor) {
    UTILS_CHECK(operand != nullptr, "operand == nullptr");
    UTILS_CHECK(operand != nullptr, "operand == nullptr");

    UTILS_CHECK_BOUNDS(operand, m_degree, m_q * input_mod_factor,
                       "operand exceeds bound " << m_q * input_mod_factor);

    if (operand != result) {
        std::memcpy(result, operand, m_degree * sizeof(uint64_t));
    }

    const uint64_t* inv_root_of_unity_powers = GetInvRootOfUnityPowersPtr();
    const uint64_t* precon_inv_root_of_unity_powers =
        GetPrecon64InvRootOfUnityPowersPtr();
    InverseTransformFromBitReverse64(
        result, m_degree, m_q, inv_root_of_unity_powers,
        precon_inv_root_of_unity_powers, input_mod_factor, output_mod_factor);
}

// NTT API
NTT::NTT() = default;

NTT::NTT(uint64_t degree, uint64_t q,
         std::shared_ptr<AllocatorBase> alloc_ptr /* = {}*/)
    : m_impl(std::make_shared<NTT::NTTImpl>(degree, q, alloc_ptr)) {}

NTT::NTT(uint64_t degree, uint64_t q, uint64_t root_of_unity,
         std::shared_ptr<AllocatorBase> alloc_ptr /* = {}*/)
    : m_impl(std::make_shared<NTT::NTTImpl>(degree, q, root_of_unity,
                                            alloc_ptr)) {}

NTT::~NTT() = default;

void NTT::ComputeForward(uint64_t* result, const uint64_t* operand,
                         uint64_t input_mod_factor,
                         uint64_t output_mod_factor) {
    UTILS_CHECK(operand != nullptr, "operand == nullptr");
    UTILS_CHECK(result != nullptr, "result == nullptr");
    UTILS_CHECK(
        input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
        "input_mod_factor must be 1, 2 or 4; got " << input_mod_factor);
    UTILS_CHECK(output_mod_factor == 1 || output_mod_factor == 4,
                "output_mod_factor must be 1 or 4; got " << output_mod_factor);

    m_impl->ComputeForward(result, operand, input_mod_factor,
                           output_mod_factor);
}

void NTT::ComputeInverse(uint64_t* result, const uint64_t* operand,
                         uint64_t input_mod_factor,
                         uint64_t output_mod_factor) {
    UTILS_CHECK(operand != nullptr, "operand == nullptr");
    UTILS_CHECK(result != nullptr, "result == nullptr");
    UTILS_CHECK(input_mod_factor == 1 || input_mod_factor == 2,
                "input_mod_factor must be 1 or 2; got " << input_mod_factor);
    UTILS_CHECK(output_mod_factor == 1 || output_mod_factor == 2,
                "output_mod_factor must be 1 or 2; got " << output_mod_factor);

    m_impl->ComputeInverse(result, operand, input_mod_factor,
                           output_mod_factor);
}

// Free functions

void ForwardTransformToBitReverse64(uint64_t* operand, uint64_t n,
                                    uint64_t modulus,
                                    const uint64_t* root_of_unity_powers,
                                    const uint64_t* precon_root_of_unity_powers,
                                    uint64_t input_mod_factor,
                                    uint64_t output_mod_factor) {
    UTILS_CHECK(CheckNTTArguments(n, modulus), "");
    UTILS_CHECK_BOUNDS(operand, n, modulus * input_mod_factor,
                       "operand exceeds bound " << modulus * input_mod_factor);
    UTILS_CHECK(root_of_unity_powers != nullptr,
                "root_of_unity_powers == nullptr");
    UTILS_CHECK(precon_root_of_unity_powers != nullptr,
                "precon_root_of_unity_powers == nullptr");
    UTILS_CHECK(
        input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
        "input_mod_factor must be 1, 2, or 4; got " << input_mod_factor);
    (void)(input_mod_factor);  // Avoid unused parameter warning
    UTILS_CHECK(output_mod_factor == 1 || output_mod_factor == 4,
                "output_mod_factor must be 1 or 4; got " << output_mod_factor);

    uint64_t twice_mod = modulus << 1;
    size_t t = (n >> 1);

    for (size_t m = 1; m < n; m <<= 1) {
        size_t j1 = 0;
        for (size_t i = 0; i < m; i++) {
            size_t j2 = j1 + t;
            const uint64_t W_op = root_of_unity_powers[m + i];
            const uint64_t W_precon = precon_root_of_unity_powers[m + i];

            uint64_t* X = operand + j1;
            uint64_t* Y = X + t;

            uint64_t tx;
            uint64_t T;

            for (size_t j = j1; j < j2; j++) {
                // The Harvey butterfly: assume X, Y in [0, 4q), and return X',
                // Y' in [0, 4q). Such that X', Y' = X + WY, X - WY (mod q). See
                // Algorithm 4 of https://arxiv.org/pdf/1205.2926.pdf
                UTILS_CHECK(*X < modulus * 4,
                            "input X " << (*X) << " too large");
                UTILS_CHECK(*Y < modulus * 4,
                            "input Y " << (*Y) << " too large");

                tx = (*X >= twice_mod) ? (*X - twice_mod) : *X;
                T = MultiplyUIntModLazy<64>(*Y, W_op, W_precon, modulus);

                *X++ = tx + T;
                *Y++ = tx + twice_mod - T;

                UTILS_CHECK(tx + T < modulus * 4,
                            "ouput X " << (tx + T) << " too large");
                UTILS_CHECK(
                    tx + twice_mod - T < modulus * 4,
                    "output Y " << (tx + twice_mod - T) << " too large");
            }
            j1 += (t << 1);
        }
        t >>= 1;
    }
    if (output_mod_factor == 1) {
        for (size_t i = 0; i < n; ++i) {
            if (operand[i] >= twice_mod) {
                operand[i] -= twice_mod;
            }
            if (operand[i] >= modulus) {
                operand[i] -= modulus;
            }
            UTILS_CHECK(operand[i] < modulus,
                        "Incorrect modulus reduction in NTT "
                            << operand[i] << " >= " << modulus);
        }
    }
}

void ReferenceForwardTransformToBitReverse(
    uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* root_of_unity_powers) {
    UTILS_CHECK(CheckNTTArguments(n, modulus), "");
    UTILS_CHECK(root_of_unity_powers != nullptr,
                "root_of_unity_powers == nullptr");
    UTILS_CHECK(operand != nullptr, "operand == nullptr");

    size_t t = (n >> 1);
    for (size_t m = 1; m < n; m <<= 1) {
        size_t j1 = 0;
        for (size_t i = 0; i < m; i++) {
            size_t j2 = j1 + t;
            const uint64_t W_op = root_of_unity_powers[m + i];

            uint64_t* X = operand + j1;
            uint64_t* Y = X + t;
            for (size_t j = j1; j < j2; j++) {
                uint64_t tx = *X;
                // X', Y' = X + WY, X - WY (mod q).
                uint64_t W_x_Y = MultiplyUIntMod(*Y, W_op, modulus);
                *X++ = AddUIntMod(tx, W_x_Y, modulus);
                *Y++ = SubUIntMod(tx, W_x_Y, modulus);
            }
            j1 += (t << 1);
        }
        t >>= 1;
    }
}

void InverseTransformFromBitReverse64(
    uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor) {
    UTILS_CHECK(CheckNTTArguments(n, modulus), "");
    UTILS_CHECK(inv_root_of_unity_powers != nullptr,
                "inv_root_of_unity_powers == nullptr");
    UTILS_CHECK(precon_inv_root_of_unity_powers != nullptr,
                "precon_inv_root_of_unity_powers == nullptr");
    UTILS_CHECK(operand != nullptr, "operand == nullptr");
    UTILS_CHECK(input_mod_factor == 1 || input_mod_factor == 2,
                "input_mod_factor must be 1 or 2; got " << input_mod_factor);
    (void)(input_mod_factor);  // Avoid unused parameter warning
    UTILS_CHECK(output_mod_factor == 1 || output_mod_factor == 2,
                "output_mod_factor must be 1 or 2; got " << output_mod_factor);

    uint64_t twice_mod = modulus << 1;
    size_t t = 1;
    size_t root_index = 1;

    for (size_t m = (n >> 1); m > 1; m >>= 1) {
        size_t j1 = 0;
        for (size_t i = 0; i < m; i++, root_index++) {
            size_t j2 = j1 + t;
            const uint64_t W_op = inv_root_of_unity_powers[root_index];
            const uint64_t W_op_precon =
                precon_inv_root_of_unity_powers[root_index];

            uint64_t* X = operand + j1;
            uint64_t* Y = X + t;

            uint64_t tx;
            uint64_t ty;

            for (size_t j = j1; j < j2; j++) {
                // The Harvey butterfly: assume X, Y in [0, 2q), and return X',
                // Y' in [0, 2q). X', Y' = X + Y (mod q), W(X - Y) (mod q).
                tx = *X + *Y;
                ty = *X + twice_mod - *Y;

                *X++ = (tx >= twice_mod) ? (tx - twice_mod) : tx;
                *Y++ = MultiplyUIntModLazy<64>(ty, W_op, W_op_precon, modulus);
            }
            j1 += (t << 1);
        }
        t <<= 1;
    }

    const uint64_t W_op = inv_root_of_unity_powers[root_index];
    const uint64_t inv_n = InverseUIntMod(n, modulus);
    const uint64_t inv_n_w = MultiplyUIntMod(inv_n, W_op, modulus);

    uint64_t* X = operand;
    uint64_t* Y = X + (n >> 1);
    uint64_t tx;
    uint64_t ty;

    for (size_t j = (n >> 1); j < n; j++) {
        tx = *X + *Y;
        if (tx >= twice_mod) {
            tx -= twice_mod;
        }
        ty = *X + twice_mod - *Y;
        *X++ = MultiplyUIntModLazy<64>(tx, inv_n, modulus);
        *Y++ = MultiplyUIntModLazy<64>(ty, inv_n_w, modulus);
    }

    if (output_mod_factor == 1) {
        // Reduce from [0, 2q) to [0,q)
        for (size_t i = 0; i < n; ++i) {
            if (operand[i] >= modulus) {
                operand[i] -= modulus;
            }
            UTILS_CHECK(operand[i] < modulus,
                        "Incorrect modulus reduction in InvNTT"
                            << operand[i] << " >= " << modulus);
        }
    }
}

bool CheckNTTArguments(uint64_t degree, uint64_t modulus) {
    // Avoid unused parameter warnings
    (void)degree;
    (void)modulus;
    UTILS_CHECK(IsPowerOfTwo(degree),
                "degree " << degree << " is not a power of 2");
    UTILS_CHECK(degree <= (1 << NTT::NTTImpl::s_max_degree_bits),
                "degree should be less than 2^"
                    << NTT::NTTImpl::s_max_degree_bits << " got " << degree);

    UTILS_CHECK(modulus % (2 * degree) == 1, "modulus mod 2n != 1");
    return true;
}

}  // namespace utils
}  // namespace hetest
