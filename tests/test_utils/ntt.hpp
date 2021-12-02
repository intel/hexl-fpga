// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "utils-test.hpp"

namespace hetest {
namespace utils {

// Stores an integer on which modular multiplication can be performed more
// efficiently, at the cost of some precomputation.
class MultiplyFactor {
public:
    MultiplyFactor() = default;

    // Computes and stores the Barrett factor (operand << bit_shift) / modulus
    MultiplyFactor(uint64_t operand, uint64_t bit_shift, uint64_t modulus)
        : m_operand(operand) {
        UTILS_CHECK(
            operand <= modulus,
            "operand " << operand << " must be less than modulus " << modulus);
        UTILS_CHECK(bit_shift == 64 || bit_shift == 52,
                    "Unsupported BitShift " << bit_shift);
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

// Computes (x * y) mod modulus, except that the output is in [0, 2 * modulus]
// @param modulus_precon Pre-computed Barrett reduction factor
template <int BitShift>
inline uint64_t MultiplyUIntModLazy(uint64_t x, uint64_t y_operand,
                                    uint64_t y_barrett_factor,
                                    uint64_t modulus) {
    UTILS_CHECK(
        y_operand < modulus,
        "y_operand " << y_operand << " must be less than modulus " << modulus);
    UTILS_CHECK(
        modulus <= MaximumValue(BitShift),
        "Modulus " << modulus << " exceeds bound " << MaximumValue(BitShift));
    UTILS_CHECK(x <= MaximumValue(BitShift),
                "Operand " << x << " exceeds bound " << MaximumValue(BitShift));

    uint64_t Q = MultiplyUInt64Hi<BitShift>(x, y_barrett_factor);
    return y_operand * x - Q * modulus;
}

// Computes (x * y) mod modulus, except that the output is in [0, 2 * modulus]
template <int BitShift>
inline uint64_t MultiplyUIntModLazy(uint64_t x, uint64_t y, uint64_t modulus) {
    UTILS_CHECK(BitShift == 64 || BitShift == 52,
                "Unsupported BitShift " << BitShift);
    UTILS_CHECK(x <= MaximumValue(BitShift),
                "Operand " << x << " exceeds bound " << MaximumValue(BitShift));
    UTILS_CHECK(y < modulus,
                "y " << y << " must be less than modulus " << modulus);
    UTILS_CHECK(
        modulus <= MaximumValue(BitShift),
        "Modulus " << modulus << " exceeds bound " << MaximumValue(BitShift));
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
    UTILS_CHECK(InputModFactor == 1 || InputModFactor == 2 ||
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
        UTILS_CHECK(twice_modulus != nullptr,
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
        UTILS_CHECK(twice_modulus != nullptr,
                    "twice_modulus should not be nullptr");
        UTILS_CHECK(four_times_modulus != nullptr,
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
    UTILS_CHECK(false, "Should be unreachable");
    return x;
}

}  // namespace utils
}  // namespace hetest

namespace hetest {
namespace utils {

/// @brief Performs negacyclic forward and inverse number-theoretic transform
/// (NTT), commonly used in RLWE cryptography.
/// @details The number-theoretic transform (NTT) specializes the discrete
/// Fourier transform (DFT) to the finite field \f$ \mathbb{Z}_q[X] / (X^N + 1)
/// \f$.
class NTT {
public:
    template <class Adaptee, class... Args>
    struct AllocatorAdapter
        : public AllocatorInterface<AllocatorAdapter<Adaptee, Args...>> {
        explicit AllocatorAdapter(Adaptee&& _a, Args&&... args);
        AllocatorAdapter(const Adaptee& _a, Args&... args);

        // interface implementation
        void* allocate_impl(size_t bytes_count);
        void deallocate_impl(void* p, size_t n);

    private:
        Adaptee alloc;
    };

    /// Initializes an empty NTT object
    NTT();

    /// Destructs the NTT object
    ~NTT();

    /// Initializes an NTT object with degree \p degree and modulus \p q.
    /// @param[in] degree also known as N. Size of the NTT transform. Must be a
    /// power of
    /// 2
    /// @param[in] q Prime modulus. Must satisfy \f$ q == 1 \mod 2N \f$
    /// @param[in] alloc_ptr Custom memory allocator used for intermediate
    /// calculations
    /// @brief Performs pre-computation necessary for forward and inverse
    /// transforms
    NTT(uint64_t degree, uint64_t q,
        std::shared_ptr<AllocatorBase> alloc_ptr = {});

    template <class Allocator, class... AllocatorArgs>
    NTT(uint64_t degree, uint64_t q, Allocator&& a, AllocatorArgs&&... args)
        : NTT(degree, q,
              std::static_pointer_cast<AllocatorBase>(
                  std::make_shared<
                      AllocatorAdapter<Allocator, AllocatorArgs...>>(
                      std::move(a), std::forward<AllocatorArgs>(args)...))) {}

    /// @brief Initializes an NTT object with degree \p degree and modulus
    /// \p q.
    /// @param[in] degree also known as N. Size of the NTT transform. Must be a
    /// power of
    /// 2
    /// @param[in] q Prime modulus. Must satisfy \f$ q == 1 \mod 2N \f$
    /// @param[in] root_of_unity 2N'th root of unity in \f$ \mathbb{Z_q} \f$.
    /// @param[in] alloc_ptr Custom memory allocator used for intermediate
    /// calculations
    /// @details  Performs pre-computation necessary for forward and inverse
    /// transforms
    NTT(uint64_t degree, uint64_t q, uint64_t root_of_unity,
        std::shared_ptr<AllocatorBase> alloc_ptr = {});

    template <class Allocator, class... AllocatorArgs>
    NTT(uint64_t degree, uint64_t q, uint64_t root_of_unity, Allocator&& a,
        AllocatorArgs&&... args)
        : NTT(degree, q, root_of_unity,
              std::static_pointer_cast<AllocatorBase>(
                  std::make_shared<
                      AllocatorAdapter<Allocator, AllocatorArgs...>>(
                      std::move(a), std::forward<AllocatorArgs>(args)...))) {}

    /// @brief Compute forward NTT. Results are bit-reversed.
    /// @param[out] result Stores the result
    /// @param[in] operand Data on which to compute the NTT
    /// @param[in] input_mod_factor Assume input \p operand are in [0,
    /// input_mod_factor * q). Must be 1, 2 or 4.
    /// @param[in] output_mod_factor Returns output \p operand in [0,
    /// output_mod_factor * q). Must be 1 or 4.
    void ComputeForward(uint64_t* result, const uint64_t* operand,
                        uint64_t input_mod_factor, uint64_t output_mod_factor);

    /// Compute inverse NTT. Results are bit-reversed.
    /// @param[out] result Stores the result
    /// @param[in] operand Data on which to compute the NTT
    /// @param[in] input_mod_factor Assume input \p operand are in [0,
    /// input_mod_factor * q). Must be 1 or 2.
    /// @param[in] output_mod_factor Returns output \p operand in [0,
    /// output_mod_factor * q). Must be 1 or 2.
    void ComputeInverse(uint64_t* result, const uint64_t* operand,
                        uint64_t input_mod_factor, uint64_t output_mod_factor);

    class NTTImpl;  /// Class implementing the NTT

public:
    std::shared_ptr<NTTImpl> m_impl;
};

}  // namespace utils
}  // namespace hetest

namespace hetest {
namespace utils {

class NTT::NTTImpl {
public:
    NTTImpl(uint64_t degree, uint64_t q, uint64_t root_of_unity,
            std::shared_ptr<AllocatorBase> alloc_ptr = {});
    NTTImpl(uint64_t degree, uint64_t q,
            std::shared_ptr<AllocatorBase> alloc_ptr = {});

    ~NTTImpl();

    uint64_t GetMinimalRootOfUnity() const { return m_w; }

    uint64_t GetDegree() const { return m_degree; }

    uint64_t GetModulus() const { return m_q; }

    AlignedVector64<uint64_t>& GetPrecon64RootOfUnityPowers() {
        return m_precon64_root_of_unity_powers;
    }

    uint64_t* GetPrecon64RootOfUnityPowersPtr() {
        return GetPrecon64RootOfUnityPowers().data();
    }

    AlignedVector64<uint64_t>& GetPrecon52RootOfUnityPowers() {
        return m_precon52_root_of_unity_powers;
    }

    uint64_t* GetPrecon52RootOfUnityPowersPtr() {
        return GetPrecon52RootOfUnityPowers().data();
    }

    uint64_t* GetRootOfUnityPowersPtr() {
        return GetRootOfUnityPowers().data();
    }

    // Returns the vector of pre-computed root of unity powers for the modulus
    // and root of unity.
    AlignedVector64<uint64_t>& GetRootOfUnityPowers() {
        return m_root_of_unity_powers;
    }

    // Returns the root of unity at index i.
    uint64_t GetRootOfUnityPower(size_t i) { return GetRootOfUnityPowers()[i]; }

    // Returns the vector of 64-bit pre-conditioned pre-computed root of unity
    // powers for the modulus and root of unity.
    AlignedVector64<uint64_t>& GetPrecon64InvRootOfUnityPowers() {
        return m_precon64_inv_root_of_unity_powers;
    }

    uint64_t* GetPrecon64InvRootOfUnityPowersPtr() {
        return GetPrecon64InvRootOfUnityPowers().data();
    }

    // Returns the vector of 52-bit pre-conditioned pre-computed root of unity
    // powers for the modulus and root of unity.
    AlignedVector64<uint64_t>& GetPrecon52InvRootOfUnityPowers() {
        return m_precon52_inv_root_of_unity_powers;
    }

    uint64_t* GetPrecon52InvRootOfUnityPowersPtr() {
        return GetPrecon52InvRootOfUnityPowers().data();
    }

    AlignedVector64<uint64_t>& GetInvRootOfUnityPowers() {
        return m_inv_root_of_unity_powers;
    }

    uint64_t* GetInvRootOfUnityPowersPtr() {
        return GetInvRootOfUnityPowers().data();
    }

    uint64_t GetInvRootOfUnityPower(size_t i) {
        return GetInvRootOfUnityPowers()[i];
    }

    void ComputeForward(uint64_t* result, const uint64_t* operand,
                        uint64_t input_mod_factor, uint64_t output_mod_factor);

    void ComputeInverse(uint64_t* result, const uint64_t* operand,
                        uint64_t input_mod_factor, uint64_t output_mod_factor);

    static const size_t s_max_degree_bits{20};  // Maximum power of 2 in degree

    // Maximum number of bits in modulus;
    static const size_t s_max_modulus_bits{62};

    // Default bit shift used in Barrett precomputation
    static const size_t s_default_shift_bits{64};

    // Bit shift used in Barrett precomputation when IFMA acceleration is
    // enabled
    static const size_t s_ifma_shift_bits{52};

    // Maximum number of bits in modulus to use IFMA acceleration for the
    // forward transform
    static const size_t s_max_fwd_ifma_modulus{1ULL << (s_ifma_shift_bits - 2)};

    // Maximum number of bits in modulus to use IFMA acceleration for the
    // inverse transform
    static const size_t s_max_inv_ifma_modulus{1ULL << (s_ifma_shift_bits - 1)};

private:
    void ComputeRootOfUnityPowers();
    uint64_t m_degree;  // N: size of NTT transform, should be power of 2
    uint64_t m_q;       // prime modulus. Must satisfy q == 1 mod 2n

    uint64_t m_degree_bits;  // log_2(m_degree)
    // Bit shift to use in computing Barrett reduction for forward transform

    uint64_t m_winv;  // Inverse of minimal root of unity
    uint64_t m_w;     // A 2N'th root of unity

    std::shared_ptr<AllocatorBase> alloc;

    // vector of floor(W * 2**52 / m_q), with W the root of unity powers
    AlignedVector64<uint64_t> m_precon52_root_of_unity_powers;
    // vector of floor(W * 2**64 / m_q), with W the root of unity powers
    AlignedVector64<uint64_t> m_precon64_root_of_unity_powers;
    // powers of the minimal root of unity
    AlignedVector64<uint64_t> m_root_of_unity_powers;

    // vector of floor(W * 2**52 / m_q), with W the inverse root of unity powers
    AlignedVector64<uint64_t> m_precon52_inv_root_of_unity_powers;
    // vector of floor(W * 2**64 / m_q), with W the inverse root of unity powers
    AlignedVector64<uint64_t> m_precon64_inv_root_of_unity_powers;

    AlignedVector64<uint64_t> m_inv_root_of_unity_powers;
};

void ForwardTransformToBitReverse64(uint64_t* operand, uint64_t n,
                                    uint64_t modulus,
                                    const uint64_t* root_of_unity_powers,
                                    const uint64_t* precon_root_of_unity_powers,
                                    uint64_t input_mod_factor = 1,
                                    uint64_t output_mod_factor = 1);

/// @brief Reference NTT which is written for clarity rather than performance
/// @param[in, out] operand Input data. Overwritten with NTT output
/// @param[in] n Size of the transfrom, i.e. the polynomial degree. Must be a
/// power of two.
/// @param[in] modulus Prime modulus. Must satisfy q == 1 mod 2n
/// @param[in] root_of_unity_powers Powers of 2n'th root of unity in F_q. In
/// bit-reversed order
void ReferenceForwardTransformToBitReverse(
    uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* root_of_unity_powers);

void InverseTransformFromBitReverse64(
    uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers,
    uint64_t input_mod_factor = 1, uint64_t output_mod_factor = 1);

// Returns true if arguments satisfy constraints for negacyclic NTT
bool CheckNTTArguments(uint64_t degree, uint64_t modulus);

}  // namespace utils
}  // namespace hetest
