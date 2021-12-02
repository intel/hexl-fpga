// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

__extension__ typedef __int128 int128_t;
__extension__ typedef unsigned __int128 uint128_t;

namespace hetest {
namespace utils {

#undef TRUE   // MSVC defines TRUE
#undef FALSE  // MSVC defines FALSE

/// @enum CMPINT
/// @brief Represents binary operations between two boolean values
enum class CMPINT {
    EQ = 0,     ///< Equal
    LT = 1,     ///< Less than
    LE = 2,     ///< Less than or equal
    FALSE = 3,  ///< False
    NE = 4,     ///< Not equal
    NLT = 5,    ///< Not less than
    NLE = 6,    ///< Not less than or equal
    TRUE = 7    ///< True
};

/// @brief Returns the logical negation of a binary operation
/// @param[in] cmp The binary operation to negate
inline CMPINT Not(CMPINT cmp) {
    switch (cmp) {
    case CMPINT::EQ:
        return CMPINT::NE;
    case CMPINT::LT:
        return CMPINT::NLT;
    case CMPINT::LE:
        return CMPINT::NLE;
    case CMPINT::FALSE:
        return CMPINT::TRUE;
    case CMPINT::NE:
        return CMPINT::EQ;
    case CMPINT::NLT:
        return CMPINT::LT;
    case CMPINT::NLE:
        return CMPINT::LE;
    case CMPINT::TRUE:
        return CMPINT::FALSE;
    default:
        return CMPINT::FALSE;
    }
}

}  // namespace utils
}  // namespace hetest

#ifdef UTILS_DEBUG

#define UTILS_CHECK(cond, expr)                                         \
    if (!(cond)) {                                                      \
        std::cerr << expr << " in function: " << __FUNCTION__           \
                  << " in file: " __FILE__ << " at line: " << __LINE__; \
        throw std::runtime_error("Error. Check log output");            \
    }

#define UTILS_CHECK_BOUNDS(arg, n, bound, expr)                                \
    for (size_t utils_check_idx = 0; utils_check_idx < n; ++utils_check_idx) { \
        UTILS_CHECK((arg)[utils_check_idx] < bound, expr);                     \
    }

#else  // UTILS_DEBUG=OFF

#define UTILS_CHECK(cond, expr) \
    {}
#define UTILS_CHECK_BOUNDS(...) \
    {}

#endif  // UTILS_DEBUG

namespace hetest {
namespace utils {

// Return x * y as 128-bit integer
// Correctness if x * y < 128 bits
inline uint128_t MultiplyUInt64(uint64_t x, uint64_t y) {
    return uint128_t(x) * uint128_t(y);
}

inline uint64_t BarrettReduce128(uint64_t input_hi, uint64_t input_lo,
                                 uint64_t modulus) {
    UTILS_CHECK(modulus != 0, "modulus == 0")
    uint128_t n = (static_cast<uint128_t>(input_hi) << 64) |
                  (static_cast<uint128_t>(input_lo));

    return static_cast<uint64_t>(n % modulus);
    // TODO(fboemer): actually use barrett reduction if performance-critical
}

// Returns low 64bit of 128b/64b where x1=high 64b, x0=low 64b
inline uint64_t DivideUInt128UInt64Lo(uint64_t x1, uint64_t x0, uint64_t y) {
    uint128_t n =
        (static_cast<uint128_t>(x1) << 64) | (static_cast<uint128_t>(x0));
    uint128_t q = n / y;

    return static_cast<uint64_t>(q);
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

// Returns most-significant bit of the input
inline uint64_t MSB(uint64_t input) {
    return static_cast<uint64_t>(std::log2l(input));
}

}  // namespace utils
}  // namespace hetest

namespace hetest {
namespace utils {

inline bool Compare(CMPINT cmp, uint64_t lhs, uint64_t rhs) {
    switch (cmp) {
    case CMPINT::EQ:
        return lhs == rhs;
    case CMPINT::LT:
        return lhs < rhs;
        break;
    case CMPINT::LE:
        return lhs <= rhs;
        break;
    case CMPINT::FALSE:
        return false;
        break;
    case CMPINT::NE:
        return lhs != rhs;
        break;
    case CMPINT::NLT:
        return lhs >= rhs;
        break;
    case CMPINT::NLE:
        return lhs > rhs;
    case CMPINT::TRUE:
        return true;
    default:
        return true;
    }
}

// Returns whether or not num is a power of two
inline bool IsPowerOfTwo(uint64_t num) { return num && !(num & (num - 1)); }

// Returns log2(x) for x a power of 2
inline uint64_t Log2(uint64_t x) {
    UTILS_CHECK(IsPowerOfTwo(x), x << " not a power of 2");
    return MSB(x);
}

// Returns the maximum value that can be represented using bits bits
inline uint64_t MaximumValue(uint64_t bits) {
    UTILS_CHECK(bits <= 64, "MaximumValue requires bits <= 64; got " << bits);
    if (bits == 64) {
        return (std::numeric_limits<uint64_t>::max)();
    }
    return (1ULL << bits) - 1;
}
}  // namespace utils
}  // namespace hetest

namespace hetest {
namespace utils {

struct AllocatorBase {
    virtual ~AllocatorBase() noexcept {}
    virtual void* allocate(size_t bytes_count) = 0;
    virtual void deallocate(void* p, size_t n) = 0;
};

template <class AllocatorImpl>
struct AllocatorInterface : public AllocatorBase {
    // override interface & delegate implementation to AllocatorImpl
    void* allocate(size_t bytes_count) override {
        return static_cast<AllocatorImpl*>(this)->allocate_impl(bytes_count);
    }
    void deallocate(void* p, size_t n) override {
        static_cast<AllocatorImpl*>(this)->deallocate_impl(p, n);
    }

private:
    // in case of AllocatorImpl doesn't provide implementations use default
    // behavior: break compilation with error
    void* allocate_impl(size_t bytes_count) {
        (void)bytes_count;
        fail_message<0>();
        return nullptr;
    }
    void deallocate_impl(void* p, size_t n) {
        (void)p;
        (void)n;
        fail_message<1>();
    }

    // pretty compilation error printing
    template <int error_code>
    static constexpr int fail_message() {
        static_assert(
            !(error_code == 0),
            "Using 'AllocatorAdapter`as interface requires to implement "
            "'::allocate_impl` method");
        static_assert(
            !(error_code == 1),
            "Using 'AllocatorAdapter`as interface requires to implement "
            "'::deallocate_impl` method");
        return 0;
    }
};
}  // namespace utils
}  // namespace hetest

namespace hetest {
namespace utils {

namespace details {
struct MallocStrategy : AllocatorBase {
    void* allocate(size_t bytes_count) final {
        return std::malloc(bytes_count);
    }

    void deallocate(void* p, size_t n) final {
        (void)n;
        std::free(p);
    }
};

struct CustomAllocStrategy {
    explicit CustomAllocStrategy(std::shared_ptr<AllocatorBase> impl)
        : p_impl(impl) {
        if (!impl) {
            throw std::runtime_error(
                "Cannot create 'CustomAllocStrategy' without `impl`");
        }
    }

    void* allocate_memory(size_t bytes_count) {
        return p_impl->allocate(bytes_count);
    }

    void deallocate_memory(void* p, size_t n) { p_impl->deallocate(p, n); }

private:
    std::shared_ptr<AllocatorBase> p_impl;
};
}  // namespace details

using AllocatorStrategyPtr = std::shared_ptr<AllocatorBase>;
extern AllocatorStrategyPtr mallocStrategy;

template <typename T, uint64_t Alignment>
class AlignedAllocator {
public:
    template <typename, uint64_t>
    friend class AlignedAllocator;

    using value_type = T;

    explicit AlignedAllocator(AllocatorStrategyPtr strategy = nullptr) noexcept
        : m_alloc_impl((strategy != nullptr) ? strategy : mallocStrategy) {}

    AlignedAllocator(const AlignedAllocator& src)
        : m_alloc_impl(src.m_alloc_impl) {}

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>& src)
        : m_alloc_impl(src.m_alloc_impl) {}

    ~AlignedAllocator() {}

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    bool operator==(const AlignedAllocator&) { return true; }

    bool operator!=(const AlignedAllocator&) { return false; }

    T* allocate(size_t n) {
        if (!IsPowerOfTwo(Alignment)) {
            return nullptr;
        }
        // Allocate enough space to ensure the alignment can be satisfied
        size_t buffer_size = sizeof(T) * n + Alignment;
        // Additionally, allocate a prefix to store the memory location of the
        // unaligned buffer
        size_t alloc_size = buffer_size + sizeof(void*);
        void* buffer = m_alloc_impl->allocate(alloc_size);
        if (!buffer) {
            return nullptr;
        }

        // Reserve first location for pointer to originally-allocated space
        void* aligned_buffer = static_cast<char*>(buffer) + sizeof(void*);
        // std::align(Alignment, sizeof(T) * n, aligned_buffer, buffer_size);
        if (!aligned_buffer) {
            return nullptr;
        }

        // Store allocated buffer address at aligned_buffer - sizeof(void*).
        void* store_buffer_addr =
            static_cast<char*>(aligned_buffer) - sizeof(void*);
        *(static_cast<void**>(store_buffer_addr)) = buffer;

        return static_cast<T*>(aligned_buffer);
    }

    void deallocate(T* p, size_t n) {
        if (!p) {
            return;
        }
        void* store_buffer_addr = (reinterpret_cast<char*>(p) - sizeof(void*));
        void* free_address = *(static_cast<void**>(store_buffer_addr));
        m_alloc_impl->deallocate(free_address, n);
    }

private:
    AllocatorStrategyPtr m_alloc_impl;
};

template <typename T>
using AlignedVector64 = std::vector<T, AlignedAllocator<T, 64> >;

}  // namespace utils
}  // namespace hetest
