
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __MOD_OPS_HPP__
#define __MOD_OPS_HPP__

#include <CL/sycl.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include "utils/kernel_assert.hpp"
typedef unsigned long ubitwidth_t;

constexpr unsigned int BITWIDTH = 64;
constexpr unsigned int BITWIDTHp1 = 1 + BITWIDTH;
constexpr unsigned int BITWIDTH2 = (2 * BITWIDTH);

using ubitwidthp1_t = ac_int<BITWIDTHp1, false>;
using ubitwidth2_t = ac_int<BITWIDTH2, false>;
using ubitwidth1_t = ac_int<BITWIDTH, false>;

ubitwidth_t AddMod(ubitwidth_t a, ubitwidth_t b, ubitwidth_t m) {
    ubitwidthp1_t sum = a + b;
    ubitwidthp1_t mm = m;

    if (sum >= mm) {
        sum -= mm;
    }
    return sum.to_uint64();
}

ubitwidth_t MultMod(ubitwidth_t a, ubitwidth_t b, ubitwidth_t m,
                    ubitwidth_t twice_m, ubitwidth_t length,
                    ubitwidth_t barr_lo) {
    ubitwidth_t x = a;
    if (x >= twice_m) {
        x = x - twice_m;
    }
    if (x >= m) {
        x = x - m;
    }

    ubitwidth_t y = b;
    if (y >= twice_m) {
        y = y - twice_m;
    }
    if (y >= m) {
        y = y - m;
    }

    ubitwidth_t a_lo = (ubitwidth_t)(uint32_t)x;
    ubitwidth_t a_hi = x >> 32;
    ubitwidth_t b_lo = (ubitwidth_t)(uint32_t)y;
    ubitwidth_t b_hi = y >> 32;

    ubitwidth_t p0 = a_lo * b_lo;
    ubitwidth_t p1 = a_lo * b_hi;
    ubitwidth_t p2 = a_hi * b_lo;
    ubitwidth_t p3 = a_hi * b_hi;

    uint32_t cy = (uint32_t)(((p0 >> 32) + (uint32_t)p1 + (uint32_t)p2) >> 32);
    ubitwidth_t lo = p0 + (p1 << 32) + (p2 << 32);
    ubitwidth_t hi = p3 + (p1 >> 32) + (p2 >> 32) + cy;

    ubitwidth_t c1 = ((lo >> length) + (hi << (64 - length)));

    ubitwidth_t c1_lo = (ubitwidth_t)(uint32_t)c1;
    ubitwidth_t c1_hi = c1 >> 32;
    ubitwidth_t barr_lo_lo = (ubitwidth_t)(uint32_t)barr_lo;
    ubitwidth_t barr_lo_hi = barr_lo >> 32;

    ubitwidth_t prod0 = c1_lo * barr_lo_lo;
    ubitwidth_t prod1 = c1_lo * barr_lo_hi;
    ubitwidth_t prod2 = c1_hi * barr_lo_lo;
    ubitwidth_t prod3 = c1_hi * barr_lo_hi;

    uint32_t prod_cy =
        (uint32_t)(((prod0 >> 32) + (uint32_t)prod1 + (uint32_t)prod2) >> 32);
    ubitwidth_t prod_hi = prod3 + (prod1 >> 32) + (prod2 >> 32) + prod_cy;
    ubitwidth_t c3 = prod_hi;

    ubitwidth_t c4 = lo - c3 * m;

    return (c4 < m) ? c4 : (c4 - m);
}

ubitwidth_t HLS_MultiplyUInt64Hi(ubitwidth_t x, ubitwidth_t y) {
    ubitwidth2_t prod = ac_int<BITWIDTH, false>(x) * ac_int<BITWIDTH, false>(y);
    ubitwidth_t ret = (prod >> BITWIDTH).to_uint64();
    return ret;
}

template <int MAX_MODULUS_BITS_>
void HLS_MultiplyUInt(ubitwidth_t x, ubitwidth_t y, ubitwidth_t* prod_hi,
                      ubitwidth_t* prod_lo) {
    ac_int<MAX_MODULUS_BITS_ * 2, false> prod =
        ac_int<MAX_MODULUS_BITS_, false>(x) *
        ac_int<MAX_MODULUS_BITS_, false>(y);
    *prod_hi = (ubitwidth_t)((prod >> BITWIDTH).to_uint64());
    *prod_lo = (ubitwidth_t)(prod.to_uint64());
}

void HLS_MultiplyUInt52(ubitwidth_t x, ubitwidth_t y, ubitwidth_t* prod_hi,
                        ubitwidth_t* prod_lo) {
    return HLS_MultiplyUInt<52>(x, y, prod_hi, prod_lo);
}

template <int MIN_K, int MAX_K, int MAX_R_K>
ubitwidth_t HLS_BarrettReduce(ubitwidth_t input_hi, ubitwidth_t input_lo,
                              ubitwidth_t modulus, unsigned long rk) {
    ac_int<MAX_K + MAX_K, false> a =
        ((ubitwidth2_t(input_hi) << BITWIDTH) | ((input_lo)));
    ac_int<MAX_K, false> n = modulus;

    ac_int<MAX_R_K, false> r = rk >> 8;
    unsigned char k = rk & 0xff;
    ac_int<7, false> k2 = 2 * k - 2 * MIN_K;

    ac_int<MAX_K + MAX_K + MAX_R_K, false> d = a * r;
    ac_int<MAX_K + MAX_K + MAX_R_K - 2 * MIN_K, false> b =
        d >> (2 * MIN_K) >> k2;
    ac_int<MAX_K + 1, false> c = a - b * n;
    if (c >= modulus) c -= modulus;
    return c.to_uint64();
}
// sabbar
ubitwidth_t HLS_BarrettReduce104(ubitwidth_t input_hi, ubitwidth_t input_lo,
                                 ubitwidth_t modulus, unsigned long rk) {
    return HLS_BarrettReduce<16, 52, 53>(input_hi, input_lo, modulus, rk);
}
ubitwidth_t HLS_BarrettReduce128(ubitwidth_t input_hi, ubitwidth_t input_lo,
                                 ubitwidth_t modulus, unsigned long rk) {
    return HLS_BarrettReduce<16, 64, 64>(input_hi, input_lo, modulus, rk);
}

ubitwidth_t MultiplyUIntModLazy3(ubitwidth_t x, ubitwidth_t y,
                                 ubitwidth_t modulus) {
    ubitwidth_t y_hi{0};
    ubitwidth_t y_lo{0};
    y_hi = y;
    y_lo = 0;

    ubitwidth_t length = ubitwidth_t(64U);
    ubitwidth2_t n = ((ubitwidth2_t(y_hi)) << length) | ubitwidth2_t(y_lo);
    ubitwidth2_t q = n / modulus;
    ubitwidth_t y_barrett_factor = ubitwidth_t((q).to_uint64());
    ubitwidth2_t product = ubitwidth2_t(x) * y_barrett_factor;
    ubitwidth_t Q = ubitwidth_t((product >> length).to_uint64());
    ubitwidth_t ret = y * x - Q * modulus;

    return ret;
}

ubitwidth_t MultiplyUIntModLazy4(ubitwidth_t x, ubitwidth_t y_operand,
                                 ubitwidth_t y_barrett_factor,
                                 ubitwidth_t modulus) {
    ubitwidth_t length = ubitwidth_t(64U);
    ubitwidth2_t prod = ubitwidth2_t(x) * ubitwidth2_t(y_operand);
    ubitwidth_t Q = (prod >> length).to_uint64();
    ubitwidth_t ret = ubitwidth_t(y_barrett_factor * x - Q * modulus);

    return ret;
}

unsigned get_ntt_log(unsigned size) {
    unsigned log = 14;
    if (size == 16384) {
        log = 14;
    } else if (size == 8192) {
        log = 13;
    } else if (size == 4096) {
        log = 12;
    } else if (size == 2048) {
        log = 11;
    } else if (size == 1024) {
        log = 10;
    } else {
        ASSERT(false, "ntt/intt size (%u) is not supported\n", size);
    }
    return log;
}

void MultiplyUInt64(uint64_t x, uint64_t y, uint64_t* prod_hi,
                    uint64_t* prod_lo) {
    ASSERT(x < MAX_MODULUS, "x >= modulus\n");
    ASSERT(y < MAX_MODULUS, "y >= modulus\n");
    HLS_MultiplyUInt52(x, y, prod_hi, prod_lo);
}

uint64_t BarrettReduce128(uint64_t prod_hi, uint64_t prod_lo, uint64_t modulus,
                          uint64_t rk) {
    uint64_t ret = HLS_BarrettReduce104(prod_hi, prod_lo, modulus, rk);
    ASSERT(ret < modulus, "BarrettReduce Failed\n");
    return ret;
}

uint64_t MultiplyUIntMod(uint64_t x, uint64_t y, uint64_t modulus,
                         uint64_t rk) {
    ASSERT(x < MAX_MODULUS, "x >= modulus\n");
    ASSERT(y < MAX_MODULUS, "y >= modulus\n");
    uint64_t prod_hi, prod_lo;
    MultiplyUInt64(x, y, &prod_hi, &prod_lo);

    return BarrettReduce128(prod_hi, prod_lo, modulus, rk);
}

uint64_t SubUIntMod(uint64_t x, uint64_t y, uint64_t modulus) {
    ASSERT(x < modulus, "x >= modulus\n");
    ASSERT(y < modulus, "y >= modulus\n");
    uint64_t diff = (x + modulus) - y;
    return (diff >= modulus) ? (diff - modulus) : diff;
}

uint64_t BarrettReduce64(uint64_t input, uint64_t modulus, uint64_t q_barr) {
    uint64_t q = HLS_MultiplyUInt64Hi(input, q_barr);
    uint64_t q_times_input = input - q * modulus;
    return q_times_input >= modulus ? q_times_input - modulus : q_times_input;
}

uint64_t AddUIntMod(uint64_t x, uint64_t y, uint64_t modulus) {
    ASSERT(x < modulus, "x >= modulus\n");
    ASSERT(y < modulus, "y >= modulus\n");
    uint64_t sum = x + y;
    return (sum >= modulus) ? (sum - modulus) : sum;
}

uint64_t ReduceMod(int InputModFactor, uint64_t x, uint64_t modulus,
                   const uint64_t* twice_modulus,
                   const uint64_t* four_times_modulus) {
    ASSERT(InputModFactor == 1 || InputModFactor == 2 || InputModFactor == 4 ||
               InputModFactor == 8,
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
        ASSERT(twice_modulus != nullptr, "twice_modulus should not be nullptr");
        if (x >= *twice_modulus) {
            x -= *twice_modulus;
        }
        if (x >= modulus) {
            x -= modulus;
        }
        return x;
    }
    if (InputModFactor == 8) {
        ASSERT(twice_modulus != nullptr, "twice_modulus should not be nullptr");
        ASSERT(four_times_modulus != nullptr,
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
    ASSERT(false, "Should be unreachable");
    return x;
}

#endif  // end of __MOD_OPS_HPP__
