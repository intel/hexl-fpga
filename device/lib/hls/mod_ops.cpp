// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mod_ops.h"

#include "HLS/hls.h"

#ifdef __INTELFPGA_COMPILER__
#include "HLS/ac_int.h"
#endif

#define ENABLE_BARRETT_REDUCTION

constexpr unsigned int BITWIDTH = 64;
constexpr unsigned int BITWIDTHp1 = 1 + BITWIDTH;
constexpr unsigned int BITWIDTH2 = (2 * BITWIDTH);

using ubitwidthp1_t = ac_int<BITWIDTHp1, false>;
using ubitwidth2_t = ac_int<BITWIDTH2, false>;
using uint32_t = unsigned int;

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

ubitwidth_t HLS_MultiplyUInt64Hi(ubitwidth_t x, ubitwidth_t y) {
    ubitwidth2_t prod = ac_int<BITWIDTH, false>(x) * ac_int<BITWIDTH, false>(y);
    ubitwidth_t ret = (prod >> BITWIDTH).to_uint64();
    return ret;
}

template <int MAX_MODULUS_BITS>
void HLS_MultiplyUInt(ubitwidth_t x, ubitwidth_t y, ubitwidth_t* prod_hi,
                      ubitwidth_t* prod_lo) {
    ac_int<MAX_MODULUS_BITS * 2, false> prod =
        ac_int<MAX_MODULUS_BITS, false>(x) * ac_int<MAX_MODULUS_BITS, false>(y);
    *prod_hi = (ubitwidth_t)((prod >> BITWIDTH).to_uint64());
    *prod_lo = (ubitwidth_t)(prod.to_uint64());
}

void HLS_MultiplyUInt52(ubitwidth_t x, ubitwidth_t y, ubitwidth_t* prod_hi,
                        ubitwidth_t* prod_lo) {
    return HLS_MultiplyUInt<52>(x, y, prod_hi, prod_lo);
}

void HLS_MultiplyUInt64(ubitwidth_t x, ubitwidth_t y, ubitwidth_t* prod_hi,
                        ubitwidth_t* prod_lo) {
    return HLS_MultiplyUInt<64>(x, y, prod_hi, prod_lo);
}

template <int MIN_K, int MAX_K, int MAX_R_K>
ubitwidth_t HLS_BarrettReduce(ubitwidth_t input_hi, ubitwidth_t input_lo,
                              ubitwidth_t modulus, unsigned long _r,
                              unsigned char k) {
#ifdef ENABLE_BARRETT_REDUCTION
    ac_int<MAX_K + MAX_K, false> a =
        ((ubitwidth2_t(input_hi) << BITWIDTH) | ((input_lo)));
    ac_int<MAX_K, false> n = modulus;

    ac_int<MAX_R_K, false> r = _r;
    ac_int<7, false> k2 = 2 * k - 2 * MIN_K;

    ac_int<MAX_K + MAX_K + MAX_R_K, false> d = a * r;
    ac_int<MAX_K + MAX_K + MAX_R_K - 2 * MIN_K, false> b =
        d >> (2 * MIN_K) >> k2;
    ac_int<MAX_K + 1, false> c = a - b * n;
    if (c >= modulus) c -= modulus;
    return c.to_uint64();
#else
    ac_int<128, false> a =
        ((ubitwidth2_t(input_hi) << BITWIDTH) | ((input_lo)));
    ac_int<64, false> n = modulus;
    return (a - a / n * n).to_uint64();
#endif
}
ubitwidth_t HLS_BarrettReduce104(ubitwidth_t input_hi, ubitwidth_t input_lo,
                                 ubitwidth_t modulus, unsigned long r,
                                 unsigned char k) {
    return HLS_BarrettReduce<16, 52, 53>(input_hi, input_lo, modulus, r, k);
}

ubitwidth_t HLS_BarrettReduce120(ubitwidth_t input_hi, ubitwidth_t input_lo,
                                 ubitwidth_t modulus, unsigned long r,
                                 unsigned char k) {
    return HLS_BarrettReduce<16, 60, 61>(input_hi, input_lo, modulus, r, k);
}
