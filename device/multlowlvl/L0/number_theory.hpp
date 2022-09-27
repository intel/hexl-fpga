#ifndef __COMMON_H__
#define __COMMON_H__
#include <CL/sycl.hpp>
#if __INTEL_LLVM_COMPILER >= 20220100
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#else
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>
#endif
#include "utils.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

#define MOD_ONCE(a, p) (a) > p ? (a)-p : (a)
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN_MODULUS_BITS 1
#define MAX_MODULUS_BITS 60

typedef unsigned long ubitwidth_t;

constexpr unsigned int BITWIDTH = 64;
constexpr unsigned int BITWIDTHp1 = 1 + BITWIDTH;
constexpr unsigned int BITWIDTH2 = (2 * BITWIDTH);

using ubitwidthp1_t = ac_int<BITWIDTHp1, false>;
using ubitwidth2_t = ac_int<BITWIDTH2, false>;
using ubitwidth1_t = ac_int<BITWIDTH, false>;

template <int MAX_MODULUS_BITS_>
static void MultiplyUInt(ubitwidth_t x, ubitwidth_t y, ubitwidth_t *prod_hi,
                         ubitwidth_t *prod_lo) {
  ac_int<MAX_MODULUS_BITS_ * 2, false> prod =
      ac_int<MAX_MODULUS_BITS_, false>(x) * ac_int<MAX_MODULUS_BITS_, false>(y);
  *prod_hi = (ubitwidth_t)((prod >> BITWIDTH).to_uint64());
  *prod_lo = (ubitwidth_t)(prod.to_uint64());
}

template <int MIN_K, int MAX_K, int MAX_R_K>
static ubitwidth_t BarrettReduce(ubitwidth_t input_hi, ubitwidth_t input_lo,
                                 ubitwidth_t modulus, unsigned long _r,
                                 unsigned long _k) {
  ac_int<MAX_K + MAX_K, false> a =
      ((ubitwidth2_t(input_hi) << BITWIDTH) | ((input_lo)));
  ac_int<MAX_K, false> n = modulus;

  ac_int<MAX_R_K, false> r = _r;
  unsigned char k = _k;
  ac_int<7, false> k2 = 2 * k - 2 * MIN_K;

  ac_int<MAX_K + MAX_K + MAX_R_K, false> d = a * r;
  ac_int<MAX_K + MAX_K + MAX_R_K - 2 *MIN_K> b = d >> (2 * MIN_K) >> k2;
  ac_int<MAX_K + 1, false> c = a - b * n;
  if (c >= modulus) c -= modulus;
  return c.to_uint64();
}

static uint64_t mulmod_naive(uint64_t a, uint64_t b, uint64_t p) {
  ac_int<128, false> prod = ac_int<64, false>(a) * ac_int<64, false>(b);
  return (prod - prod / p * p).to_uint64();
}

static uint64_t mulmod(uint64_t x, ulong2 y, uint64_t p) {
  uint64_t z = x * y[0];
  ac_int<128, false> xy = ac_int<64, false>(x) * ac_int<64, false>(y[1]);
  xy = xy >> 64;
  uint64_t t = xy.to_uint64();
  uint64_t ze = t * p;
  z = z - ze;
  if (z >= p) z -= p;
#if 0
  // ASSERT(x < p, "%lu > %lu\n", x, p);
  // ASSERT(y[0] < p, "%lu > %lu\n", y[0], p);
  if (z != mulmod_naive(x, y[0], p)) {
    PRINTF("mulmod Failed: %lu,%lu,%lu,%lu,%lu\n", x, y[0], p, z,
           mulmod_naive(x, y[0], p));
  } else {
    // PRINTF("mulmod Succes: %lu,%lu,%lu,%lu,%lu\n", x, y[0], p, z,
    //       mulmod_naive(x, y[0], p));
  }
#endif
  return z;
}

static uint64_t MultiplyUIntMod(uint64_t a, uint64_t b, uint64_t p, uint64_t r,
                                uint64_t k) {
  ubitwidth_t prod_hi, prod_lo;
  MultiplyUInt<MAX_MODULUS_BITS>(a, b, &prod_hi, &prod_lo);
  uint64_t ret = BarrettReduce<0, MAX_MODULUS_BITS, MAX_MODULUS_BITS + 1>(
      prod_hi, prod_lo, p, r, k);
#ifdef FPGA_EMULATOR
  if (ret != mulmod_naive(a, b, p)) {
    PRINTF("Failed to MultiplyUIntMod %llu * %llu mod %llu != %llu\n", a, b, p,
           ret);
  }
#endif
  return ret;
}

static ubitwidth_t MultiplyUInt64Hi(ubitwidth_t x, ubitwidth_t y) {
  ubitwidth2_t prod = ac_int<BITWIDTH, false>(x) * ac_int<BITWIDTH, false>(y);
  ubitwidth_t ret = (prod >> BITWIDTH).to_uint64();
  return ret;
}

static uint64_t BarrettReduce64(uint64_t input, uint64_t modulus,
                                uint64_t q_barr) {
  uint64_t q = MultiplyUInt64Hi(input, q_barr);
  uint64_t q_times_input = input - q * modulus;
  return q_times_input >= modulus ? q_times_input - modulus : q_times_input;
}

static uint64_t AddUIntMod(uint64_t x, uint64_t y, uint64_t modulus) {
  ASSERT(x < modulus, "%llu >= %llu\n", x, modulus);
  ASSERT(y < modulus, "%llu >= %llu\n", y, modulus);
  uint64_t sum = x + y;
  return (sum >= modulus) ? (sum - modulus) : sum;
}

static uint64_t SubUIntMod(uint64_t x, uint64_t y, uint64_t modulus) {
  ASSERT(x < modulus, "%llu >= %llu\n", x, modulus);
  ASSERT(y < modulus, "%llu >= %llu\n", y, modulus);
  uint64_t diff = (x + modulus) - y;
  return (diff >= modulus) ? (diff - modulus) : diff;
}
#endif
