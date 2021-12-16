#ifndef __COMMON_H__
#define __COMMON_H__
#include "mod_ops.h"

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
#endif
