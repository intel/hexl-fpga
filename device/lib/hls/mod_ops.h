// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __MOD_OPS_H__
#define __MOD_OPS_H__

typedef unsigned long ubitwidth_t;

#ifdef __cplusplus
extern "C" {
#endif

ubitwidth_t AddMod(ubitwidth_t a, ubitwidth_t b, ubitwidth_t m);
ubitwidth_t MultMod(ubitwidth_t a, ubitwidth_t b, ubitwidth_t m,
                    ubitwidth_t twice_m, ubitwidth_t length,
                    ubitwidth_t barr_lo);
ubitwidth_t MultiplyUIntModLazy4(ubitwidth_t x, ubitwidth_t y_operand,
                                 ubitwidth_t y_barrett_factor,
                                 ubitwidth_t modulus);
ubitwidth_t MultiplyUIntModLazy3(ubitwidth_t x, ubitwidth_t y,
                                 ubitwidth_t modulus);
ubitwidth_t HLS_MultiplyUInt64Hi(ubitwidth_t x, ubitwidth_t y);
void HLS_MultiplyUInt52(ubitwidth_t x, ubitwidth_t y, ubitwidth_t* prod_hi,
                        ubitwidth_t* prod_lo);
void HLS_MultiplyUInt64(ubitwidth_t x, ubitwidth_t y, ubitwidth_t* prod_hi,
                        ubitwidth_t* prod_lo);
ubitwidth_t HLS_BarrettReduce104(ubitwidth_t input_hi, ubitwidth_t input_lo,
                                 ubitwidth_t modulus, unsigned long r,
                                 unsigned char k);
ubitwidth_t HLS_BarrettReduce120(ubitwidth_t input_hi, ubitwidth_t input_lo,
                                 ubitwidth_t modulus, unsigned long r,
                                 unsigned char k);
#ifdef __cplusplus
}
#endif

#endif
