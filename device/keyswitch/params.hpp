
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef NUM_CORES
#define NUM_CORES 1
#endif

#define VEC_LOG_INTT2 1
#define VEC_INTT2 2
// Acquire the size of the NTT
#ifndef FPGA_NTT_SIZE
#define FPGA_NTT_SIZE_LOG 14
#define FPGA_NTT_SIZE (1 << FPGA_NTT_SIZE_LOG)
#endif

#define MAX_MODULUS_BITS 52
#define MAX_MODULUS (1UL << MAX_MODULUS_BITS)

#define MAX_MODULUS_BITS 52
#define MAX_MODULUS (1UL << MAX_MODULUS_BITS)
#define MAX_KEY (1UL << MAX_MODULUS_BITS)

#define BIT_MASK(BITS) ((1UL << BITS) - 1)
#define BIT_MASK_52 (uint256_t) BIT_MASK(52)
#define BIT_MASK_4 (uint256_t) BIT_MASK(4)
#define BIT_MASK_8 BIT_MASK(8)
#define MODULUS_BIT_MASK BIT_MASK(52)
#define GET_COEFF_COUNT(mod) ((mod >> MAX_MODULUS_BITS) << 10)

#define MAX_KEY_MODULUS_SIZE 7
#define MAX_DECOMP_MODULUS_SIZE 6
#define MAX_RNS_MODULUS_SIZE 7
#define MAX_KEY_COMPONENT_SIZE 2

#define INTT_INS 3
#define NTT_ENGINES (MAX_RNS_MODULUS_SIZE + MAX_KEY_COMPONENT_SIZE)
#define MAX_COFF_COUNT 16384
#define DEFAULT_DEPTH 8

#define STEP(n, max) n = n == (max - 1) ? 0 : n + 1
#define STEP2(n, max) n = n == ((max)-1) ? -1 : n + 1
