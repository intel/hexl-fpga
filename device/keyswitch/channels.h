// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __CHANNEL_H__
#define __CHANNEL_H__
// Enabling Double Precision Floating-Point Operations
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// Enabling Intel pipes
#pragma OPENCL EXTENSION cl_intel_channels : enable

#ifndef VEC
#define VEC_LOG 3
#define VEC (1 << VEC_LOG)
#endif

#ifndef CORES
#define CORES 1
#endif

#define MAX_MODULUS_BITS 52
#define MAX_MODULUS (1UL << MAX_MODULUS_BITS)
#define MAX_KEY (1UL << MAX_MODULUS_BITS)

#define BIT_MASK(BITS) ((1UL << BITS) - 1)
#define BIT_MASK_52 BIT_MASK(52)
#define BIT_MASK_4 BIT_MASK(4)
#define BIT_MASK_8 BIT_MASK(8)
#define MODULUS_BIT_MASK BIT_MASK_52
#define GET_COEFF_COUNT(mod) ((mod >> MAX_MODULUS_BITS) << 10)

#ifdef EMULATOR
#define ASSERT(cond, message, ...)             \
    if (!(cond)) {                             \
        printf("%s#%d: ", __FILE__, __LINE__); \
        printf(message, ##__VA_ARGS__);        \
    }
#define debug(message, ...) printf(message, ##__VA_ARGS__);
#else
#define ASSERT(cond, message, ...)
#define debug(message, ...)
#endif

typedef unsigned long uint64_t;
typedef long int64_t;
#define NULL 0
#define nullptr 0

typedef unsigned int __attribute__((__ap_int(64))) uint54_t;
typedef unsigned int __attribute__((__ap_int(52))) uint52_t;
typedef unsigned int __attribute__((__ap_int(256))) uint256_t;

#ifdef PAC_S10_USM
#pragma message("Using the USM BSP")
#define HOST_MEM __attribute__((buffer_location("host")))
#define DEVICE_MEM __attribute__((buffer_location("device")))
#else
#pragma message("Using the non-USM BSP")
#define HOST_MEM
#define DEVICE_MEM
#endif

#define __single_task                       \
    __attribute__((max_global_work_dim(0))) \
        __attribute__((uses_global_work_offset(0))) __kernel

#define __autorun __attribute__((autorun))
#define INTT_CHANNEL 1
#define NTT_CHANNEL 1
#define MAX_KEY_MODULUS_SIZE 7
#define MAX_DECOMP_MODULUS_SIZE 6
#define MAX_RNS_MODULUS_SIZE 7
#define MAX_KEY_COMPONENT_SIZE 2

#define INTT_INS 3
#define NTT_ENGINES (MAX_RNS_MODULUS_SIZE + MAX_KEY_COMPONENT_SIZE)
#define MAX_COFF_COUNT 16384
#define DEFAULT_DEPTH 1
#define MAX_U32 (unsigned)(-1)

#define STEP(n, max) n = n == ((max)-1) ? 0 : n + 1
#define STEP2(n, max) n = n == (max) ? MAX_U32 : n + 1

typedef struct {
    uint64_t data[VEC * 2];
} ntt_elements;

typedef struct {
    ulong4 data[8];
} moduli_t;

typedef struct {
    ulong4 data[8];
} invn_t;

typedef struct {
    unsigned size;
    __global uint256_t* restrict k_switch_keys1;
    __global uint256_t* restrict k_switch_keys2;
    __global uint256_t* restrict k_switch_keys3;
} keyswitch_params;

typedef struct {
    bool load_twiddle_factors;
    unsigned coeff_count;
    __global uint64_t* restrict data;
} twiddle_factors_t;

channel keyswitch_params ch_keyswitch_params __attribute__((depth(32)));
channel twiddle_factors_t ch_twiddle_factors;
channel ulong4 ch_intt_redu_params[CORES][MAX_RNS_MODULUS_SIZE]
    __attribute__((depth(32)));
channel ulong4 ch_intt2_redu_params[CORES][MAX_KEY_COMPONENT_SIZE]
    __attribute__((depth(32)));
channel ulong4 ch_dyadmult_params[CORES][MAX_RNS_MODULUS_SIZE]
    __attribute__((depth(32)));
channel ulong4 ch_ms_params[CORES][MAX_KEY_COMPONENT_SIZE]
    __attribute__((depth(32)));

channel unsigned ch_ntt2_decomp_size;
channel unsigned ch_intt1_decomp_size;

typedef struct {
    uint64_t data[VEC];
} twiddle_factor;

typedef struct {
    uint64_t data[2];
} twiddle_factor2;

channel twiddle_factor ch_twiddle_factor[CORES][NTT_ENGINES]
    __attribute__((depth(4)));
channel twiddle_factor ch_twiddle_factor_rep[NTT_ENGINES - 1]
    __attribute__((depth(4)));

channel twiddle_factor ch_intt1_twiddle_factor[CORES][1]
    __attribute__((depth(4)));
channel twiddle_factor ch_intt1_twiddle_factor_rep __attribute__((depth(4)));

channel twiddle_factor2 ch_intt2_twiddle_factor[CORES][MAX_KEY_COMPONENT_SIZE]
    __attribute__((depth(4)));
channel twiddle_factor ch_intt2_twiddle_factor_rep __attribute__((depth(4)));

// intt1 and intt2
channel uint64_t ch_intt_elements_in[CORES][INTT_INS]
    __attribute__((depth(DEFAULT_DEPTH)));
channel uint64_t ch_intt_elements_out[CORES][INTT_INS]
    __attribute__((depth(DEFAULT_DEPTH)));
channel uint64_t ch_intt_elements_out_inter[CORES][INTT_INS]
    __attribute__((depth(DEFAULT_DEPTH)));

channel ulong4 ch_normalize[CORES][INTT_INS]
    __attribute__((depth(DEFAULT_DEPTH)));

channel ulong4 ch_intt_modulus[CORES][INTT_INS]
    __attribute__((depth(DEFAULT_DEPTH)));

// dyadmult keys
channel ulong2 ch_dyadmult_keys[CORES][MAX_RNS_MODULUS_SIZE]
    __attribute__((depth(DEFAULT_DEPTH)));

// intt1 reduction
channel uint64_t ch_intt_elements_out_rep[CORES][MAX_RNS_MODULUS_SIZE]
    __attribute__((depth(DEFAULT_DEPTH)));

// ntt engines
channel uint64_t ch_ntt_elements_in[CORES][NTT_ENGINES]
    __attribute__((depth(DEFAULT_DEPTH)));
channel uint64_t ch_ntt_elements_out[CORES][NTT_ENGINES]
    __attribute__((depth(DEFAULT_DEPTH)));
channel ntt_elements ch_ntt_elements[CORES][NTT_ENGINES * 2]
    __attribute__((depth(MAX_COFF_COUNT / VEC / 2)));
channel ulong4 ch_ntt_modulus[CORES][NTT_ENGINES]
    __attribute__((depth(DEFAULT_DEPTH)));
channel unsigned int ch_ntt_key_modulus_idx[CORES][NTT_ENGINES]
    __attribute__((depth(DEFAULT_DEPTH)));

// ms
channel uint64_t
    ch_t_poly_prod_iter[CORES][MAX_RNS_MODULUS_SIZE - 1][MAX_KEY_COMPONENT_SIZE]
    __attribute__((depth(MAX_COFF_COUNT * 2)));
channel uint64_t ch_t_poly_prod_iter_last[CORES][MAX_KEY_COMPONENT_SIZE]
    __attribute__((depth(DEFAULT_DEPTH)));
channel uint64_t ch_result[CORES][2];
#endif
