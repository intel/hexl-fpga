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

// Acquire the size of the NTT
#ifndef FPGA_NTT_SIZE
#define FPGA_NTT_SIZE_LOG 14
#define FPGA_NTT_SIZE (1 << FPGA_NTT_SIZE_LOG)
#endif

#define MAX_MODULUS_BITS 52
#define MAX_MODULUS (1UL << MAX_MODULUS_BITS)
#define MAX_KEY (1UL << MAX_MODULUS_BITS)

#ifdef EMULATOR
#define ASSERT(cond, message, ...)             \
    if (!(cond)) {                             \
        printf("%s#%d: ", __FILE__, __LINE__); \
        printf(message, ##__VA_ARGS__);        \
    }
#else
#define ASSERT(cond, message, ...)
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
#define DEFAULT_DEPTH 4

#define STEP(n, max) n = n == (max - 1) ? 0 : n + 1

typedef struct {
    uint64_t data[VEC * 2];
} ntt_elements;

typedef struct {
    uint64_t data[VEC * 2];
} intt_elements;

typedef struct {
    ulong4 data[8];
} moduli_t;

typedef struct {
    ulong4 data[8];
} invn_t;

// intt1 and intt2
channel uint64_t ch_intt_elements_in[INTT_INS]
    __attribute__((depth(DEFAULT_DEPTH)));
channel uint64_t ch_intt_elements_out[INTT_INS]
    __attribute__((depth(DEFAULT_DEPTH)));
channel uint64_t ch_intt_elements_out_inter[INTT_INS]
    __attribute__((depth(DEFAULT_DEPTH)));
channel intt_elements ch_intt_elements[INTT_INS * 2]
    __attribute__((depth(MAX_COFF_COUNT / VEC / 2)));

channel ulong4 ch_normalize[INTT_INS] __attribute__((depth(DEFAULT_DEPTH)));

channel ulong4 ch_intt_modulus[INTT_INS] __attribute__((depth(DEFAULT_DEPTH)));

// dyadmult keys
channel ulong2 ch_dyadmult_keys[MAX_RNS_MODULUS_SIZE]
    __attribute__((depth(256)));

// intt1 reduction
channel uint64_t ch_intt_elements_out_rep[MAX_RNS_MODULUS_SIZE]
    __attribute__((depth(DEFAULT_DEPTH)));

// ntt engines
channel uint64_t ch_ntt_elements_in[NTT_ENGINES]
    __attribute__((depth(DEFAULT_DEPTH)));
channel uint64_t ch_ntt_elements_out[NTT_ENGINES]
    __attribute__((depth(DEFAULT_DEPTH)));
channel ntt_elements ch_ntt_elements[NTT_ENGINES * 2]
    __attribute__((depth(MAX_COFF_COUNT / VEC / 2)));
channel ulong4 ch_ntt_modulus[NTT_ENGINES]
    __attribute__((depth(DEFAULT_DEPTH)));
channel unsigned int ch_ntt_key_modulus_idx[NTT_ENGINES]
    __attribute__((depth(DEFAULT_DEPTH)));

// ws
channel uint64_t
    ch_t_poly_prod_iter[MAX_RNS_MODULUS_SIZE][MAX_KEY_COMPONENT_SIZE]
    __attribute__((depth(MAX_COFF_COUNT * 2)));

channel uint64_t ch_result[2] __attribute__((depth(DEFAULT_DEPTH)));
#endif
