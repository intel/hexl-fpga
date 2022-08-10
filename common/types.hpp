
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

/// @brief
/// Struct moduli_info_t
/// @param[in] modulus stores the polynomial modulus
/// @param[in] len stores the the modulus size in bits
/// @param[in] barr_lo stores n / modulus where n is the polynomial size
///
typedef struct {
    uint64_t moduli;
    uint64_t len;
    uint64_t barr_lo;
} moduli_info_t;
using uint256_t = ac_int<256, false>;

/// @brief
/// Struct moduli_t
/// @param[in] data stores the KeySwitch modulus data

typedef struct {
    sycl::ulong4 data[8];
} moduli_t;

/// @brief
/// Struct invn_t
/// @param[in] data stores the KeySwitch invn data
///
typedef struct {
    sycl::ulong4 data[8];
} invn_t;
