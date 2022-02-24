// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <iostream>

#include "number_theory_util.h"

namespace intel {
namespace hexl {
namespace fpga {

#define MAX_DEGREE 16384

void ComputeRootOfUnityPowers(uint64_t m_q, uint64_t m_degree,
                              uint64_t m_degree_bits, uint64_t m_w,
                              uint64_t* inv_root_of_unity_powers,
                              uint64_t* precon64_inv_root_of_unity_powers,
                              uint64_t* root_of_unity_powers,
                              uint64_t* precon64_root_of_unity_powers) {
    uint64_t inv_root_of_unity_powers_pre[MAX_DEGREE];

    // 64-bit preconditioning
    root_of_unity_powers[0] = 1;
    inv_root_of_unity_powers_pre[0] = 1;
    uint64_t idx = 0;
    uint64_t prev_idx = idx;

    for (size_t i = 1; i < m_degree; i++) {
        idx = ReverseBitsUInt(i, m_degree_bits);
        root_of_unity_powers[idx] =
            MultiplyUIntMod(root_of_unity_powers[prev_idx], m_w, m_q);
        inv_root_of_unity_powers_pre[idx] =
            InverseUIntMod(root_of_unity_powers[idx], m_q);

        prev_idx = idx;
    }

    precon64_root_of_unity_powers[0] = 0;
    for (size_t i = 1; i < m_degree; i++) {
        precon64_root_of_unity_powers[i] =
            MultiplyFactor(root_of_unity_powers[i], 64, m_q).BarrettFactor();
    }

    idx = 0;

    for (size_t m = (m_degree >> 1); m > 0; m >>= 1) {
        for (size_t i = 0; i < m; i++) {
            inv_root_of_unity_powers[idx] = inv_root_of_unity_powers_pre[m + i];
            idx++;
        }
    }

    inv_root_of_unity_powers[m_degree - 1] = 0;

    for (uint64_t i = 0; i < m_degree; i++) {
        precon64_inv_root_of_unity_powers[i] =
            MultiplyFactor(inv_root_of_unity_powers[i], 64, m_q)
                .BarrettFactor();
    }
}

}  // namespace fpga
}  // namespace hexl
}  // namespace intel
