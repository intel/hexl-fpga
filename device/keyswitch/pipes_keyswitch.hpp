
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// info All pipe declarations for kernel connectivity

// info: Pipe connects load kernel to twiddle central/global
// info: twiddle table generator.

#include <CL/sycl.hpp>
#include <CL/__spirv/spirv_types.hpp>
#include "../../common/types.hpp"
#include "../utils/pipe_array.hpp"
#include "params.hpp"
#include "../device_types.hpp"
#include "../utils/pipe_def_marcos.hpp"
defLongPipe(ch_keyswitch_params, unsigned, 32);
defPipe2d(ch_intt_redu_params, sycl::ulong4, 32, NUM_CORES,
          MAX_RNS_MODULUS_SIZE);
defPipe2d(ch_intt2_redu_params, sycl::ulong4, 32, NUM_CORES,
          MAX_KEY_COMPONENT_SIZE);
defPipe2d(ch_dyadmult_params, sycl::ulong4, 32, NUM_CORES,
          MAX_RNS_MODULUS_SIZE);
defPipe2d(ch_ms_params, sycl::ulong4, 32, NUM_CORES, MAX_KEY_COMPONENT_SIZE);
defPipe(ch_ntt2_decomp_size, unsigned);
defPipe(ch_intt1_decomp_size, unsigned);
defPipe2d(ch_twiddle_factor, TwiddleFactor_t, 4, NUM_CORES, NTT_ENGINES);
defPipe1d(ch_twiddle_factor_rep, TwiddleFactor_t, 4, NTT_ENGINES - 1);
defPipe2d(ch_intt1_twiddle_factor, TwiddleFactor_t, 4, NUM_CORES, 1);
defLongPipe(ch_intt1_twiddle_factor_rep, TwiddleFactor_t, 4);
using WVec_t = TwiddleFactor2_t;
defPipe2d(ch_intt2_twiddle_factor, WVec_t, 4, NUM_CORES,
          MAX_KEY_COMPONENT_SIZE);
defLongPipe(ch_intt2_twiddle_factor_rep, TwiddleFactor_t, 4);

// intt1 and intt2
defPipe2d(ch_intt_elements_in, uint64_t, DEFAULT_DEPTH, NUM_CORES, INTT_INS);
defPipe2d(ch_intt_elements_out, uint64_t, DEFAULT_DEPTH, NUM_CORES, INTT_INS);
defPipe2d(ch_intt_elements_out_inter, uint64_t, DEFAULT_DEPTH, NUM_CORES,
          INTT_INS);
defPipe2d(ch_normalize, sycl::ulong4, DEFAULT_DEPTH, NUM_CORES, INTT_INS);
defPipe2d(ch_intt_modulus, sycl::ulong4, DEFAULT_DEPTH, NUM_CORES, INTT_INS);

// dyadmult keys
defPipe2d(ch_dyadmult_keys, sycl::ulong2, DEFAULT_DEPTH, NUM_CORES,
          MAX_RNS_MODULUS_SIZE);

// intt1 reduction
defPipe2d(ch_intt_elements_out_rep, uint64_t, DEFAULT_DEPTH, NUM_CORES,
          MAX_RNS_MODULUS_SIZE);

// ntt engines
defPipe2d(ch_ntt_elements_in, uint64_t, DEFAULT_DEPTH, NUM_CORES, NTT_ENGINES);
defPipe2d(ch_ntt_elements_out, uint64_t, DEFAULT_DEPTH, NUM_CORES, NTT_ENGINES);
defPipe2d(ch_ntt_elements, WideVec_t, (MAX_COFF_COUNT / VEC / 2), NUM_CORES,
          NTT_ENGINES * 2);
defPipe2d(ch_ntt_modulus, sycl::ulong4, DEFAULT_DEPTH, NUM_CORES, NTT_ENGINES);
defPipe2d(ch_ntt_key_modulus_idx, unsigned int, DEFAULT_DEPTH, NUM_CORES,
          NTT_ENGINES);

// ms
defPipe3d(ch_t_poly_prod_iter, uint64_t, (MAX_COFF_COUNT * 2), NUM_CORES,
          MAX_RNS_MODULUS_SIZE - 1, MAX_KEY_COMPONENT_SIZE);
defPipe2d(ch_t_poly_prod_iter_last, uint64_t, DEFAULT_DEPTH, NUM_CORES,
          MAX_KEY_COMPONENT_SIZE);
defPipe2d(ch_result, uint64_t, 4, NUM_CORES, 2);
defPipe2d(ch_intt2_elements, WideVecINTT2_t, (MAX_COFF_COUNT / VEC / 2),
          NUM_CORES, (2 * 2));
defPipe2d(ch_intt_elements, WideVec_t, (MAX_COFF_COUNT / VEC / 2), NUM_CORES,
          2);
