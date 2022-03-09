// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "channels.h"

__single_task __autorun void dispatch_twiddle_factors() {
    uint64_t twiddle_factors0[MAX_COFF_COUNT / VEC][VEC];
    uint64_t twiddle_factors1[MAX_COFF_COUNT / VEC][VEC];
    uint64_t twiddle_factors2[MAX_COFF_COUNT / VEC][VEC];
    uint64_t twiddle_factors3[MAX_COFF_COUNT / VEC][VEC];
    uint64_t twiddle_factors4[MAX_COFF_COUNT / VEC][VEC];
    uint64_t twiddle_factors5[MAX_COFF_COUNT / VEC][VEC];
    uint64_t twiddle_factors6[MAX_COFF_COUNT / VEC][VEC];

    uint64_t intt1_twiddle_factors[(MAX_RNS_MODULUS_SIZE - 1) * MAX_COFF_COUNT /
                                   VEC][VEC];
    uint64_t intt2_twiddle_factors[MAX_COFF_COUNT / VEC][VEC];

    twiddle_factors_t twiddle_factors = read_channel_intel(ch_twiddle_factors);

    unsigned coeff_count = twiddle_factors.coeff_count;
    for (int k = 0; k < MAX_KEY_MODULUS_SIZE; k++) {
        for (int i = 0; i < coeff_count / VEC; i++) {
#pragma unroll
            for (int j = 0; j < VEC; j++) {
                uint64_t data =
                    twiddle_factors
                        .data[coeff_count * (k * 4 + 2) + i * VEC + j];
                switch (k) {
                case 6:
                    twiddle_factors6[i][j] = data;
                    break;
                case 5:
                    twiddle_factors5[i][j] = data;
                    break;
                case 4:
                    twiddle_factors4[i][j] = data;
                    break;
                case 3:
                    twiddle_factors3[i][j] = data;
                    break;
                case 2:
                    twiddle_factors2[i][j] = data;
                    break;
                case 1:
                    twiddle_factors1[i][j] = data;
                    break;
                default:
                    twiddle_factors0[i][j] = data;
                    break;
                }
            }

// load intt twiddle factors
#pragma unroll
            for (int j = 0; j < VEC; j++) {
                uint64_t data =
                    twiddle_factors.data[coeff_count * k * 4 + i * VEC + j];
                if (k < (MAX_RNS_MODULUS_SIZE - 1)) {
                    intt1_twiddle_factors[k * (coeff_count / VEC) + i][j] =
                        data;
                } else {
                    intt2_twiddle_factors[i][j] = data;
                }
            }
        }
    }

    unsigned ntt1_index[MAX_KEY_MODULUS_SIZE] = {0, 0, 0, 0, 0, 0, 0};
    unsigned ntt2_index = MAX_U32;
    unsigned ntt2_decomp_size;
    unsigned intt1_decomp_size;

    unsigned intt1_index = MAX_U32;
    unsigned intt2_index = 0;
    bool success;

    while (true) {
        if (ntt2_index == MAX_U32) {
            bool valid_read;
            ntt2_decomp_size =
                read_channel_nb_intel(ch_ntt2_decomp_size, &valid_read);
            if (valid_read) {
                ntt2_index = 0;
            }
        }

        if (intt1_index == MAX_U32) {
            bool valid_read;
            intt1_decomp_size =
                read_channel_nb_intel(ch_intt1_decomp_size, &valid_read);
            if (valid_read) {
                intt1_index = 0;
            }
        }

        unsigned ntt1_tf_index[MAX_KEY_MODULUS_SIZE];
        unsigned coeff_log = get_ntt_log(coeff_count);
#pragma unroll
        for (int i = 0; i < MAX_KEY_MODULUS_SIZE; i++) {
            unsigned fly_index = ntt1_index[i] / (coeff_count / 2 / VEC);
            unsigned i0 =
                ((ntt1_index[i] & ((coeff_count / 2 / VEC)) - 1) * VEC) >>
                (coeff_log - 1 - fly_index);
            unsigned m = 1 << fly_index;
            ntt1_tf_index[i] = (m + i0) / VEC;
        }
        unsigned ntt1_tf_size = coeff_log * coeff_count / 2 / VEC;

        {
            int k = 0;
            twiddle_factor tf;
#pragma unroll
            for (int j = 0; j < VEC; j++) {
                tf.data[j] = twiddle_factors0[ntt1_tf_index[k]][j];
            }
            bool success = write_channel_nb_intel(ch_twiddle_factor_rep[k], tf);
            if (success) STEP(ntt1_index[k], ntt1_tf_size);
        }

        {
            int k = 1;
            twiddle_factor tf;
#pragma unroll
            for (int j = 0; j < VEC; j++) {
                tf.data[j] = twiddle_factors1[ntt1_tf_index[k]][j];
            }
            bool success = write_channel_nb_intel(ch_twiddle_factor_rep[k], tf);
            if (success) STEP(ntt1_index[k], ntt1_tf_size);
        }

        {
            int k = 2;
            twiddle_factor tf;
#pragma unroll
            for (int j = 0; j < VEC; j++) {
                tf.data[j] = twiddle_factors2[ntt1_tf_index[k]][j];
            }
            bool success = write_channel_nb_intel(ch_twiddle_factor_rep[k], tf);
            if (success) STEP(ntt1_index[k], ntt1_tf_size);
        }

        {
            int k = 3;
            twiddle_factor tf;
#pragma unroll
            for (int j = 0; j < VEC; j++) {
                tf.data[j] = twiddle_factors3[ntt1_tf_index[k]][j];
            }
            bool success = write_channel_nb_intel(ch_twiddle_factor_rep[k], tf);
            if (success) STEP(ntt1_index[k], ntt1_tf_size);
        }

        {
            int k = 4;
            twiddle_factor tf;
#pragma unroll
            for (int j = 0; j < VEC; j++) {
                tf.data[j] = twiddle_factors4[ntt1_tf_index[k]][j];
            }
            bool success = write_channel_nb_intel(ch_twiddle_factor_rep[k], tf);
            if (success) STEP(ntt1_index[k], ntt1_tf_size);
        }

        {
            int k = 5;
            twiddle_factor tf;
#pragma unroll
            for (int j = 0; j < VEC; j++) {
                tf.data[j] = twiddle_factors5[ntt1_tf_index[k]][j];
            }
            bool success = write_channel_nb_intel(ch_twiddle_factor_rep[k], tf);
            if (success) STEP(ntt1_index[k], ntt1_tf_size);
        }

        {
            int k = 6;
            twiddle_factor tf;
#pragma unroll
            for (int j = 0; j < VEC; j++) {
                tf.data[j] = twiddle_factors6[ntt1_tf_index[k]][j];
            }
            bool success = write_channel_nb_intel(ch_twiddle_factor_rep[k], tf);
            if (success) STEP(ntt1_index[k], ntt1_tf_size);
        }

        unsigned ntt_loops = coeff_log * coeff_count / 2 / VEC;
        unsigned ntt2_decomp_index =
            ntt2_index == MAX_U32 ? 0 : ntt2_index / ntt_loops;
        unsigned ntt2_coeff_index =
            ntt2_index == MAX_U32 ? 0 : ntt2_index % ntt_loops;

        unsigned fly_index = ntt2_coeff_index / (coeff_count / 2 / VEC);
        unsigned i0 =
            ((ntt2_coeff_index & ((coeff_count / 2 / VEC)) - 1) * VEC) >>
            (coeff_log - 1 - fly_index);
        unsigned m = 1 << fly_index;
        unsigned ntt2_tf_index = (m + i0) / VEC;

        twiddle_factor tf;
#pragma unroll
        for (int j = 0; j < VEC; j++) {
            switch (ntt2_decomp_index) {
            case 5:
                tf.data[j] = twiddle_factors5[ntt2_tf_index][j];
                break;
            case 4:
                tf.data[j] = twiddle_factors4[ntt2_tf_index][j];
                break;
            case 3:
                tf.data[j] = twiddle_factors3[ntt2_tf_index][j];
                break;
            case 2:
                tf.data[j] = twiddle_factors2[ntt2_tf_index][j];
                break;
            case 1:
                tf.data[j] = twiddle_factors1[ntt2_tf_index][j];
                break;
            default:
                tf.data[j] = twiddle_factors0[ntt2_tf_index][j];
                break;
            }
        }

        // write ntt2
        if (ntt2_index != MAX_U32) {
            bool success = write_channel_nb_intel(
                ch_twiddle_factor_rep[NTT_ENGINES - 2], tf);
            if (success) STEP2(ntt2_index, ntt2_decomp_size);
        }

        // write intt1
        twiddle_factor intt1_tf;
#pragma unroll
        for (int j = 0; j < VEC; j++) {
            intt1_tf.data[j] =
                intt1_twiddle_factors[intt1_index == MAX_U32 ? 0 : intt1_index]
                                     [j];
        }
        if (intt1_index != MAX_U32) {
            success =
                write_channel_nb_intel(ch_intt1_twiddle_factor_rep, intt1_tf);
            if (success) STEP2(intt1_index, intt1_decomp_size);
        }

        // write intt2
        twiddle_factor intt2_tf;
#pragma unroll
        for (int j = 0; j < VEC; j++) {
            intt2_tf.data[j] = intt2_twiddle_factors[intt2_index][j];
        }

        success = write_channel_nb_intel(ch_intt2_twiddle_factor_rep, intt2_tf);
        if (success) STEP(intt2_index, coeff_count / VEC);
    }
}

__single_task __autorun void dispatch_intt1_twiddle_factor() {
    while (true) {
        twiddle_factor tf = read_channel_intel(ch_intt1_twiddle_factor_rep);
#pragma unroll
        for (int core = 0; core < CORES; core++) {
            write_channel_intel(ch_intt1_twiddle_factor[core][0], tf);
        }
    }
}

__single_task __autorun void dispatch_ntt1_twiddle_factor() {
    while (true) {
#pragma unroll
        for (int i = 0; i < MAX_RNS_MODULUS_SIZE; i++) {
            twiddle_factor tf = read_channel_intel(ch_twiddle_factor_rep[i]);
#pragma unroll
            for (int core = 0; core < CORES; core++) {
                write_channel_intel(ch_twiddle_factor[core][i], tf);
            }
        }
    }
}

__single_task __autorun void dispatch_intt2_twiddle_factor() {
    while (true) {
        twiddle_factor tf = read_channel_intel(ch_intt2_twiddle_factor_rep);
        twiddle_factor2 tf2;
        tf2.data[0] = tf.data[0];
        tf2.data[1] = tf.data[1];
#pragma unroll
        for (int core = 0; core < CORES; core++) {
            write_channel_intel(ch_intt2_twiddle_factor[core][0], tf2);
            write_channel_intel(ch_intt2_twiddle_factor[core][1], tf2);
        }

        tf2.data[0] = tf.data[2];
        tf2.data[1] = tf.data[3];
#pragma unroll
        for (int core = 0; core < CORES; core++) {
            write_channel_intel(ch_intt2_twiddle_factor[core][0], tf2);
            write_channel_intel(ch_intt2_twiddle_factor[core][1], tf2);
        }

        tf2.data[0] = tf.data[4];
        tf2.data[1] = tf.data[5];
#pragma unroll
        for (int core = 0; core < CORES; core++) {
            write_channel_intel(ch_intt2_twiddle_factor[core][0], tf2);
            write_channel_intel(ch_intt2_twiddle_factor[core][1], tf2);
        }

        tf2.data[0] = tf.data[6];
        tf2.data[1] = tf.data[7];
#pragma unroll
        for (int core = 0; core < CORES; core++) {
            write_channel_intel(ch_intt2_twiddle_factor[core][0], tf2);
            write_channel_intel(ch_intt2_twiddle_factor[core][1], tf2);
        }
    }
}

__single_task __autorun void dispatch_ntt2_twiddle_factor() {
    while (true) {
        twiddle_factor tf =
            read_channel_intel(ch_twiddle_factor_rep[NTT_ENGINES - 2]);
#pragma unroll
        for (int core = 0; core < CORES; core++) {
            write_channel_intel(ch_twiddle_factor[core][MAX_RNS_MODULUS_SIZE],
                                tf);
            write_channel_intel(
                ch_twiddle_factor[core][MAX_RNS_MODULUS_SIZE + 1], tf);
        }
    }
}
