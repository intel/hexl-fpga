
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
// info ----- kernels for genrating the roots of unity tables --------
// info ----- also included are the kernels for streaming     --------

#include "pipes_keyswitch.hpp"

// info old name : dispatch_twiddle_factors

template <class tt_kernelNameClass, class tt_ch_intt1_decomp_size,
          class tt_ch_ntt2_decomp_size, class tt_ch_twiddle_factor_rep,
          class tt_ch_intt1_twiddle_factor_rep,
          class tt_ch_intt2_twiddle_factor_rep>
void dispatch_twiddle_factors(sycl::queue& q,
                              sycl::buffer<uint64_t>* buff_twiddles,
                              unsigned coeff_count, bool load_twiddle_factors) {
    auto qSubLambda = [&](sycl::handler& h) {
        sycl::accessor twiddle_factors_data(*buff_twiddles, h, sycl::read_only);
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            uint64_t twiddle_factors0[MAX_COFF_COUNT / VEC][VEC];
            uint64_t twiddle_factors1[MAX_COFF_COUNT / VEC][VEC];
            uint64_t twiddle_factors2[MAX_COFF_COUNT / VEC][VEC];
            uint64_t twiddle_factors3[MAX_COFF_COUNT / VEC][VEC];
            uint64_t twiddle_factors4[MAX_COFF_COUNT / VEC][VEC];
            uint64_t twiddle_factors5[MAX_COFF_COUNT / VEC][VEC];
            uint64_t twiddle_factors6[MAX_COFF_COUNT / VEC][VEC];

            uint64_t intt1_twiddle_factors[(MAX_RNS_MODULUS_SIZE - 1) *
                                           MAX_COFF_COUNT / VEC][VEC];
            uint64_t intt2_twiddle_factors[MAX_COFF_COUNT / VEC][VEC];
            for (int k = 0; k < MAX_KEY_MODULUS_SIZE; k++) {
                for (int i = 0; i < coeff_count / VEC; i++) {
#pragma unroll
                    for (int j = 0; j < VEC; j++) {
                        uint64_t data =
                            twiddle_factors_data[coeff_count * (k * 4 + 2) +
                                                 i * VEC + j];
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
                            twiddle_factors_data[coeff_count * k * 4 + i * VEC +
                                                 j];
                        if (k < (MAX_RNS_MODULUS_SIZE - 1)) {
                            intt1_twiddle_factors[k * (coeff_count / VEC) + i]
                                                 [j] = data;
                        } else {
                            intt2_twiddle_factors[i][j] = data;
                        }
                    }
                }
            }

            unsigned ntt1_index[MAX_KEY_MODULUS_SIZE] = {0, 0, 0, 0, 0, 0, 0};
            short ntt2_index = -1;
            unsigned ntt2_decomp_size;
            unsigned intt1_decomp_size;

            short intt1_index = -1;
            unsigned intt2_index = 0;
            bool success;

            while (true) {
                if (ntt2_index == -1) {
                    bool valid_read;
                    ntt2_decomp_size = tt_ch_ntt2_decomp_size::read(valid_read);
                    if (valid_read) {
                        ntt2_index = 0;
                    }
                }

                if (intt1_index == -1) {
                    bool valid_read;
                    intt1_decomp_size =
                        tt_ch_intt1_decomp_size::read(valid_read);
                    if (valid_read) {
                        intt1_index = 0;
                    }
                }
                {
                    const int k = 0;
                    TwiddleFactor_t tf;
#pragma unroll
                    for (int j = 0; j < VEC; j++) {
                        tf.data[j] = twiddle_factors0[ntt1_index[k]][j];
                    }
                    bool success;
                    using twPipe =
                        typename tt_ch_twiddle_factor_rep::template PipeAt<k>;
                    twPipe::write(tf, success);
                    if (success) STEP(ntt1_index[k], coeff_count / VEC);
                }

                {
                    const int k = 1;
                    TwiddleFactor_t tf;
#pragma unroll
                    for (int j = 0; j < VEC; j++) {
                        tf.data[j] = twiddle_factors1[ntt1_index[k]][j];
                    }
                    bool success;
                    using twPipe2 =
                        typename tt_ch_twiddle_factor_rep::template PipeAt<k>;
                    twPipe2::write(tf, success);
                    if (success) STEP(ntt1_index[k], coeff_count / VEC);
                }

                {
                    const int k = 2;
                    TwiddleFactor_t tf;
#pragma unroll
                    for (int j = 0; j < VEC; j++) {
                        tf.data[j] = twiddle_factors2[ntt1_index[k]][j];
                    }
                    bool success;
                    using twPipe3 =
                        typename tt_ch_twiddle_factor_rep::template PipeAt<k>;
                    twPipe3::write(tf, success);
                    if (success) STEP(ntt1_index[k], coeff_count / VEC);
                }

                {
                    const int k = 3;
                    TwiddleFactor_t tf;
#pragma unroll
                    for (int j = 0; j < VEC; j++) {
                        tf.data[j] = twiddle_factors3[ntt1_index[k]][j];
                    }
                    bool success;
                    using twPipe4 =
                        typename tt_ch_twiddle_factor_rep::template PipeAt<k>;
                    twPipe4::write(tf, success);
                    if (success) STEP(ntt1_index[k], coeff_count / VEC);
                }

                {
                    const int k = 4;
                    TwiddleFactor_t tf;
#pragma unroll
                    for (int j = 0; j < VEC; j++) {
                        tf.data[j] = twiddle_factors4[ntt1_index[k]][j];
                    }
                    bool success;
                    using twPipe5 =
                        typename tt_ch_twiddle_factor_rep::template PipeAt<k>;
                    twPipe5::write(tf, success);
                    if (success) STEP(ntt1_index[k], coeff_count / VEC);
                }

                {
                    const int k = 5;
                    TwiddleFactor_t tf;
#pragma unroll
                    for (int j = 0; j < VEC; j++) {
                        tf.data[j] = twiddle_factors5[ntt1_index[k]][j];
                    }
                    bool success;
                    using twPipe6 =
                        typename tt_ch_twiddle_factor_rep::template PipeAt<k>;
                    twPipe6::write(tf, success);
                    if (success) STEP(ntt1_index[k], coeff_count / VEC);
                }

                {
                    const int k = 6;
                    TwiddleFactor_t tf;
#pragma unroll
                    for (int j = 0; j < VEC; j++) {
                        tf.data[j] = twiddle_factors6[ntt1_index[k]][j];
                    }
                    bool success;
                    using twPipe7 =
                        typename tt_ch_twiddle_factor_rep::template PipeAt<k>;
                    twPipe7::write(tf, success);
                    if (success) STEP(ntt1_index[k], coeff_count / VEC);
                }

                unsigned ntt2_decomp_index =
                    ntt2_index == -1 ? 0 : ntt2_index / (coeff_count / VEC);
                unsigned ntt2_coeff_index =
                    ntt2_index == -1 ? 0 : ntt2_index % (coeff_count / VEC);
                TwiddleFactor_t tf;
#pragma unroll
                for (int j = 0; j < VEC; j++) {
                    switch (ntt2_decomp_index) {
                    case 5:
                        tf.data[j] = twiddle_factors5[ntt2_coeff_index][j];
                        break;
                    case 4:
                        tf.data[j] = twiddle_factors4[ntt2_coeff_index][j];
                        break;
                    case 3:
                        tf.data[j] = twiddle_factors3[ntt2_coeff_index][j];
                        break;
                    case 2:
                        tf.data[j] = twiddle_factors2[ntt2_coeff_index][j];
                        break;
                    case 1:
                        tf.data[j] = twiddle_factors1[ntt2_coeff_index][j];
                        break;
                    default:
                        tf.data[j] = twiddle_factors0[ntt2_coeff_index][j];
                        break;
                    }
                }

                // write ntt2
                if (ntt2_index >= 0) {
                    bool success;
                    using twPipe8 =
                        typename tt_ch_twiddle_factor_rep::template PipeAt<
                            NTT_ENGINES - 2>;
                    twPipe8::write(tf, success);
                    short max_tmp = ntt2_decomp_size / VEC - 1;
                    if (success) STEP3(ntt2_index, max_tmp);
                }
                // write intt1
                TwiddleFactor_t intt1_tf;
#pragma unroll
                for (int j = 0; j < VEC; j++) {
                    intt1_tf.data[j] =
                        intt1_twiddle_factors[intt1_index == -1 ? 0
                                                                : intt1_index]
                                             [j];
                }
                if (intt1_index >= 0) {
                    tt_ch_intt1_twiddle_factor_rep::write(intt1_tf, success);
                    short max_tmp = intt1_decomp_size / VEC - 1;
                    if (success) STEP3(intt1_index, max_tmp);
                }

                // write intt2
                TwiddleFactor_t intt2_tf;
#pragma unroll
                for (int j = 0; j < VEC; j++) {
                    intt2_tf.data[j] = intt2_twiddle_factors[intt2_index][j];
                }
                tt_ch_intt2_twiddle_factor_rep::write(intt2_tf, success);
                if (success) STEP(intt2_index, coeff_count / VEC);
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}

// info : kernel broadcasts the same twiddle to multiple cores
// info: old name : dispatch_intt1_twiddle_factor
template <class tt_kernelNameClass, class tt_ch_intt1_twiddle_factor_rep,
          class tt_ch_intt1_twiddle_factor>

void dispatch_intt1_twiddle_factor(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            while (true) {
                TwiddleFactor_t tf = tt_ch_intt1_twiddle_factor_rep::read();

                // info Unrol and broadcast to all cores
                Unroller<0, NUM_CORES>::Step([&](auto core_num) {
                    using pipe =
                        typename tt_ch_intt1_twiddle_factor::template PipeAt<
                            core_num, 0>;
                    pipe::write(tf);
                });
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}

// INFO: twiddle factor dispatch for ntt1

template <class tt_kernelNameClass, class tt_ch_twiddle_factor_rep,
          class tt_ch_twiddle_factor>
void dispatch_ntt1_twiddle_factor(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            while (true) {
                Unroller<0, MAX_RNS_MODULUS_SIZE>::Step([&](auto i) {
                    using readPipe =
                        typename tt_ch_twiddle_factor_rep::template PipeAt<i>;
                    TwiddleFactor_t tf = readPipe::read();
                    Unroller<0, NUM_CORES>::Step([&](auto core) {
                        using writePipe =
                            typename tt_ch_twiddle_factor::template PipeAt<core,
                                                                           i>;
                        writePipe::write(tf);
                    });
                });
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}

// INFO: kernel dispatch_intt2_twiddle_factor
template <class tt_kernelNameClass, class tt_ch_intt2_twiddle_factor_rep,
          class tt_ch_intt2_twiddle_factor>
void dispatch_intt2_twiddle_factor(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            while (true) {
                TwiddleFactor_t tf = tt_ch_intt2_twiddle_factor_rep::read();
                TwiddleFactor2_t tf2;
                tf2.data[0] = tf.data[0];
                tf2.data[1] = tf.data[1];
                Unroller<0, NUM_CORES>::Step([&](auto core) {
                    using writePipe1 =
                        typename tt_ch_intt2_twiddle_factor::template PipeAt<
                            core, 0>;
                    writePipe1::write(tf2);
                    using writePipe2 =
                        typename tt_ch_intt2_twiddle_factor::template PipeAt<
                            core, 1>;
                    writePipe2::write(tf2);
                });

                tf2.data[0] = tf.data[2];
                tf2.data[1] = tf.data[3];
                Unroller<0, NUM_CORES>::Step([&](auto core) {
                    using writePipe3 =
                        typename tt_ch_intt2_twiddle_factor::template PipeAt<
                            core, 0>;
                    writePipe3::write(tf2);
                    using writePipe4 =
                        typename tt_ch_intt2_twiddle_factor::template PipeAt<
                            core, 1>;
                    writePipe4::write(tf2);
                });
                tf2.data[0] = tf.data[4];
                tf2.data[1] = tf.data[5];
                Unroller<0, NUM_CORES>::Step([&](auto core) {
                    using writePipe5 =
                        typename tt_ch_intt2_twiddle_factor::template PipeAt<
                            core, 0>;
                    writePipe5::write(tf2);
                    using writePipe6 =
                        typename tt_ch_intt2_twiddle_factor::template PipeAt<
                            core, 1>;
                    writePipe6::write(tf2);
                });

                tf2.data[0] = tf.data[6];
                tf2.data[1] = tf.data[7];
                Unroller<0, NUM_CORES>::Step([&](auto core) {
                    using writePipe7 =
                        typename tt_ch_intt2_twiddle_factor::template PipeAt<
                            core, 0>;
                    writePipe7::write(tf2);
                    using writePipe8 =
                        typename tt_ch_intt2_twiddle_factor::template PipeAt<
                            core, 1>;
                    writePipe8::write(tf2);
                });
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}

// info : kernel broadcasts the same twiddle to multiple cores
// info: old name : dispatch_ntt2_twiddle_factor

template <class tt_kernelNameClass, class tt_ch_twiddle_factor_rep,
          class tt_ch_twiddle_factor>
void dispatch_ntt2_twiddle_factor(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]] {
            while (true) {
                using readPipe1 =
                    typename tt_ch_twiddle_factor_rep::template PipeAt<
                        NTT_ENGINES - 2>;

                TwiddleFactor_t tf = readPipe1::read();
                Unroller<0, NUM_CORES>::Step([&](auto core_num) {
                    using writePipe1 =
                        typename tt_ch_twiddle_factor::template PipeAt<
                            core_num, MAX_RNS_MODULUS_SIZE>;
                    using writePipe2 =
                        typename tt_ch_twiddle_factor::template PipeAt<
                            core_num, MAX_RNS_MODULUS_SIZE + 1>;
                    writePipe1::write(tf);
                    writePipe2::write(tf);
                });
            }
        };
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}

namespace temp {
template <class tt_kernelNameClass

          >
void inttStreamingKernel(sycl::queue& q) {
    auto qSubLambda = [&](sycl::handler& h) {
        auto kernelLambda = [=]()
            [[intel::kernel_args_restrict]] [[intel::max_global_work_dim(0)]]{};
        h.single_task<tt_kernelNameClass>(kernelLambda);
    };
    q.submit(qSubLambda);
}
};  // namespace temp
