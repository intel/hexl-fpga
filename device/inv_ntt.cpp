// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "./utils/pipe_def_marcos.hpp"
#include "./utils/pipe_array.hpp"
#include "./utils/unroller.hpp"
#include "mod_ops.hpp"

#include "dpc_common.hpp"

#ifndef FPGA_INTT_SIZE
#define FPGA_INTT_SIZE 16384
#endif

#ifndef NUM_INTT_COMPUTE_UNITS
#define NUM_INTT_COMPUTE_UNITS 1
#endif

#ifndef VEC_INTT
#define VEC_INTT 8
#endif

typedef unsigned long ulong64Bit;
typedef unsigned int uint32Bit;
typedef struct {
    ulong64Bit data[VEC_INTT * 2];
} elements_in_intt_t;

typedef struct {
    ulong64Bit data[VEC_INTT * 2];
} elements_out_intt_t;

typedef struct {
    ulong64Bit data[VEC_INTT * 2];
} WideVecInType;

typedef struct {
    ulong64Bit data[VEC_INTT];
} WideVecOutType;

typedef struct {
    ulong64Bit data[64 / sizeof(ulong64Bit)];
} Wide64ByteType;

#ifndef __FULL_DEPTH_CHANNELS__
defPipe1d(modulusPipeINTT, ulong64Bit, 16, NUM_INTT_COMPUTE_UNITS);
defPipe1d(miniBatchPipeINTT, ulong64Bit, 16, NUM_INTT_COMPUTE_UNITS);
defPipe1d(inDataPipeINTT, WideVecInType, 16, NUM_INTT_COMPUTE_UNITS);
defPipe1d(outDataPipeINTT, WideVecOutType, (FPGA_INTT_SIZE / VEC_INTT / 2),
          NUM_INTT_COMPUTE_UNITS);
defPipe1d(outDataPipeINTT2, WideVecOutType, (FPGA_INTT_SIZE / VEC_INTT / 2),
          NUM_INTT_COMPUTE_UNITS);
defPipe1d(inv_ni_pipe, ulong64Bit, 16, NUM_INTT_COMPUTE_UNITS);
defPipe1d(inv_n_wi_pipe, ulong64Bit, 16, NUM_INTT_COMPUTE_UNITS);
defPipe1d(twiddleFactorsPipeINTT, Wide64ByteType, 16, NUM_INTT_COMPUTE_UNITS);
defPipe1d(barrettTwiddleFactorsPipeINTT, Wide64ByteType, 16,
          NUM_INTT_COMPUTE_UNITS);
#else
////////////////////////// Full depth
/// pipes//////////////////////////////////////////////////////
defPipe1d(modulusPipeINTT, ulong64Bit, FPGA_INTT_SIZE, NUM_INTT_COMPUTE_UNITS);
defPipe1d(miniBatchPipeINTT, uint32Bit, FPGA_INTT_SIZE, NUM_INTT_COMPUTE_UNITS);
defPipe1d(inDataPipeINTT, WideVecInType, FPGA_INTT_SIZE,
          NUM_INTT_COMPUTE_UNITS);
defPipe1d(outDataPipeINTT, WideVecOutType, FPGA_INTT_SIZE,
          NUM_INTT_COMPUTE_UNITS);
defPipe1d(outDataPipeINTT2, WideVecOutType, FPGA_INTT_SIZE,
          NUM_INTT_COMPUTE_UNITS);
defPipe1d(inv_ni_pipe, ulong64Bit, FPGA_INTT_SIZE, NUM_INTT_COMPUTE_UNITS);
defPipe1d(inv_n_wi_pipe, ulong64Bit, FPGA_INTT_SIZE, NUM_INTT_COMPUTE_UNITS);
defPipe1d(twiddleFactorsPipeINTT, Wide64ByteType, FPGA_INTT_SIZE,
          NUM_INTT_COMPUTE_UNITS);
defPipe1d(barrettTwiddleFactorsPipeINTT, Wide64ByteType, FPGA_INTT_SIZE,
          NUM_INTT_COMPUTE_UNITS);
#endif

template <size_t id>
class INV_NTT_KERN;

template <size_t id>
void inv_ntt_kernel(sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.single_task<INV_NTT_KERN<id>>([=]() {
            constexpr int computeUnitID = id;
            int n = (int)FPGA_INTT_SIZE;
            unsigned char Xm_val;
            unsigned int roots_acc;
            int t;
            int logt;

            [[intel::numbanks(VEC_INTT)]] [[intel::max_replicates(
                2)]] unsigned long X[FPGA_INTT_SIZE / VEC_INTT][VEC_INTT];
            [[intel::numbanks(VEC_INTT)]] [[intel::max_replicates(
                2)]] unsigned long X2[FPGA_INTT_SIZE / VEC_INTT][VEC_INTT];

            unsigned long Xm[FPGA_INTT_SIZE / VEC_INTT];
            unsigned long local_roots[FPGA_INTT_SIZE];
            unsigned long local_precons[FPGA_INTT_SIZE];

            for (int i = 0; i < FPGA_INTT_SIZE / VEC_INTT; i++) {
                Xm[i] = 0;
            }

            ulong64Bit prime;

            [[intel::disable_loop_pipelining]] while (true) {
                uint32Bit miniBatchSize =
                    miniBatchPipeINTT::PipeAt<computeUnitID>::read();
                prime = modulusPipeINTT::PipeAt<computeUnitID>::read();
                ulong64Bit inv_n = inv_ni_pipe::PipeAt<computeUnitID>::read();
                ulong64Bit inv_n_w =
                    inv_n_wi_pipe::PipeAt<computeUnitID>::read();

                unsigned long twice_mod = prime << 1;
                unsigned long output_mod_factor = 1;
                constexpr size_t numTwiddlePerWord =
                    sizeof(Wide64ByteType) / sizeof(ulong64Bit);

                for (int i = 0; i < FPGA_INTT_SIZE / numTwiddlePerWord; i++) {
                    Wide64ByteType VEC_INTTTwiddle =
                        twiddleFactorsPipeINTT::PipeAt<computeUnitID>::read();
#pragma unroll
                    for (size_t j = 0; j < numTwiddlePerWord; ++j) {
                        local_roots[i * numTwiddlePerWord + j] =
                            VEC_INTTTwiddle.data[j];
                    }
                }

                for (int i = 0; i < FPGA_INTT_SIZE / numTwiddlePerWord; i++) {
                    Wide64ByteType VEC_INTTExponent =
                        barrettTwiddleFactorsPipeINTT::PipeAt<
                            computeUnitID>::read();
#pragma unroll
                    for (size_t j = 0; j < numTwiddlePerWord; ++j) {
                        local_precons[i * numTwiddlePerWord + j] =
                            VEC_INTTExponent.data[j];
                    }
                }

                for (int i = 0; i < miniBatchSize; i++) {
                    Xm_val = 0;
                    roots_acc = 1;
                    t = 1;
                    logt = 0;
                    // Normalize the Transform by N

                    for (int m = (n >> 1); m > 1; m >>= 1) {
                        Xm_val++;
                        bool b_first_stage = Xm_val == 1;

                        [[intel::ivdep(X)]] [[intel::ivdep(X2)]] [[intel::ivdep(
                            Xm)]] for (int k = 0;
                                       k < FPGA_INTT_SIZE / 2 / VEC_INTT; k++) {
                            [[intel::fpga_register]] unsigned long
                                curX[VEC_INTT * 2];
                            [[intel::fpga_register]] unsigned long
                                curX_rep[VEC_INTT * 2];

                            size_t i0 = (k * VEC_INTT + 0) >>
                                        logt;  // i is the index of groups
                            size_t j0 =
                                (k * VEC_INTT + 0) &
                                (t - 1);  // j is the position of a group
                            size_t j10 = i0 * 2 * t;

                            // fetch the next VEC_INTTtor if the VEC_INTT index
                            // is the same
                            bool b_same_VEC_INTT = ((j10 + j0) / VEC_INTT) ==
                                                   ((j10 + j0 + t) / VEC_INTT);
                            size_t X_ind = (j10 + j0) / VEC_INTT;
                            size_t Xt_ind =
                                (j10 + j0 + t) / VEC_INTT + b_same_VEC_INTT;

                            bool b_X = Xm[X_ind] == (Xm_val - 1) || Xm_val == 1;
                            bool b_Xt =
                                Xm[Xt_ind] == (Xm_val - 1) || Xm_val == 1;

                            WideVecInType elements_in;
                            if (b_first_stage) {
                                elements_in = inDataPipeINTT::PipeAt<
                                    computeUnitID>::read();
                            }

#pragma unroll
                            for (int n = 0; n < VEC_INTT; n++) {
                                size_t i = (k * VEC_INTT + n) >>
                                           logt;  // i is the index of groups
                                size_t j =
                                    (k * VEC_INTT + n) &
                                    (t - 1);  // j is the position of a group
                                size_t j1 = i * 2 * t;
                                if (b_first_stage) {
                                    curX[n] = elements_in.data[n];
                                    curX[n + VEC_INTT] =
                                        elements_in.data[n + VEC_INTT];
                                } else {
                                    curX[n] = b_X ? X[X_ind][n] : X2[X_ind][n];
                                    curX[n + VEC_INTT] =
                                        b_Xt ? X[Xt_ind][n] : X2[Xt_ind][n];
                                }
                            }

                            if (t == 1) {
#pragma unroll
                                for (int n = 0; n < VEC_INTT; n++) {
                                    const int cur_t = 1;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn +
                                        ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                                    curX_rep[n] = curX[Xn];
                                    curX_rep[VEC_INTT + n] = curX[Xnt];
                                }
#if VEC_INTT >= 4
                            } else if (t == 2) {
#pragma unroll
                                for (int n = 0; n < VEC_INTT; n++) {
                                    const int cur_t = 2;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn +
                                        ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                                    curX_rep[n] = curX[Xn];
                                    curX_rep[VEC_INTT + n] = curX[Xnt];
                                }
#endif
#if VEC_INTT >= 8
                            } else if (t == 4) {
#pragma unroll
                                for (int n = 0; n < VEC_INTT; n++) {
                                    const int cur_t = 4;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn +
                                        ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                                    curX_rep[n] = curX[Xn];
                                    curX_rep[VEC_INTT + n] = curX[Xnt];
                                }
#endif
#if VEC_INTT >= 16
                            } else if (t == 8) {
#pragma unroll
                                for (int n = 0; n < VEC_INTT; n++) {
                                    const int cur_t = 8;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn +
                                        ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                                    curX_rep[n] = curX[Xn];
                                    curX_rep[VEC_INTT + n] = curX[Xnt];
                                }
#endif
#if VEC_INTT >= 32
                            } else if (t == 16) {
#pragma unroll
                                for (int n = 0; n < VEC_INTT; n++) {
                                    const int cur_t = 16;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn +
                                        ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                                    curX_rep[n] = curX[Xn];
                                    curX_rep[VEC_INTT + n] = curX[Xnt];
                                }
#endif
                            } else {
#pragma unroll
                                for (int n = 0; n < VEC_INTT; n++) {
                                    curX_rep[n] = curX[n];
                                    curX_rep[VEC_INTT + n] = curX[VEC_INTT + n];
                                }
                            }

#pragma unroll
                            for (int n = 0; n < VEC_INTT; n++) {
                                int i = (k * VEC_INTT + n) >>
                                        logt;  // i is the index of groups
                                int j =
                                    (k * VEC_INTT + n) &
                                    (t - 1);  // j is the position of a group
                                int j1 = i * 2 * t + j;
                                int j2 = j1 + t;
                                unsigned long W_op = local_roots[roots_acc + i];
                                unsigned long W_op_precon =
                                    local_precons[roots_acc + i];
                                unsigned long tx = 0;
                                unsigned long ty = 0;

                                // Butterfly
                                unsigned long x_j1 = curX_rep[n];
                                unsigned long x_j2 = curX_rep[VEC_INTT + n];

                                tx = x_j1 + x_j2;
                                ty = x_j1 + twice_mod - x_j2;
                                curX[n] = (unsigned long)(tx >= twice_mod)
                                              ? (tx - twice_mod)
                                              : tx;
                                curX[VEC_INTT + n] = MultiplyUIntModLazy4(
                                    ty, W_op_precon, W_op, prime);
                            }

                            // reorder back
                            if (t == 1) {
#pragma unroll
                                for (int n = 0; n < VEC_INTT; n++) {
                                    const int cur_t = 1;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn +
                                        ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                                    curX_rep[Xn] = curX[n];
                                    curX_rep[Xnt] = curX[VEC_INTT + n];
                                }
#if VEC_INTT >= 4
                            } else if (t == 2) {
#pragma unroll
                                for (int n = 0; n < VEC_INTT; n++) {
                                    const int cur_t = 2;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn +
                                        ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                                    curX_rep[Xn] = curX[n];
                                    curX_rep[Xnt] = curX[VEC_INTT + n];
                                }
#endif
#if VEC_INTT >= 8
                            } else if (t == 4) {
#pragma unroll
                                for (int n = 0; n < VEC_INTT; n++) {
                                    const int cur_t = 4;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn +
                                        ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                                    curX_rep[Xn] = curX[n];
                                    curX_rep[Xnt] = curX[VEC_INTT + n];
                                }
#endif
#if VEC_INTT >= 16
                            } else if (t == 8) {
#pragma unroll
                                for (int n = 0; n < VEC_INTT; n++) {
                                    const int cur_t = 8;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn +
                                        ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                                    curX_rep[Xn] = curX[n];
                                    curX_rep[Xnt] = curX[VEC_INTT + n];
                                }
#endif
#if VEC_INTT >= 32
                            } else if (t == 16) {
#pragma unroll
                                for (int n = 0; n < VEC_INTT; n++) {
                                    const int cur_t = 16;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn +
                                        ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                                    curX_rep[Xn] = curX[n];
                                    curX_rep[Xnt] = curX[VEC_INTT + n];
                                }
#endif
                            } else {
#pragma unroll
                                for (int n = 0; n < VEC_INTT; n++) {
                                    curX_rep[n] = curX[n];
                                    curX_rep[VEC_INTT + n] = curX[VEC_INTT + n];
                                }
                            }

#pragma unroll
                            for (int n = 0; n < VEC_INTT; n++) {
                                X[X_ind][n] = curX_rep[n];
                                X2[Xt_ind][n] = curX_rep[n + VEC_INTT];
                            }
                            // update Xm
                            Xm[X_ind] = Xm_val;
                        }

                        roots_acc += m;
                        t <<= 1;
                        logt++;
                    }

                    // Normalization modulo
                    for (int i = 0; i < n / 2 / VEC_INTT;
                         i++) {  // line 564 of hexl
                        WideVecOutType out;
                        WideVecOutType out2;
#pragma unroll
                        for (int j = 0; j < VEC_INTT; j++) {
                            unsigned long x =
                                Xm[i] == Xm_val ? X[i][j] : X2[i][j];
                            unsigned long xt =
                                Xm[(n >> 1) / VEC_INTT + i] == Xm_val
                                    ? X[(n >> 1) / VEC_INTT + i][j]
                                    : X2[(n >> 1) / VEC_INTT + i][j];
                            unsigned long tx = x + xt;
                            unsigned long ty = x + twice_mod - xt;

                            tx = (uint64_t)(tx >= twice_mod) ? (tx - twice_mod)
                                                             : tx;

                            unsigned long nx =
                                MultiplyUIntModLazy3(tx, inv_n, prime);
                            unsigned long nx2 =
                                MultiplyUIntModLazy3(ty, inv_n_w, prime);

                            if (nx >= prime) {
                                nx -= prime;
                            }
                            if (nx2 >= prime) {
                                nx2 -= prime;
                            }

                            out.data[j] = nx;
                            out2.data[j] = nx2;
                        }

                        outDataPipeINTT::PipeAt<computeUnitID>::write(out);
                        outDataPipeINTT2::PipeAt<computeUnitID>::write(out2);
                    }  // end for
                }
            }
        });
    });
}

void intt_input_kernel(unsigned int numFrames, uint64_t* k_inData,
                       uint64_t* k_modulus, uint64_t* k_inv_ni,
                       uint64_t* k_inv_n_wi, uint64_t* k_twiddleFactors,
                       uint64_t* k_barrettTwiddleFactors) {
    sycl::host_ptr<uint64_t> inData(k_inData);
    sycl::host_ptr<uint64_t> modulus(k_modulus);
    sycl::host_ptr<uint64_t> inv_ni(k_inv_ni);
    sycl::host_ptr<uint64_t> inv_n_wi(k_inv_n_wi);
    sycl::host_ptr<uint64_t> twiddleFactors(k_twiddleFactors);
    sycl::host_ptr<uint64_t> barrettTwiddleFactors(k_barrettTwiddleFactors);

    // Create mini batches for every INTT instance
    Unroller<0, NUM_INTT_COMPUTE_UNITS>::Step([&](auto i) {
        uint32Bit fractionalMiniBatch =
            (numFrames % NUM_INTT_COMPUTE_UNITS) / (i + 1);
        if (fractionalMiniBatch > 0)
            fractionalMiniBatch = 1;
        else
            fractionalMiniBatch = 0;

        uint32Bit miniBatchSize =
            (numFrames / NUM_INTT_COMPUTE_UNITS) + fractionalMiniBatch;
        miniBatchPipeINTT::PipeAt<i>::write(miniBatchSize);
    });

    // Assuming the twiddle factors and the complex root of unity are similar
    // broadcast roots of unity to each kernel
    uint64_t temp = modulus[0];
    Unroller<0, NUM_INTT_COMPUTE_UNITS>::Step(
        [&](auto c) { modulusPipeINTT::PipeAt<c>::write(temp); });

    uint64_t inv_ni_temp = inv_ni[0];
    Unroller<0, NUM_INTT_COMPUTE_UNITS>::Step(
        [&](auto c) { inv_ni_pipe::PipeAt<c>::write(inv_ni_temp); });

    uint64_t inv_n_wi_temp = inv_n_wi[0];
    Unroller<0, NUM_INTT_COMPUTE_UNITS>::Step(
        [&](auto c) { inv_n_wi_pipe::PipeAt<c>::write(inv_n_wi_temp); });

    constexpr size_t numTwiddlePerWord =
        sizeof(Wide64ByteType) / sizeof(ulong64Bit);
    constexpr unsigned int iterations = FPGA_INTT_SIZE / numTwiddlePerWord;

    for (size_t i = 0; i < iterations; i++) {
        Wide64ByteType tw;
#pragma unroll
        for (size_t j = 0; j < numTwiddlePerWord; j++) {
            tw.data[j] = twiddleFactors[i * numTwiddlePerWord + j];
        }

        // broadcast twiddles to all compute units
        Unroller<0, NUM_INTT_COMPUTE_UNITS>::Step(
            [&](auto c) { twiddleFactorsPipeINTT::PipeAt<c>::write(tw); });
    }

    for (size_t i = 0; i < FPGA_INTT_SIZE / numTwiddlePerWord; i++) {
        Wide64ByteType tw;
#pragma unroll
        for (size_t j = 0; j < numTwiddlePerWord; j++) {
            tw.data[j] = barrettTwiddleFactors[i * numTwiddlePerWord + j];
        }
        // broadcast twiddles to all compute units
        Unroller<0, NUM_INTT_COMPUTE_UNITS>::Step([&](auto c) {
            barrettTwiddleFactorsPipeINTT::PipeAt<c>::write(tw);
        });
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // Retrieve one INTT data and stream to different kernels, per iteration for
    // top level loop

    constexpr unsigned int numElementsInVec =
        sizeof(WideVecInType) / sizeof(ulong64Bit);

    Unroller<0, NUM_INTT_COMPUTE_UNITS>::Step([&](auto computeUnitIndex) {
        for (unsigned int b = 0; b < numFrames; b++) {
            if (b % NUM_INTT_COMPUTE_UNITS == computeUnitIndex) {
                for (size_t i = 0; i < FPGA_INTT_SIZE / numElementsInVec; i++) {
                    WideVecInType inVec;
                    unsigned long offset =
                        b * FPGA_INTT_SIZE + i * numElementsInVec;
#pragma unroll
                    for (size_t j = 0; j < numElementsInVec / 2; j++) {
                        inVec.data[j] = inData[offset + j];
                    }

#pragma unroll
                    for (size_t j = 0; j < numElementsInVec / 2; j++) {
                        inVec.data[j + VEC_INTT] =
                            inData[offset + j + numElementsInVec / 2];
                    }

                    inDataPipeINTT::PipeAt<computeUnitIndex>::write(inVec);
                }
            }
        }
    });
}

void intt_output_kernel(unsigned int numFrames, uint64_t* k_outData) {
    sycl::host_ptr<uint64_t> outData(k_outData);

    Unroller<0, NUM_INTT_COMPUTE_UNITS>::Step([&](auto computeUnitIndex) {
        for (size_t i = 0; i < numFrames; i++) {
            if (i % NUM_INTT_COMPUTE_UNITS == computeUnitIndex) {
                size_t frameOffset = i * FPGA_INTT_SIZE;
                for (unsigned int k = 0; k < FPGA_INTT_SIZE / VEC_INTT; k++) {
                    size_t offset = frameOffset + k * VEC_INTT;
                    WideVecOutType elements_out;
                    if (k < FPGA_INTT_SIZE / VEC_INTT / 2) {
                        elements_out =
                            outDataPipeINTT::PipeAt<computeUnitIndex>::read();
                    } else {
                        elements_out =
                            outDataPipeINTT2::PipeAt<computeUnitIndex>::read();
                    }
#pragma unroll
                    for (int j = 0; j < VEC_INTT; j++) {
                        outData[offset + j] = elements_out.data[j];
                    }
                }
            }
        }
    });
}

class INV_NTT_INPUT;
class INV_NTT_OUTPUT;

extern "C" {

// the interface for inv ntt, aligned with hexl-fpga inv_ntt.cl file.

void inv_ntt(sycl::queue& q) {
    Unroller<0, NUM_INTT_COMPUTE_UNITS>::Step(
        [&](auto idx) { inv_ntt_kernel<idx>(q); });
}

sycl::event intt_input(sycl::queue& q, unsigned int numFrames,
                       uint64_t* inData_svm, uint64_t* modulus_svm,
                       uint64_t* inv_ni_svm, uint64_t* inv_n_wi_svm,
                       uint64_t* twiddleFactors_svm,
                       uint64_t* barrettTwiddleFactors_svm) {
    auto e = q.submit([&](sycl::handler& h) {
        h.single_task<INV_NTT_INPUT>([=]() [[intel::kernel_args_restrict]] {
            intt_input_kernel(numFrames, inData_svm, modulus_svm, inv_ni_svm,
                              inv_n_wi_svm, twiddleFactors_svm,
                              barrettTwiddleFactors_svm);
        });
    });

    return e;
}

sycl::event intt_output(sycl::queue& q, unsigned int numFrames,
                        uint64_t* outData_svm) {
    auto e = q.submit([&](sycl::handler& h) {
        h.single_task<INV_NTT_OUTPUT>([=]() [[intel::kernel_args_restrict]] {
            intt_output_kernel(numFrames, outData_svm);
        });
    });

    return e;
}

}  // end of extern C
