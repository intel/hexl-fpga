// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ihc_apint.h"
#include "mod_ops.h"
#ifdef __DEBUG_REV_NTT__
#define hwiprintf(...)           \
    printf("\033[1;33m");        \
    printf("\n");                \
    printf("[ ---DEBUG-INTT: "); \
    printf(__VA_ARGS__);         \
    printf(" ]\n");              \
    printf("\033[0m");
#pragma message \
    "============== Building Debug version of INTT for emulation Mode Only ==============\n"
#else
#define hwiprintf(...)
#endif

#define INTT_TASK __attribute__((max_global_work_dim(0)))
#define INTT_HOST_MEM __attribute__((buffer_location("host")))
#define INTT_DEVICE_MEM __attribute__((buffer_location("device")))

#ifndef FPGA_INTT_SIZE
#define FPGA_INTT_SIZE 16384
#endif

#ifndef NUM_INTT_COMPUTE_UNITS
#define NUM_INTT_COMPUTE_UNITS 2
#endif

#ifndef VEC_INTT
#define VEC_INTT 16
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

#pragma OPENCL EXTENSION cl_intel_channels : enable
//#define __FULL_DEPTH_CHANNELS__
#ifndef __FULL_DEPTH_CHANNELS__
//////////////////////////// Optimized depth channels
/////////////////////////////////////////////////////////
channel ulong64Bit modulusChannelINTT[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(16)));
channel ulong64Bit miniBatchChannelINTT[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(16)));
channel WideVecInType inDataChannelINTT[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(16)));
channel WideVecOutType outDataChannelINTT[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(FPGA_INTT_SIZE / VEC_INTT / 2)));
channel WideVecOutType outDataChannelINTT2[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(FPGA_INTT_SIZE / VEC_INTT / 2)));
channel ulong64Bit inv_ni_channel[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(16)));
channel ulong64Bit inv_n_wi_channel[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(16)));
channel Wide64ByteType twiddleFactorsChannelINTT[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(16)));
channel Wide64ByteType barrettTwiddleFactorsChannelINTT[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(16)));
/////////////////////////////////////////////////////////////////////////////////////////////////////////
#else
////////////////////////// Full depth
/// channels//////////////////////////////////////////////////////
channel ulong64Bit modulusChannelINTT[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(FPGA_INTT_SIZE)));
channel uint32Bit miniBatchChannelINTT[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(FPGA_INTT_SIZE)));
channel WideVecInType inDataChannelINTT[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(FPGA_INTT_SIZE)));
channel WideVecOutType outDataChannelINTT[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(FPGA_INTT_SIZE)));
channel WideVecOutType outDataChannelINTT2[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(FPGA_INTT_SIZE)));
channel ulong64Bit inv_ni_channel[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(FPGA_INTT_SIZE)));
channel ulong64Bit inv_n_wi_channel[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(FPGA_INTT_SIZE)));
channel Wide64ByteType twiddleFactorsChannelINTT[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(FPGA_INTT_SIZE)));
channel Wide64ByteType barrettTwiddleFactorsChannelINTT[NUM_INTT_COMPUTE_UNITS]
    __attribute__((depth(FPGA_INTT_SIZE)));
///////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif

INTT_TASK kernel void intt_input_kernel(
    unsigned int numFrames, __global INTT_HOST_MEM ulong64Bit* restrict inData,
    __global INTT_HOST_MEM ulong64Bit* restrict modulus,
    __global INTT_HOST_MEM ulong64Bit* restrict inv_ni,
    __global INTT_HOST_MEM ulong64Bit* restrict inv_n_wi,
    __global INTT_HOST_MEM ulong64Bit* restrict
        twiddleFactors,  // rootsOfUnity,
    __global INTT_HOST_MEM ulong64Bit* restrict
        barrettTwiddleFactors  // precon_root_of_unity

) {
// Create mini batches for every INTT instance
#pragma unroll
    for (int i = 0; i < NUM_INTT_COMPUTE_UNITS; i++) {
        uint32Bit fractionalMiniBatch =
            (numFrames % NUM_INTT_COMPUTE_UNITS) / (i + 1);
        if (fractionalMiniBatch > 0)
            fractionalMiniBatch = 1;
        else
            fractionalMiniBatch = 0;
        uint32Bit miniBatchSize =
            (numFrames / NUM_INTT_COMPUTE_UNITS) + fractionalMiniBatch;
        write_channel_intel(miniBatchChannelINTT[i], miniBatchSize);
    }
    // Assuming the twiddle factors and the complex root of unity are similar
    // broadcast roots of unity to each kernel
    ulong64Bit temp = modulus[0];
#pragma unroll
    for (size_t c = 0; c < NUM_INTT_COMPUTE_UNITS; c++) {
        write_channel_intel(modulusChannelINTT[c], temp);
    }

#pragma unroll
    for (size_t c = 0; c < NUM_INTT_COMPUTE_UNITS; c++) {
        write_channel_intel(inv_ni_channel[c], inv_ni[0]);
    }

#pragma unroll
    for (size_t c = 0; c < NUM_INTT_COMPUTE_UNITS; c++) {
        write_channel_intel(inv_n_wi_channel[c], inv_n_wi[0]);
    }
    const size_t numTwiddlePerWord =
        sizeof(Wide64ByteType) / sizeof(ulong64Bit);
    const unsigned int iterations = FPGA_INTT_SIZE / numTwiddlePerWord;
    for (size_t i = 0; i < iterations; i++) {
        Wide64ByteType tw;
#pragma unroll
        for (size_t j = 0; j < numTwiddlePerWord; j++) {
            tw.data[j] = twiddleFactors[i * numTwiddlePerWord + j];
        }
// broadcast twiddles to all compute units
#pragma unroll
        for (size_t c = 0; c < NUM_INTT_COMPUTE_UNITS; c++) {
            write_channel_intel(twiddleFactorsChannelINTT[c], tw);
        }
    }

    for (size_t i = 0; i < FPGA_INTT_SIZE / numTwiddlePerWord; i++) {
        Wide64ByteType tw;
#pragma unroll
        for (size_t j = 0; j < numTwiddlePerWord; j++) {
            tw.data[j] = barrettTwiddleFactors[i * numTwiddlePerWord + j];
        }
// broadcast twiddles to all compute units
#pragma unroll
        for (size_t c = 0; c < NUM_INTT_COMPUTE_UNITS; c++) {
            write_channel_intel(barrettTwiddleFactorsChannelINTT[c], tw);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////
    // Retrieve one INTT data and stream to different kernels, per iteration for
    // top level loop

    const unsigned int numElementsInVec =
        sizeof(WideVecInType) / sizeof(ulong64Bit);
    for (size_t b = 0; b < numFrames; b++) {
        unsigned int computeUnitIndex = b % NUM_INTT_COMPUTE_UNITS;
        for (size_t i = 0; i < FPGA_INTT_SIZE / numElementsInVec; i++) {
            WideVecInType inVec;
            unsigned long offset = b * FPGA_INTT_SIZE + i * numElementsInVec;
#pragma unroll
            for (size_t j = 0; j < numElementsInVec / 2; j++) {
                inVec.data[j] = inData[offset + j];
            }
#pragma unroll
            for (size_t j = 0; j < numElementsInVec / 2; j++) {
                inVec.data[j + VEC_INTT] =
                    inData[offset + j + numElementsInVec / 2];
            }
            write_channel_intel(inDataChannelINTT[computeUnitIndex], inVec);
        }
    }
}

INTT_TASK kernel void intt_output_kernel(
    unsigned int numFrames,
    __global INTT_HOST_MEM unsigned long* restrict outData) {
    for (unsigned int i = 0; i < numFrames; i++) {
        unsigned int computeUnitIndex = i % NUM_INTT_COMPUTE_UNITS;
        size_t frameOffset = i * FPGA_INTT_SIZE;
        for (unsigned int k = 0; k < FPGA_INTT_SIZE / VEC_INTT; k++) {
            size_t offset = frameOffset + k * VEC_INTT;
            WideVecOutType elements_out;
            if (k < FPGA_INTT_SIZE / VEC_INTT / 2) {
                elements_out =
                    read_channel_intel(outDataChannelINTT[computeUnitIndex]);
            } else {
                elements_out =
                    read_channel_intel(outDataChannelINTT2[computeUnitIndex]);
            }
#pragma unroll
            for (int j = 0; j < VEC_INTT; j++) {
                outData[offset + j] = elements_out.data[j];
            }
        }
    }
}

__attribute__((autorun))
__attribute__((num_compute_units(NUM_INTT_COMPUTE_UNITS))) INTT_TASK kernel void
InverseTransformToBitReverse64_16384() {
    const int computeUnitID = get_compute_id(0);
    int n = (int)FPGA_INTT_SIZE;
    unsigned char Xm_val;
    unsigned int roots_acc;
    int t;
    int logt;

    unsigned long X[FPGA_INTT_SIZE / VEC_INTT][VEC_INTT]
        __attribute__((numbanks(VEC_INTT), max_replicates(2)));
    unsigned long X2[FPGA_INTT_SIZE / VEC_INTT][VEC_INTT]
        __attribute__((numbanks(VEC_INTT), max_replicates(2)));
    unsigned long Xm[FPGA_INTT_SIZE / VEC_INTT];
    unsigned long local_roots[FPGA_INTT_SIZE];
    unsigned long local_precons[FPGA_INTT_SIZE];
    for (int i = 0; i < FPGA_INTT_SIZE / VEC_INTT; i++) {
        Xm[i] = 0;
    }

    ulong64Bit prime;

#pragma disable_loop_pipelining
    while (true) {
        uint32Bit miniBatchSize =
            read_channel_intel(miniBatchChannelINTT[computeUnitID]);
        prime = read_channel_intel(modulusChannelINTT[computeUnitID]);
        ulong64Bit inv_n = read_channel_intel(inv_ni_channel[computeUnitID]);
        ulong64Bit inv_n_w =
            read_channel_intel(inv_n_wi_channel[computeUnitID]);
        unsigned long twice_mod = prime << 1;
        unsigned long output_mod_factor = 1;
        const size_t numTwiddlePerWord =
            sizeof(Wide64ByteType) / sizeof(ulong64Bit);
        for (int i = 0; i < FPGA_INTT_SIZE / numTwiddlePerWord; i++) {
            Wide64ByteType VEC_INTTTwiddle =
                read_channel_intel(twiddleFactorsChannelINTT[computeUnitID]);
            for (size_t j = 0; j < numTwiddlePerWord; ++j) {
                local_roots[i * numTwiddlePerWord + j] =
                    VEC_INTTTwiddle.data[j];
            }
        }

        for (int i = 0; i < FPGA_INTT_SIZE / numTwiddlePerWord; i++) {
            Wide64ByteType VEC_INTTExponent = read_channel_intel(
                barrettTwiddleFactorsChannelINTT[computeUnitID]);
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

// Flights
#pragma ivdep array(X)
#pragma ivdep array(X2)
#pragma ivdep array(Xm)
                for (int k = 0; k < FPGA_INTT_SIZE / 2 / VEC_INTT; k++) {
                    unsigned long curX[VEC_INTT * 2] __attribute__((register));
                    unsigned long curX_rep[VEC_INTT * 2]
                        __attribute__((register));
                    size_t i0 =
                        (k * VEC_INTT + 0) >> logt;  // i is the index of groups
                    size_t j0 = (k * VEC_INTT + 0) &
                                (t - 1);  // j is the position of a group
                    size_t j10 = i0 * 2 * t;

                    // fetch the next VEC_INTTtor if the VEC_INTT index is the
                    // same
                    bool b_same_VEC_INTT =
                        ((j10 + j0) / VEC_INTT) == ((j10 + j0 + t) / VEC_INTT);
                    size_t X_ind = (j10 + j0) / VEC_INTT;
                    size_t Xt_ind = (j10 + j0 + t) / VEC_INTT + b_same_VEC_INTT;

                    bool b_X = Xm[X_ind] == (Xm_val - 1) || Xm_val == 1;
                    bool b_Xt = Xm[Xt_ind] == (Xm_val - 1) || Xm_val == 1;

                    WideVecInType elements_in;
                    if (b_first_stage) {
                        elements_in = read_channel_intel(
                            inDataChannelINTT[computeUnitID]);
                    }

#pragma unroll
                    for (int n = 0; n < VEC_INTT; n++) {
                        size_t i = (k * VEC_INTT + n) >>
                                   logt;  // i is the index of groups
                        size_t j = (k * VEC_INTT + n) &
                                   (t - 1);  // j is the position of a group
                        size_t j1 = i * 2 * t;
                        if (b_first_stage) {
                            curX[n] = elements_in.data[n];
                            curX[n + VEC_INTT] = elements_in.data[n + VEC_INTT];
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
                            const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                            const int Xnt =
                                Xn + ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                            curX_rep[n] = curX[Xn];
                            curX_rep[VEC_INTT + n] = curX[Xnt];
                        }
#if VEC_INTT >= 4
                    } else if (t == 2) {
#pragma unroll
                        for (int n = 0; n < VEC_INTT; n++) {
                            const int cur_t = 2;
                            const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                            const int Xnt =
                                Xn + ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                            curX_rep[n] = curX[Xn];
                            curX_rep[VEC_INTT + n] = curX[Xnt];
                        }
#endif
#if VEC_INTT >= 8
                    } else if (t == 4) {
#pragma unroll
                        for (int n = 0; n < VEC_INTT; n++) {
                            const int cur_t = 4;
                            const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                            const int Xnt =
                                Xn + ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                            curX_rep[n] = curX[Xn];
                            curX_rep[VEC_INTT + n] = curX[Xnt];
                        }
#endif
#if VEC_INTT >= 16
                    } else if (t == 8) {
#pragma unroll
                        for (int n = 0; n < VEC_INTT; n++) {
                            const int cur_t = 8;
                            const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                            const int Xnt =
                                Xn + ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                            curX_rep[n] = curX[Xn];
                            curX_rep[VEC_INTT + n] = curX[Xnt];
                        }
#endif
#if VEC_INTT >= 32
                    } else if (t == 16) {
#pragma unroll
                        for (int n = 0; n < VEC_INTT; n++) {
                            const int cur_t = 16;
                            const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                            const int Xnt =
                                Xn + ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
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
                        int j = (k * VEC_INTT + n) &
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
                        curX[VEC_INTT + n] =
                            MultiplyUIntModLazy4(ty, W_op_precon, W_op, prime);
                    }

                    // reorder back
                    if (t == 1) {
#pragma unroll
                        for (int n = 0; n < VEC_INTT; n++) {
                            const int cur_t = 1;
                            const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                            const int Xnt =
                                Xn + ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                            curX_rep[Xn] = curX[n];
                            curX_rep[Xnt] = curX[VEC_INTT + n];
                        }
#if VEC_INTT >= 4
                    } else if (t == 2) {
#pragma unroll
                        for (int n = 0; n < VEC_INTT; n++) {
                            const int cur_t = 2;
                            const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                            const int Xnt =
                                Xn + ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                            curX_rep[Xn] = curX[n];
                            curX_rep[Xnt] = curX[VEC_INTT + n];
                        }
#endif
#if VEC_INTT >= 8
                    } else if (t == 4) {
#pragma unroll
                        for (int n = 0; n < VEC_INTT; n++) {
                            const int cur_t = 4;
                            const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                            const int Xnt =
                                Xn + ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                            curX_rep[Xn] = curX[n];
                            curX_rep[Xnt] = curX[VEC_INTT + n];
                        }
#endif
#if VEC_INTT >= 16
                    } else if (t == 8) {
#pragma unroll
                        for (int n = 0; n < VEC_INTT; n++) {
                            const int cur_t = 8;
                            const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                            const int Xnt =
                                Xn + ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
                            curX_rep[Xn] = curX[n];
                            curX_rep[Xnt] = curX[VEC_INTT + n];
                        }
#endif
#if VEC_INTT >= 32
                    } else if (t == 16) {
#pragma unroll
                        for (int n = 0; n < VEC_INTT; n++) {
                            const int cur_t = 16;
                            const int Xn = n / cur_t * (2 * cur_t) + n % cur_t;
                            const int Xnt =
                                Xn + ((cur_t < VEC_INTT) ? cur_t : VEC_INTT);
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
            for (int i = 0; i < n / 2 / VEC_INTT; i++) {  // line 564 of hexl
                WideVecOutType out;
                WideVecOutType out2;
#pragma unroll
                for (int j = 0; j < VEC_INTT; j++) {
                    unsigned long x = Xm[i] == Xm_val ? X[i][j] : X2[i][j];
                    unsigned long xt = Xm[(n >> 1) / VEC_INTT + i] == Xm_val
                                           ? X[(n >> 1) / VEC_INTT + i][j]
                                           : X2[(n >> 1) / VEC_INTT + i][j];
                    unsigned long tx = x + xt;
                    unsigned long ty = x + twice_mod - xt;

                    tx = (uint64_t)(tx >= twice_mod) ? (tx - twice_mod) : tx;

                    unsigned long nx = MultiplyUIntModLazy3(tx, inv_n, prime);
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
                write_channel_intel(outDataChannelINTT[computeUnitID], out);
                write_channel_intel(outDataChannelINTT2[computeUnitID], out2);
            }  // end for
        }
    }
    return;
}
