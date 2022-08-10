// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "./utils/pipe_def_marcos.hpp"
#include "./utils/pipe_array.hpp"
#include "./utils/unroller.hpp"

#include "dpc_common.hpp"
#ifndef NUM_NTT_COMPUTE_UNITS
#define NUM_NTT_COMPUTE_UNITS 1
#else
#pragma clang diagnostic warning "Compiling with external NUM_NTT_COMPUTE_UNITS"
#endif

#ifndef VEC
#define VEC 8
#endif
#define REORDER 1
#define PRINT_ROW_RESULT 0

#ifndef FPGA_NTT_SIZE
#define FPGA_NTT_SIZE 16384
#else
#pragma clang diagnostic warning "Compiling with external FPGA_NTT_SIZE"
#endif

#define LHIGH(num, type) \
    ((type)(num) & ~((((type)1) << (sizeof(type) * 8 / 2)) - (type)1))
#define LOW(num, type) \
    ((type)(num) & ((((type)1) << (sizeof(type) * 8 / 2)) - (type)1))
#define HIGH(num, type) ((type)(num) >> (sizeof(type) * 8 / 2))

#define HEXL_FPGA_USE_64BIT_MULT
#ifdef HEXL_FPGA_USE_64BIT_MULT
#pragma clang diagnostic warning "Compiling with HEXL_FPGA_USE_64BIT_MULT"
#endif

typedef uint64_t unsigned64Bits_t;
typedef unsigned int unsigned32Bits_t;
typedef struct {
    unsigned64Bits_t data[VEC * 2];
} elements_in_t;

typedef struct {
    unsigned64Bits_t data[VEC * 2];
} elements_out_t;

typedef struct {
    unsigned64Bits_t data[VEC * 2];
} WideVecType;

typedef struct {
    unsigned64Bits_t data[64 / sizeof(unsigned64Bits_t)];
} Wide64BytesType;

defPipe1d(inDataPipe, WideVecType, 16, NUM_NTT_COMPUTE_UNITS);
defPipe1d(miniBatchSizePipeNTT, unsigned32Bits_t, 16, NUM_NTT_COMPUTE_UNITS);
defPipe1d(outDataPipe, WideVecType, 16, NUM_NTT_COMPUTE_UNITS);
defPipe1d(modulusPipe, unsigned64Bits_t, 16, NUM_NTT_COMPUTE_UNITS);
defPipe1d(twiddleFactorsPipe, Wide64BytesType, 16, NUM_NTT_COMPUTE_UNITS);
defPipe1d(barrettTwiddleFactorsPipe, Wide64BytesType, 16,
          NUM_NTT_COMPUTE_UNITS);

#if 32 == FPGA_NTT_SIZE
#define FPGA_NTT_SIZE_LOG 5
#elif 1024 == FPGA_NTT_SIZE
#define FPGA_NTT_SIZE_LOG 10
#elif 8192 == FPGA_NTT_SIZE
#define FPGA_NTT_SIZE_LOG 13
#elif 16384 == FPGA_NTT_SIZE
#define FPGA_NTT_SIZE_LOG 14
#elif 32768 == FPGA_NTT_SIZE
#define FPGA_NTT_SIZE_LOG 15
#endif

template <size_t idx>
class FWD_NTT;

template <size_t id>
void fwd_ntt_kernel(sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        h.single_task<FWD_NTT<id>>([=]() [[intel::kernel_args_restrict]] {
            [[intel::fpga_memory("BLOCK_RAM")]] [[intel::numbanks(VEC)]] [
                [intel::max_replicates(
                    2)]] unsigned long X[FPGA_NTT_SIZE / VEC][VEC];
            [[intel::fpga_memory("BLOCK_RAM")]] [[intel::numbanks(VEC)]] [
                [intel::max_replicates(
                    2)]] unsigned long X2[FPGA_NTT_SIZE / VEC][VEC];
            [[intel::fpga_memory("BLOCK_RAM")]] [[intel::numbanks(VEC)]] [
                [intel::max_replicates(
                    2)]] unsigned char Xm[FPGA_NTT_SIZE / VEC][VEC];

            unsigned long local_roots[FPGA_NTT_SIZE];
            unsigned long local_precons[FPGA_NTT_SIZE];

            constexpr int computeUnitID = id;
            constexpr size_t numTwiddlePerWord =
                sizeof(Wide64BytesType) / sizeof(unsigned64Bits_t);

            for (int i = 0; i < FPGA_NTT_SIZE / VEC; i++) {
#pragma unroll
                for (int j = 0; j < VEC; j++) {
                    Xm[i][j] = 0;
                }
            }

            while (true) {
                unsigned32Bits_t miniBatchSize =
                    miniBatchSizePipeNTT::PipeAt<computeUnitID>::read();

                for (int i = 0; i < FPGA_NTT_SIZE / numTwiddlePerWord; i++) {
                    Wide64BytesType vecTwiddle =
                        twiddleFactorsPipe::PipeAt<computeUnitID>::read();
#pragma unroll
                    for (size_t j = 0; j < numTwiddlePerWord; ++j) {
                        local_roots[i * numTwiddlePerWord + j] =
                            vecTwiddle.data[j];
                    }
                }

                for (int i = 0; i < FPGA_NTT_SIZE / numTwiddlePerWord; i++) {
                    Wide64BytesType vecExponent =
                        barrettTwiddleFactorsPipe::PipeAt<
                            computeUnitID>::read();
#pragma unroll
                    for (size_t j = 0; j < numTwiddlePerWord; ++j) {
                        local_precons[i * numTwiddlePerWord + j] =
                            vecExponent.data[j];
                    }
                }

                unsigned64Bits_t modulus =
                    modulusPipe::PipeAt<computeUnitID>::read();

                for (int mb = 0; mb < miniBatchSize; mb++) {
                    unsigned64Bits_t coeff_mod = modulus;
                    unsigned64Bits_t twice_mod = modulus << 1;
                    unsigned64Bits_t t = (FPGA_NTT_SIZE >> 1);

                    unsigned int t_log = FPGA_NTT_SIZE_LOG - 1;
                    unsigned char Xm_val = 0;
                    size_t s_index = 0;

                    for (unsigned int m = 1; m < FPGA_NTT_SIZE; m <<= 1) {
                        Xm_val++;
                        [[intel::ivdep(X)]] [[intel::ivdep(X2)]] [[intel::ivdep(
                            Xm)]] for (unsigned int k = 0;
                                       k < FPGA_NTT_SIZE / 2 / VEC; k++) {
                            [[intel::fpga_register]] unsigned long
                                curX[VEC * 2];
                            [[intel::fpga_register]] unsigned long
                                curX2[VEC * 2];
                            [[intel::fpga_register]] unsigned long
                                curX_rep[VEC * 2];
                            [[intel::fpga_register]] unsigned long
                                curX2_rep[VEC * 2];

                            size_t i0 = (k * VEC + 0) >>
                                        t_log;  // i is the index of groups
                            size_t j0 =
                                (k * VEC + 0) &
                                (t - 1);  // j is the position of a group
                            size_t j10 = i0 * 2 * t;
                            // fetch the next vector if the vec index is the
                            // same
                            bool b_same_vec =
                                ((j10 + j0) / VEC) == ((j10 + j0 + t) / VEC);
                            size_t X_ind = (j10 + j0) / VEC;
                            size_t Xt_ind = (j10 + j0 + t) / VEC + b_same_vec;

                            WideVecType elements_in;
                            if (m == 1) {
                                elements_in =
                                    inDataPipe::PipeAt<computeUnitID>::read();
                            }

#pragma unroll
                            for (int n = 0; n < VEC; n++) {
                                size_t i = (k * VEC + n) >>
                                           t_log;  // i is the index of groups
                                size_t j =
                                    (k * VEC + n) &
                                    (t - 1);  // j is the position of a group
                                size_t j1 = i * 2 * t;
                                if (m == 1) {
                                    curX[n] = elements_in.data[n];
                                    curX[n + VEC] = elements_in.data[VEC + n];
                                } else {
                                    curX[n] = X[X_ind][n];
                                    curX[n + VEC] = X[Xt_ind][n];
                                    curX2[n] = X2[X_ind][n];
                                    curX2[n + VEC] = X2[Xt_ind][n];
                                }
                            }

                            WideVecType elements_out;
                            if (t == 1) {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    const int cur_t = 1;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn + ((cur_t < VEC) ? cur_t : VEC);
                                    curX_rep[n] = curX[Xn];
                                    curX2_rep[n] = curX2[Xn];
                                    curX_rep[VEC + n] = curX[Xnt];
                                    curX2_rep[VEC + n] = curX2[Xnt];
                                }
#if VEC >= 4
                            } else if (t == 2) {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    const int cur_t = 2;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn + ((cur_t < VEC) ? cur_t : VEC);
                                    curX_rep[n] = curX[Xn];
                                    curX2_rep[n] = curX2[Xn];
                                    curX_rep[VEC + n] = curX[Xnt];
                                    curX2_rep[VEC + n] = curX2[Xnt];
                                }
#endif
#if VEC >= 8
                            } else if (t == 4) {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    const int cur_t = 4;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn + ((cur_t < VEC) ? cur_t : VEC);
                                    curX_rep[n] = curX[Xn];
                                    curX2_rep[n] = curX2[Xn];
                                    curX_rep[VEC + n] = curX[Xnt];
                                    curX2_rep[VEC + n] = curX2[Xnt];
                                }
#endif
#if VEC >= 16
                            } else if (t == 8) {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    const int cur_t = 8;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn + ((cur_t < VEC) ? cur_t : VEC);
                                    curX_rep[n] = curX[Xn];
                                    curX2_rep[n] = curX2[Xn];
                                    curX_rep[VEC + n] = curX[Xnt];
                                    curX2_rep[VEC + n] = curX2[Xnt];
                                }
#endif
#if VEC >= 32
                            } else if (t == 16) {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    const int cur_t = 16;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn + ((cur_t < VEC) ? cur_t : VEC);
                                    curX_rep[n] = curX[Xn];
                                    curX2_rep[n] = curX2[Xn];
                                    curX_rep[VEC + n] = curX[Xnt];
                                    curX2_rep[VEC + n] = curX2[Xnt];
                                }
#endif
                            } else {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    curX_rep[n] = curX[n];
                                    curX2_rep[n] = curX2[n];
                                    curX_rep[VEC + n] = curX[VEC + n];
                                    curX2_rep[VEC + n] = curX2[VEC + n];
                                }
                            }
#pragma unroll
                            for (int n = 0; n < VEC; n++) {
                                size_t i = (k * VEC + n) >>
                                           t_log;  // i is the index of groups
                                size_t j =
                                    (k * VEC + n) &
                                    (t - 1);  // j is the position of a group
                                size_t j1 = i * 2 * t;
                                const unsigned long W_op = local_roots[m + i];
                                const unsigned long W_precon =
                                    local_precons[m + i];

                                unsigned long tx;
                                unsigned long Q;
                                unsigned long a, b;
                                unsigned long a_0, b_0, a_1, b_1;
                                unsigned long a_0b_0, a_0b_1, a_1b_0, a_1b_1;
                                unsigned long a0b0_1, a0b1_0, a1b0_0, a0b1_1,
                                    a1b0_1;

#if REORDER
                                const int Xn = n / t * (2 * t) + n % t;
                                unsigned long tx1 = curX_rep[n];
                                unsigned long tx2 = curX2_rep[n];
#else
                                unsigned long tx1 =
                                X[(j1 + j) / VEC][(j1 + j) % VEC];
                                unsigned long tx2 =
                                X2[(j1 + j) / VEC][(j1 + j) % VEC];
#endif

#if REORDER
                                // anyplace in current 2 VEC is equal
                                bool b_X =
                                    Xm[(j1 + j) / VEC][n] == (Xm_val - 1) ||
                                    m == 1;
#else
                                bool b_X =
                                    Xm[(j1 + j) / VEC][(j1 + j) % VEC] ==
                                     (Xm_val - 1);
#endif
                                tx = b_X ? tx1 : tx2;
                                if (tx >= twice_mod) tx -= twice_mod;

#if REORDER
                                int Xnt = Xn + ((t < VEC) ? t : VEC);
                                unsigned long a1 = curX_rep[VEC + n];
                                unsigned long a2 = curX2_rep[VEC + n];
#else
                                unsigned long a1 =
                                    X[(j1 + j + t) / VEC][(j1 + j + t) % VEC];
                                unsigned long a2 =
                                    X2[(j1 + j + t) / VEC][(j1 + j + t) % VEC];
#endif
                                a = b_X ? a1 : a2;
                                b = W_precon;
                                a_0 = LOW(a, unsigned long);
                                b_1 = HIGH(b, unsigned long);
                                b_0 = LOW(b, unsigned long);
                                a_1 = HIGH(a, unsigned long);
                                a_0b_0 = a_0 * b_0;
                                a_0b_1 = a_0 * b_1;
                                a_1b_0 = a_1 * b_0;
                                a_1b_1 = a_1 * b_1;
                                a0b0_1 = HIGH(a_0b_0, unsigned long);
                                a0b1_0 = LOW(a_1b_0, unsigned long);
                                a1b0_0 = LOW(a_0b_1, unsigned long);
                                a0b1_1 = HIGH(a_1b_0, unsigned long);
                                a1b0_1 = HIGH(a_0b_1, unsigned long);
                                unsigned long m2 = a0b0_1 + a0b1_0 + a1b0_0;
                                unsigned long m_1 = HIGH(m2, unsigned long);
                                unsigned long c_1 =
                                    a_1b_1 + a0b1_1 + a1b0_1 + m_1;
                                Q = W_op * a - c_1 * coeff_mod;

#if REORDER
                                // curX[Xn] = tx + Q;
                                // curX[Xnt] = tx + twice_mod - Q;
                                curX[n] = tx + Q;
                                curX[VEC + n] = tx + twice_mod - Q;
#else
                                X[(j1 + j) / VEC][(j1 + j) % VEC] = tx + Q;
                                X2[(j1 + j + t) / VEC][(j1 + j + t) % VEC] =
                                    tx + twice_mod - Q;
                                Xm[(j1 + j) / VEC][(j1 + j) % VEC] = Xm_val;
#endif
                                // the last outer loop, t == 1
                                if (m == (FPGA_NTT_SIZE / 2)) {
                                    unsigned long val = tx + Q;
                                    if (val >= twice_mod) {
                                        val -= twice_mod;
                                    }
                                    if (val >= coeff_mod) {
                                        val -= coeff_mod;
                                    }
                                    elements_out.data[n * 2] = val;
                                    unsigned long val2 = tx + twice_mod - Q;
                                    if (val2 >= twice_mod) {
                                        val2 -= twice_mod;
                                    }
                                    if (val2 >= coeff_mod) {
                                        val2 -= coeff_mod;
                                    }
                                    elements_out.data[n * 2 + 1] = val2;
                                }
                            }

                            // reorder back
                            if (t == 1) {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    const int cur_t = 1;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn + ((cur_t < VEC) ? cur_t : VEC);
                                    curX_rep[Xn] = curX[n];
                                    curX2_rep[Xn] = curX2[n];
                                    curX_rep[Xnt] = curX[VEC + n];
                                    curX2_rep[Xnt] = curX2[VEC + n];
                                }
#if VEC >= 4
                            } else if (t == 2) {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    const int cur_t = 2;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn + ((cur_t < VEC) ? cur_t : VEC);
                                    curX_rep[Xn] = curX[n];
                                    curX2_rep[Xn] = curX2[n];
                                    curX_rep[Xnt] = curX[VEC + n];
                                    curX2_rep[Xnt] = curX2[VEC + n];
                                }
#endif
#if VEC >= 8
                            } else if (t == 4) {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    const int cur_t = 4;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn + ((cur_t < VEC) ? cur_t : VEC);
                                    curX_rep[Xn] = curX[n];
                                    curX2_rep[Xn] = curX2[n];
                                    curX_rep[Xnt] = curX[VEC + n];
                                    curX2_rep[Xnt] = curX2[VEC + n];
                                }
#endif
#if VEC >= 16
                            } else if (t == 8) {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    const int cur_t = 8;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn + ((cur_t < VEC) ? cur_t : VEC);
                                    curX_rep[Xn] = curX[n];
                                    curX2_rep[Xn] = curX2[n];
                                    curX_rep[Xnt] = curX[VEC + n];
                                    curX2_rep[Xnt] = curX2[VEC + n];
                                }
#endif
#if VEC >= 32
                            } else if (t == 16) {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    const int cur_t = 16;
                                    const int Xn =
                                        n / cur_t * (2 * cur_t) + n % cur_t;
                                    const int Xnt =
                                        Xn + ((cur_t < VEC) ? cur_t : VEC);
                                    curX_rep[Xn] = curX[n];
                                    curX2_rep[Xn] = curX2[n];
                                    curX_rep[Xnt] = curX[VEC + n];
                                    curX2_rep[Xnt] = curX2[VEC + n];
                                }
#endif
                            } else {
#pragma unroll
                                for (int n = 0; n < VEC; n++) {
                                    curX_rep[n] = curX[n];
                                    curX2_rep[n] = curX2[n];
                                    curX_rep[VEC + n] = curX[VEC + n];
                                    curX2_rep[VEC + n] = curX2[VEC + n];
                                }
                            }

                            if (m == (FPGA_NTT_SIZE / 2)) {
                                s_index = k * (VEC * 2);
                                outDataPipe::PipeAt<computeUnitID>::write(
                                    elements_out);
                            }
#pragma unroll
                            for (int n = 0; n < VEC; n++) {
#if REORDER
                                X[X_ind][n] = curX_rep[n];
#if PRINT_ROW_RESULT
                                x[Xt_ind][n] = curX[n + VEC];
#endif
                                X2[Xt_ind][n] = curX_rep[n + VEC];
                                Xm[X_ind][n] = Xm_val;
#endif
                            }
                        }

                        t >>= 1;
                        t_log -= 1;
                    }
                }
            }
        });
    });
}

void ntt_input_kernel(unsigned int numFrames, uint64_t* k_inData,
                      uint64_t* k_inData2, uint64_t* k_modulus,
                      uint64_t* k_twiddleFactors,
                      uint64_t* k_barrettTwiddleFactors) {
    sycl::host_ptr<uint64_t> inData(k_inData);
    sycl::host_ptr<uint64_t> inData2(k_inData2);
    sycl::host_ptr<uint64_t> modulus(k_modulus);
    sycl::host_ptr<uint64_t> twiddleFactors(k_twiddleFactors);
    sycl::host_ptr<uint64_t> barrettTwiddleFactors(k_barrettTwiddleFactors);

    // Boardcast send miniBatchSize to each NTT autorun kernel instances
    Unroller<0, NUM_NTT_COMPUTE_UNITS>::Step([&](auto i) {
        unsigned32Bits_t fractionalMiniBatch =
            (numFrames % NUM_NTT_COMPUTE_UNITS) / (i + 1);
        if (fractionalMiniBatch > 0)
            fractionalMiniBatch = 1;
        else
            fractionalMiniBatch = 0;
        unsigned32Bits_t miniBatchSize =
            (numFrames / NUM_NTT_COMPUTE_UNITS) + fractionalMiniBatch;
        miniBatchSizePipeNTT::PipeAt<i>::write(miniBatchSize);
    });

    // Assuming the twiddle factors and the complex root of unity are similar
    // distribute roots of unity to each kernel
    constexpr size_t numTwiddlePerWord =
        sizeof(Wide64BytesType) / sizeof(unsigned64Bits_t);
    constexpr unsigned int iterations = FPGA_NTT_SIZE / numTwiddlePerWord;

    for (size_t i = 0; i < iterations; i++) {
        Wide64BytesType tw;
#pragma unroll
        for (size_t j = 0; j < numTwiddlePerWord; j++) {
            tw.data[j] = twiddleFactors[i * numTwiddlePerWord + j];
        }

        // broadcast twiddles to all compute units
        Unroller<0, NUM_NTT_COMPUTE_UNITS>::Step(
            [&](auto c) { twiddleFactorsPipe::PipeAt<c>::write(tw); });
    }

    for (size_t i = 0; i < FPGA_NTT_SIZE / numTwiddlePerWord; i++) {
        Wide64BytesType tw;
#pragma unroll
        for (size_t j = 0; j < numTwiddlePerWord; j++) {
            tw.data[j] = barrettTwiddleFactors[i * numTwiddlePerWord + j];
        }

        // broadcast twiddles to all compute units
        Unroller<0, NUM_NTT_COMPUTE_UNITS>::Step(
            [&](auto c) { barrettTwiddleFactorsPipe::PipeAt<c>::write(tw); });
    }

    // boardcast modulus to all the kernels
    unsigned64Bits_t mod = modulus[0];
    Unroller<0, NUM_NTT_COMPUTE_UNITS>::Step(
        [&](auto c) { modulusPipe::PipeAt<c>::write(mod); });

    ////////////////////////////////////////////////////////////////////////////////////
    // Retrieve one NTT data and stream to different kernels, per iteration for
    // top level loop

    constexpr unsigned int numElementsInVec =
        sizeof(WideVecType) / sizeof(unsigned64Bits_t);
    Unroller<0, NUM_NTT_COMPUTE_UNITS>::Step([&](auto computeUnitID) {
        for (unsigned int b = 0; b < numFrames; b++) {
            if (b % NUM_NTT_COMPUTE_UNITS == computeUnitID) {
                for (size_t i = 0; i < FPGA_NTT_SIZE / numElementsInVec; i++) {
                    WideVecType inVec;
                    unsigned long offset = b * FPGA_NTT_SIZE + i * VEC;
#pragma unroll
                    for (size_t j = 0; j < VEC; j++) {
                        inVec.data[j] = inData[offset + j];
                        inVec.data[j + VEC] =
                            inData2[offset + FPGA_NTT_SIZE / 2 + j];
                    }
                    inDataPipe::PipeAt<computeUnitID>::write(inVec);
                }
            }
        }
    });
}

void ntt_output_kernel(int numFrames, uint64_t* k_outData) {
    sycl::host_ptr<uint64_t> outData(k_outData);

    constexpr unsigned int numElementsInVec =
        sizeof(WideVecType) / sizeof(unsigned64Bits_t);

    Unroller<0, NUM_NTT_COMPUTE_UNITS>::Step([&](auto computeUnitID) {
        for (size_t b = 0; b < numFrames; b++) {
            if (b % NUM_NTT_COMPUTE_UNITS == computeUnitID) {
                for (size_t i = 0; i < FPGA_NTT_SIZE / numElementsInVec; i++) {
                    WideVecType oVec =
                        outDataPipe::PipeAt<computeUnitID>::read();
                    unsigned long offset =
                        b * FPGA_NTT_SIZE + i * numElementsInVec;
#pragma unroll
                    for (size_t j = 0; j < numElementsInVec; j++) {
                        outData[offset + j] = oVec.data[j];
                    }
                }
            }
        }
    });
}

class FWD_NTT_INPUT;
class FWD_NTT_OUTPUT;

extern "C" {

// fwd ntt interface, aligned with hexl-fpga fwd_ntt.cl file.

/**
 * @brief submit 4 ntt kernel compute units in default. e.g.
 * NUM_NTT_COMPUTE_UNITS = 4
 *
 * @param q
 */
void fwd_ntt(sycl::queue& q) {
    Unroller<0, NUM_NTT_COMPUTE_UNITS>::Step(
        [&](auto idx) { fwd_ntt_kernel<idx>(q); });
}

sycl::event ntt_input(sycl::queue& q, unsigned int numFrames, uint64_t* inData,
                      uint64_t* inData2, uint64_t* modulus,
                      uint64_t* twiddleFactors,
                      uint64_t* barrettTwiddleFactors) {
    auto e = q.submit([&](sycl::handler& h) {
        h.single_task<FWD_NTT_INPUT>([=]() [[intel::kernel_args_restrict]] {
            ntt_input_kernel(numFrames, inData, inData2, modulus,
                             twiddleFactors, barrettTwiddleFactors);
        });
    });

    return e;
}

sycl::event ntt_output(sycl::queue& q, int numFrames,
                       uint64_t* outData_in_svm) {
    auto e = q.submit([&](sycl::handler& h) {
        h.single_task<FWD_NTT_OUTPUT>([=]() [[intel::kernel_args_restrict]] {
            ntt_output_kernel(numFrames, outData_in_svm);
        });
    });
    return e;
}
}  // end of extern "C"
