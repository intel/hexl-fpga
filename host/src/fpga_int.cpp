// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

#include "dyadic_multiply_int.h"
#include "fpga.h"
#include "fpga_assert.h"
#include "intt_int.h"
#include "ntt_int.h"

namespace intel {
namespace hexl {
namespace fpga {

static std::mutex muNTT_;
static std::mutex muINTT_;
static std::unordered_set<Object*> outstanding_objects;
static std::unordered_set<Object*> ntt_outstanding_objects;
static std::unordered_set<Object*> intt_outstanding_objects;
static DevicePool* pool;
static std::promise<bool> exit_signal;

static DEV_TYPE get_device() {
    DEV_TYPE d = FPGA;
    char* env = getenv("RUN_CHOICE");
    if (env) {
        int e = atoi(env);
        FPGA_ASSERT((e >= 1) && (e <= 2));
        d = DEV_TYPE(e);
    }
    return d;
}

static int g_choice = get_device();

// DYADIC_MULTIPLY section
static uint64_t get_coeff_size() {
    char* env = getenv("COEFF_SIZE");
    uint64_t size = env ? strtoul(env, NULL, 10) : 16384;
    return size;
}

static uint64_t g_coeff_size = get_coeff_size();

static uint32_t get_modulus_size() {
    char* env = getenv("MODULUS_SIZE");
    uint32_t size = env ? uint32_t(atoi(env)) : 14;
    return size;
}

static uint32_t g_modulus_size = get_modulus_size();

static uint64_t get_batch_size_dyadic_mult() {
    char* env = getenv("BATCH_SIZE_DYADIC_MULTIPLY");
    uint64_t size = env ? strtoul(env, NULL, 10) : 1;
    return size;
}

static uint64_t g_batch_size_dyadic_mult = get_batch_size_dyadic_mult();

static uint64_t get_batch_size_ntt() {
    char* env = getenv("BATCH_SIZE_NTT");
    uint64_t size = env ? strtoul(env, NULL, 10) : 1;
    return size;
}

static uint64_t g_batch_size_ntt = get_batch_size_ntt();

static uint64_t get_batch_size_intt() {
    char* env = getenv("BATCH_SIZE_INTT");
    uint64_t size = env ? strtoul(env, NULL, 10) : 1;
    return size;
}

static uint64_t g_batch_size_intt = get_batch_size_intt();

static uint32_t get_fpga_debug() {
    char* env = getenv("FPGA_DEBUG");
    uint32_t debug = env ? atoi(env) : 0;
    return debug;
}

static uint32_t g_fpga_debug = get_fpga_debug();

static uint32_t get_fpga_bufsize() {
    char* env = getenv("FPGA_BUFSIZE");
    uint32_t bufsize = env ? atoi(env) : 1024;
    return bufsize;
}

static uint32_t g_fpga_bufsize = get_fpga_bufsize();

static Buffer fpga_buffer(g_fpga_bufsize, g_batch_size_dyadic_mult,
                          g_batch_size_ntt, g_batch_size_intt);

void attach_fpga_pooling() {
    std::cout << "Running on FPGA ... " << std::endl;
    exit_signal = std::promise<bool>();
    auto f = exit_signal.get_future();
    pool = new DevicePool(g_choice, fpga_buffer, f, g_coeff_size,
                          g_modulus_size, g_batch_size_dyadic_mult,
                          g_batch_size_ntt, g_batch_size_intt, g_fpga_debug);
}

void detach_fpga_pooling() {
    exit_signal.set_value(true);
    delete pool;
    pool = nullptr;
}

void set_worksize_DyadicMultiply_int(uint64_t n) {
    fpga_buffer.set_worksize_DyadicMultiply(n);
}

static void fpga_dyadic_mult(uint64_t* results, const uint64_t* operand1,
                             const uint64_t* operand2, uint64_t n,
                             const uint64_t* moduli, uint64_t n_moduli) {
    Object* obj = new Object_DyadicMultiply(results, operand1, operand2, n,
                                            moduli, n_moduli);

    fpga_buffer.push(obj);

    outstanding_objects.insert(obj);

    if (fpga_buffer.get_worksize_DyadicMultiply() == 1) {
        DyadicMultiplyCompleted_int();
    }
}

bool DyadicMultiplyCompleted_int() {
    bool all_done = false;
    while (!all_done) {
        bool done = true;
        auto iter = outstanding_objects.begin();
        while (iter != outstanding_objects.end()) {
            Object* obj = *iter;
            if (obj->ready_) {
                delete obj;
                obj = nullptr;
                iter = outstanding_objects.erase(iter);
            } else {
                done = false;
                iter++;
            }
        }
        all_done = done;
    }
    outstanding_objects.clear();

    fpga_buffer.set_worksize_DyadicMultiply(1);

    return all_done;
}

void DyadicMultiply_int(uint64_t* results, const uint64_t* operand1,
                        const uint64_t* operand2, uint64_t n,
                        const uint64_t* moduli, uint64_t n_moduli) {
    switch (g_choice) {
    case EMU:
    case FPGA:
        fpga_dyadic_mult(results, operand1, operand2, n, moduli, n_moduli);
        break;
    default:
        std::cerr << "ERROR: Invalid RUN_CHOICE envvar. Set to a valid "
                     "value {1, or 2}, where 1:EMU, 2:FPGA."
                  << std::endl;
        FPGA_ASSERT(0);
        break;
    }
}

void set_worksize_INTT_int(uint64_t n) { fpga_buffer.set_worksize_INTT(n); }

static void fpga_INTT(uint64_t* coeff_poly,
                      const uint64_t* inv_root_of_unity_powers,
                      const uint64_t* precon_inv_root_of_unity_powers,
                      uint64_t coeff_modulus, uint64_t inv_n, uint64_t inv_n_w,
                      uint64_t n) {
    std::unique_lock<std::mutex> locker(muINTT_);

    Object* obj = new Object_INTT(coeff_poly, inv_root_of_unity_powers,
                                  precon_inv_root_of_unity_powers,
                                  coeff_modulus, inv_n, inv_n_w, n);

    fpga_buffer.push(obj);

    intt_outstanding_objects.insert(obj);

    if (fpga_buffer.get_worksize_INTT() == 1) {
        INTTCompleted_int();
    }
}

bool INTTCompleted_int() {
    bool all_done = false;
    while (!all_done) {
        bool done = true;
        auto iter = intt_outstanding_objects.begin();
        while (iter != intt_outstanding_objects.end()) {
            Object* obj = *iter;
            if (obj->ready_) {
                delete obj;
                obj = nullptr;
                iter = intt_outstanding_objects.erase(iter);
            } else {
                done = false;
                iter++;
            }
        }
        all_done = done;
    }
    intt_outstanding_objects.clear();

    fpga_buffer.set_worksize_INTT(1);

    return all_done;
}

void INTT_int(uint64_t* coeff_poly, const uint64_t* inv_root_of_unity_powers,
              const uint64_t* precon_inv_root_of_unity_powers,
              uint64_t coeff_modulus, uint64_t inv_n, uint64_t inv_n_w,
              uint64_t n) {
    switch (g_choice) {
    case EMU:
    case FPGA:
        fpga_INTT(coeff_poly, inv_root_of_unity_powers,
                  precon_inv_root_of_unity_powers, coeff_modulus, inv_n,
                  inv_n_w, n);
        break;
    default:
        std::cerr << "ERROR: Invalid RUN_CHOICE envvar. Set to a valid "
                     "value {1, or 2}, where 1:EMU, 2:FPGA."
                  << std::endl;
        FPGA_ASSERT(0);
        break;
    }
}

void set_worksize_NTT_int(uint64_t n) { fpga_buffer.set_worksize_NTT(n); }

static void fpga_NTT(uint64_t* coeff_poly, const uint64_t* root_of_unity_powers,
                     const uint64_t* precon_root_of_unity_powers,
                     uint64_t coeff_modulus, uint64_t n) {
    std::unique_lock<std::mutex> locker(muNTT_);

    Object* obj = new Object_NTT(coeff_poly, root_of_unity_powers,
                                 precon_root_of_unity_powers, coeff_modulus, n);

    fpga_buffer.push(obj);

    ntt_outstanding_objects.insert(obj);

    if (fpga_buffer.get_worksize_NTT() == 1) {
        NTTCompleted_int();
    }
}

bool NTTCompleted_int() {
    bool all_done = false;
    while (!all_done) {
        bool done = true;
        auto iter = ntt_outstanding_objects.begin();
        while (iter != ntt_outstanding_objects.end()) {
            Object* obj = *iter;
            if (obj->ready_) {
                delete obj;
                obj = nullptr;
                iter = ntt_outstanding_objects.erase(iter);
            } else {
                done = false;
                iter++;
            }
        }
        all_done = done;
    }
    ntt_outstanding_objects.clear();

    fpga_buffer.set_worksize_NTT(1);

    return all_done;
}

void NTT_int(uint64_t* coeff_poly, const uint64_t* root_of_unity_powers,
             const uint64_t* precon_root_of_unity_powers,
             uint64_t coeff_modulus, uint64_t n) {
    switch (g_choice) {
    case EMU:
    case FPGA:
        fpga_NTT(coeff_poly, root_of_unity_powers, precon_root_of_unity_powers,
                 coeff_modulus, n);
        break;
    default:
        std::cerr << "ERROR: Invalid RUN_CHOICE envvar. Set to a valid "
                     "value {0, 1, or 2}, where 1:EMU, 2:FPGA."
                  << std::endl;
        FPGA_ASSERT(0);
        break;
    }
}
}  // namespace fpga
}  // namespace hexl
}  // namespace intel
