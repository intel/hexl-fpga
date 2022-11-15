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
#include "keyswitch_int.h"
#include "ntt_int.h"
#include "multlowlvl_int.h"

#ifdef FPGA_USE_INTEL_HEXL
#include "hexl/hexl.hpp"
#endif

namespace intel {
namespace hexl {
namespace fpga {

static std::mutex muNTT;
static std::mutex muINTT;
static std::mutex muDyadicMultiply;
static std::mutex muKeySwitch;
static std::mutex muMultLowLvl;
static std::unordered_set<Object*> outstanding_objects_DyadicMultiply;
static std::unordered_set<Object*> outstanding_objects_NTT;
static std::unordered_set<Object*> outstanding_objects_INTT;
static std::unordered_set<Object*> outstanding_objects_KeySwitch;
static std::unordered_set<Object*> outstanding_objects_MultLowLvl;
static DevicePool* pool;
static std::promise<bool> exit_signal;

static DEV_TYPE get_device() {
    DEV_TYPE d = FPGA;
    char* env = getenv("RUN_CHOICE");
    if (env) {
        int e = atoi(env);
        FPGA_ASSERT((e >= 0) && (e <= 2));
        d = DEV_TYPE(e);
    }
    switch (d) {
    case CPU:
        std::cout << "Running using HEXL CPU ..." << std::endl;
        break;
    case EMU:
        std::cout << "Running using FPGA Emulator ..." << std::endl;
        break;
    case FPGA:
        std::cout << "Running using Physical FPGA device accelerator ..."
                  << std::endl;
        break;
    default:
        FPGA_ASSERT(0);
        break;
    }
    return d;
}

static int g_choice = get_device();

// DYADIC_MULTIPLY section
static uint64_t get_coeff_size() {
    char* env = getenv("COEFF_SIZE");
    // uint64_t size = env ? strtoul(env, NULL, 10) : 16384;
    uint64_t size = env ? strtoul(env, NULL, 10) : 65536;
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

static uint64_t get_batch_size_KeySwitch() {
    char* env = getenv("BATCH_SIZE_KEYSWITCH");
    uint64_t size = env ? strtoul(env, NULL, 10) : 1;
    if (size > 1024) {
        std::cerr << "Error: BATCH_SIZE_KEYSWITCH is " << size << std::endl;
        std::cerr << "       Maxiaml supported BATCH_SIZE_KEYSWITCH is 1024."
                  << std::endl;
        exit(1);
    }
    return size;
}

static uint64_t g_batch_size_KeySwitch = get_batch_size_KeySwitch();

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
                          g_batch_size_ntt, g_batch_size_intt,
                          g_batch_size_KeySwitch, 1);

void attach_fpga_pooling() {
    if (g_choice == CPU) {
        return;
    }
    std::cout << "Running on FPGA: Creating Static FPGA Device Context ... "
              << std::endl;
    exit_signal = std::promise<bool>();
    auto f = exit_signal.get_future();
    pool =
        new DevicePool(g_choice, fpga_buffer, f, g_coeff_size, g_modulus_size,
                       g_batch_size_dyadic_mult, g_batch_size_ntt,
                       g_batch_size_intt, g_batch_size_KeySwitch, 1, g_fpga_debug);
}

void detach_fpga_pooling() {
    if (g_choice == CPU) {
        return;
    }
    exit_signal.set_value(true);
    delete pool;
    pool = nullptr;
}

void set_worksize_DyadicMultiply_int(uint64_t n) {
    fpga_buffer.set_worksize_DyadicMultiply(n);
}

static void fpga_DyadicMultiply(uint64_t* results, const uint64_t* operand1,
                                const uint64_t* operand2, uint64_t n,
                                const uint64_t* moduli, uint64_t n_moduli) {
    std::lock_guard<std::mutex> locker(muDyadicMultiply);

    bool fence = (fpga_buffer.size() == 0);

    if (!fence) {
        Object* obj = fpga_buffer.back();
        fence |= (obj->type_ != kernel_t::DYADIC_MULTIPLY);
    }

    Object* obj = new Object_DyadicMultiply(results, operand1, operand2, n,
                                            moduli, n_moduli, fence);

    fpga_buffer.push(obj);

    outstanding_objects_DyadicMultiply.insert(obj);

    if (fpga_buffer.get_worksize_DyadicMultiply() == 1) {
        DyadicMultiplyCompleted_int();
    }
}

static void cpu_DyadicMultiply(uint64_t* results, const uint64_t* operand1,
                               const uint64_t* operand2, uint64_t n,
                               const uint64_t* moduli, uint64_t n_moduli) {
    FPGA_ASSERT(g_choice == CPU);

#ifdef FPGA_USE_INTEL_HEXL
    namespace ns = intel::hexl::internal;
    ns::DyadicMultiply(results, operand1, operand2, n, moduli, n_moduli);
#else
    std::cerr << "HEXL CPU version not supported" << std::endl;
    exit(1);
#endif
}

bool DyadicMultiplyCompleted_int() {
    bool all_done = false;
    while (!all_done) {
        bool done = true;
        auto iter = outstanding_objects_DyadicMultiply.begin();
        while (iter != outstanding_objects_DyadicMultiply.end()) {
            Object* obj = *iter;
            if (obj->ready_) {
                delete obj;
                obj = nullptr;
                iter = outstanding_objects_DyadicMultiply.erase(iter);
            } else {
                done = false;
                iter++;
            }
        }
        all_done = done;
    }
    outstanding_objects_DyadicMultiply.clear();

    fpga_buffer.set_worksize_DyadicMultiply(1);

    return all_done;
}

void DyadicMultiply_int(uint64_t* results, const uint64_t* operand1,
                        const uint64_t* operand2, uint64_t n,
                        const uint64_t* moduli, uint64_t n_moduli) {
    switch (g_choice) {
    case CPU:
        cpu_DyadicMultiply(results, operand1, operand2, n, moduli, n_moduli);
        break;
    case EMU:
    case FPGA:
        fpga_DyadicMultiply(results, operand1, operand2, n, moduli, n_moduli);
        break;
    default:
        std::cerr << "ERROR: Invalid RUN_CHOICE envvar. Set to a valid "
                     "value {0, 1, or 2}, where 0:CPU, 1:EMU, 2:FPGA."
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
    std::lock_guard<std::mutex> locker(muINTT);

    bool fence = (fpga_buffer.size() == 0);

    if (!fence) {
        Object* obj = fpga_buffer.back();
        fence |= (obj->type_ != kernel_t::INTT);
        if (!fence) {
            FPGA_ASSERT(obj->type_ == kernel_t::INTT);
            Object_INTT* obj_INTT = dynamic_cast<Object_INTT*>(obj);
            fence |= (coeff_modulus != obj_INTT->coeff_modulus_);
        }
    }

    Object* obj = new Object_INTT(coeff_poly, inv_root_of_unity_powers,
                                  precon_inv_root_of_unity_powers,
                                  coeff_modulus, inv_n, inv_n_w, n, fence);

    fpga_buffer.push(obj);

    outstanding_objects_INTT.insert(obj);

    if (fpga_buffer.get_worksize_INTT() == 1) {
        INTTCompleted_int();
    }
}

bool INTTCompleted_int() {
    bool all_done = false;
    while (!all_done) {
        bool done = true;
        auto iter = outstanding_objects_INTT.begin();
        while (iter != outstanding_objects_INTT.end()) {
            Object* obj = *iter;
            if (obj->ready_) {
                delete obj;
                obj = nullptr;
                iter = outstanding_objects_INTT.erase(iter);
            } else {
                done = false;
                iter++;
            }
        }
        all_done = done;
    }
    outstanding_objects_INTT.clear();

    fpga_buffer.set_worksize_INTT(1);

    return all_done;
}

void INTT_int(uint64_t* coeff_poly, const uint64_t* inv_root_of_unity_powers,
              const uint64_t* precon_inv_root_of_unity_powers,
              uint64_t coeff_modulus, uint64_t inv_n, uint64_t inv_n_w,
              uint64_t n) {
    switch (g_choice) {
    case CPU:
        std::cerr << "HEXL CPU version not supported" << std::endl;
        FPGA_ASSERT(0);
        break;
    case EMU:
    case FPGA:
        fpga_INTT(coeff_poly, inv_root_of_unity_powers,
                  precon_inv_root_of_unity_powers, coeff_modulus, inv_n,
                  inv_n_w, n);
        break;
    default:
        std::cerr << "ERROR: Invalid RUN_CHOICE envvar. Set to a valid "
                     "value {0, 1, or 2}, where 0:CPU, 1:EMU, 2:FPGA."
                  << std::endl;
        FPGA_ASSERT(0);
        break;
    }
}

void set_worksize_NTT_int(uint64_t n) { fpga_buffer.set_worksize_NTT(n); }

static void fpga_NTT(uint64_t* coeff_poly, const uint64_t* root_of_unity_powers,
                     const uint64_t* precon_root_of_unity_powers,
                     uint64_t coeff_modulus, uint64_t n) {
    std::lock_guard<std::mutex> locker(muNTT);

    bool fence = (fpga_buffer.size() == 0);

    if (!fence) {
        Object* obj = fpga_buffer.back();
        fence |= (obj->type_ != kernel_t::NTT);
        if (!fence) {
            FPGA_ASSERT(obj->type_ == kernel_t::NTT);
            Object_NTT* obj_NTT = dynamic_cast<Object_NTT*>(obj);
            fence |= (coeff_modulus != obj_NTT->coeff_modulus_);
        }
    }

    Object* obj =
        new Object_NTT(coeff_poly, root_of_unity_powers,
                       precon_root_of_unity_powers, coeff_modulus, n, fence);

    fpga_buffer.push(obj);

    outstanding_objects_NTT.insert(obj);

    if (fpga_buffer.get_worksize_NTT() == 1) {
        NTTCompleted_int();
    }
}

bool NTTCompleted_int() {
    bool all_done = false;
    while (!all_done) {
        bool done = true;
        auto iter = outstanding_objects_NTT.begin();
        while (iter != outstanding_objects_NTT.end()) {
            Object* obj = *iter;
            if (obj->ready_) {
                delete obj;
                obj = nullptr;
                iter = outstanding_objects_NTT.erase(iter);
            } else {
                done = false;
                iter++;
            }
        }
        all_done = done;
    }
    outstanding_objects_NTT.clear();

    fpga_buffer.set_worksize_NTT(1);

    return all_done;
}

void NTT_int(uint64_t* coeff_poly, const uint64_t* root_of_unity_powers,
             const uint64_t* precon_root_of_unity_powers,
             uint64_t coeff_modulus, uint64_t n) {
    switch (g_choice) {
    case CPU:
        std::cerr << "HEXL CPU version not supported" << std::endl;
        FPGA_ASSERT(0);
        break;
    case EMU:
    case FPGA:
        fpga_NTT(coeff_poly, root_of_unity_powers, precon_root_of_unity_powers,
                 coeff_modulus, n);
        break;
    default:
        std::cerr << "ERROR: Invalid RUN_CHOICE envvar. Set to a valid "
                     "value {0, 1, or 2}, where 0:CPU, 1:EMU, 2:FPGA."
                  << std::endl;
        FPGA_ASSERT(0);
        break;
    }
}

void set_worksize_KeySwitch_int(uint64_t n) {
    fpga_buffer.set_worksize_KeySwitch(n);
}

static void fpga_KeySwitch(uint64_t* result, const uint64_t* t_target_iter_ptr,
                           uint64_t n, uint64_t decomp_modulus_size,
                           uint64_t key_modulus_size, uint64_t rns_modulus_size,
                           uint64_t key_component_count, const uint64_t* moduli,
                           const uint64_t** k_switch_keys,
                           const uint64_t* modswitch_factors,
                           const uint64_t* twiddle_factors) {
    std::lock_guard<std::mutex> locker(muKeySwitch);

    bool fence = (fpga_buffer.size() == 0);

    if (!fence) {
        Object* obj = fpga_buffer.back();
        FPGA_ASSERT(obj);
        fence |= (obj->type_ != kernel_t::KEYSWITCH);
        if (!fence) {
            FPGA_ASSERT(obj->type_ == kernel_t::KEYSWITCH);
            Object_KeySwitch* obj_KeySwitch =
                dynamic_cast<Object_KeySwitch*>(obj);
            fence |= (n != obj_KeySwitch->n_);
            fence |=
                (decomp_modulus_size != obj_KeySwitch->decomp_modulus_size_);
            fence |= (key_modulus_size != obj_KeySwitch->key_modulus_size_);
            fence |= (rns_modulus_size != obj_KeySwitch->rns_modulus_size_);
            fence |=
                (key_component_count != obj_KeySwitch->key_component_count_);
            fence |= (moduli != obj_KeySwitch->moduli_);
            fence |= (k_switch_keys != obj_KeySwitch->k_switch_keys_);
            fence |= (modswitch_factors != obj_KeySwitch->modswitch_factors_);
        }
    }

    Object* obj = new Object_KeySwitch(
        result, t_target_iter_ptr, n, decomp_modulus_size, key_modulus_size,
        rns_modulus_size, key_component_count, moduli, k_switch_keys,
        modswitch_factors, twiddle_factors, fence);

    fpga_buffer.push(obj);

    outstanding_objects_KeySwitch.insert(obj);

    if (fpga_buffer.get_worksize_KeySwitch() == 1) {
        KeySwitchCompleted_int();
    }
}

static void cpu_KeySwitch(uint64_t* result, const uint64_t* t_target_iter_ptr,
                          uint64_t n, uint64_t decomp_modulus_size,
                          uint64_t key_modulus_size, uint64_t rns_modulus_size,
                          uint64_t key_component_count, const uint64_t* moduli,
                          const uint64_t** k_switch_keys,
                          const uint64_t* modswitch_factors,
                          const uint64_t* twiddle_factors) {
    FPGA_ASSERT(g_choice == CPU);

#ifdef FPGA_USE_INTEL_HEXL
    namespace ns = intel::hexl::internal;
    ns::KeySwitch(result, t_target_iter_ptr, n, decomp_modulus_size,
                  key_modulus_size, rns_modulus_size, key_component_count,
                  moduli, k_switch_keys, modswitch_factors, twiddle_factors);
#else
    std::cerr << "HEXL CPU version not supported" << std::endl;
    exit(1);
#endif
}

bool KeySwitchCompleted_int() {
    bool all_done = false;
    while (!all_done) {
        bool done = true;
        auto iter = outstanding_objects_KeySwitch.begin();
        while (iter != outstanding_objects_KeySwitch.end()) {
            Object* obj = *iter;
            if (obj->ready_) {
                delete obj;
                obj = nullptr;
                iter = outstanding_objects_KeySwitch.erase(iter);
            } else {
                done = false;
                iter++;
            }
        }
        all_done = done;
    }
    outstanding_objects_KeySwitch.clear();

    fpga_buffer.set_worksize_KeySwitch(1);

    return all_done;
}

void KeySwitch_int(uint64_t* result, const uint64_t* t_target_iter_ptr,
                   uint64_t n, uint64_t decomp_modulus_size,
                   uint64_t key_modulus_size, uint64_t rns_modulus_size,
                   uint64_t key_component_count, const uint64_t* moduli,
                   const uint64_t** k_switch_keys,
                   const uint64_t* modswitch_factors,
                   const uint64_t* twiddle_factors) {
    switch (g_choice) {
    case CPU:
        cpu_KeySwitch(result, t_target_iter_ptr, n, decomp_modulus_size,
                      key_modulus_size, rns_modulus_size, key_component_count,
                      moduli, k_switch_keys, modswitch_factors,
                      twiddle_factors);
        break;
    case EMU:
    case FPGA:
        fpga_KeySwitch(result, t_target_iter_ptr, n, decomp_modulus_size,
                       key_modulus_size, rns_modulus_size, key_component_count,
                       moduli, k_switch_keys, modswitch_factors,
                       twiddle_factors);
        break;
    default:
        std::cerr << "ERROR: Invalid RUN_CHOICE envvar. Set to a valid "
                     "value {0, 1, or 2}, where 0:CPU, 1:EMU, 2:FPGA."
                  << std::endl;
        FPGA_ASSERT(0);
        break;
    }
}


static void fpga_MultLowLvl(uint64_t* a0, uint64_t* a1, uint64_t a_primes_size, uint8_t* a_primes_index,
                            uint64_t* b0, uint64_t* b1, uint64_t b_primes_size, uint8_t* b_primes_index,
                            uint64_t plainText, uint64_t coeff_count, 
                            uint64_t* c0, uint64_t* c1, uint64_t* c2, uint64_t c_primes_size, 
                            uint8_t* output_primes_index, uint64_t* primes, uint64_t primes_size) {
    bool fence = (fpga_buffer.size() == 0);
    
    if (!fence) {
        Object* obj = fpga_buffer.back();
        FPGA_ASSERT(obj);
        fence |= (obj->type_ != kernel_t::MULTLOWLVL);
        std::cout << "kernel type = " << (unsigned long)kernel_t::MULTLOWLVL << std::endl;;
        if (!fence) {
            FPGA_ASSERT(obj->type_ == kernel_t::MULTLOWLVL);
            Object_MultLowLvl* object_MultLowLvl = dynamic_cast<Object_MultLowLvl*>(obj);
            fence |= (a0 != object_MultLowLvl->a0_);
            fence |= (a1 != object_MultLowLvl->a1_);
            fence |= (b0 != object_MultLowLvl->b0_);
            fence |= (b1 != object_MultLowLvl->b1_);
            fence |= (a_primes_size != object_MultLowLvl->a_primes_size_);
            fence |= (a_primes_index != object_MultLowLvl->a_primes_index_);
            fence |= (b_primes_size != object_MultLowLvl->b_primes_size_);
            fence |= (b_primes_index != object_MultLowLvl->b_primes_index_);
            fence |= (c_primes_size != object_MultLowLvl->c_primes_size_);
            fence |= (output_primes_index != object_MultLowLvl->output_primes_index_);
            fence |= (c0 != object_MultLowLvl->c0_);
            fence |= (c1 != object_MultLowLvl->c1_);
            fence |= (c2 != object_MultLowLvl->c2_);
            fence |= (primes != object_MultLowLvl->primes_);
            fence |= (primes_size != object_MultLowLvl->primes_size_);

        }
    }

    Object* obj = new Object_MultLowLvl(a0, a1, a_primes_size, a_primes_index,
                                        b0, b1, b_primes_size, b_primes_index,
                                        plainText, coeff_count,
                                        c0, c1, c2, c_primes_size, output_primes_index, 
                                        primes, primes_size, fence);
    std::cout << "Object_multlowlvl construct successfully.\n";
    
    fpga_buffer.push(obj);

    std::cout << "push Object_multlowlvl construct successfully.\n";

    outstanding_objects_MultLowLvl.insert(obj);

    std::cout << "insert Object_multlowlvl construct successfully.\n";

    if (fpga_buffer.get_worksize_MultLowLvl() == 1) {
        MultLowLvlCompleted_int();
    }

} 

static void cpu_MultLowLvl(uint64_t* a0, uint64_t* a1, uint64_t a_primes_size, uint8_t* a_primes_index,
                            uint64_t* b0, uint64_t* b1, uint64_t b_primes_size, uint8_t* b_primes_index,
                            uint64_t plainText, uint64_t coeff_count, 
                            uint64_t* c0, uint64_t* c1, uint64_t* c2, uint64_t c_primes_size, 
                            uint8_t* output_primes_index, uint64_t* primes, uint64_t primes_size) {
    FPGA_ASSERT(g_choice == CPU);

    // todo, confirm CPU version existed.
    std::cerr << "CPU VERSION not supported!" << std::endl;
    exit(-1);
}



void set_worksize_MultLowLvl_int(uint64_t n) {
    fpga_buffer.set_worksize_MultLowLvl(n);
}

bool MultLowLvlCompleted_int() {
    bool all_done = false;
    while (!all_done) {
        bool done = true;
        auto iter = outstanding_objects_MultLowLvl.begin();
        while (iter != outstanding_objects_MultLowLvl.end()) {
            Object* obj = *iter;
            if (obj->ready_) {
                delete obj;
                obj = nullptr;
                iter = outstanding_objects_MultLowLvl.erase(iter);
            } else {
                done = false;
                iter++;
            }
        }
        all_done = done;
    }

    outstanding_objects_MultLowLvl.clear();
    fpga_buffer.set_worksize_MultLowLvl(1);

    return all_done;
}


void MultLowLvl_int(uint64_t* a0, uint64_t* a1, uint64_t a_primes_size, uint8_t* a_primes_index,
                    uint64_t* b0, uint64_t* b1, uint64_t b_primes_size, uint8_t* b_primes_index,
                    uint64_t plainText, uint64_t coeff_count, 
                    uint64_t* c0, uint64_t* c1, uint64_t* c2, 
                    uint64_t c_primes_size, uint8_t* output_primes_index,
                    uint64_t* primes, uint64_t primes_size) {
    std::cout << __func__ << ", g_choice = " << g_choice << std::endl;
    switch (g_choice) {
    case CPU:
        cpu_MultLowLvl(a0, a1, a_primes_size, a_primes_index,
                       b0, b1, b_primes_size, b_primes_index,
                       plainText, coeff_count,
                       c0, c1, c2, c_primes_size, output_primes_index, 
                       primes, primes_size);
        break;
    case EMU:
    case FPGA:
        fpga_MultLowLvl(a0, a1, a_primes_size, a_primes_index,
                       b0, b1, b_primes_size, b_primes_index,
                       plainText, coeff_count,
                       c0, c1, c2, c_primes_size, output_primes_index,
                       primes, primes_size );
        break;
    default:
        std::cerr << "ERROR: Invalid RUN_CHOICE envvar. Set to a valid value {0, 1, 2}, \
        where 0:CPU, 1:EMU, 2:FPGA."  << std::endl;
        break;
    }
}



}  // namespace fpga
}  // namespace hexl
}  // namespace intel
