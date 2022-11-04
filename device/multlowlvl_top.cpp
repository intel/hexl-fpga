// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include "multlowlvl/src/L0/load.hpp"
#include "multlowlvl/src/L0/scale.hpp"
#include "multlowlvl/src/L0/store.hpp"
#include "multlowlvl/src/L0/tensorProduct.hpp"

#include "multlowlvl/include/L1/multLowLvl.h"
#include "multlowlvl/include/L1/tensorProduct.h"
#include "multlowlvl/include/L1/helib_bgv.h"

// TODO:
// In general, L2/ files should not be include in
// this file, because L2 is for runtime, let's try to only include
// L1/ files in this file, and remove link hexl to the shared library.

//#include "multlowlvl/include/L2/utils.h"
// #include "multlowlvl/include/L2/ntt.hpp"
// #include "multlowlvl/include/L2/intt.hpp"
// #include "multlowlvl/include/L2/utils.h"


#include "multlowlvl/include/L1/ntt.h"
#include "multlowlvl/include/L1/intt.h"
#include "multlowlvl/include/L1/tensorProduct.h"

using namespace L1::helib::bgv;

// This file is the interface of the BGV multLowLvl operation. 
// This file is a rewritten version of L1/ folder of the hexl-fpga-helib repo.
// All the sycl::pipe should be determined at compile-time. 

extern "C" {

//L2 interfaces for intt and ntt

void launch_intt1_IF_L2(std::vector<uint64_t> &primes) {
    // std::cout << "intt1 address: " << &L1::helib::bgv::GetINTT1() << std::endl;
    //L2::helib::bgv::launch_intt(L1::helib::bgv::GetINTT1(), primes, COEFF_COUNT);
}

void launch_intt2_IF_L2(std::vector<uint64_t> &primes) {
    // std::cout << "intt2 address: " << &L1::helib::bgv::GetINTT2() << std::endl;
    //L2::helib::bgv::launch_intt(L1::helib::bgv::GetINTT2(), primes, COEFF_COUNT);
}

void launch_ntt1_IF_L2(std::vector<uint64_t> &primes) {
    // std::cout << "ntt1 address: " << &L1::helib::bgv::GetTensorProductNTT1() << std::endl;
    //L2::helib::bgv::launch_ntt(L1::helib::bgv::GetTensorProductNTT1(), primes, COEFF_COUNT);
}

void launch_ntt2_IF_L2(std::vector<uint64_t> &primes) {
    // std::cout << "ntt2 address: " << &L1::helib::bgv::GetTensorProductNTT2() << std::endl;
    //L2::helib::bgv::launch_ntt(L1::helib::bgv::GetTensorProductNTT2(), primes, COEFF_COUNT);
}


intt1_t& launch_intt1_IF() {
    //std::cout << "intt1 address IF " << &L1::helib::bgv::GetINTT1() << std::endl;
    //return L1::helib::bgv::GetINTT1();
    //L2::helib::bgv::launch_intt(, primes, COEFF_COUNT);
}

intt2_t& launch_intt2_IF() {
    //std::cout << "intt2 address IF " << &L1::helib::bgv::GetINTT2() << std::endl;
    //return L1::helib::bgv::GetINTT2();
    //L2::helib::bgv::launch_intt(L1::helib::bgv::GetINTT2(), primes, COEFF_COUNT);
}

tensor_product_ntt1_t& launch_ntt1_IF() {
    //std::cout << "ntt1 address IF " << &L1::helib::bgv::GetTensorProductNTT1() << std::endl;
    //return L1::helib::bgv::GetTensorProductNTT1();
    //L2::helib::bgv::launch_ntt(L1::helib::bgv::GetTensorProductNTT1(), primes, COEFF_COUNT);
}

tensor_product_ntt2_t& launch_ntt2_IF() {
    //std::cout << "ntt2 address IF " << &L1::helib::bgv::GetTensorProductNTT2() << std::endl;
    //return L1::helib::bgv::GetTensorProductNTT2();
    //L2::helib::bgv::launch_ntt(L1::helib::bgv::GetTensorProductNTT2(), primes, COEFF_COUNT);
}


// multLowLvl "load" interface exposed to runtime

// declare and define a new interface, this interface call functions defined in L1/multlowlovl.cpp
// and L1/tensorProduct.cpp, this interface will be exposed to runtime, loaded by dynamic loading 
// C library function call (e.g. dlopen(), dlsym() ).

sycl::event BringToSetLoad_IF(sycl::queue &q, sycl::event &depends,
                           sycl::buffer<uint64_t> &c,
                           sycl::buffer<uint8_t> &prime_index_set_buf) {
    return BringToSetLoad(q, depends, c, prime_index_set_buf);
}

sycl::event BringToSetLoad2_IF(sycl::queue &q, sycl::event &depends,
                            sycl::buffer<uint64_t> &c,
                            sycl::buffer<uint8_t> &prime_index_set_buf) {
    return BringToSetLoad2(q, depends, c, prime_index_set_buf);
}



// multLowLvl "store" interface exposed to runtime
// The pipes in the template parameter must be determined at compile time.

sycl::event TensorProductStore0_IF(sycl::queue &q,
                                sycl::buffer<ulong> &output_c0) {
    return TensorProductStore0(q, output_c0);
}

sycl::event TensorProductStore12_IF(sycl::queue &q, sycl::buffer<ulong> &output_c1,
                                 sycl::buffer<ulong> &output_c2) {
    return TensorProductStore12(q, output_c1, output_c2);
}

// multLowLvl "BringToSet" kernel interface exposed to runtime

sycl::event BringToSet_IF(sycl::queue &q, uint32_t coeff_count,
                 sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
                 uint32_t Q, uint I, uint64_t t) {
    return BringToSet(q, coeff_count, scale_param_set_buf, P, Q, I, t);
}

sycl::event BringToSet2_IF(sycl::queue &q, uint32_t coeff_count,
                  sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
                  uint32_t Q, uint I, uint64_t t) {
    return BringToSet2(q, coeff_count, scale_param_set_buf, P, Q, I, t);
}

sycl::event TensorProduct_IF(sycl::queue &q, sycl::buffer<ulong4> &primes) {
  return TensorProduct(q, primes);
}

BringToSet_t& BringToSet_struct_IF() {
    return L1::helib::bgv::BringToSet_struct();
}

INTT_Method& intt1_method_IF() {
    return L1::helib::bgv::intt1_method();
}

INTT_Method& intt2_method_IF() {
    return L1::helib::bgv::intt2_method();
}

NTT_Method& ntt1_method_IF() {
    return L1::helib::bgv::ntt1_method();
}

NTT_Method& ntt2_method_IF() {
    return L1::helib::bgv::ntt2_method();
}

}