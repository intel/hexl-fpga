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

// #include "multlowlvl/include/L2/utils.h"
// #include "multlowlvl/include/L2/ntt.hpp"
// #include "multlowlvl/include/L2/intt.hpp"

#include "multlowlvl/include/L2/utils.h"
#include "multlowlvl/include/L1/ntt.h"
#include "multlowlvl/include/L1/intt.h"
#include "multlowlvl/include/L1/tensorProduct.h"

using namespace L1::helib::bgv;

// This file is the interface of the BGV multLowLvl operation. 
// This file is a rewritten version of L1/ folder of the hexl-fpga-helib repo.
// All the sycl::pipe should be determined at compile-time. 

extern "C" {

//L2 interfaces for intt and ntt

// void launch_intt1_IF(std::vector<uint64_t> &primes) {
//     L2::helib::bgv::launch_intt(L1::helib::bgv::GetINTT1(), primes, COEFF_COUNT);
// }

// void launch_intt2_IF(std::vector<uint64_t> &primes) {
//     L2::helib::bgv::launch_intt(L1::helib::bgv::GetINTT2(), primes, COEFF_COUNT);
// }

// void launch_ntt1_IF(std::vector<uint64_t> &primes) {
//     L2::helib::bgv::launch_ntt(L1::helib::bgv::GetTensorProductNTT1(), primes, COEFF_COUNT);
// }

// void launch_ntt2_IF(std::vector<uint64_t> &primes) {
//     L2::helib::bgv::launch_ntt(L1::helib::bgv::GetTensorProductNTT2(), primes, COEFF_COUNT);
// }




intt1_t& launch_intt1_IF() {
    return L1::helib::bgv::GetINTT1();
    //L2::helib::bgv::launch_intt(, primes, COEFF_COUNT);
}

intt2_t& launch_intt2_IF() {
    return L1::helib::bgv::GetINTT2();
    //L2::helib::bgv::launch_intt(L1::helib::bgv::GetINTT2(), primes, COEFF_COUNT);
}

tensor_product_ntt1_t& launch_ntt1_IF() {
    return L1::helib::bgv::GetTensorProductNTT1();
    //L2::helib::bgv::launch_ntt(L1::helib::bgv::GetTensorProductNTT1(), primes, COEFF_COUNT);
}

tensor_product_ntt2_t& launch_ntt2_IF() {
    return L1::helib::bgv::GetTensorProductNTT2();
    //L2::helib::bgv::launch_ntt(L1::helib::bgv::GetTensorProductNTT2(), primes, COEFF_COUNT);
}


// multLowLvl "load" interface exposed to runtime

// sycl::event BringToSetLoad(sycl::queue &q, sycl::event &depends,
//                            sycl::buffer<uint64_t> &c,
//                            sycl::buffer<uint8_t> &prime_index_set_buf) {
//     return L0::load<class BringToSetLoad, L1::helib::bgv::pipe_intt1_input,
//                   L1::helib::bgv::pipe_intt1_primes_index, COEFF_COUNT>(q, depends, c,
//                                                         prime_index_set_buf);
// }

// declare and define a new interface, this interface call functions defined in L1/multlowlovl.cpp
// and L1/tensorProduct.cpp, this interface will be exposed to runtime, loaded by dynamic loading 
// C library function call (e.g. dlopen(), dlsym() ).

sycl::event BringToSetLoad_IF(sycl::queue &q, sycl::event &depends,
                           sycl::buffer<uint64_t> &c,
                           sycl::buffer<uint8_t> &prime_index_set_buf) {
    return BringToSetLoad(q, depends, c, prime_index_set_buf);
}

// sycl::event BringToSetLoad2(sycl::queue &q, sycl::event &depends,
//                             sycl::buffer<uint64_t> &c,
//                             sycl::buffer<uint8_t> &prime_index_set_buf) {
//   return L0::load<class BringToSetLoad2, L1::helib::bgv::pipe_intt2_input,
//                   L1::helib::bgv::pipe_intt2_primes_index, COEFF_COUNT>(q, depends, c,
//                                                         prime_index_set_buf);
// }

sycl::event BringToSetLoad2_IF(sycl::queue &q, sycl::event &depends,
                            sycl::buffer<uint64_t> &c,
                            sycl::buffer<uint8_t> &prime_index_set_buf) {
    return BringToSetLoad2(q, depends, c, prime_index_set_buf);
}



// multLowLvl "store" interface exposed to runtime
// The pipes in the template parameter must be determined at compile time.

// sycl::event TensorProductStore0(sycl::queue &q,
//                                 sycl::buffer<ulong> &output_c0) {
//   return L0::store<class TensorProductStore0, L1::helib::bgv::pipe_tensor_product_store0>(
//       q, output_c0);
// }

sycl::event TensorProductStore0_IF(sycl::queue &q,
                                sycl::buffer<ulong> &output_c0) {
    return TensorProductStore0(q, output_c0);
}

// sycl::event TensorProductStore12(sycl::queue &q, sycl::buffer<ulong> &output_c1,
//                                  sycl::buffer<ulong> &output_c2) {
//   return L0::store2<class TensorProductStore12, L1::helib::bgv::pipe_tensor_product_store12>(
//       q, output_c1, output_c2);
// }

sycl::event TensorProductStore12_IF(sycl::queue &q, sycl::buffer<ulong> &output_c1,
                                 sycl::buffer<ulong> &output_c2) {
    return TensorProductStore12(q, output_c1, output_c2);
}

// multLowLvl "BringToSet" kernel interface exposed to runtime

// sycl::event BringToSet(sycl::queue &q, uint32_t coeff_count,
//                  sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
//                  uint32_t Q, uint I, uint64_t t) {
//     // check inputs
//     assert(P <= MAX_MULT_LOW_LVL_BRING_TO_SET_P);
//     assert(Q <= MAX_MULT_LOW_LVL_BRING_TO_SET_Q);
//     assert(coeff_count = COEFF_COUNT);

//     return L0::scale<
//       class MultLowLvlBringToSet, MAX_MULT_LOW_LVL_BRING_TO_SET_P,
//       MAX_MULT_LOW_LVL_BRING_TO_SET_P_BANKS, MAX_MULT_LOW_LVL_BRING_TO_SET_Q,
//       COEFF_COUNT, L1::helib::bgv::pipe_scale_input, L1::helib::bgv::pipe_scale_output,
//       L1::helib::bgv::pipe_tensor_product_prime_index1, false, /* added primes not at end */
//       false /* do not add special primes*/>(q, coeff_count, scale_param_set_buf,
//                                             P, Q, I, t);

// }

sycl::event BringToSet_IF(sycl::queue &q, uint32_t coeff_count,
                 sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
                 uint32_t Q, uint I, uint64_t t) {
    return BringToSet(q, coeff_count, scale_param_set_buf, P, Q, I, t);
}

// sycl::event BringToSet2(sycl::queue &q, uint32_t coeff_count,
//                   sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
//                   uint32_t Q, uint I, uint64_t t) {
//   // check inputs
//   assert(P <= MAX_MULT_LOW_LVL_BRING_TO_SET_P);
//   assert(Q <= MAX_MULT_LOW_LVL_BRING_TO_SET_Q);
//   assert(coeff_count = COEFF_COUNT);
//   return L0::scale<
//       class MultLowLvlBringToSet2, MAX_MULT_LOW_LVL_BRING_TO_SET_P,
//       MAX_MULT_LOW_LVL_BRING_TO_SET_P_BANKS, MAX_MULT_LOW_LVL_BRING_TO_SET_Q,
//       COEFF_COUNT, L1::helib::bgv::pipe_scale_input2, L1::helib::bgv::pipe_scale_output2,
//       L1::helib::bgv::pipe_tensor_product_prime_index2, false, /* added primes not at end */
//       false /* do not add special primes*/>(q, coeff_count, scale_param_set_buf,
//                                             P, Q, I, t);
// }

sycl::event BringToSet2_IF(sycl::queue &q, uint32_t coeff_count,
                  sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
                  uint32_t Q, uint I, uint64_t t) {
    return BringToSet2(q, coeff_count, scale_param_set_buf, P, Q, I, t);
}


// MultLowLvl "TensorProduct" kernel interface exposed to the runtime

// sycl::event TensorProduct(sycl::queue &q, sycl::buffer<ulong4> &primes) {
//   return L0::TensorProduct<
//       COEFF_COUNT, L1::helib::bgv::pipe_tensor_product_input1, L1::helib::bgv::pipe_tensor_product_input2,
//       L1::helib::bgv::pipe_tensor_product_store0, L1::helib::bgv::pipe_tensor_product_store12>(q, primes);
// }

sycl::event TensorProduct_IF(sycl::queue &q, sycl::buffer<ulong4> &primes) {
  return TensorProduct(q, primes);
}

// MultLowLvl lunch_intt "intt1_t" requires a intt1_t object as input. define and expose
// intt1 and intt2 objects to runtime 

// see intt1_t & intt2_t definition in L1/multlowlvl.h file.

// L1::helib::bgv::intt1_t &GetINTT1_IF() {
//   return L1::helib::bgv::GetINTT1();
//   // static L1::helib::bgv::intt1_t intt;
//   // return intt;
// }

// L1::helib::bgv::intt2_t &GetINTT2_IF() {
//   return L1::helib::bgv::GetINTT2();
//   // static L1::helib::bgv::intt2_t intt;
//   // return intt;
// }


// launch "tensor_product_ntt" kernels. see src/L1/tensorProduct.cpp file.

// using tensor_product_ntt1_t =
//     ntt<10, 8, COEFF_COUNT, pipe_scale_output, pipe_tensor_product_prime_index1,
//         pipe_tensor_product_input1>;

// using tensor_product_ntt2_t =
//     ntt<11, 8, COEFF_COUNT, pipe_scale_output2,
//         pipe_tensor_product_prime_index2, pipe_tensor_product_input2>;


// L1::helib::bgv::tensor_product_ntt1_t &GetTensorProductNTT1_IF() {
//   return L1::helib::bgv::GetTensorProductNTT1();
//   // static L1::helib::bgv::tensor_product_ntt1_t ntt;
//   // return ntt;
// }

// L1::helib::bgv::tensor_product_ntt2_t &GetTensorProductNTT2_IF() {
//   return L1::helib::bgv::GetTensorProductNTT2();
//   // static L1::helib::bgv::tensor_product_ntt2_t ntt;
//   // return ntt;
// }

}