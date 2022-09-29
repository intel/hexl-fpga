// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include <L0/load.hpp>
#include <L0/scale.hpp>
#include <L0/store.hpp>
#include <L0/tensorProduct.hpp>
#include <L1/tensorProduct.h>
#include <L1/helib_bgv.h>

// This file is the interface of the BGV multLowLvl operation. 
// This file is a rewritten version of L1/ folder of the hexl-fpga-helib repo.

extern "C" {
// multLowLvl "load" interface exposed to runtime

sycl::event BringToSetLoad(sycl::queue &q, sycl::event &depends,
                           sycl::buffer<uint64_t> &c,
                           sycl::buffer<uint8_t> &prime_index_set_buf) {
    return L0::load<class BringToSetLoad, pipe_intt1_input,
                  pipe_intt1_primes_index, COEFF_COUNT>(q, depends, c,
                                                        prime_index_set_buf);
}


sycl::event BringToSetLoad2(sycl::queue &q, sycl::event &depends,
                            sycl::buffer<uint64_t> &c,
                            sycl::buffer<uint8_t> &prime_index_set_buf) {
  return L0::load<class BringToSetLoad2, pipe_intt2_input,
                  pipe_intt2_primes_index, COEFF_COUNT>(q, depends, c,
                                                        prime_index_set_buf);
}


// multLowLvl "store" interface exposed to runtime
// The pipes in the template parameter will be determined at compile time at this time.

sycl::event TensorProductStore0(sycl::queue &q,
                                sycl::buffer<ulong> &output_c0) {
  return L0::store<class TensorProductStore0, pipe_tensor_product_store0>(
      q, output_c0);
}

sycl::event TensorProductStore12(sycl::queue &q, sycl::buffer<ulong> &output_c1,
                                 sycl::buffer<ulong> &output_c2) {
  return L0::store2<class TensorProductStore12, pipe_tensor_product_store12>(
      q, output_c1, output_c2);
}

// multLowLvl "BringToSet" kernel interface exposed to runtime

sycl::event BringToSet(sycl::queue &q, uint32_t coeff_count,
                 sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
                 uint32_t Q, uint I, uint64_t t) {
    // check inputs
    assert(P <= MAX_MULT_LOW_LVL_BRING_TO_SET_P);
    assert(Q <= MAX_MULT_LOW_LVL_BRING_TO_SET_Q);
    assert(coeff_count = COEFF_COUNT);

    return L0::scale<
      class MultLowLvlBringToSet, MAX_MULT_LOW_LVL_BRING_TO_SET_P,
      MAX_MULT_LOW_LVL_BRING_TO_SET_P_BANKS, MAX_MULT_LOW_LVL_BRING_TO_SET_Q,
      COEFF_COUNT, pipe_scale_input, pipe_scale_output,
      pipe_tensor_product_prime_index1, false, /* added primes not at end */
      false /* do not add special primes*/>(q, coeff_count, scale_param_set_buf,
                                            P, Q, I, t);

}

sycl::event BringToSet2(sycl::queue &q, uint32_t coeff_count,
                  sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
                  uint32_t Q, uint I, uint64_t t) {
  // check inputs
  assert(P <= MAX_MULT_LOW_LVL_BRING_TO_SET_P);
  assert(Q <= MAX_MULT_LOW_LVL_BRING_TO_SET_Q);
  assert(coeff_count = COEFF_COUNT);
  return L0::scale<
      class MultLowLvlBringToSet2, MAX_MULT_LOW_LVL_BRING_TO_SET_P,
      MAX_MULT_LOW_LVL_BRING_TO_SET_P_BANKS, MAX_MULT_LOW_LVL_BRING_TO_SET_Q,
      COEFF_COUNT, pipe_scale_input2, pipe_scale_output2,
      pipe_tensor_product_prime_index2, false, /* added primes not at end */
      false /* do not add special primes*/>(q, coeff_count, scale_param_set_buf,
                                            P, Q, I, t);
}


// MultLowLvl "TensorProduct" kernel interface exposed to the runtime

sycl::event TensorProduct(sycl::queue &q, sycl::buffer<ulong4> &primes) {
  return L0::TensorProduct<
      COEFF_COUNT, pipe_tensor_product_input1, pipe_tensor_product_input2,
      pipe_tensor_product_store0, pipe_tensor_product_store12>(q, primes);
}


// MultLowLvl lunch_intt "intt1_t" requires a intt1_t object as input. define and expose
// intt1 and intt2 objects to runtime 

// using intt1_t = intt<1, 8, COEFF_COUNT, pipe_intt1_input,
//                      pipe_intt1_primes_index, pipe_scale_input>;

// using intt2_t = intt<2, 8, COEFF_COUNT, pipe_intt2_input,
//                      pipe_intt2_primes_index, pipe_scale_input2>;

intt1_t &GetINTT1() {
  static intt1_t intt;
  return intt;
}

intt2_t &GetINTT2() {
  static intt2_t intt;
  return intt;
}


}

