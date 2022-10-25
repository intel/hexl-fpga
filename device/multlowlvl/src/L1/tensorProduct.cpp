#include "../L0/load.hpp"
#include "../L0/scale.hpp"
#include "../L0/store.hpp"
#include "../L0/tensorProduct.hpp"
#include "../../include/L1/tensorProduct.h"

namespace L1 {
namespace helib {
namespace bgv {

event TensorProduct(sycl::queue &q, sycl::buffer<ulong4> &primes) {
  return L0::TensorProduct<
      COEFF_COUNT, pipe_tensor_product_input1, pipe_tensor_product_input2,
      pipe_tensor_product_store0, pipe_tensor_product_store12>(q, primes);
}

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

/*
sycl::event TensorProductLoad1(sycl::queue &q, sycl::buffer<uint64_t> &c) {
  return L0::generic_load<class TensorProductLoad1,
                          pipe_tensor_product_ntt_input1>(q, c);
}
sycl::event TensorProductLoad2(sycl::queue &q, sycl::buffer<uint64_t> &c) {
  return L0::generic_load<class TensorProductLoad2,
                          pipe_tensor_product_ntt_input2>(q, c);
}
*/



// tensor_product_ntt1_t &GetTensorProductNTT1() {
//   static tensor_product_ntt1_t ntt;
//   return ntt;
// }

// tensor_product_ntt2_t &GetTensorProductNTT2() {
//   static tensor_product_ntt2_t ntt;
//   return ntt;
// }

/*
sycl::event TensorProductNTT1LoadPrimeIndex(
    sycl::queue &q, sycl::buffer<uint8_t> &primes_index) {
  return L0::LoadPrimeIndexGeneric2<class TensorProductNTT1LoadPrimeIndex,
                                    pipe_tensor_product_prime_index1, 2>(
      q, primes_index);
}

sycl::event TensorProductNTT2LoadPrimeIndex(
    sycl::queue &q, sycl::buffer<uint8_t> &primes_index) {
  return L0::LoadPrimeIndexGeneric2<class TensorProductNTT2LoadPrimeIndex,
                                    pipe_tensor_product_prime_index2, 2>(
      q, primes_index);
}
*/

#define NTT_VEC 8

int get_ntt_VEC()
{
  static int vec = 8;
  return vec;
}

sycl::event ntt1_read(sycl::queue &q) {
    return L0::read<NTTRead<10>, pipe_scale_output, 
            ntt_pipe_generator<10, 8, COEFF_COUNT>::pipe_NTT_read_out, NTT_VEC>(q);
}

sycl::event ntt1_write(sycl::queue &q) {
    return L0::write<NTTWrite<10>, 
              ntt_pipe_generator<10, 8, COEFF_COUNT>::pipe_NTT_write_in, 
              pipe_tensor_product_input1, NTT_VEC>(q);
}

sycl::event ntt1_compute_forward(sycl::queue &q,
                              const std::vector<ulong4> &config) {
    return L0::NTT::ntt<NTTNTT<10>, 
            ntt_pipe_generator<10, 8, COEFF_COUNT>::pipe_ntt_prime_index_forward,
            ntt_pipe_generator<10, 8, COEFF_COUNT>::pipe_NTT_read_out, 
            ntt_pipe_generator<10, 8, COEFF_COUNT>::pipe_NTT_write_in, 
            ntt_pipe_generator<10, 8, COEFF_COUNT>::pipe_NTT_tf, 
            NTT_VEC, COEFF_COUNT>(q, config);
}

sycl::event ntt1_config_tf(sycl::queue &q, const std::vector<uint64_t> &tf_set) {
    return L0::TwiddleFactor_NTT<NTTTF<10>, 
            pipe_tensor_product_prime_index1,
            ntt_pipe_generator<10, 8, COEFF_COUNT>::pipe_ntt_prime_index_forward, 
            ntt_pipe_generator<10, 8, COEFF_COUNT>::pipe_NTT_tf, 
            NTT_VEC, COEFF_COUNT>(q, tf_set);
}



sycl::event ntt2_read(sycl::queue &q) {
    return L0::read<NTTRead<11>, pipe_scale_output2, 
            ntt_pipe_generator<11, 8, COEFF_COUNT>::pipe_NTT_read_out, NTT_VEC>(q);
}

sycl::event ntt2_write(sycl::queue &q) {
    return L0::write<NTTWrite<11>, 
              ntt_pipe_generator<11, 8, COEFF_COUNT>::pipe_NTT_write_in, 
              pipe_tensor_product_input2, NTT_VEC>(q);
}

sycl::event ntt2_compute_forward(sycl::queue &q,
                              const std::vector<ulong4> &config) {
    return L0::NTT::ntt<NTTNTT<11>, 
            ntt_pipe_generator<11, 8, COEFF_COUNT>::pipe_ntt_prime_index_forward,
            ntt_pipe_generator<11, 8, COEFF_COUNT>::pipe_NTT_read_out, 
            ntt_pipe_generator<11, 8, COEFF_COUNT>::pipe_NTT_write_in, 
            ntt_pipe_generator<11, 8, COEFF_COUNT>::pipe_NTT_tf, 
            NTT_VEC, COEFF_COUNT>(q, config);
}

sycl::event ntt2_config_tf(sycl::queue &q, const std::vector<uint64_t> &tf_set) {
    return L0::TwiddleFactor_NTT<NTTTF<11>, 
            pipe_tensor_product_prime_index2,
            ntt_pipe_generator<11, 8, COEFF_COUNT>::pipe_ntt_prime_index_forward, 
            ntt_pipe_generator<11, 8, COEFF_COUNT>::pipe_NTT_tf, 
            NTT_VEC, COEFF_COUNT>(q, tf_set);
}


NTT_Method& ntt1_method()
{
  static NTT_Method ntt1_method_obj = {
    .get_VEC = &get_ntt_VEC,
    .read = &ntt1_read,
    .write = &ntt1_write,
    .compute_forward = &ntt1_compute_forward,
    .config_tf = &ntt1_config_tf
  };

  return ntt1_method_obj;
}


NTT_Method& ntt2_method()
{
  static NTT_Method ntt2_method_obj = {
    .get_VEC = &get_ntt_VEC,
    .read = &ntt2_read,
    .write = &ntt2_write,
    .compute_forward = &ntt2_compute_forward,
    .config_tf = &ntt2_config_tf
  };

  return ntt2_method_obj;
}



}  // namespace bgv
}  // namespace helib
}  // namespace L1
