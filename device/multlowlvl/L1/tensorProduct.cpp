#include <L0/load.hpp>
#include <L0/scale.hpp>
#include <L0/store.hpp>
#include <L0/tensorProduct.hpp>
#include <L1/tensorProduct.h>

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

tensor_product_ntt1_t &GetTensorProductNTT1() {
  static tensor_product_ntt1_t ntt;
  return ntt;
}

tensor_product_ntt2_t &GetTensorProductNTT2() {
  static tensor_product_ntt2_t ntt;
  return ntt;
}
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
}  // namespace bgv
}  // namespace helib
}  // namespace L1
