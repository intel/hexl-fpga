#pragma once
#include "intt.h"
#include "ntt.h"
#include "common.h"
#include "pipes.h"

namespace L1 {
namespace helib {
namespace bgv {

// template class ntt<10, 8, COEFF_COUNT, pipe_scale_output, 
//         pipe_tensor_product_prime_index1, pipe_tensor_product_input1>;

// template class ntt<11, 8, COEFF_COUNT, pipe_scale_output2,
//         pipe_tensor_product_prime_index2, pipe_tensor_product_input2>;

/**
 * @brief instance the ntt template
 *
 */
using tensor_product_ntt1_t =
    ntt<10, 8, COEFF_COUNT, pipe_scale_output, 
    pipe_tensor_product_prime_index1, pipe_tensor_product_input1>;

using tensor_product_ntt2_t =
    ntt<11, 8, COEFF_COUNT, pipe_scale_output2,
    pipe_tensor_product_prime_index2, pipe_tensor_product_input2>;



/**
 * @brief Get the tensor_product_ntt1_t instance
 *
 * @return tensor_product_ntt1_t&
 */
tensor_product_ntt1_t &GetTensorProductNTT1();

/**
 * @brief Get the tensor_product_ntt1_t instance
 *
 * @return tensor_product_ntt2_t&
 */
tensor_product_ntt2_t &GetTensorProductNTT2();


#if 0
/**
 * @brief TensorProductNTT1LoadPrimeIndex
 *
 * @param q
 * @param prime_index_start_end
 * @return sycl::event
 */
sycl::event TensorProductNTT1LoadPrimeIndex(
    sycl::queue &q, sycl::buffer<uint8_t> &primes_index);

/**
 * @brief TensorProductNTT2LoadPrimeIndex
 *
 * @param q
 * @param prime_index_start_end
 * @return sycl::event
 */
sycl::event TensorProductNTT2LoadPrimeIndex(
    sycl::queue &q, sycl::buffer<uint8_t> &primes_index);
#endif

/**
 * @brief TensorProduct
 *
 * @param q
 * @param primes
 * @param output_c0
 * @param output_c1
 * @param output_c2
 * @return event
 */
sycl::event TensorProduct(sycl::queue &q, sycl::buffer<ulong4> &primes);

sycl::event TensorProductStore0(sycl::queue &q, sycl::buffer<ulong> &output_c0);
sycl::event TensorProductStore12(sycl::queue &q, sycl::buffer<ulong> &output_c1,
                                 sycl::buffer<ulong> &output_c2);

#if 0

/**
 * @brief TensorProductLoad1
 *
 * @param q
 * @param c
 * @return sycl::event
 */
sycl::event TensorProductLoad1(sycl::queue &q, sycl::buffer<uint64_t> &c);

/**
 * @brief TensorProductLoad2
 *
 * @param q
 * @param c
 * @return sycl::event
 */
sycl::event TensorProductLoad2(sycl::queue &q, sycl::buffer<uint64_t> &c);

#endif

struct NTT_Method {
    int (*get_VEC)();
    sycl::event (*read)(sycl::queue &q);
    sycl::event (*write)(sycl::queue &q);
    sycl::event (*compute_forward)(sycl::queue &q,
                              const std::vector<ulong4> &config);
    sycl::event (*config_tf)(sycl::queue &q, const std::vector<uint64_t> &tf_set);
};

NTT_Method& ntt1_method();
NTT_Method& ntt2_method();

}  // namespace bgv
}  // namespace helib
}  // namespace L1
