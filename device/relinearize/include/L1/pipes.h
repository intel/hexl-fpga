#pragma once

namespace L1 {

/**
 * @brief pipes for BringToSet (or called scale)
 *
 */
using pipe_intt1_input = ext::intel::pipe<class INTT1InputPipeId, uint64_t, 4>;
using pipe_intt2_input = ext::intel::pipe<class INTT2InputPipeId, uint64_t, 4>;

using pipe_intt1_primes_index =
    ext::intel::pipe<class INTT1PrimeIndexPipeId, uint64_t, 4>;
using pipe_intt2_primes_index =
    ext::intel::pipe<class INTT2PrimeIndexPipeId, uint64_t, 4>;

using pipe_scale_input = ext::intel::pipe<class ScaleInputPipeId, uint64_t, 4>;
using pipe_scale_output =
    ext::intel::pipe<class ScaleOutputPipeId, uint64_t, 4>;

using pipe_scale_input2 =
    ext::intel::pipe<class ScaleInput2PipeId, uint64_t, 4>;
using pipe_scale_output2 =
    ext::intel::pipe<class ScaleOutput2PipeId, uint64_t, 4>;

/**
 * @brief pipes for TensorProduct
 *
 */
using pipe_tensor_product_ntt_input1 =
    ext::intel::pipe<class TensorProductNTTInput1PipeId, uint64_t, 4>;
using pipe_tensor_product_ntt_input2 =
    ext::intel::pipe<class TensorProductNTTInput2PipeId, uint64_t, 4>;
using pipe_tensor_product_input1 =
    ext::intel::pipe<class TensorProductInput1PipeId, uint64_t, 4>;
using pipe_tensor_product_input2 =
    ext::intel::pipe<class TensorProductInput2PipeId, uint64_t, 4>;

using pipe_tensor_product_prime_index1 =
    ext::intel::pipe<class TensorProductPrimeIndex1PipeId, uint8_t, 4>;
using pipe_tensor_product_prime_index2 =
    ext::intel::pipe<class TensorProductPrimeIndex2PipeId, uint8_t, 4>;

using pipe_tensor_product_store0 =
    ext::intel::pipe<class TensorProductStore0PipeId, uint64_t, 2048>;
using pipe_tensor_product_store12 =
    ext::intel::pipe<class TensorProductStore12PipeId, sycl::ulong2, 2048>;

}  // namespace L1
