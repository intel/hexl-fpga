#include <L0/tensorproduct.hpp>
#include <L1/common.h>
#include <L1/pipes.h>

extern "C" {
using pipe_output =
    ext::intel::pipe<class StoreTensorProductOutputPipe, ulong4>;
sycl::event TensorProduct(sycl::queue &q, sycl::buffer<ulong> &a0,
                          sycl::buffer<ulong> &a1, sycl::buffer<ulong> &b0,
                          sycl::buffer<ulong> &b1, sycl::buffer<ulong> &c01,
                          sycl::buffer<ulong> &c2, sycl::buffer<ulong4> &primes,
                          unsigned num_primes, int offset1, int offset2,
                          int offset3, int offset4, sycl::event &depend_event,
                          int flag) {
  return L0::TensorProduct<pipe_tensor_product_ready, COEFF_COUNT, pipe_output>(
      q, a0, a1, b0, b1, c01, c2, primes, num_primes, offset1, offset2, offset3,
      offset4, depend_event, flag);
}

sycl::event StoreTensorProduct(sycl::queue &q, sycl::buffer<uint64_t> &data1,
                               sycl::buffer<uint64_t> &data2,
                               sycl::buffer<uint64_t> &data3, unsigned size,
                               int flag) {
  return L0::StoreTensorProduct<class StoreTensorProduct, pipe_output,
                                pipe_data_ptr_c01,
                                pipe_break_into_digits_ready>(
      q, data1, data2, data3, size, flag);
}
}