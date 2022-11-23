#include <L0/tensorproduct.hpp>
#include <L1/tensorproduct.h>
#include <L1/common.h>

namespace L1 {
sycl::event TensorProduct(sycl::queue &q, sycl::buffer<ulong> &a0,
                          sycl::buffer<ulong> &a1, sycl::buffer<ulong> &b0,
                          sycl::buffer<ulong> &b1, sycl::buffer<ulong> &c0,
                          sycl::buffer<ulong> &c1, sycl::buffer<ulong> &c2,
                          sycl::buffer<ulong4> &primes, unsigned num_primes,
                          int offset1, int offset2, int offset3, int offset4,
                          sycl::event &depend_event, int flag) {
  return L0::TensorProduct<COEFF_COUNT>(q, a0, a1, b0, b1, c0, c1, c2, primes,
                                        num_primes, offset1, offset2, offset3,
                                        offset4, depend_event, flag);
}
}  // namespace L1
