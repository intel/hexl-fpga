#pragma once
#include <L1/common.h>

extern "C" {
/**
 * @brief Tensor Product
 *
 * @param q
 * @param a0
 * @param a1
 * @param b0
 * @param b1
 * @param c0
 * @param c1
 * @param c2
 * @param primes
 * @param num_primes
 * @param offset1
 * @param offset2
 * @param offset3
 * @param offset4
 * @param flag
 * @return sycl::event
 */
sycl::event TensorProduct(
    sycl::queue &q, sycl::buffer<ulong> &a0, sycl::buffer<sycl::ulong> &a1,
    sycl::buffer<sycl::ulong> &b0, sycl::buffer<sycl::ulong> &b1,
    sycl::buffer<sycl::ulong> &c01, sycl::buffer<sycl::ulong> &c2,
    sycl::buffer<sycl::ulong4> &primes, unsigned num_primes, int offset1,
    int offset2, int offset3, int offset4, sycl::event &depend_event, int flag);
sycl::event StoreTensorProduct(sycl::queue &q, sycl::buffer<uint64_t> &data1,
                               sycl::buffer<uint64_t> &data2,
                               sycl::buffer<uint64_t> &data3, unsigned size,
                               int flag);
}
