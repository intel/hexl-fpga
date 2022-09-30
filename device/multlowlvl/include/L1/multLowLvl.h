#pragma once
#include <L1/intt.h>
#include <L1/ntt.h>
#include <L1/pipes.h>

namespace L1 {
namespace helib {
namespace bgv {
/* 4 special primes and 4 normal primes */
#define MAX_MULT_LOW_LVL_BRING_TO_SET_P 6
#define MAX_MULT_LOW_LVL_BRING_TO_SET_P_BANKS 8
#define MAX_MULT_LOW_LVL_BRING_TO_SET_Q 23

/**
 * @brief instance the intt template
 *
 */
using intt1_t = intt<1, 8, COEFF_COUNT, pipe_intt1_input,
                     pipe_intt1_primes_index, pipe_scale_input>;

/**
 * @brief intt2_t
 *
 */
using intt2_t = intt<2, 8, COEFF_COUNT, pipe_intt2_input,
                     pipe_intt2_primes_index, pipe_scale_input2>;

/**
 * @brief GetINTT1
 *
 * @return intt1_t&
 */
intt1_t &GetINTT1();

/**
 * @brief GetINTT2
 *
 * @return intt2_t&
 */
intt2_t &GetINTT2();

/**
 * @brief INTT1LoadPrimesIndex
 *
 * @param q
 * @param prime_index_set
 * @return sycl::event
 */
sycl::event INTT1LoadPrimesIndex(sycl::queue &q,
                                 sycl::buffer<uint8_t> &primes_index);

/**
 * @brief INTT2LoadPrimesIndex
 *
 * @param q
 * @param prime_index_set
 * @return sycl::event
 */
sycl::event INTT2LoadPrimesIndex(sycl::queue &q,
                                 sycl::buffer<uint8_t> &primes_index);

/**
 * @brief scale up and scale down, we also call it bringToSet
 *
 * @param q
 * @param coeff_count
 * @param scale_param_set_buf
 * @param P
 * @param Q
 * @param I
 * @param t
 * @return event
 */
event BringToSet(sycl::queue &q, uint32_t coeff_count,
                 sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
                 uint32_t Q, uint I, uint64_t t);

/**
 * @brief Load the input for bringToSet
 *
 * @param q
 * @param c
 * @return sycl::event
 */
sycl::event BringToSetLoad(sycl::queue &q, sycl::event &depends,
                           sycl::buffer<uint64_t> &c,
                           sycl::buffer<uint8_t> &prime_index_set_buf);

/**
 * @brief store the output for bringToSet
 *
 * @param q
 * @param c
 * @return sycl::event
 */
sycl::event BringToSetStore(sycl::queue &q, sycl::buffer<uint64_t> &c);

/**
 * @brief scale up and scale down, we also call it bringToSet
 *
 * @param q
 * @param coeff_count
 * @param scale_param_set_buf
 * @param P
 * @param Q
 * @param I
 * @param t
 * @return event
 */
event BringToSet2(sycl::queue &q, uint32_t coeff_count,
                  sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
                  uint32_t Q, uint I, uint64_t t);

/**
 * @brief Load the input for bringToSet
 *
 * @param q
 * @param c
 * @return sycl::event
 */
sycl::event BringToSetLoad2(sycl::queue &q, sycl::event &depends,
                            sycl::buffer<uint64_t> &c,
                            sycl::buffer<uint8_t> &prime_index_set_buf);

/**
 * @brief store the output for bringToSet
 *
 * @param q
 * @param c
 * @return sycl::event
 */
sycl::event BringToSetStore2(sycl::queue &q, sycl::buffer<uint64_t> &c);
}  // namespace bgv
}  // namespace helib
}  // namespace L1
