#pragma once
#include <CL/sycl.hpp>

using namespace sycl;

namespace L1 {
namespace helib {
namespace bgv {
// relinearlize
#define MAX_NORMAL_PRIMES 18
#define MAX_SPECIAL_PRIMES 9
#define MAX_C2_DROP_SMALL_P 4
#define MAX_C2_DROP_SMALL_P_BANKS 4
#define MAX_C2_DROP_SMALL_Q 23

/**
 * @brief pipes for reLinearize
 *
 */
using pipe_break_into_digits_input =
    ext::intel::pipe<class BreakIntoDigitsInputPipeId, uint64_t, 4>;
using pipe_break_into_digits_output =
    ext::intel::pipe<class BreakIntoDigitsOutputPipeId, uint64_t, 4>;

using pipe_c0_drop_small_input =
    ext::intel::pipe<class C0DropSmallInputPipeId, uint64_t, 4>;
using pipe_c0_drop_small_output =
    ext::intel::pipe<class C0DropSmallOutputPipeId, uint64_t, 4>;

using pipe_reLinearizeC01_ntt_output =
    ext::intel::pipe<class ReLinearizeC01NTTOutputPipeId, uint64_t, 4>;

using pipe_reLinearizeC01_prime_index =
    ext::intel::pipe<class ReLinearizeC01PrimeIndexPipeId, uint8_t, 4>;

/**
 * @brief pipe for the input of DropSmall
 *
 */
using pipe_c2_drop_small_input =
    ext::intel::pipe<class C2DropSmallInputPipeId, uint64_t, 4>;

/**
 * @brief pipe for the output of DropSmall
 *
 */
using pipe_c2_drop_small_output =
    ext::intel::pipe<class C2DropSmallOutputPipeId, uint64_t, 4>;

/**
 * @brief drop small primes and break into digits
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
event C2DropSmall(sycl::queue &q, uint32_t coeff_count,
                  sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
                  uint32_t Q, uint I, uint64_t t);

/**
 * @brief load input for DropSmall
 *
 * @param q
 * @param c
 * @return sycl::event
 */
sycl::event C2DropSmallLoad(sycl::queue &q, sycl::buffer<uint64_t> &c);

/**
 * @brief BreakIntoDigits
 *
 * @param q
 * @param packed_precomuted_params_buf
 * @param num_digit1_primes
 * @param num_digit2_primes
 * @param num_special_primes
 * @return event
 */
event BreakIntoDigits(sycl::queue &q,
                      sycl::buffer<ulong2> &packed_precomuted_params_buf,
                      uint num_digit1_primes, uint num_digit2_primes,
                      uint num_special_primes);

/**
 * @brief BreakIntoDigits_store
 *
 * @param q
 * @param c
 * @return sycl::event
 */
sycl::event BreakIntoDigits_store(sycl::queue &q, sycl::buffer<uint64_t> &c);

/**
 * @brief load the input for C0 drop small
 *
 * @param q
 * @param c
 * @return sycl::event
 */
sycl::event C0DropSmallLoad(sycl::queue &q, sycl::buffer<uint64_t> &c);

/**
 * @brief drop small for c0
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
event C0DropSmall(sycl::queue &q, uint32_t coeff_count,
                  sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
                  uint32_t Q, uint I, uint64_t t);

/**
 * @brief store c0
 *
 * @param q
 * @param c
 * @return sycl::event
 */
sycl::event C0Store(sycl::queue &q, sycl::buffer<uint64_t> &c);

/**
 * @brief instance the ntt template
 *
 */
using reLinearizeC01_ntt_t =
    ntt<1, 8, COEFF_COUNT, pipe_c0_drop_small_output,
        pipe_reLinearizeC01_prime_index, pipe_reLinearizeC01_ntt_output>;

/**
 * @brief Get the reLinearizeC01 NTT instance
 *
 * @return reLinearizeC01_ntt_t&
 */
reLinearizeC01_ntt_t &GetReLinearizeC01NTT();

/**
 * @brief ReLinearizeC01LoadPrimeIndex
 *
 * @param q
 * @param prime_index_set
 * @return sycl::event
 */
sycl::event ReLinearizeC01LoadPrimeIndex(sycl::queue &q,
                                         sycl::uchar2 prime_index_start_end);
}  // namespace bgv
}  // namespace helib
}  // namespace L1
