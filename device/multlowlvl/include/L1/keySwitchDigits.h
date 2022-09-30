#pragma once
#include "common.h"
#include <L1/ntt.h>
#include <CL/sycl.hpp>

using namespace sycl;

namespace L1 {
namespace helib {
namespace bgv {
/**
 * @brief pipes for the KeySwitchDigits module
 *
 */
using pipe_keySwitchDigits_input =
    ext::intel::pipe<class KeySwitchDigitsInputPipeId, uint64_t, 4>;
using pipe_keySwitchDigits_ntt_output =
    ext::intel::pipe<class KeySwitchDigitsNTTOutputPipeId, uint64_t, 4>;
using pipe_keySwitchDigits_output_c1 =
    ext::intel::pipe<class KeySwitchDigitsOutputAPipeId, uint64_t, 4>;
using pipe_keySwitchDigits_output_c2 =
    ext::intel::pipe<class KeySwitchDigitsOutputBPipeId, uint64_t, 4>;
using pipe_keySwitchDigits_prime_index =
    ext::intel::pipe<class KeySwitchDigitsPrimeIndexPipeId, uint8_t, 4>;

/**
 * @brief KeySwitchDigits kernel
 *
 * @param q
 * @param primes
 * @param wa
 * @param wb
 * @return event
 */
event keySwitchDigits(sycl::queue &q, sycl::buffer<ulong4> &primes,
                      sycl::buffer<ulong> &wa, sycl::buffer<ulong> &wb);

/**
 * @brief load the input for KeySwitchDigits
 *
 * @param q
 * @param input
 * @param primes_index
 * @return sycl::event
 */
sycl::event keySwitchDigits_load(sycl::queue &q, sycl::buffer<uint64_t> &input,
                                 sycl::buffer<uint8_t> &primes_index);

/**
 * @brief store the output c1 of KeySwitchDigits
 *
 * @param q
 * @param c1
 * @return sycl::event
 */
sycl::event keySwitchDigits_store_c1(sycl::queue &q,
                                     sycl::buffer<uint64_t> &c1);

/**
 * @brief store the output c2 of KeySwitchDigits
 *
 * @param q
 * @param c2
 * @return sycl::event
 */
sycl::event keySwitchDigits_store_c2(sycl::queue &q,
                                     sycl::buffer<uint64_t> &c2);

/**
 * @brief instance the ntt template
 *
 */
using keySwitchDigits_ntt_t =
    ntt<2, 8, COEFF_COUNT, pipe_keySwitchDigits_input,
        pipe_keySwitchDigits_prime_index, pipe_keySwitchDigits_ntt_output>;

/**
 * @brief Get the Key Switch Digits NTT instance
 *
 * @return keySwitchDigits_ntt_t&
 */
keySwitchDigits_ntt_t &GetKeySwitchDigitsNTT();
}  // namespace bgv
}  // namespace helib
}  // namespace L1
