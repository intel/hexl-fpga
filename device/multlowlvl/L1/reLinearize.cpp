#include <L0/load.hpp>
#include <L0/scale.hpp>
#include <L0/store.hpp>
#include <L0/breakIntoDigits.hpp>
#include <L1/helib_bgv.h>
#include <L1/reLinearize.h>

namespace L1 {
namespace helib {
namespace bgv {
event C2DropSmall(sycl::queue &q, uint32_t coeff_count,
                  sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
                  uint32_t Q, uint I, uint64_t t) {
  // check inputs
  assert(P <= MAX_C2_DROP_SMALL_P);
  assert(Q <= MAX_C2_DROP_SMALL_Q);
  assert(coeff_count = COEFF_COUNT);
  return L0::scale<class C2DropSmallKernel, MAX_C2_DROP_SMALL_P,
                   MAX_C2_DROP_SMALL_P_BANKS, MAX_C2_DROP_SMALL_Q, COEFF_COUNT,
                   pipe_c2_drop_small_input, pipe_break_into_digits_input,
                   true, /* added primes at end */
                   false /* do not add special primes*/>(
      q, coeff_count, scale_param_set_buf, P, Q, I, t);
}

sycl::event C2DropSmallLoad(sycl::queue &q, sycl::buffer<uint64_t> &c) {
  return L0::generic_load<class C2DropSmallLoadKernel,
                          pipe_c2_drop_small_input>(q, c);
}

event BreakIntoDigits(sycl::queue &q,
                      sycl::buffer<ulong2> &packed_precomuted_params_buf,
                      uint num_digit1_primes, uint num_digit2_primes,
                      uint num_special_primes) {
  return L0::BreakIntoDigits<
      MAX_NORMAL_PRIMES, MAX_SPECIAL_PRIMES, BREAK_INTO_DIGITS_COEFF_COUNT,
      pipe_break_into_digits_input, pipe_break_into_digits_output>(
      q, packed_precomuted_params_buf, num_digit1_primes, num_digit2_primes,
      num_special_primes);
}

sycl::event BreakIntoDigits_store(sycl::queue &q, sycl::buffer<uint64_t> &c) {
  return L0::store<class BreakIntoDigitsStore, pipe_break_into_digits_output>(
      q, c);
}

sycl::event C0DropSmallLoad(sycl::queue &q, sycl::buffer<uint64_t> &c) {
  return L0::generic_load<class C0DropSmallLoadKernel,
                          pipe_c0_drop_small_input>(q, c);
}

event C0DropSmall(sycl::queue &q, uint32_t coeff_count,
                  sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
                  uint32_t Q, uint I, uint64_t t) {
  // check inputs
  assert(P <= MAX_C2_DROP_SMALL_P);
  assert(Q <= MAX_C2_DROP_SMALL_Q);
  assert(coeff_count = COEFF_COUNT);
  return L0::scale<class C0DropSmallKernel, MAX_C2_DROP_SMALL_P,
                   MAX_C2_DROP_SMALL_P_BANKS, MAX_C2_DROP_SMALL_Q, COEFF_COUNT,
                   pipe_c0_drop_small_input, pipe_c0_drop_small_output, true,
                   true /* add special primes */>(
      q, coeff_count, scale_param_set_buf, P, Q, I, t);
}

sycl::event C0Store(sycl::queue &q, sycl::buffer<uint64_t> &c) {
  return L0::store<class C0StoreKernel, pipe_reLinearizeC01_ntt_output>(q, c);
}

reLinearizeC01_ntt_t &GetReLinearizeC01NTT() {
  static reLinearizeC01_ntt_t ntt;
  return ntt;
}
sycl::event ReLinearizeC01LoadPrimeIndex(sycl::queue &q,
                                         sycl::uchar2 prime_index_start_end) {
  return L0::LoadPrimeIndexGeneric<class ReLinearizeC01LoadPrimeIndexKernel,
                                   pipe_reLinearizeC01_prime_index>(
      q, prime_index_start_end);
}

}  // namespace bgv
}  // namespace helib
}  // namespace L1
