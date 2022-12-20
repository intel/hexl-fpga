#include <L0/load.hpp>
#include <L0/scale.hpp>
#include <L0/store.hpp>
#include <L0/breakintodigits.hpp>
#include <L1/common.h>
#include <L1/intt.h>
#include <L1/ntt.h>
#include <L1/pipes.h>
#include <L1/launch_ntt.hpp>

using namespace L1;

extern "C" {
/**
 * @brief pipes for reLinearize
 *
 */
using pipe_break_into_digits_input =
    ext::intel::pipe<class BreakIntoDigitsInputPipeId, uint64_t, 4>;
using pipe_break_into_digits_output =
    ext::intel::pipe<class BreakIntoDigitsOutputPipeId, uint64_t, 4>;

using pipe_break_into_digits_intt_input =
    ext::intel::pipe<class BreakIntoDigitsINTTInputPipeId, uint64_t, 4>;
using pipe_break_into_digits_intt_primes_index =
    ext::intel::pipe<class BreakIntoDigitsINTTPrimeIndexPipeId, uint64_t, 4>;
using pipe_break_into_digits_store_offset =
    ext::intel::pipe<class BreakIntoDigitsStoreOffsetPipeId, uint, 4>;

using pipe_breakIntoDigits_prime_index =
    ext::intel::pipe<class BreakIntoDigitsPrimeIndexPipeId, uint8_t, 128>;
using pipe_breakIntoDigits_ntt_output =
    ext::intel::pipe<class BreakIntoDigitsNTTOutputPipeId, uint64_t, 128>;

/**
 * @brief instance the intt template
 *
 */
using break_into_digits_intt_t =
    L1::intt::intt<3, 8, COEFF_COUNT, pipe_break_into_digits_intt_input,
                   pipe_break_into_digits_intt_primes_index,
                   pipe_break_into_digits_input>;

/**
 * @brief instance the ntt template
 *
 */
using breakIntoDigits_ntt_t =
    L1::ntt::ntt<4, 8, COEFF_COUNT, pipe_break_into_digits_output,
                 pipe_breakIntoDigits_prime_index,
                 pipe_breakIntoDigits_ntt_output>;

sycl::event LoadBreakIntoDigits(sycl::queue &q, sycl::buffer<uint64_t> &c,
                                unsigned size, int flag, sycl::event depend) {
  return L0::generic_load_with_prime_index<
      class BreakIntoDigitsLoad, pipe_break_into_digits_intt_input,
      pipe_break_into_digits_intt_primes_index, pipe_break_into_digits_ready,
      COEFF_COUNT>(q, c, size, flag, depend);
}

sycl::event BreakIntoDigits(sycl::queue &q,
                            sycl::buffer<ulong2> &packed_precomuted_params_buf,
                            uint num_digits, uint num_digit1_primes,
                            uint num_digit2_primes, uint num_digit3_primes,
                            uint num_digit4_primes, uint num_special_primes,
                            uint special_primes_offset, sycl::event depend,
                            int flag) {
  return L0::BreakIntoDigits<
      MAX_NORMAL_PRIMES, MAX_SPECIAL_PRIMES, COEFF_COUNT, MAX_DIGITS,
      MAX_DIGIT_SIZE, MAX_DIGIT_SIZE_POW_2, pipe_break_into_digits_input,
      pipe_break_into_digits_output, pipe_breakIntoDigits_prime_index,
      pipe_break_into_digits_store_offset>(
      q, packed_precomuted_params_buf, num_digits, num_digit1_primes,
      num_digit2_primes, num_digit3_primes, num_digit4_primes,
      num_special_primes, special_primes_offset, depend, flag);
}

sycl::event StoreBreakIntoDigits(sycl::queue &q, sycl::buffer<uint64_t> &c,
                                 unsigned size, int flag) {
  return L0::store<class BreakIntoDigitsStore, pipe_breakIntoDigits_ntt_output,
                   pipe_break_into_digits_store_offset,
                   pipe_key_switch_digits_ready, COEFF_COUNT>(q, c, size, flag);
}

void LaunchBreakIntoDigitsINTT(const std::vector<uint64_t> &primes,
                               uint64_t coeff_count, int flag) {
  static break_into_digits_intt_t intt;
  launch_intt(intt, primes, coeff_count, flag);
}

void LaunchBreakIntoDigitsNTT(const std::vector<uint64_t> &primes,
                              uint64_t coeff_count, int flag) {
  static breakIntoDigits_ntt_t ntt;
  launch_ntt(ntt, primes, coeff_count, flag);
}
}