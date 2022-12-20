#include <L0/keyswitchdigits.hpp>
#include <L0/load.hpp>
#include <L0/store.hpp>
#include <L1/common.h>
#include <L1/pipes.h>

extern "C" {
using pipe_output = ext::intel::pipe<class KeySwitchDigitsOutput, ulong2>;
sycl::event KeySwitchDigits(
    sycl::queue &q, sycl::event depend_event, sycl::buffer<ulong4> &primes,
    sycl::buffer<ulong2> &keys, sycl::buffer<uint64_t> &digits,
    sycl::buffer<uint64_t> &c_input, sycl::buffer<uint64_t> &output1,
    sycl::buffer<uint64_t> &output2, unsigned num_digits, unsigned num_primes,
    unsigned num_normal_primes, unsigned num_all_primes, unsigned flag) {
  return L0::keySwitchDigits<COEFF_COUNT, pipe_key_switch_digits_ready,
                             pipe_data_ptr_c01, pipe_output>(
      q, depend_event, primes, keys, digits, c_input, output1, output2,
      num_digits, num_primes, num_normal_primes, num_all_primes, flag);
}

sycl::event StoreKeySwitchDigits(sycl::queue &q,
                                 sycl::buffer<uint64_t> &output1,
                                 sycl::buffer<uint64_t> &output2, unsigned size,
                                 unsigned flag) {
  return L0::StoreKeySwitchDigits<class StoreKeySwitchDigits, pipe_output>(
      q, output1, output2, size, flag);
}
}
