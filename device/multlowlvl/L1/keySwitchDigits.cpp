#include <L0/keySwitchDigits.hpp>
#include <L0/store.hpp>
#include <L1/common.h>
#include <L1/keySwitchDigits.h>

namespace L1 {
namespace helib {
namespace bgv {
event keySwitchDigits(sycl::queue &q, sycl::buffer<ulong4> &primes,
                      sycl::buffer<ulong> &wa, sycl::buffer<ulong> &wb) {
  return L0::keySwitchDigits<COEFF_COUNT, pipe_keySwitchDigits_ntt_output,
                             pipe_keySwitchDigits_output_c1,
                             pipe_keySwitchDigits_output_c2>(q, primes, wa, wb);
}

sycl::event keySwitchDigits_load(sycl::queue &q, sycl::buffer<uint64_t> &input,
                                 sycl::buffer<uint8_t> &primes_index) {
  return L0::keySwitchDigitsLoad<class keySwitchDigitsLoad, COEFF_COUNT,
                                 pipe_keySwitchDigits_input,
                                 pipe_keySwitchDigits_prime_index>(
      q, input, primes_index);
}
sycl::event keySwitchDigits_store_c1(sycl::queue &q,
                                     sycl::buffer<uint64_t> &c1) {
  return L0::store<class keySwitchDigitsStoreC1,
                   pipe_keySwitchDigits_output_c1>(q, c1);
}
sycl::event keySwitchDigits_store_c2(sycl::queue &q,
                                     sycl::buffer<uint64_t> &c2) {
  return L0::store<class keySwitchDigitsStoreC2,
                   pipe_keySwitchDigits_output_c2>(q, c2);
}

keySwitchDigits_ntt_t &GetKeySwitchDigitsNTT() {
  static keySwitchDigits_ntt_t ntt;
  return ntt;
}
}  // namespace bgv
}  // namespace helib
}  // namespace L1
