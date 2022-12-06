#include <L0/keyswitchdigits.hpp>
#include <L0/load.hpp>
#include <L0/store.hpp>
#include <L1/common.h>
#include <L1/keyswitchdigits.h>

namespace L1 {
sycl::event keySwitchDigits(
    sycl::queue &q, sycl::buffer<ulong4> primes, sycl::buffer<ulong2> &keys1,
    sycl::buffer<ulong2> &keys2, sycl::buffer<ulong2> &keys3,
    sycl::buffer<ulong2> &keys4, sycl::buffer<uint64_t> &digit1,
    sycl::buffer<uint64_t> &digit2, sycl::buffer<uint64_t> &digit3,
    sycl::buffer<uint64_t> &digit4, sycl::buffer<uint64_t> &c0,
    sycl::buffer<uint64_t> &c1, unsigned num_digits, unsigned num_primes,
    ulong4 digits_offset, sycl::event depend_event, unsigned flag) {
  return L0::keySwitchDigits<COEFF_COUNT>(
      q, primes, keys1, keys2, keys3, keys4, digit1, digit2, digit3, digit4, c0,
      c1, num_digits, num_primes, digits_offset, depend_event, flag);
}

typedef struct keyswitchdigits_ops {
    sycl::event (*keySwitchDigits)(
        sycl::queue &q, sycl::buffer<ulong4> primes, sycl::buffer<ulong2> &keys1,
        sycl::buffer<ulong2> &keys2, sycl::buffer<ulong2> &keys3,
        sycl::buffer<ulong2> &keys4, sycl::buffer<uint64_t> &digit1,
        sycl::buffer<uint64_t> &digit2, sycl::buffer<uint64_t> &digit3,
        sycl::buffer<uint64_t> &digit4, sycl::buffer<uint64_t> &c0,
        sycl::buffer<uint64_t> &c1, unsigned num_digits, unsigned num_primes,
        ulong4 digits_offset, sycl::event depend_event, unsigned flag);
} keyswitchdigits_ops_t;

keyswitchdigits_ops_t& get_keyswitchdigits_ops() {
  static keyswitchdigits_ops_t keyswitch_ops_obj = {
    .keySwitchDigits = keySwitchDigits
  };
  return keyswitch_ops_obj;
}

}  // namespace L1
