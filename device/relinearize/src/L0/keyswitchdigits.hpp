#pragma once
#include "number-theory.hpp"
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

namespace L0 {
using PrefetchingLSU = ext::intel::lsu<ext::intel::prefetch<true>,
                                       ext::intel::statically_coalesce<false>>;

template <int coeff_count>
sycl::event keySwitchDigits(
    sycl::queue &q, sycl::buffer<ulong4> primes, sycl::buffer<ulong2> &keys1,
    sycl::buffer<ulong2> &keys2, sycl::buffer<ulong2> &keys3,
    sycl::buffer<ulong2> &keys4, sycl::buffer<uint64_t> &digit1,
    sycl::buffer<uint64_t> &digit2, sycl::buffer<uint64_t> &digit3,
    sycl::buffer<uint64_t> &digit4, sycl::buffer<uint64_t> &c0,
    sycl::buffer<uint64_t> &c1, unsigned num_digits, unsigned num_primes,
    ulong4 digits_offset, sycl::event depend_event, unsigned flag) {
  unsigned c0_size = num_primes * coeff_count;
  sycl::event e = q.submit([&](handler &h) {
    h.depends_on(depend_event);
    accessor acc_primes(primes, h, read_only);
    accessor acc_keys1(keys1, h, read_only);
    accessor acc_keys2(keys2, h, read_only);
    accessor acc_keys3(keys3, h, read_only);
    accessor acc_keys4(keys4, h, read_only);
    accessor acc_digit1(digit1, h, read_only);
    accessor acc_digit2(digit2, h, read_only);
    accessor acc_digit3(digit3, h, read_only);
    accessor acc_digit4(digit4, h, read_only);
    accessor acc_c0(c0, h, sycl::write_only, sycl::no_init);
    accessor acc_c1(c1, h, sycl::write_only, sycl::no_init);

    h.single_task<class keySwitchDigits>([=]() [[intel::kernel_args_restrict]] {
      PRINTF("num_digits = %d\n", num_digits);
      PRINTF("num_primes = %d\n", num_primes);
      auto acc_digit1_ptr = reinterpret_cast<ushort *>(
          acc_digit1.get_pointer().get() + digits_offset.s0());
      auto acc_digit2_ptr = reinterpret_cast<ushort *>(
          acc_digit2.get_pointer().get() + digits_offset.s1());
      auto acc_digit3_ptr = reinterpret_cast<ushort *>(
          acc_digit3.get_pointer().get() + digits_offset.s2());
      auto acc_digit4_ptr = reinterpret_cast<ushort *>(
          acc_digit4.get_pointer().get() + digits_offset.s3());

      ushort *acc_digit_ptr[4] = {acc_digit1_ptr, acc_digit2_ptr,
                                  acc_digit3_ptr, acc_digit4_ptr};

      auto acc_keys1_ptr =
          reinterpret_cast<uint *>(acc_keys1.get_pointer().get());
      auto acc_keys2_ptr =
          reinterpret_cast<uint *>(acc_keys2.get_pointer().get());
      auto acc_keys3_ptr =
          reinterpret_cast<uint *>(acc_keys3.get_pointer().get());
      auto acc_keys4_ptr =
          reinterpret_cast<uint *>(acc_keys4.get_pointer().get());
      uint *acc_keys_ptr[4] = {acc_keys1_ptr, acc_keys2_ptr, acc_keys3_ptr,
                               acc_keys4_ptr};

      uint64_t data[4];
      typedef ac_int<128, false> uint128_t;
      uint128_t keys[4];
      [[intel::disable_loop_pipelining]] for (int j = 0; j < num_primes; j++) {
        ulong4 tmp = acc_primes[j];
        ulong prime = tmp.s0();
        ulong prime_index_offset = tmp.s1();

        for (int m = 0; m < coeff_count * 4; m++) {
#pragma unroll
          for (int k = 0; k < 4; k++) {
            data[k] >>= 16;
            ulong r = BIT(flag, k) ? acc_digit_ptr[k][m] : 0;
            data[k] |= (r << 48);

            keys[k] >>= 32;
            uint128_t key = BIT(flag, k + 4) ? acc_keys_ptr[k][m] : 0;
            keys[k] |= (key << (128 - 32));
          }

          if ((m % 4) == 3) {
            ulong c0_elem = 0;
            ulong c1_elem = 0;
#pragma unroll
            for (int i = 0; i < 4; i++) {
              if (i < num_digits) {
                ulong key0 = *(ulong *)(&(keys[i]));
                keys[i] >>= 64;
                ulong key1 = *(ulong *)(&(keys[i]));
                c0_elem +=
                    MultiplyUIntMod(data[i], key0, prime, tmp.s2(), tmp.s3());
                c0_elem = MOD_ONCE(c0_elem, prime);
                c1_elem +=
                    MultiplyUIntMod(data[i], key1, prime, tmp.s2(), tmp.s3());
                c1_elem = MOD_ONCE(c1_elem, prime);
              }
            }
            acc_c0[j * coeff_count + m / 4] = c0_elem;
            acc_c1[c0_size + j * coeff_count + m / 4] = c1_elem;
          }
        }

#pragma unroll
        for (int k = 0; k < 4; k++) {
          acc_digit_ptr[k] += (coeff_count * 4);
          acc_keys_ptr[k] += (prime_index_offset + 1) * coeff_count * 4;
        }
      }
    });
  });
  return e;
}  // namespace L0
}  // namespace L0
