#pragma once
#include "number-theory.hpp"
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

namespace L0 {
using PrefetchingLSU = ext::intel::lsu<ext::intel::prefetch<true>,
                                       ext::intel::statically_coalesce<false>>;

template <int coeff_count, class pipe_ready, class pipe_data_ptr_c01,
          class pipe_output>
sycl::event keySwitchDigits(
    sycl::queue &q, sycl::event depend_event, sycl::buffer<ulong4> &primes,
    sycl::buffer<ulong2> &keys, sycl::buffer<uint64_t> &digits,
    sycl::buffer<uint64_t> &c_input, sycl::buffer<uint64_t> &c_output1,
    sycl::buffer<uint64_t> &c_output2, unsigned num_digits, unsigned num_primes,
    unsigned num_normal_primes, unsigned num_all_primes, unsigned flag) {
  unsigned c0_size = num_primes * coeff_count;
  unsigned c_input_offset = num_normal_primes * coeff_count;
  sycl::event e = q.submit([&](handler &h) {
    // h.depends_on(depend_event);
    accessor acc_primes(primes, h, read_only);
    accessor acc_keys1(keys, h, read_only);
    accessor acc_keys2(keys, h, read_only);
    accessor acc_keys3(keys, h, read_only);
    accessor acc_keys4(keys, h, read_only);
#if 0
    accessor acc_digit1(digits, h, read_only);
    accessor acc_digit2(digits, h, read_only);
    accessor acc_digit3(digits, h, read_only);
    accessor acc_digit4(digits, h, read_only);
    accessor acc_c0_in(c_input, h, read_only);
    accessor acc_c1_in(c_input, h, read_only);
    accessor acc_c0_ou(c_output1, h, sycl::write_only, sycl::no_init);
    accessor acc_c1_ou(c_output2, h, sycl::write_only, sycl::no_init);
#endif
    h.single_task<class keySwitchDigits>([=]() [[intel::kernel_args_restrict]] {
      PRINTF("num_digits = %d\n", num_digits);
      PRINTF("num_primes = %d\n", num_primes);
      // wait on the blocking pipe_read until notified by the producer
      uint64_t data_ptr_c01_int = pipe_data_ptr_c01::read();
      uint64_t *data_ptr_c01 = (uint64_t *)data_ptr_c01_int;
      uint64_t *digits_ptr = (uint64_t *)pipe_ready::read();

      // use atomic_fence to ensure memory ordering
      atomic_fence(memory_order::seq_cst, memory_scope::device);

      auto digit_offset = num_primes * coeff_count;
      auto acc_digit1_ptr = reinterpret_cast<ushort *>(digits_ptr);
      auto acc_digit2_ptr =
          reinterpret_cast<ushort *>(digits_ptr + digit_offset);
      auto acc_digit3_ptr =
          reinterpret_cast<ushort *>(digits_ptr + digit_offset * 2);
      auto acc_digit4_ptr =
          reinterpret_cast<ushort *>(digits_ptr + digit_offset * 3);

      ushort *acc_digit_ptr[4] = {acc_digit1_ptr, acc_digit2_ptr,
                                  acc_digit3_ptr, acc_digit4_ptr};

      auto keys_offset = num_all_primes * coeff_count;
      auto acc_keys1_ptr =
          reinterpret_cast<uint *>(acc_keys1.get_pointer().get());
      auto acc_keys2_ptr =
          reinterpret_cast<uint *>(acc_keys2.get_pointer().get() + keys_offset);
      auto acc_keys3_ptr = reinterpret_cast<uint *>(
          acc_keys3.get_pointer().get() + keys_offset * 2);
      auto acc_keys4_ptr = reinterpret_cast<uint *>(
          acc_keys4.get_pointer().get() + keys_offset * 3);
      uint *acc_keys_ptr[4] = {acc_keys1_ptr, acc_keys2_ptr, acc_keys3_ptr,
                               acc_keys4_ptr};
      auto c0_in_ptr = reinterpret_cast<ushort *>(data_ptr_c01);
      auto c1_in_ptr =
          reinterpret_cast<ushort *>(data_ptr_c01 + c_input_offset);

      uint64_t data[4];
      typedef ac_int<128, false> uint128_t;
      uint128_t keys[4];
      uint64_t c0_in, c1_in;
      [[intel::disable_loop_pipelining]] for (int j = 0; j < num_primes; j++) {
        auto tmp = acc_primes[j];
        ulong prime = tmp.s0() & BIT_MASK(60);
        ulong prime_index_offset = tmp.s0() >> 60;
        ulong P_pi = tmp.s1();  // prod of special primes mod pi

        [[intel::ivdep]] for (int m = 0; m < coeff_count * 4; m++) {
#pragma unroll
          for (int k = 0; k < 4; k++) {
            data[k] >>= 16;
            ulong r = BIT(flag, k) ? acc_digit_ptr[k][m] : 0;
            data[k] |= (r << 48);

            keys[k] >>= 32;
            uint128_t key = BIT(flag, k + 4) ? acc_keys_ptr[k][m] : 0;
            keys[k] |= (key << (128 - 32));
          }

          c0_in >>= 16;
          c1_in >>= 16;
          uint64_t c0_in_ushort =
              j < num_normal_primes && BIT(flag, 10) ? c0_in_ptr[m] : 0;
          uint64_t c1_in_ushort =
              j < num_normal_primes && BIT(flag, 11) ? c1_in_ptr[m] : 0;
          c0_in_ushort <<= 48;
          c1_in_ushort <<= 48;
          c0_in |= c0_in_ushort;
          c1_in |= c1_in_ushort;

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
            auto c0_orig_val =
                MultiplyUIntMod(c0_in, P_pi, prime, tmp.s2(), tmp.s3());
            auto c1_orig_val =
                MultiplyUIntMod(c1_in, P_pi, prime, tmp.s2(), tmp.s3());
            pipe_output::write({MOD_ONCE(c0_elem + c0_orig_val, prime),
                                MOD_ONCE(c1_elem + c1_orig_val, prime)});
          }
        }

#pragma unroll
        for (int k = 0; k < 4; k++) {
          acc_digit_ptr[k] += (coeff_count * 4);
          acc_keys_ptr[k] += (prime_index_offset + 1) * coeff_count * 4;
        }

        c0_in_ptr += (coeff_count * 4);
        c1_in_ptr += (coeff_count * 4);
      }
    });
  });
  return e;
}

template <class KernelNameClass, class pipe_output>
event StoreKeySwitchDigits(sycl::queue &q, sycl::buffer<uint64_t> &data1,
                           sycl::buffer<uint64_t> &data2, unsigned size,
                           int flag) {
  event e = q.submit([&](handler &h) {
    accessor acc1(data1, h, sycl::write_only, sycl::no_init);
    accessor acc2(data2, h, sycl::write_only, sycl::no_init);
    h.single_task<KernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      for (unsigned i = 0; i < size; i++) {
        auto tmp = pipe_output::read();
        if (BIT(flag, 0)) {
          acc1[i] = tmp.s0();
        }
        if (BIT(flag, 1)) {
          acc2[size + i] = tmp.s1();
        }
      }
    });
  });
  return e;
}
}  // namespace L0
