#pragma once
#include "number_theory.hpp"
#include <CL/sycl.hpp>

using namespace sycl;

namespace L0 {

// interleaved read
template <class kernelNameClass, int coeff_count, class pipe_load,
          class pipe_prime_index>
event keySwitchDigitsLoad(sycl::queue &q, sycl::buffer<uint64_t> &input,
                          sycl::buffer<uint8_t> &primes_index) {
  assert(primes_index.size() > 0);
  event e = q.submit([&](handler &h) {
    accessor _input(input, h, read_only);
    accessor primes_index_acc(primes_index, h, read_only);

    // get the num of primes according to the size of input
    int num_primes = input.size() / 2 / coeff_count;
    int digit2_offset = input.size() / 2;

    // launch kernel
    h.single_task<kernelNameClass>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < num_primes * 2; i++) {
        bool b_digit_0 = (i % 2) == 0;
        // two digits have the same prime index, just load the same
        auto prime_index = primes_index_acc[i / 2];
        pipe_prime_index::write(prime_index);
        for (int j = 0; j < coeff_count; j++) {
          auto addr = b_digit_0 ? i / 2 * coeff_count + j
                                : digit2_offset + i / 2 * coeff_count + j;
          auto val = _input[addr];
          pipe_load::write(val);
        }
      }
    });
  });
  return e;
}

template <int coeff_count, class pipe_input, class pipe_output_a,
          class pipe_output_b>
event keySwitchDigits(sycl::queue &q, sycl::buffer<ulong4> &primes,
                      sycl::buffer<ulong> &wa, sycl::buffer<ulong> &wb) {
  event e = q.submit([&](handler &h) {
    accessor _primes(primes, h, read_only);
    accessor _wa(wa, h, read_only);
    accessor _wb(wb, h, read_only);

    const int num_primes = primes.size();
    const int num_wa_primes = wa.size() / coeff_count / 2;
    const int num_wb_primes = wb.size() / coeff_count / 2;
    assert(num_wa_primes == num_wb_primes);
    printf("num_primes = %d, num_wa_primes = %d\n", num_primes, num_wa_primes);
    assert(num_primes <= num_wa_primes);

    h.single_task<class DyadMult>([=]() [[intel::kernel_args_restrict]] {
      ulong digit0_coeffs[coeff_count][2];
      for (int i = 0; i < num_primes * 2; i++) {
        // buffer all coeffs for the first digit as we need to sum them with the
        // second digit
        bool b_digit_0 = (i % 2) == 0;
        auto tmp = _primes[i / 2];
        auto prime = tmp.s0();

        // we need prime index to access the key-switching keys
        auto prime_index = tmp.s1();
        ASSERT(prime_index < num_wa_primes, "prime_index %d >= num_wa_primes\n",
               prime_index);

        // primes pre-computed parameters to optimize the multiplication mod
        auto prime_r = tmp.s2();
        auto prime_k = tmp.s3();
        PRINTF("i=%d, prime_index = %d\n", i, prime_index);

        for (int j = 0; j < coeff_count; j++) {
          ulong data = pipe_input::read();

          // dyadic multiplication
          // digit 0 + digit 1 + digit 0 + digit 1 + ....
          auto offset = b_digit_0 == false ? coeff_count : 0;
          auto res_a = MultiplyUIntMod(
              data, _wa[prime_index * coeff_count * 2 + offset + j], prime,
              prime_r, prime_k);
          auto res_b = MultiplyUIntMod(
              data, _wb[prime_index * coeff_count * 2 + offset + j], prime,
              prime_r, prime_k);

          // always read, but only use it for the second digit
          auto coeff_a = digit0_coeffs[j][0];
          auto coeff_b = digit0_coeffs[j][1];

          // update every time
          digit0_coeffs[j][0] = res_a;
          digit0_coeffs[j][1] = res_b;

          if (!b_digit_0) {
            res_a += coeff_a;
            res_a = MOD_ONCE(res_a, prime);
            pipe_output_a::write(res_a);
            res_b += coeff_b;
            res_b = MOD_ONCE(res_b, prime);
            pipe_output_b::write(res_b);
          }
        }
      }
    });
  });
  return e;
}
}  // namespace L0
