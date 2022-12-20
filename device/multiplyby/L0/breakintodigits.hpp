#pragma once
#include "number-theory.hpp"
#include <CL/sycl.hpp>

using namespace sycl;

namespace L0 {
// see the math
// An Improved RNS Variant of the BFV Homomorphic Encryption Scheme
// https://eprint.iacr.org/2018/117.pdf
template <int MAX_NORMAL_PRIMES, int MAX_SPECIAL_PRIMES, int coeff_count,
          int MAX_DIGITS, int MAX_DIGIT_SIZE, int MAX_DIGIT_SIZE_POW_2,
          class pipe_input, class pipe_output, class pipe_prime_index_output,
          class pipe_break_into_digits_store_offset>
event BreakIntoDigits(sycl::queue &q,
                      sycl::buffer<ulong2> &packed_precomuted_params_buf,
                      uint num_digits, uint num_digit1_primes,
                      uint num_digit2_primes, uint num_digit3_primes,
                      uint num_digit4_primes, uint num_special_primes,
                      uint special_primes_offset, sycl::event depend,
                      int flag) {
  event e = q.submit([&](handler &h) {
    // h.depends_on(depend);
    accessor packed_precomuted_params_buf_acc(packed_precomuted_params_buf, h,
                                              read_only);

    assert((num_digit2_primes + num_digit1_primes + num_digit3_primes +
            num_digit4_primes) < MAX_NORMAL_PRIMES);
    assert(num_digits <= MAX_DIGITS);
    assert(num_digit1_primes <= MAX_DIGIT_SIZE);
    assert(num_digit2_primes <= MAX_DIGIT_SIZE);
    assert(num_digit3_primes <= MAX_DIGIT_SIZE);
    assert(num_digit4_primes <= MAX_DIGIT_SIZE);
    assert(MAX_DIGIT_SIZE <= MAX_DIGIT_SIZE_POW_2);

    h.single_task<class BreakIntoDigits>([=]() [[intel::kernel_args_restrict]] {
      uint64_t pi[MAX_NORMAL_PRIMES + MAX_SPECIAL_PRIMES];
      double pi_recip[MAX_NORMAL_PRIMES + MAX_SPECIAL_PRIMES];
      ulong2 pstar_inv[MAX_NORMAL_PRIMES];
      ulong2 pstar_inv_recip[MAX_NORMAL_PRIMES];
      // pi*^qj
      [[intel::numbanks(MAX_DIGIT_SIZE_POW_2)]] ulong2
          pstar_qj[(MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES) * MAX_DIGITS]
                  [MAX_DIGIT_SIZE_POW_2];
      // P (prod of each digit) mod qj
      ulong P_qj[(MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES) * MAX_DIGITS];

      [[intel::numbanks(
          MAX_DIGIT_SIZE_POW_2)]] ulong c[coeff_count][MAX_DIGIT_SIZE_POW_2];

      double v[coeff_count];
      ulong2 qhat_inv[MAX_NORMAL_PRIMES];

      PRINTF("num_digits = %d\n", num_digits);
      PRINTF("num_digit1_primes = %d\n", num_digit1_primes);
      PRINTF("num_digit2_primes = %d\n", num_digit2_primes);
      PRINTF("num_digit3_primes = %d\n", num_digit3_primes);
      PRINTF("num_digit4_primes = %d\n", num_digit4_primes);
      PRINTF("num_special_primes = %d\n", num_special_primes);

      int num_all_normal_primes = num_digit1_primes + num_digit2_primes +
                                  num_digit3_primes + num_digit4_primes;
      int num_all_primes = num_all_normal_primes + num_special_primes;

      int offset1 = num_all_primes;
      int offset2 = offset1 + num_all_normal_primes;
      int offset3 = offset2 + num_all_normal_primes;
      int offset4 = offset3 + num_all_primes * num_digits;
      int offset5 = offset4 + num_all_normal_primes;

      // pstar_qj
      int offset6 = offset5 + (MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES) *
                                  num_digits * MAX_DIGIT_SIZE;

      int packed_params_size = offset6;
      // ASSERT2(packed_params_size == packed_precomuted_params_buf_acc.size());
      for (int i = 0; i < packed_params_size; i++) {
        auto data = packed_precomuted_params_buf_acc[i];
        if (i < offset1) {
          pi[i] = data.s0();
          pi_recip[i] = *(double *)&(data.s1());
        } else if (i < offset2) {
          pstar_inv[i - offset1] = data;
        } else if (i < offset3) {
          pstar_inv_recip[i - offset2] = data;
        } else if (i < offset4) {
          P_qj[i - offset3] = data.s0();
        } else if (i < offset5) {
          qhat_inv[i - offset4] = data;
        } else {
          ushort pos = i - offset5;
          pstar_qj[pos / MAX_DIGIT_SIZE][pos % MAX_DIGIT_SIZE] = data;
        }
      }

      int digit_read_size[MAX_DIGITS];
      digit_read_size[0] = num_digit1_primes;
      digit_read_size[1] = num_digit2_primes;
      digit_read_size[2] = num_digit3_primes;
      digit_read_size[3] = num_digit4_primes;

      int qj_pi_index_offset[MAX_DIGITS];
      qj_pi_index_offset[0] = num_digit1_primes;
      qj_pi_index_offset[1] = qj_pi_index_offset[0] + num_digit2_primes;
      qj_pi_index_offset[2] = qj_pi_index_offset[1] + num_digit3_primes;
      qj_pi_index_offset[3] = qj_pi_index_offset[2] + num_digit4_primes;

      uchar prime_index_reordered[MAX_NORMAL_PRIMES + MAX_SPECIAL_PRIMES]
                                 [MAX_DIGITS - 1];

      [[intel::disable_loop_pipelining]] for (int i = 0;
                                              i < num_digits * num_all_primes;
                                              i++) {
        int digit_row_index = i / num_all_primes;
        int prime_index = i % num_all_primes;

        // read digit base primes
        bool b_read = prime_index < digit_read_size[digit_row_index];

        // qj index, skip the number of digit base primes
        auto qj_index = prime_index - digit_read_size[digit_row_index];

        // the pi index for qj
        // 9 ~ MAX
        // 0 ~ 9, 18 ~ MAX
        // 0 ~ 18, 26 ~ MAX
        uint64_t qj_pi_index =
            prime_index < qj_pi_index_offset[digit_row_index]
                ? prime_index - digit_read_size[digit_row_index]
                : prime_index;

        // fixed size is MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES per digit
        auto pstar_qj_i =
            (MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES) * digit_row_index +
            qj_index;

        //
        uchar ntt_prime_index;
        if (b_read) {
          ntt_prime_index =
              digit_row_index == 0
                  ? prime_index
                  : prime_index + qj_pi_index_offset[digit_row_index - 1];
        } else {
          ntt_prime_index = qj_pi_index;
        }

        // store offset
        if (BIT(flag, 1)) {
          pipe_break_into_digits_store_offset::write(
              ntt_prime_index + digit_row_index * num_all_primes);
        }

        // fix the offset of special primes
        if (ntt_prime_index >= num_all_normal_primes) {
          ntt_prime_index =
              special_primes_offset + (ntt_prime_index - num_all_normal_primes);
        }
        if (BIT(flag, 2)) {
          pipe_prime_index_output::write(ntt_prime_index);
        }
#if DEBUG_BREAK_INTO_DIGITS
        PRINTF(
            "BreakIntoDigits: digit_row_index = %d, qj_index = %d, qj_pi_index "
            "= %d, ntt_prime_index = % d\n ",
            digit_row_index, qj_index, qj_pi_index, ntt_prime_index);
#endif
        for (int j = 0; j < coeff_count; j++) {
          uint64_t out;

          if (b_read) {
            // pi base
            int pi_offset =
                digit_row_index == 0
                    ? prime_index
                    : qj_pi_index_offset[digit_row_index - 1] + prime_index;
            auto c_i = BIT(flag, 0) ? pipe_input::read() : 0;
            auto c_i_qhat_inv = mulmod(c_i, qhat_inv[pi_offset], pi[pi_offset]);
            auto c_pi =
                mulmod(c_i_qhat_inv, pstar_inv[pi_offset], pi[pi_offset]);

            ASSERT2(prime_index <= MAX_DIGIT_SIZE);
            c[j][prime_index] = c_pi & BIT_MASK(MAX_PRIME_BITS);
            double tmp = c_pi * pi_recip[pi_offset];
            v[j] = prime_index == 0 ? tmp : v[j] + tmp;

            out = c_i_qhat_inv;
          } else {
            // ext
            double cur_v = v[j];
            // if (j == 0) PRINTF("cur_v = %f\n", cur_v);
            int cur_v_int = (int)cur_v;
            auto P_qj_cur = P_qj[qj_index + digit_row_index * num_all_primes];
            ulong part2 = cur_v_int * P_qj_cur;
            uint64_t x = 0;
            uint64_t qj = pi[qj_pi_index];

#pragma unroll
            for (int k = 0; k < MAX_DIGIT_SIZE; k++) {
              ASSERT2(pstar_qj_i <
                      ((MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES) * num_digits));
              x += mulmod(c[j][k] & BIT_MASK(MAX_PRIME_BITS),
                          pstar_qj[pstar_qj_i][k], qj);
              x = MOD_ONCE(x, qj);
              part2 = MOD_ONCE(part2, qj);
            }

            // substract the integer part, for example (3.4-3)*P = 0.4P
            x = x - part2;
            x = MOD_ONCE(x + qj, qj);

            // x is in the range of [-P_qj_cur/2, P_qj_cur/2]
            double rem = cur_v - cur_v_int;
            if (rem > 0.5) x = x - P_qj_cur;
            x = MOD_ONCE(x + qj, qj);

            out = x;
          }
          if (out == 139691621315616L) {
            PRINTF("-------------------%d,%d\n", i, j);
          }
          if (BIT(flag, 3)) {
            pipe_output::write(out);
          }
        }
      }
    });
  });
  return e;
}  // namespace L0
}  // namespace L0
