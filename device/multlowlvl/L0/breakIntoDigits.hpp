#pragma once
#include "number_theory.hpp"
#include <CL/sycl.hpp>

using namespace sycl;

namespace L0 {
#define PSTAR_QJ_BANKS 16

// see the math
// An Improved RNS Variant of the BFV Homomorphic Encryption Scheme
// https://eprint.iacr.org/2018/117.pdf
template <int MAX_NORMAL_PRIMES, int MAX_SPECIAL_PRIMES, int coeff_count,
          class pipe_input, class pipe_output>
event BreakIntoDigits(sycl::queue &q,
                      sycl::buffer<ulong2> &packed_precomuted_params_buf,
                      uint num_digit1_primes, uint num_digit2_primes,
                      uint num_special_primes) {
  event e = q.submit([&](handler &h) {
    accessor packed_precomuted_params_buf_acc(packed_precomuted_params_buf, h,
                                              read_only);

    assert((num_digit2_primes + num_digit1_primes) < MAX_NORMAL_PRIMES);
    assert((MAX_NORMAL_PRIMES / 2) <= PSTAR_QJ_BANKS);
    h.single_task<class BreakIntoDigits>([=]() [[intel::kernel_args_restrict]] {
      uint64_t pi[MAX_NORMAL_PRIMES + MAX_SPECIAL_PRIMES];
      double pi_recip[MAX_NORMAL_PRIMES + MAX_SPECIAL_PRIMES];
      // MAX_DIGIT_SIZE is the maximum number of the digit primes
      ulong2 pstar_inv[MAX_NORMAL_PRIMES];
      ulong2 pstar_inv_recip[MAX_NORMAL_PRIMES];
      // pi*^qj
      // MAX_NORMAL_PRIMES should be an even number otherwise we need one more
      // space
      [[intel::numbanks(PSTAR_QJ_BANKS)]] ulong2
          pstar_qj[(MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES / 2) * 2]
                  [PSTAR_QJ_BANKS];
      // P (prod of digit 1 or digit 2) mod qj
      ulong P_qj[MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES / 2][2];
      ulong2 P_inv[MAX_NORMAL_PRIMES / 2];

      // MAX_P should be smaller than MAX_P_BANKS
      [[intel::numbanks(8)]] ulong c[coeff_count * 2][8];
      ulong c2[coeff_count * 2];

      ASSERT((MAX_NORMAL_PRIMES / 2) <= 9,
             "Each digit should be no more than 9 primes");
      double v[coeff_count][2];

      int num_all_primes =
          num_digit1_primes + num_digit2_primes + num_special_primes;
      int num_all_normal_primes = num_digit1_primes + num_digit2_primes;

      int offset1 = num_all_primes;
      int offset2 = offset1 + num_all_normal_primes;
      int offset3 = offset2 + num_all_normal_primes;
      int offset4 = offset3 + MAX(num_digit1_primes, num_digit2_primes) +
                    num_special_primes;
      int offset5 = offset4 + num_digit2_primes;

      // pstar_qj
      int offset6 = offset5 + (num_digit2_primes + num_special_primes) *
                                  num_digit1_primes;
      int packed_params_size =
          offset6 +
          (num_digit1_primes + num_special_primes) * num_digit2_primes;
      ASSERT(packed_params_size <= packed_precomuted_params_buf_acc.size(),
             "overflow");
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
          P_qj[i - offset3][0] = data.s0();
          P_qj[i - offset3][1] = data.s1();
        } else if (i < offset5) {
          P_inv[i - offset4] = data;
        } else {
          ushort pos;
          ushort cols;
          if (i < offset6) {
            pos = i - offset5;
            cols = num_digit1_primes;
          } else {
            pos = i - offset6 +
                  (MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES / 2) *
                      num_digit2_primes;
            cols = num_digit2_primes;
          }
          pstar_qj[pos / cols][pos % cols] = data;
        }
      }

      uint num_stage_primes = num_all_primes * 2;

      [[intel::disable_loop_pipelining]] for (int i = 0; i < num_stage_primes;
                                              i++) {
        // stage 1: digit 1 part 1 = P1(1)
        // stage 2: digit 1 part 1 ext to part 2 P1(2)
        //          (digit 2 part 2 - P1(2)) / pi(prod of P1 primes)
        // stage 3: digit 1 part 3 = P1(3)
        // stage 4: digit 2 part 1 = P2(1)
        // stage 5: write P2(2)
        // stage 6: digit 2 part 3 = P2(3)
        // so the order of digit 1 is P1(1,2,3), and
        // the order of digit 2 is P2(2,1,3)
        bool b_stage_1 = i < num_digit1_primes;
        bool b_stage_2 =
            (!b_stage_1) && (i < (num_digit1_primes + num_digit2_primes));
        bool b_stage_3 = (!b_stage_1) && (!b_stage_2) && (i < num_all_primes);
        bool b_stage_4 = (!b_stage_1) && (!b_stage_2) && (!b_stage_3) &&
                         (i < (num_all_primes + num_digit1_primes));
        bool b_stage_5 =
            (!b_stage_1) && (!b_stage_2) && (!b_stage_3) && (!b_stage_4) &&
            (i < (num_all_primes + num_digit1_primes + num_digit2_primes));

        [[intel::ivdep(c)]] [[intel::ivdep(c2)]] [
            [intel::ivdep(v)]] for (uint n = 0; n < coeff_count; n++) {
          ulong c_i;
          // only the first and second stage read the input
          if (b_stage_1 || b_stage_2) {
            c_i = pipe_input::read();
          }

          // variable for the output
          ulong out;

          int P_base_ptr;
          int pstar_qj_i = 0;
          ulong P_qj_cur;
          uint64_t qj;
          double cur_v;

          long x = 0;

          bool b_ext = true;

          if (b_stage_1) {
            // stage 1: digit 1 part 1 - P1(1)
            // write the same value
            out = c_i;
            b_ext = false;
          } else if (b_stage_2) {
            // perform P1(2)
            // P_base_ptr starts from 0 of each base
            P_base_ptr = i - num_digit1_primes;
            pstar_qj_i = P_base_ptr;
            P_qj_cur = P_qj[P_base_ptr][0];
            qj = pi[i];
            cur_v = v[n][0];
          } else if (b_stage_3) {
            // stage 3: digit 1 part 3 = P1(3)
            pstar_qj_i = i - num_digit1_primes;
            P_qj_cur = P_qj[i - num_digit1_primes][0];
            qj = pi[i];
            cur_v = v[n][0];
          } else if (b_stage_4) {
            // Perform P2(1)
            pstar_qj_i =
                i - num_all_primes + MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES / 2;
            P_qj_cur = P_qj[i - num_all_primes][1];
            qj = pi[i - num_all_primes];
            cur_v = v[n][1];
          } else if (b_stage_5) {
            // write P2(2)
            P_base_ptr = i - num_all_primes - num_digit1_primes;

            b_ext = false;
          } else {
            // Perform P2(3)
            pstar_qj_i = i - num_all_primes - num_digit2_primes +
                         MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES / 2;
            P_qj_cur = P_qj[i - num_all_primes - num_digit2_primes][1];
            qj = pi[i - num_all_primes];
            cur_v = v[n][1];
          }

          // part2 maybe larger than qj, so have to mod once with multiple
          // times
          int cur_v_int = (int)cur_v;
          ulong part2 = cur_v_int * P_qj_cur;

          uint64_t c_list[8];
#pragma unroll
          for (int j = 0; j < 8; j++) {
            c_list[j] = c[(b_stage_2 || b_stage_3) ? n : n + coeff_count][j];
          }
#pragma unroll
          for (int j = 0; j < 8; j++) {
            ASSERT(
                pstar_qj_i < ((MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES / 2) * 2),
                "OVERFLOW %d\n", pstar_qj_i);
            x += mulmod(c_list[j], pstar_qj[pstar_qj_i][j], qj);
            x = MOD_ONCE(x, qj);
            part2 = MOD_ONCE(part2, qj);
          }
          if ((MAX_NORMAL_PRIMES / 2) == 9) {
            x += mulmod(c2[(b_stage_2 || b_stage_3) ? n : n + coeff_count],
                        pstar_qj[pstar_qj_i][8], qj);
            x = MOD_ONCE(x, qj);
            part2 = MOD_ONCE(part2, qj);
          }

          // PRINTF("x = %ld, v = %f, P_qj_cur = %ld\n", x, cur_v, P_qj_cur);

          // substract the integer part, for example (3.4-3)*P = 0.4P
          x = x - part2;
          x = MOD_ONCE(x + qj, qj);

          // x is in the range of [-P_qj_cur/2, P_qj_cur/2]
          double rem = cur_v - cur_v_int;
          if (rem > 0.5) x = x - P_qj_cur;
          x = MOD_ONCE(x + qj, qj);

          if (b_ext) {
            out = x;
            // PRINTF("Send x %ld, rem = %f\n", x, rem);
          }

          if (b_stage_5) {
            out =
                mulmod(c_list[P_base_ptr], pstar_inv_recip[i - num_all_primes],
                       pi[i - num_all_primes]);
          }

          if (b_stage_2) {
            // perform P2(2)
            c_i = MOD_ONCE(c_i + qj - x, qj);
            c_i = mulmod(c_i, P_inv[i - num_digit1_primes], qj);
          }

          if (b_stage_1 || b_stage_2) {
            ulong c_pi = mulmod(c_i, pstar_inv[i], pi[i]);
            auto tmp = (c_pi * pi_recip[i]);
            int c_prime_index = b_stage_2 ? P_base_ptr : i;
            if (c_prime_index < 8) {
              c[b_stage_2 ? (n + coeff_count) : n][b_stage_2 ? P_base_ptr : i] =
                  c_pi & BIT_MASK(MAX_PRIME_BITS);
            } else {
              c2[b_stage_2 ? (n + coeff_count) : n] =
                  c_pi & BIT_MASK(MAX_PRIME_BITS);
            }
            if (b_stage_2) {
              if (num_digit1_primes == i)
                v[n][1] = tmp;
              else
                v[n][1] += tmp;
            } else {
              if (0 == i) {
                v[n][0] = tmp;
              } else {
                v[n][0] += tmp;
              }
            }
          }

          pipe_output::write(out);
        }
      }
    });
  });
  return e;
}  // namespace L0
}  // namespace L0
