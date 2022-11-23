#pragma once
#include "number-theory.hpp"
#include <CL/sycl.hpp>

using namespace sycl;

namespace L0 {
// a none pipe
using pipe_none = ext::intel::pipe<class NonePipeId, uint8_t, 0>;

// see the math
// https://github.com/intel-sandbox/hexl-fpga-helib/blob/master/docs/scale.md
template <class KernelName, int MAX_P, int MAX_P_BANKS, int MAX_Q,
          int MAX_COEFF_COUNT, class pipe_scale_input, class pipe_scale_output,
          class pipe_scale_output_prime_index, bool added_primes_at_end,
          bool b_add_special_primes = false,
          bool pipe_scale_output_prime_index_valid = true>
event scale(sycl::queue &q, uint32_t coeff_count,
            sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P, uint32_t Q,
            uint I, uint64_t t) {
  event e = q.submit([&](handler &h) {
    accessor scale_param_set_acc(scale_param_set_buf, h, read_only);
    int scale_param_set_acc_size = scale_param_set_acc.size();

    // MAX_P should not larger than MAX_P_BANKS
    assert(MAX_P <= MAX_P_BANKS);

    h.single_task<KernelName>([=]() [[intel::kernel_args_restrict]] {
      // MAX_P is the maximum diff primes, usually it is 4
      ulong2 pstar_inv[MAX_P];
      // pi^qj
      ulong2 pstar_qj[MAX_Q][MAX_P_BANKS];
      // (pi^* mod t * P^-1 mod t) mod t
      ulong2 pt[MAX_P];
      // qj, the second element is related rk for barrett reduction
      uint64_t qj[MAX_Q];
      uint8_t qj_prime_index[MAX_Q];
      // P mod qj
      ulong2 P_qj[MAX_Q];
      // Prod of Special Primes mod qj
      ulong2 R_qj[MAX_Q];
      // pi, the second element is related rk for barrett reduction
      uint64_t pi[MAX_P];
      double pi_recip[MAX_P];
      // MAX_P should be smaller than MAX_P_BANKS
      [[intel::numbanks(MAX_P_BANKS)]] ulong c[MAX_COEFF_COUNT][MAX_P_BANKS];
      int ks[MAX_COEFF_COUNT];
      double v[MAX_COEFF_COUNT];

      // pi - P
      // qj - Q
      // pt - P
      // pstar_inv - P
      // P_qj - Q
      // R_qj - Q if applicable
      // pstar_qj - Q*P
      ushort offset1 = P;
      ushort offset2 = offset1 + Q;
      ushort offset3 = offset2 + P;
      ushort offset4 = offset3 + P;
      ushort offset5 = offset4 + Q;
      ushort offset6 = offset5 + (b_add_special_primes ? Q : 0);
      ushort scale_param_set_size = offset6 + Q * P;
      ASSERT(scale_param_set_acc_size == scale_param_set_size,
             "Warning - param size is mismatched\n");
      for (int i = 0; i < scale_param_set_size; i++) {
        auto data = scale_param_set_acc[i];
        if (i < offset1) {
          pi[i] = data.s0();
          pi_recip[i] = *(double *)&(data.s1());
        } else if (i < offset2) {
          qj[i - offset1] = data.s0();
          qj_prime_index[i - offset1] = data.s1();
        } else if (i < offset3) {
          pt[i - offset2] = data;
        } else if (i < offset4) {
          pstar_inv[i - offset3] = data;
        } else if (i < offset5) {
          P_qj[i - offset4] = data;
        } else if (i < offset6) {
          R_qj[i - offset5] = data;
        } else {
          ushort pos = i - offset6;
          pstar_qj[pos / (ushort)P][pos % (ushort)P] = data;
        }
      }

      [[intel::disable_loop_pipelining]] for (int poly = 0; poly < 2; poly++) {
        // cache P's coeffs
        [[intel::disable_loop_pipelining]] for (uint i = 0; i < P; i++) {
          for (uint n = 0; n < coeff_count; n++) {
            ulong c_i = pipe_scale_input::read();
            ASSERT(c_i < pi[i], "%lu > %lu\n", c_i, pi[i]);
            ulong c_pi = mulmod(c_i, pstar_inv[i], pi[i]);
            c[n][i] = c_pi & BIT_MASK(MAX_PRIME_BITS);
            ks[n] = (i == 0 ? 0 : ks[n]) + mulmod(c_pi, pt[i], t);
            v[n] = (i == 0 ? 0 : v[n]) + c_pi * pi_recip[i];
          }
        }

        [[intel::disable_loop_pipelining]] for (uint32_t j = 0; j < Q; j++) {
          uint64_t qj_prime = qj[j];
          if (pipe_scale_output_prime_index_valid) {
            pipe_scale_output_prime_index::write(qj_prime_index[j]);
          }
          [[intel::ivdep]] for (uint32_t n = 0; n < coeff_count; n++) {
            uint64_t part1 = 0;
            int k = ks[n];

#pragma unroll
            for (uint32_t i = 0; i < MAX_P; i++) {
              uint64_t c_pi = i < P ? (c[n][i] & BIT_MASK(MAX_PRIME_BITS)) : 0;
              part1 += mulmod(c_pi, pstar_qj[j][i], qj_prime);
              // mod once
              part1 = MOD_ONCE(part1, qj_prime);
              k = MOD_ONCE(k, t);
            }

            int round_v = round(v[n]);
            k = k - round_v;

            // make k positive if it is negative
            k = k < 0 ? t + k : k;

            // balance the remainder
            k = k > t / 2 ? k - t : k;
            k += round_v;

            // mulmod only supports positve number
            uint64_t part2 = k < 0 ? qj_prime + k : k;
            uint64_t d = MOD_ONCE(qj_prime + part1 - part2, qj_prime);
            ulong temp = 0;
            if (added_primes_at_end ? j < (Q - I) : j >= I) {
              uint64_t c = pipe_scale_input::read();
              ASSERT(c < qj_prime, "%lu > %lu at n = %d, I = %d, j = %d\n", c,
                     qj_prime, n, I, j);
              temp = mulmod(c, P_qj[j], qj_prime);
            }
            uint64_t out = MOD_ONCE(qj_prime + temp - d, qj_prime);

            if (b_add_special_primes) {
              out = mulmod(out, R_qj[j], qj_prime);
            }
            pipe_scale_output::write(out);
          }
        }
      }
    });
  });
  return e;
}
}  // namespace L0
