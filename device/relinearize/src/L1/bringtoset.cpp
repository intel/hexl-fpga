#include <L0/load.hpp>
#include <L0/scale.hpp>
#include <L0/store.hpp>
#include <L1/common.h>
#include <L1/bringtoset.h>
#include <L1/intt.h>
#include <L1/ntt.h>
#include <L1/pipes.h>
#include <L1/launch_ntt.hpp>

namespace L1 {
namespace BringToSet {
/**
 * @brief pipes
 *
 */
using pipe_rescale_intt_input =
    ext::intel::pipe<class RescaleINTTInputPipeId, uint64_t, 4>;
using pipe_scale_intt_prime_index =
    ext::intel::pipe<class RescaleINTTPrimeIndexPipeId, uint8_t, 4>;
using pipe_rescale_ntt_prime_index =
    ext::intel::pipe<class RescaleNTTPrimeIndexPipeId, uint8_t, 4>;
using pipe_rescale_input =
    ext::intel::pipe<class RescaleInputPipeId, uint64_t, 4>;
using pipe_rescale_output =
    ext::intel::pipe<class RescaleOutputPipeId, uint64_t, 4>;
using pipe_rescale_ntt_output =
    ext::intel::pipe<class RescaleNTTOutputPipeId, uint64_t, 4>;

/**
 * @brief rescale INTT definition
 *
 */
using rescale_intt_t =
    L1::intt::intt<RESCALE_INTT_ID, RESCALE_INTT_VEC, COEFF_COUNT,
                   pipe_rescale_intt_input, pipe_scale_intt_prime_index,
                   pipe_rescale_input>;

/**
 * @brief rescale NTT definition
 *
 */
using rescale_ntt_t =
    L1::ntt::ntt<RESCALE_NTT_ID, RESCALE_NTT_VEC, COEFF_COUNT,
                 pipe_rescale_output, pipe_rescale_ntt_prime_index,
                 pipe_rescale_ntt_output>;

sycl::event load(sycl::queue &q, sycl::buffer<uint64_t> &c,
                 sycl::buffer<uint8_t> &prime_index_set_buf,
                 unsigned prime_size) {
  return L0::load<class RescaleLoad, pipe_rescale_intt_input,
                  pipe_scale_intt_prime_index, COEFF_COUNT>(
      q, c, prime_index_set_buf, prime_size);
}

sycl::event kernel(sycl::queue &q, uint32_t coeff_count,
                   sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
                   uint32_t Q, uint I, uint64_t t) {
  // check inputs
  assert(P <= MAX_RESCALE_P);
  assert(Q <= MAX_RESCALE_Q);
  assert(coeff_count = COEFF_COUNT);
  return L0::scale<class Rescale, MAX_RESCALE_P, MAX_RESCALE_P_BANKS,
                   MAX_RESCALE_Q, COEFF_COUNT, pipe_rescale_input,
                   pipe_rescale_output, pipe_rescale_ntt_prime_index,
                   false, /* added primes not at end */
                   false /* do not add special primes*/>(
      q, coeff_count, scale_param_set_buf, P, Q, I, t);
}
sycl::event store(sycl::queue &q, sycl::buffer<uint64_t> &c, unsigned size) {
  return L0::store<class RescaleStore, pipe_rescale_ntt_output>(q, c, size);
}

void intt(const std::vector<uint64_t> &primes, uint64_t coeff_count, int flag) {
  static rescale_intt_t intt;
  launch_intt(intt, primes, coeff_count, flag);
}

void ntt(const std::vector<uint64_t> &primes, uint64_t coeff_count, int flag) {
  static rescale_ntt_t ntt;
  launch_ntt(ntt, primes, coeff_count, flag);
}
}  // namespace BringToSet
}  // namespace L1
