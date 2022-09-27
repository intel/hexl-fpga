#include <L0/load.hpp>
#include <L0/scale.hpp>
#include <L0/store.hpp>
#include <L1/helib_bgv.h>

namespace L1 {
namespace helib {
namespace bgv {
event scale(sycl::queue &q, uint32_t coeff_count, sycl::buffer<ulong2> &pt_buf,
            sycl::buffer<ulong2> &pstar_qj_buf,
            sycl::buffer<ulong2> &pstar_inv_buf, sycl::buffer<ulong2> &P_qj_buf,
            sycl::buffer<uint64_t> &pi_buf, sycl::buffer<uint64_t> &qj_buf,
            uint32_t P, uint32_t Q, uint I, uint64_t t) {
  // check inputs
  assert(P <= MAX_P);
  assert(Q <= MAX_Q);
  assert(coeff_count = COEFF_COUNT);
  return L0::scale<class ScaleKernel, MAX_P, MAX_Q, COEFF_COUNT,
                   pipe_scale_input, pipe_ntt1_input, false>(
      q, coeff_count, pt_buf, pstar_qj_buf, pstar_inv_buf, P_qj_buf, pi_buf,
      qj_buf, P, Q, I, t);
}

// create the intt1 instance
static intt1_t intt1;

// create the intt1 getter
intt1_t &get_intt1() { return intt1; }

// create the ntt1 instance
static ntt1_t ntt1;

// create the intt1 getter
ntt1_t &get_ntt1() { return ntt1; }

sycl::event intt1_load(sycl::queue &q, sycl::buffer<uint64_t> &c,
                       sycl::buffer<uint8_t> &prime_index_set) {
  return L0::load<class INTT1Load, pipe_intt1_input, pipe_intt1_prime_index,
                  COEFF_COUNT>(q, c, prime_index_set);
}
sycl::event ntt1_store(sycl::queue &q, sycl::buffer<uint64_t> &c) {
  return L0::store<class NTT1Store, pipe_ntt1_store>(q, c);
}

sycl::event ntt1_load_prime_index(sycl::queue &q,
                                  sycl::buffer<uint8_t> &prime_index_set) {
  return L0::generic_load_prime_index<class NTT1LoadPrimeIndex,
                                      pipe_ntt1_prime_index>(q,
                                                             prime_index_set);
}
}  // namespace bgv
}  // namespace helib
}  // namespace L1
