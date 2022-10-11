#include <L0/load.hpp>
#include <L0/scale.hpp>
#include <L0/store.hpp>
#include <L1/helib_bgv.h>

namespace L1 {
namespace helib {
namespace bgv {

// intt1_t &GetINTT1() {
//   static intt1_t intt;
//   return intt;
// }

// intt2_t &GetINTT2() {
//   static intt2_t intt;
//   return intt;
// }

/*
sycl::event INTT1LoadPrimesIndex(sycl::queue &q,
                                 sycl::buffer<uint8_t> &primes_index) {
  return L0::LoadPrimeIndexGeneric2<class INTT1LoadPrimesIndex,
                                    pipe_intt1_primes_index, 2>(q,
                                                                primes_index);
}

sycl::event INTT2LoadPrimesIndex(sycl::queue &q,
                                 sycl::buffer<uint8_t> &primes_index) {
  return L0::LoadPrimeIndexGeneric2<class INTT2LoadPrimesIndex,
                                    pipe_intt2_primes_index, 2>(q,
                                                                primes_index);
}
*/

event BringToSet(sycl::queue &q, uint32_t coeff_count,
                 sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
                 uint32_t Q, uint I, uint64_t t) {
  // check inputs
  assert(P <= MAX_MULT_LOW_LVL_BRING_TO_SET_P);
  assert(Q <= MAX_MULT_LOW_LVL_BRING_TO_SET_Q);
  assert(coeff_count = COEFF_COUNT);
  return L0::scale<
      class MultLowLvlBringToSet, MAX_MULT_LOW_LVL_BRING_TO_SET_P,
      MAX_MULT_LOW_LVL_BRING_TO_SET_P_BANKS, MAX_MULT_LOW_LVL_BRING_TO_SET_Q,
      COEFF_COUNT, pipe_scale_input, pipe_scale_output,
      pipe_tensor_product_prime_index1, false, /* added primes not at end */
      false /* do not add special primes*/>(q, coeff_count, scale_param_set_buf,
                                            P, Q, I, t);
}

sycl::event BringToSetLoad(sycl::queue &q, sycl::event &depends,
                           sycl::buffer<uint64_t> &c,
                           sycl::buffer<uint8_t> &prime_index_set_buf) {
  return L0::load<class BringToSetLoad, pipe_intt1_input,
                  pipe_intt1_primes_index, COEFF_COUNT>(q, depends, c,
                                                        prime_index_set_buf);
}
/*
sycl::event BringToSetStore(sycl::queue &q, sycl::buffer<uint64_t> &c) {
  return L0::store<class BringToSetStore, pipe_scale_output>(q, c);
}
*/
event BringToSet2(sycl::queue &q, uint32_t coeff_count,
                  sycl::buffer<ulong2> &scale_param_set_buf, uint32_t P,
                  uint32_t Q, uint I, uint64_t t) {
  // check inputs
  assert(P <= MAX_MULT_LOW_LVL_BRING_TO_SET_P);
  assert(Q <= MAX_MULT_LOW_LVL_BRING_TO_SET_Q);
  assert(coeff_count = COEFF_COUNT);
  return L0::scale<
      class MultLowLvlBringToSet2, MAX_MULT_LOW_LVL_BRING_TO_SET_P,
      MAX_MULT_LOW_LVL_BRING_TO_SET_P_BANKS, MAX_MULT_LOW_LVL_BRING_TO_SET_Q,
      COEFF_COUNT, pipe_scale_input2, pipe_scale_output2,
      pipe_tensor_product_prime_index2, false, /* added primes not at end */
      false /* do not add special primes*/>(q, coeff_count, scale_param_set_buf,
                                            P, Q, I, t);
}

sycl::event BringToSetLoad2(sycl::queue &q, sycl::event &depends,
                            sycl::buffer<uint64_t> &c,
                            sycl::buffer<uint8_t> &prime_index_set_buf) {
  return L0::load<class BringToSetLoad2, pipe_intt2_input,
                  pipe_intt2_primes_index, COEFF_COUNT>(q, depends, c,
                                                        prime_index_set_buf);
}

/*
sycl::event BringToSetStore2(sycl::queue &q, sycl::buffer<uint64_t> &c) {
  return L0::store<class BringToSetStore2, pipe_scale_output2>(q, c);
}
*/

}  // namespace bgv
}  // namespace helib
}  // namespace L1
