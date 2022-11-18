#include "../L0/load.hpp"
#include "../L0/scale.hpp"
#include "../L0/store.hpp"
#include "../../include/L1/helib_bgv.h"

namespace L1 {
namespace helib {
namespace bgv {


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


BringToSet_t& BringToSet_struct()
{
  static BringToSet_t BringtoSet_obj = {
    .BringToSet = &BringToSet,
    .BringToSetLoad = &BringToSetLoad,
    .BringToSet2 = &BringToSet2,
    .BringToSetLoad2 = &BringToSetLoad2
  };

  return BringtoSet_obj;
}

#define VEC 8


int get_intt_VEC()
{
  static int vec = 8;
  return vec;
}

// generator intt1 and intt2 internal pipes
// intt_pipe_generator<1, 8, COEFF_COUNT> intt1_pipes;
// intt_pipe_generator<2, 8, COEFF_COUNT> intt2_pipes;

sycl::event intt1_read(sycl::queue&q)
{
  return L0::read<INTTRead<1>, pipe_intt1_input, 
          intt_pipe_generator<1, 8, COEFF_COUNT>::pipe_intt_read_out, 
          8>(q);
}

sycl::event intt1_write(sycl::queue&q)
{
  return L0::write<INTTWrite<1>, 
          intt_pipe_generator<1, 8, COEFF_COUNT>::pipe_intt_write_in, 
          intt_pipe_generator<1, 8, COEFF_COUNT>::pipe_intt_inter, VEC>(q);
}

sycl::event intt1_compute_inverse(sycl::queue &q,
                            const std::vector<ulong4> &configs) {
  return L0::INTT::intt<INTTINTT<1>, 
          intt_pipe_generator<1, 8, COEFF_COUNT>::pipe_prime_index_inverse,
          intt_pipe_generator<1, 8, COEFF_COUNT>::pipe_intt_read_out, 
          intt_pipe_generator<1, 8, COEFF_COUNT>::pipe_intt_write_in,
          intt_pipe_generator<1, 8, COEFF_COUNT>::pipe_intt_norm, 
          intt_pipe_generator<1, 8, COEFF_COUNT>::pipe_intt_tf, 
          COEFF_COUNT, VEC>(q, configs);
}

sycl::event intt1_norm(sycl::queue &q) {
    return L0::INTT::norm<INTTNorm<1>, 
            intt_pipe_generator<1, 8, COEFF_COUNT>::pipe_intt_inter, 
            pipe_scale_input,
            intt_pipe_generator<1, 8, COEFF_COUNT>::pipe_intt_norm, COEFF_COUNT>(q);
}


sycl::event intt1_config_tf(sycl::queue &q, const std::vector<uint64_t> &tf_set) {
    return L0::TwiddleFactor<INTTTF<1>, 
             pipe_intt1_primes_index,
             intt_pipe_generator<1, 8, COEFF_COUNT>::pipe_prime_index_inverse, 
             intt_pipe_generator<1, 8, COEFF_COUNT>::pipe_intt_tf, 
             VEC, COEFF_COUNT>(q, tf_set);
}


sycl::event intt2_read(sycl::queue&q)
{
  return L0::read<INTTRead<2>, pipe_intt2_input, 
          intt_pipe_generator<2, 8, COEFF_COUNT>::pipe_intt_read_out, 8>(q);
}

sycl::event intt2_write(sycl::queue&q)
{
  return L0::write<INTTWrite<2>, 
          intt_pipe_generator<2, 8, COEFF_COUNT>::pipe_intt_write_in, 
          intt_pipe_generator<2, 8, COEFF_COUNT>::pipe_intt_inter, 8>(q);
}

sycl::event intt2_compute_inverse(sycl::queue &q,
                            const std::vector<ulong4> &configs) {
  return L0::INTT::intt<INTTINTT<2>, 
          intt_pipe_generator<2, 8, COEFF_COUNT>::pipe_prime_index_inverse,
          intt_pipe_generator<2, 8, COEFF_COUNT>::pipe_intt_read_out, 
          intt_pipe_generator<2, 8, COEFF_COUNT>::pipe_intt_write_in,
          intt_pipe_generator<2, 8, COEFF_COUNT>::pipe_intt_norm, 
          intt_pipe_generator<2, 8, COEFF_COUNT>::pipe_intt_tf, 
          COEFF_COUNT, VEC>(q, configs);
}

sycl::event intt2_norm(sycl::queue &q) {
    return L0::INTT::norm<INTTNorm<2>, 
            intt_pipe_generator<2, 8, COEFF_COUNT>::pipe_intt_inter, 
            pipe_scale_input2,
            intt_pipe_generator<2, 8, COEFF_COUNT>::pipe_intt_norm, COEFF_COUNT>(q);
}


sycl::event intt2_config_tf(sycl::queue &q, const std::vector<uint64_t> &tf_set) {
    return L0::TwiddleFactor<INTTTF<2>, 
             pipe_intt2_primes_index,
             intt_pipe_generator<2, 8, COEFF_COUNT>::pipe_prime_index_inverse, 
             intt_pipe_generator<2, 8, COEFF_COUNT>::pipe_intt_tf, 
             VEC, COEFF_COUNT>(q, tf_set);
}



INTT_Method& intt1_method() {
  static INTT_Method intt1_method_obj = {
    .get_VEC = &get_intt_VEC,
    .read = &intt1_read,
    .write = &intt1_write,
    .compute_inverse = &intt1_compute_inverse,
    .norm = &intt1_norm,
    .config_tf = &intt1_config_tf
  };

  return intt1_method_obj;
}

INTT_Method& intt2_method() {
  static INTT_Method intt2_method_obj = {
    .get_VEC = &get_intt_VEC,
    .read = &intt2_read,
    .write = &intt2_write,
    .compute_inverse = &intt2_compute_inverse,
    .norm = &intt2_norm,
    .config_tf = &intt2_config_tf
  };

  return intt2_method_obj;
}



/*
sycl::event BringToSetStore2(sycl::queue &q, sycl::buffer<uint64_t> &c) {
  return L0::store<class BringToSetStore2, pipe_scale_output2>(q, c);
}
*/

}  // namespace bgv
}  // namespace helib
}  // namespace L1
