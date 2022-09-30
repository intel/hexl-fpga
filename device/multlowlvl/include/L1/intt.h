#pragma once
#include <L0/intt.hpp>
#include <L0/load.hpp>
#include <L0/ntt.hpp>
#include <L0/scale.hpp>
#include <L0/store.hpp>
#include <L0/twiddle_factors.hpp>

namespace L1 {
namespace helib {
namespace bgv {
// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
template <int id>
class INTTRead;
template <int id>
class INTTWrite;

template <int id>
class INTTReadOutPipeId;
template <int id>
class INTTWriteInPipeId;
template <int id>
class INTTTFPipeId;
template <int id>
class INTTPrimeIndexPipeId2;

template <int id>
class INTTINTT;
template <int id>
class INTTTF;
template <int id>
class INTTNorm;

template <int id>
class INTTInterPipeId;
template <int id>
class INTTNormPipeId;

template <int id, int VEC, int coeff_count, class pipe_intt_read_in,
          class pipe_intt_prime_index, class pipe_intt_write_out>
class intt {
 private:
  // internal pipes
  using pipe_intt_read_out =
      ext::intel::pipe<INTTReadOutPipeId<id>, L0::WideVector_t<VEC>,
                       coeff_count / 2 / VEC>;
  using pipe_intt_inter = ext::intel::pipe<INTTInterPipeId<id>, uint64_t, 4>;
  using pipe_intt_write_in =
      ext::intel::pipe<INTTWriteInPipeId<id>, L0::WideVector_t<VEC>,
                       coeff_count / 2 / VEC>;
  using pipe_intt_tf =
      ext::intel::pipe<INTTTFPipeId<id>, L0::TwiddleFactor_t<VEC>, 4>;
  using pipe_intt_norm = ext::intel::pipe<INTTNormPipeId<id>, ulong4, 4>;
  using pipe_prime_index_inverse =
      ext::intel::pipe<INTTPrimeIndexPipeId2<id>, uint8_t, 4>;

 public:
  int get_VEC() { return VEC; }
  sycl::event read(sycl::queue &q) {
    return L0::read<INTTRead<id>, pipe_intt_read_in, pipe_intt_read_out, VEC>(
        q);
  }
  sycl::event write(sycl::queue &q) {
    return L0::write<INTTWrite<id>, pipe_intt_write_in, pipe_intt_inter, VEC>(
        q);
  }
  sycl::event compute_inverse(sycl::queue &q,
                              const std::vector<ulong4> &configs) {
    return L0::INTT::intt<INTTINTT<id>, pipe_prime_index_inverse,
                          pipe_intt_read_out, pipe_intt_write_in,
                          pipe_intt_norm, pipe_intt_tf, coeff_count, VEC>(
        q, configs);
  }
  sycl::event norm(sycl::queue &q) {
    return L0::INTT::norm<INTTNorm<id>, pipe_intt_inter, pipe_intt_write_out,
                          pipe_intt_norm, coeff_count>(q);
  }
  sycl::event config_tf(sycl::queue &q, const std::vector<uint64_t> &tf_set) {
    return L0::TwiddleFactor<INTTTF<id>, pipe_intt_prime_index,
                             pipe_prime_index_inverse, pipe_intt_tf, VEC,
                             coeff_count>(q, tf_set);
  }
};
}  // namespace bgv
}  // namespace helib
}  // namespace L1
