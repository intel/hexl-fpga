#pragma once
#include <L0/load.hpp>
#include <L0/ntt.hpp>
#include <L0/store.hpp>
#include <L0/twiddle_factors.hpp>

namespace L1 {
namespace helib {
namespace bgv {
// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
template <int id>
class NTTRead;
template <int id>
class NTTWrite;

template <int id>
class NTTReadOutPipeId;
template <int id>
class NTTWriteInPipeId;
template <int id>
class NTTTFPipeId;
template <int id>
class NTTPrimeIndexPipeId2;

template <int id>
class NTTNTT;
template <int id>
class NTTTF;

template <int id, int VEC, int coeff_count, class pipe_NTT_read_in,
          class pipe_ntt_prime_index, class pipe_NTT_write_out>
class ntt {
 private:
  // internal pipes
  using pipe_NTT_read_out =
      ext::intel::pipe<NTTReadOutPipeId<id>, L0::WideVector_t<VEC>,
                       coeff_count / 2 / VEC>;
  using pipe_NTT_write_in =
      ext::intel::pipe<NTTWriteInPipeId<id>, L0::WideVector_t<VEC>,
                       coeff_count / 2 / VEC>;
  using pipe_NTT_tf =
      ext::intel::pipe<NTTTFPipeId<id>, L0::TwiddleFactor_t<VEC>, 4>;
  using pipe_ntt_prime_index_forward =
      ext::intel::pipe<NTTPrimeIndexPipeId2<id>, unsigned char, 4>;

 public:
  int get_VEC() { return VEC; }

  sycl::event read(sycl::queue &q) {
    return L0::read<NTTRead<id>, pipe_NTT_read_in, pipe_NTT_read_out, VEC>(q);
  }
  sycl::event write(sycl::queue &q) {
    return L0::write<NTTWrite<id>, pipe_NTT_write_in, pipe_NTT_write_out, VEC>(
        q);
  }
  sycl::event compute_forward(sycl::queue &q,
                              const std::vector<ulong4> &config) {
    return L0::NTT::ntt<NTTNTT<id>, pipe_ntt_prime_index_forward,
                        pipe_NTT_read_out, pipe_NTT_write_in, pipe_NTT_tf, VEC,
                        coeff_count>(q, config);
  }
  sycl::event config_tf(sycl::queue &q, const std::vector<uint64_t> &tf_set) {
    return L0::TwiddleFactor_NTT<NTTTF<id>, pipe_ntt_prime_index,
                                 pipe_ntt_prime_index_forward, pipe_NTT_tf, VEC,
                                 coeff_count>(q, tf_set);
  }
};
}  // namespace bgv
}  // namespace helib
}  // namespace L1
