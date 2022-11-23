#include <algorithm>
#include <CL/sycl.hpp>
#include <NTL/ZZ.h>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <L1/breakintodigits.h>
#include <L2/utils.h>

#define MAX_PRIMES 32
#define MAX_BUFF_DEPTH 4
#define MAX_PACKED_PRECOMPUTED_PARAMS_SIZE 2048
#define SYNC_MODE 0

using namespace sycl;

namespace L2 {
namespace helib {
namespace bgv {
namespace BreakIntoDigits {
class BreakIntoDigitsImpl {
 public:
  static BreakIntoDigitsImpl &GetInstance() {
    static BreakIntoDigitsImpl impl;
    return impl;
  }

  void Init(std::vector<uint64_t> &all_primes, uint32_t input_mem_channel,
            uint32_t output_mem_channel, int buf_depth = MAX_BUFF_DEPTH) {
    // Create queue, get platform and device
#if defined(FPGA_EMULATOR)
    sycl::ext::intel::fpga_emulator_selector device_selector;
#else
    sycl::ext::intel::fpga_selector device_selector;
#endif

    this->all_primes_ = all_primes;
    this->buf_depth_ = buf_depth;
    this->buf_index_ = 0;
    this->debug_ = getenv("DEBUG") ? atoi(getenv("DEBUG")) : 0;

    auto prop_list = property_list{property::queue::enable_profiling()};
    q_load = new sycl::queue(device_selector, prop_list);
    q_load_data = new sycl::queue(device_selector, prop_list);
    q_breakintodigits = new sycl::queue(device_selector, prop_list);
    q_store = new sycl::queue(device_selector, prop_list);
    q_store_data = new sycl::queue(device_selector, prop_list);

    // create buffers ahead
    for (int i = 0; i < buf_depth_; i++) {
      input_buffer_[i] = new sycl::buffer<uint64_t>(
          MAX_PRIMES * COEFF_COUNT,
          {sycl::property::buffer::mem_channel{input_mem_channel}});
      packed_precomuted_params_buf_[i] = new buffer<ulong2>(
          MAX_PACKED_PRECOMPUTED_PARAMS_SIZE,
          {sycl::property::buffer::mem_channel{input_mem_channel}});
      output_buffer_[i] = new sycl::buffer<uint64_t>(
          MAX_PRIMES * COEFF_COUNT * 4,
          {sycl::property::buffer::mem_channel{output_mem_channel}});
      output_buffer_[i]->set_write_back(false);

      output_ptr_[i] = NULL;
    }

    // only launch once
    L1::BreakIntoDigits::intt(all_primes_, COEFF_COUNT, 0b111);
    L1::BreakIntoDigits::ntt(all_primes_, COEFF_COUNT, 0b111);
  }

  sycl::event ProcessInput(std::vector<uint64_t> &input,
                           std::vector<uint64_t> &pi,
                           std::vector<unsigned> num_designed_digits_primes,
                           unsigned num_special_primes) {
    Timer t_process_input("BreakIntoDigits::ProcessInput", debug_);
    store_events_[buf_index_].wait();
    // Process load kernel
    unsigned input_size = input.size();
    assert(input_buffer_[buf_index_]->size() >= input_size);
    queue_copy(*q_load_data, input, *input_buffer_[buf_index_]);

    L1::BreakIntoDigits::load(*q_load, *input_buffer_[buf_index_], input_size);

    // Process BreakIntoDigits kernel
    std::vector<sycl::ulong2> packed_precomuted_params;
    std::vector<unsigned> num_digits_primes;

    PreComputeParams(pi, all_primes_, num_designed_digits_primes,
                     num_special_primes, packed_precomuted_params,
                     num_digits_primes);
    queue_copy(*q_load_data, packed_precomuted_params,
               *packed_precomuted_params_buf_[buf_index_]);

    // launch breakIntoDigits
    unsigned num_digits = num_digits_primes.size();
    unsigned num_digit1_primes = num_digits_primes[0];
    unsigned num_digit2_primes = num_digits > 1 ? num_digits_primes[1] : 0;
    unsigned num_digit3_primes = num_digits > 2 ? num_digits_primes[2] : 0;
    unsigned num_digit4_primes = num_digits > 3 ? num_digits_primes[3] : 0;

    // desinged means all normal primes
    uint num_designed_normal_primes = 0;
    for (int i = 0; i < num_designed_digits_primes.size(); i++) {
      num_designed_normal_primes += num_designed_digits_primes[i];
    }

    // pi includes the special primes
    auto num_output_primes = pi.size();
    auto output_size = num_output_primes * COEFF_COUNT * num_digits;

    assert(output_size <= output_buffer_[buf_index_]->size());

    L1::BreakIntoDigits::kernel(
        *q_breakintodigits, *packed_precomuted_params_buf_[buf_index_],
        num_digits, num_digit1_primes, num_digit2_primes, num_digit3_primes,
        num_digit4_primes, num_special_primes, num_designed_normal_primes,
        0b1111);
    if (debug_) {
      std::cout << "BreakIntoDigits::ProcessInput - " << buf_index_
                << std::endl;
    }
    store_events_[buf_index_] = L1::BreakIntoDigits::store(
        *q_store, *output_buffer_[buf_index_], output_size, 0b111);
#if SYNC_MODE
    store_events_[buf_index_].wait();
    if (debug_) {
      PrintEventTime(store_events_[buf_index_], "BreakIntoDigits - Kernel");
    }
#endif
    return store_events_[buf_index_];
  }

  sycl::buffer<uint64_t> &GetLastOutputBuff() {
    auto last_buf_index = GetLastBufferIndex();
    return *output_buffer_[last_buf_index];
  }

  void ProcessOutput(int output_buf_index) {
    if (!output_ptr_[output_buf_index]) return;
    Timer t_process_output("BreakIntoDigits::ProcessOutput", debug_);
    if (debug_) {
      std::cout << "BreakIntoDigits::ProcessOutput - " << output_buf_index
                << std::endl;
    }
    q_store_data->submit([&](sycl::handler &h) {
      h.depends_on(store_events_[output_buf_index]);
      h.copy(output_buffer_[output_buf_index]
                 ->template get_access<sycl::access::mode::read>(
                     h, sycl::range<1>(output_size_[output_buf_index])),
             output_ptr_[output_buf_index]);
    });
#if 1
    q_store_data->wait();
    output_ptr_[output_buf_index] = NULL;
#endif
  }

  sycl::event Enqueue(std::vector<uint64_t> &input, std::vector<uint64_t> &pi,
                      std::vector<unsigned> num_designed_digits_primes,
                      unsigned num_special_primes,
                      std::vector<uint64_t> &output) {
    output_ptr_[buf_index_] = output.data();
    output_size_[buf_index_] = output.size();
    ProcessOutput(GetNextBufferIndex());
    return Enqueue(input, pi, num_designed_digits_primes, num_special_primes);
  }

  sycl::event Enqueue(std::vector<uint64_t> &input, std::vector<uint64_t> &pi,
                      std::vector<unsigned> num_designed_digits_primes,
                      unsigned num_special_primes) {
    auto e =
        ProcessInput(input, pi, num_designed_digits_primes, num_special_primes);
    buf_index_ = (buf_index_ + 1) % buf_depth_;
    return e;
  }

  void ProcessLeftOutput() {
    for (int i = 0; i < buf_depth_; i++) {
      ProcessOutput((buf_index_ + i) % buf_depth_);
    }
  }

  int GetLastBufferIndex() {
    return (buf_depth_ + buf_index_ - 1) % buf_depth_;
  }
  int GetNextBufferIndex() { return (buf_index_ + 1) % buf_depth_; }

  void PreComputeParams(std::vector<uint64_t> &pi,
                        std::vector<uint64_t> &all_primes,
                        std::vector<unsigned> num_designed_digits_primes,
                        unsigned num_special_primes,
                        std::vector<sycl::ulong2> &packed_precomuted_params,
                        std::vector<unsigned> &num_digits_primes) {
    std::vector<sycl::ulong4> pstar_inv;
    std::vector<sycl::ulong2> pstar_qj;
    std::vector<sycl::ulong> P_qj;

    /* packed parameters */
    // FORMAT:
    // pi and pi recip - all normal primes and special primes
    // pstar_inv and pstar_inv_recip - all normal primes
    // P_qj - num_digit_primes (normal_primes/2) + special primes
    long num_normal_primes = pi.size() - num_special_primes;

    // compute the actual prime size of each digit
    // the last digit size maybe smaller than designed digit size
    int i = 0;
    int num_left_primes = num_normal_primes;
    while (num_left_primes > 0) {
      num_digits_primes.push_back(
          std::min(num_designed_digits_primes[i], (unsigned)num_left_primes));
      num_left_primes -= num_designed_digits_primes[i];
      i++;
    }

    // compute the num of small primes
    long num_small_primes = 0;
    for (size_t i = 0; i < all_primes.size(); i++) {
      if (all_primes[i] == pi[0]) {
        // all the primes before the normal primes are small primes
        num_small_primes = i;
        break;
      }
    }

    // compute the offset of each digit
    std::vector<int> digits_offset;
    int lastDigitsOffset = num_small_primes;
    long num_digits = num_digits_primes.size();
    for (long i = 0; i < num_digits; i++) {
      digits_offset.push_back(lastDigitsOffset);
      lastDigitsOffset += num_digits_primes[i];
    }

    // compute the prod of each digit
    std::vector<NTL::ZZ> P;
    for (long i = 0; i < num_digits; i++) {
      NTL::ZZ tmp{1};
      for (int j = 0; j < num_digits_primes[i]; j++) {
        tmp *= all_primes[digits_offset[i] + j];
      }
      P.push_back(tmp);
    }

    // compute digitsQHatInv
    std::vector<NTL::ZZ> prodOfDesignedDigits;
    std::vector<NTL::ZZ> digitsQHatInv;

    for (long i = 0; i < num_digits; i++) {
      NTL::ZZ tmp{1};
      for (int j = 0; j < num_designed_digits_primes[i]; j++) {
        tmp *= all_primes[digits_offset[i] + j];
      }

      prodOfDesignedDigits.push_back(tmp);
    }

    // compute QHatInv
    for (long i = 0; i < num_digits; i++) {
      NTL::ZZ qhat{1};
      for (long j = 0; j < num_digits; j++) {
        if (j != i) {
          qhat *= prodOfDesignedDigits[j];
        }
      }
      auto qhat_inv =
          NTL::InvMod(qhat % prodOfDesignedDigits[i], prodOfDesignedDigits[i]);
      digitsQHatInv.push_back(qhat_inv);
    }

    // gererate the qj primes of each digits
    std::vector<uint64_t> digit_qj_primes[MAX_DIGITS];

    for (long j = 0; j < num_digits; j++) {
      for (long i = 0; i < pi.size(); i++) {
        if (i < digits_offset[j] ||
            i >= (digits_offset[j] + num_digits_primes[j]))
          digit_qj_primes[j].push_back(pi[i]);
      }
    }

    // pstar_inv has all the primes
    for (long j = 0; j < num_digits; j++) {
      for (long i = digits_offset[j];
           i < digits_offset[j] + num_digits_primes[j]; i++) {
        ulong p_star_inv_i = NTL::InvMod(NTL::rem(P[j] / pi[i], pi[i]), pi[i]);
        auto tmp = mulmod_y_ext(p_star_inv_i, pi[i]);
        auto tmp2 = mulmod_y_ext(NTL::InvMod(p_star_inv_i, pi[i]), pi[i]);

        pstar_inv.push_back({tmp.s0(), tmp.s1(), tmp2.s0(), tmp2.s1()});
      }
    }

    // std::cout << "pstar_inv.size() = " << pstar_inv.size() << std::endl;
    assert(num_normal_primes == pstar_inv.size());

    // compute pstar_qj
    for (int i = 0; i < MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES / 2; i++) {
      for (long k = 0; k < num_digits; k++) {
        for (int j = 0; j < MAX_NORMAL_PRIMES / 2; j++) {
          // P* mod qj
          auto tmp = i < digit_qj_primes[k].size() && j < num_digits_primes[k]
                         ? NTL::rem(P[k] / all_primes[digits_offset[k] + j],
                                    digit_qj_primes[k][i])
                         : 0;
          pstar_qj.push_back(mulmod_y_ext(tmp, digit_qj_primes[k][i]));
        }
      }
    }

    // comput P_qj
    for (long j = 0; j < num_digits; j++)
      for (int i = 0; i < pi.size(); i++) {
        P_qj.push_back(i < digit_qj_primes[j].size()
                           ? NTL::rem(P[j], digit_qj_primes[j][i])
                           : 0);
      }

    // compute pi_recip
    std::vector<ulong2> pi_with_recip;
    for (int i = 0; i < pi.size(); i++) {
      double pi_recip = (double)1 / pi[i];
      pi_with_recip.push_back({pi[i], *(ulong *)&pi_recip});
    }

    // packing now
    // pi and pi recip - all normal primes and special primes
    for (size_t i = 0; i < pi.size(); i++) {
      packed_precomuted_params.push_back(pi_with_recip[i]);
    }

    // pstar_inv and pstar_inv_recip - all normal primes
    for (size_t i = 0; i < pstar_inv.size(); i++) {
      auto tmp = pstar_inv[i];
      packed_precomuted_params.push_back({tmp.s0(), tmp.s1()});
    }

    // pstar_inv_recip - all normal primes
    for (size_t i = 0; i < pstar_inv.size(); i++) {
      auto tmp = pstar_inv[i];
      packed_precomuted_params.push_back({tmp.s2(), tmp.s3()});
    }

    // P_qj - pi.size() * 2
    for (size_t i = 0; i < P_qj.size(); i++) {
      packed_precomuted_params.push_back({P_qj[i], 0});
    }

    for (long j = 0; j < num_digits; j++) {
      for (int i = 0; i < num_digits_primes[j]; i++) {
        auto pi = all_primes[digits_offset[j] + i];
        long qhat_inv_pi = NTL::rem(digitsQHatInv[j], pi);
        packed_precomuted_params.push_back(mulmod_y_ext(qhat_inv_pi, pi));
      }
    }

    for (long k = 0; k < num_digits; k++) {
      for (int i = 0; i < MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES; i++) {
        for (int j = 0; j < MAX_DIGIT_SIZE; j++) {
          if (i < digit_qj_primes[k].size() && j < num_digits_primes[k]) {
            // P mod qj
            auto tmp = NTL::rem(P[k] / all_primes[digits_offset[k] + j],
                                digit_qj_primes[k][i]);
            packed_precomuted_params.push_back(
                mulmod_y_ext(tmp, digit_qj_primes[k][i]));
          } else {
            packed_precomuted_params.push_back({0, 0});
          }
        }
      }
    }
  }

 private:
  sycl::queue *q_load;
  sycl::queue *q_load_data;
  sycl::queue *q_breakintodigits;
  sycl::queue *q_store;
  sycl::queue *q_store_data;
  sycl::buffer<uint64_t> *input_buffer_[MAX_BUFF_DEPTH];
  sycl::buffer<uint64_t> *output_buffer_[MAX_BUFF_DEPTH];
  sycl::buffer<ulong2> *packed_precomuted_params_buf_[MAX_BUFF_DEPTH];
  sycl::event store_events_[MAX_BUFF_DEPTH];
  std::vector<uint64_t> all_primes_;
  void *output_ptr_[MAX_BUFF_DEPTH];
  size_t output_size_[MAX_BUFF_DEPTH];

  int buf_depth_;
  int buf_index_;
  bool debug_;
};
}  // namespace BreakIntoDigits
}  // namespace bgv
}  // namespace helib
}  // namespace L2
