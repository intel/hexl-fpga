#include <CL/sycl.hpp>
#include <NTL/ZZ.h>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <L1/keyswitchdigits.h>
#include <L2/utils.h>
#include <L2/keyswitchdigits.h>

#define MAX_PRIMES 32
#define MAX_BUFF_DEPTH 4

using namespace sycl;

namespace L2 {
namespace helib {
namespace bgv {
namespace KeySwitchDigits {
class KeySwitchDigitsImpl {
 public:
  static KeySwitchDigitsImpl &GetInstance() {
    static KeySwitchDigitsImpl impl;
    return impl;
  }

  void Init(const std::vector<ulong> &all_primes,
            const std::vector<uint64_t> &keys1,
            const std::vector<uint64_t> &keys2,
            const std::vector<uint64_t> &keys3,
            const std::vector<uint64_t> &keys4, uint32_t input_mem_channel,
            uint32_t output_mem_channel, uint32_t keys_mem_channel,
            int buf_depth = MAX_BUFF_DEPTH) {
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
    q_load_data = new sycl::queue(device_selector, prop_list);
    q_kernel = new sycl::queue(device_selector, prop_list);
    q_store_data = new sycl::queue(device_selector, prop_list);

    // create buffers ahead
    for (int i = 0; i < buf_depth_; i++) {
      input_buffer_[i] = new sycl::buffer<uint64_t>(
          MAX_PRIMES * COEFF_COUNT * 4,
          {sycl::property::buffer::mem_channel{input_mem_channel}});
      packed_precomuted_params_buf_[i] = new buffer<ulong4>(
          MAX_PRIMES, {sycl::property::buffer::mem_channel{input_mem_channel}});
      output_buffer_[i] = new sycl::buffer<uint64_t>(
          MAX_PRIMES * COEFF_COUNT * 2,
          {sycl::property::buffer::mem_channel{output_mem_channel}});
      output_buffer_[i]->set_write_back(false);
      output_ptr_[i] = NULL;
    }
    for (int i = 0; i < 4; i++) {
      keys_buffer_[i] = new sycl::buffer<ulong2>(
          MAX_PRIMES * COEFF_COUNT,
          {sycl::property::buffer::mem_channel{keys_mem_channel}});
    }

    queue_copy(*q_load_data, keys1, *keys_buffer_[0], keys1.size() / 2);
    queue_copy(*q_load_data, keys2, *keys_buffer_[1], keys2.size() / 2);
    queue_copy(*q_load_data, keys3, *keys_buffer_[2], keys3.size() / 2);
    queue_copy(*q_load_data, keys4, *keys_buffer_[3], keys4.size() / 2);
  }

  sycl::event ProcessInput(const std::vector<uint8_t> &primes_index,
                           const std::vector<uint64_t> &digits,
                           sycl::ulong4 digits_offset,
                           sycl::event depend_event) {
    Timer t_process_input("KeySwitchDigits::ProcessInput::vector", debug_);
    queue_copy(*q_load_data, digits, *input_buffer_[buf_index_]);
    return ProcessInput(primes_index, *input_buffer_[buf_index_], digits_offset,
                        depend_event);
  }

  sycl::event ProcessInput(const std::vector<uint8_t> &primes_index,
                           sycl::buffer<uint64_t> &digits,
                           sycl::ulong4 digits_offset,
                           sycl::event depend_event) {
    Timer t_process_input("KeySwitchDigits::ProcessInput::buffer", debug_);
    store_events_[buf_index_].wait();

    if (debug_) {
      std::cout << "KeySwitchDigits::ProcessInput - " << buf_index_
                << std::endl;
    }

    // compute the diff value to prepare for the next prime
    // 0,0,0,1,1 -> 0,0,1,0,1
    // the last one doesn't matter
    auto num_primes = primes_index.size();
    std::vector<sycl::ulong4> primes_keyswitch_digits;
    std::vector<int> primes_index_offset(primes_index.size());
    for (size_t i = 0; i < primes_index.size(); i++) {
      primes_index_offset[i] = primes_index[i] - i;
    }
    for (size_t i = 0; i < primes_index.size() - 1; i++) {
      primes_index_offset[i] =
          primes_index_offset[i + 1] - primes_index_offset[i];
    }

    // pre-computing r and k for primes
    for (size_t i = 0; i < primes_index.size(); i++) {
      auto prime = all_primes_[primes_index[i]];
      ulong4 tmp;
      tmp.s0() = prime;
      tmp.s1() = primes_index_offset[i];
      tmp.s2() = precompute_modulus_r(prime);
      tmp.s3() = precompute_modulus_k(prime);
      primes_keyswitch_digits.push_back(tmp);
    }

    queue_copy(*q_load_data, primes_keyswitch_digits,
               *packed_precomuted_params_buf_[buf_index_]);

    int num_digits = 4;

    store_events_[buf_index_] = L1::keySwitchDigits(
        *q_kernel, *packed_precomuted_params_buf_[buf_index_], *keys_buffer_[0],
        *keys_buffer_[1], *keys_buffer_[2], *keys_buffer_[3], digits, digits,
        digits, digits, *output_buffer_[buf_index_],
        *output_buffer_[buf_index_], num_digits, num_primes, digits_offset,
        depend_event, 0xff);

#if SYNC_MODE
    store_events_[buf_index_].wait();
    if (debug_) {
      PrintEventTime(store_events_[buf_index_], "keySwitchDigits - Kernel");
    }
#endif
    return store_events_[buf_index_];
  }

  void ProcessOutput(int output_buf_index) {
    if (!output_ptr_[output_buf_index]) return;
    Timer t_process_output("KeySwitchDigits::ProcessOutput", debug_);
    if (debug_) {
      std::cout << "KeySwitchDigits::ProcessOutput - " << output_buf_index
                << std::endl;
    }
    assert(output_size_[output_buf_index] <=
           output_buffer_[output_buf_index]->size());
    {
      Timer t_process_output("KeySwitchDigits::store_events_::wait", debug_);
      store_events_[output_buf_index].wait();
      if (debug_)
        PrintEventTime(store_events_[output_buf_index],
                       "KeySwitchDigits::kernel");
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

  void Enqueue(const std::vector<uint8_t> &primes_index,
               const std::vector<uint64_t> &digits, sycl::ulong4 digits_offset,
               std::vector<uint64_t> &output, sycl::event depend_event) {
    output_ptr_[buf_index_] = output.data();
    output_size_[buf_index_] = output.size();
    ProcessOutput(GetNextBufferIndex());
    ProcessInput(primes_index, digits, digits_offset, depend_event);
    buf_index_ = (buf_index_ + 1) % buf_depth_;
  }

  void Enqueue(const std::vector<uint8_t> &primes_index,
               sycl::buffer<uint64_t> &digits, sycl::ulong4 digits_offset,
               std::vector<uint64_t> &output, sycl::event depend_event) {
    output_ptr_[buf_index_] = output.data();
    output_size_[buf_index_] = output.size();
    ProcessOutput(GetNextBufferIndex());
    ProcessInput(primes_index, digits, digits_offset, depend_event);
    buf_index_ = (buf_index_ + 1) % buf_depth_;
  }

  void ProcessLeftOutput() {
    for (int i = 0; i < buf_depth_; i++) {
      ProcessOutput((buf_index_ + i) % buf_depth_);
    }
  }

  int GetNextBufferIndex() { return (buf_index_ + 1) % buf_depth_; }

 private:
  sycl::queue *q_load_data;
  sycl::queue *q_kernel;
  sycl::queue *q_store_data;
  sycl::buffer<uint64_t> *input_buffer_[MAX_BUFF_DEPTH];
  sycl::buffer<ulong2> *keys_buffer_[4];
  sycl::buffer<uint64_t> *output_buffer_[MAX_BUFF_DEPTH];
  sycl::buffer<sycl::ulong4> *packed_precomuted_params_buf_[MAX_BUFF_DEPTH];
  sycl::event store_events_[MAX_BUFF_DEPTH];
  std::vector<uint64_t> all_primes_;
  void *output_ptr_[MAX_BUFF_DEPTH];
  size_t output_size_[MAX_BUFF_DEPTH];

  int buf_depth_;
  int buf_index_;
  bool debug_;
};
}  // namespace KeySwitchDigits
}  // namespace bgv
}  // namespace helib
}  // namespace L2
