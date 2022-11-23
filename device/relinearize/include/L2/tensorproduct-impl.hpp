#include <CL/sycl.hpp>
#include <NTL/ZZ.h>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <L1/tensorproduct.h>
#include <L2/utils.h>
#include <L2/tensorproduct.h>

#define MAX_PRIMES 32
#define MAX_BUFF_DEPTH 4

using namespace sycl;

namespace L2 {
namespace helib {
namespace bgv {
namespace TensorProduct {
class TensorProductImpl {
 public:
  static TensorProductImpl &GetInstance() {
    static TensorProductImpl impl;
    return impl;
  }

  void Init(const std::vector<ulong> &all_primes, uint32_t input_mem_channel,
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
    q_load_data = new sycl::queue(device_selector, prop_list);
    q_kernel = new sycl::queue(device_selector, prop_list);
    q_store_data = new sycl::queue(device_selector, prop_list);

    const property_list input_mem_channel_prop = {
        sycl::property::buffer::mem_channel{input_mem_channel}};
    const property_list output_mem_channel_prop = {
        sycl::property::buffer::mem_channel{output_mem_channel}};

    for (int i = 0; i < buf_depth_; i++) {
      buf_primes_[i] = new buffer<ulong4>(MAX_PRIMES, input_mem_channel_prop);
      input_buffer_[i] = new buffer<uint64_t>(4 * MAX_PRIMES * COEFF_COUNT,
                                              input_mem_channel_prop);
      output_buffer_[i][0] = new buffer<uint64_t>(MAX_PRIMES * COEFF_COUNT,
                                                  output_mem_channel_prop);
      output_buffer_[i][1] = new buffer<uint64_t>(MAX_PRIMES * COEFF_COUNT,
                                                  output_mem_channel_prop);
      output_buffer_[i][2] = new buffer<uint64_t>(MAX_PRIMES * COEFF_COUNT,
                                                  output_mem_channel_prop);

      // explicitly copy
      output_buffer_[i][0]->set_write_back(false);
      output_buffer_[i][1]->set_write_back(false);
      output_buffer_[i][2]->set_write_back(false);
      output_ptr_[i][0] = NULL;
      output_ptr_[i][1] = NULL;
      output_ptr_[i][2] = NULL;
    }
  }

  sycl::event ProcessInput(const std::vector<uint8_t> &primes_index,
                           const std::vector<uint64_t> &inputs,
                           sycl::ulong4 inputs_offset,
                           sycl::event &depend_event) {
    Timer t_process_input("TensorProduct::ProcessInput::vector", debug_);
    queue_copy(*q_load_data, inputs, *input_buffer_[buf_index_]);
    return ProcessInput(primes_index, *input_buffer_[buf_index_], inputs_offset,
                        depend_event);
  }

  sycl::event ProcessInput(const std::vector<uint8_t> &primes_index,
                           sycl::buffer<uint64_t> &buf_inputs,
                           sycl::ulong4 inputs_offset,
                           sycl::event &depend_event) {
    Timer t_process_input("TensorProduct::ProcessInput::buffer", debug_);
    store_events_[buf_index_].wait();

    if (debug_) {
      std::cout << "TensorProduct::ProcessInput - " << buf_index_ << std::endl;
    }

    auto num_primes = primes_index.size();
    auto &buf_primes = *buf_primes_[buf_index_];
    std::vector<ulong4> primes_ext;
    for (auto prime_index : primes_index) {
      auto prime = all_primes_[prime_index];
      primes_ext.push_back(
          {prime, precompute_modulus_r(prime), precompute_modulus_k(prime), 0});
    }

    queue_copy(*q_load_data, primes_ext, buf_primes);

    // Process kernel
    Timer timer_perform("TensorProduct::ProcessKernel");
    auto &buf_c0 = *output_buffer_[buf_index_][0];
    auto &buf_c1 = *output_buffer_[buf_index_][1];
    auto &buf_c2 = *output_buffer_[buf_index_][2];
    store_events_[buf_index_] = L1::TensorProduct(
        *q_kernel, buf_inputs, buf_inputs, buf_inputs, buf_inputs, buf_c0,
        buf_c1, buf_c2, buf_primes, num_primes, inputs_offset.s0(),
        inputs_offset.s1(), inputs_offset.s2(), inputs_offset.s3(),
        depend_event, 0xff);

#if SYNC_MODE
    store_events_[buf_index_].wait();
    if (debug_) {
      PrintEventTime(store_events_[buf_index_], "TensorProduct - Kernel");
    }
#endif
    return store_events_[buf_index_];
  }

  void ProcessOutput(int output_buf_index) {
    if (!output_ptr_[output_buf_index][0]) return;
    Timer t_process_output("TensorProduct::ProcessOutput", debug_);
    if (debug_) {
      std::cout << "TensorProduct::ProcessOutput - " << output_buf_index
                << std::endl;
    }
    assert(output_size_[output_buf_index] <=
           output_buffer_[output_buf_index][0]->size());
    {
      Timer t_process_output("TensorProduct::store_events_::wait", debug_);
      store_events_[output_buf_index].wait();
      if (debug_)
        PrintEventTime(store_events_[output_buf_index],
                       "TensorProduct::kernel");
    }

    q_store_data->submit([&](sycl::handler &h) {
      h.depends_on(store_events_[output_buf_index]);
      h.copy(output_buffer_[output_buf_index][0]
                 ->template get_access<sycl::access::mode::read>(
                     h, sycl::range<1>(output_size_[output_buf_index])),
             output_ptr_[output_buf_index][0]);
    });
    q_store_data->submit([&](sycl::handler &h) {
      h.depends_on(store_events_[output_buf_index]);
      h.copy(output_buffer_[output_buf_index][1]
                 ->template get_access<sycl::access::mode::read>(
                     h, sycl::range<1>(output_size_[output_buf_index])),
             output_ptr_[output_buf_index][1]);
    });
    q_store_data->submit([&](sycl::handler &h) {
      h.depends_on(store_events_[output_buf_index]);
      h.copy(output_buffer_[output_buf_index][2]
                 ->template get_access<sycl::access::mode::read>(
                     h, sycl::range<1>(output_size_[output_buf_index])),
             output_ptr_[output_buf_index][2]);
    });
#if 1
    q_store_data->wait();
    output_ptr_[output_buf_index][0] = NULL;
    output_ptr_[output_buf_index][1] = NULL;
    output_ptr_[output_buf_index][2] = NULL;
#endif
  }

  void Enqueue(const std::vector<uint8_t> &primes_index,
               const std::vector<uint64_t> &inputs, sycl::ulong4 inputs_offset,
               std::vector<uint64_t> &c0, std::vector<uint64_t> &c1,
               std::vector<uint64_t> &c2, sycl::event depend_event) {
    output_ptr_[buf_index_][0] = c0.data();
    output_ptr_[buf_index_][1] = c1.data();
    output_ptr_[buf_index_][2] = c2.data();
    output_size_[buf_index_] = c0.size();
    ProcessOutput(GetNextBufferIndex());
    ProcessInput(primes_index, inputs, inputs_offset, depend_event);
    buf_index_ = (buf_index_ + 1) % buf_depth_;
  }

  void Enqueue(const std::vector<uint8_t> &primes_index,
               sycl::buffer<uint64_t> &inputs, sycl::ulong4 inputs_offset,
               std::vector<uint64_t> &c0, std::vector<uint64_t> &c1,
               std::vector<uint64_t> &c2, sycl::event depend_event) {
    output_ptr_[buf_index_][0] = c0.data();
    output_ptr_[buf_index_][1] = c1.data();
    output_ptr_[buf_index_][2] = c2.data();
    output_size_[buf_index_] = c0.size();
    ProcessOutput(GetNextBufferIndex());
    ProcessInput(primes_index, inputs, inputs_offset, depend_event);
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
  sycl::buffer<uint64_t> *output_buffer_[MAX_BUFF_DEPTH][3];
  sycl::event store_events_[MAX_BUFF_DEPTH];
  std::vector<uint64_t> all_primes_;
  sycl::buffer<ulong4> *buf_primes_[MAX_BUFF_DEPTH];
  void *output_ptr_[MAX_BUFF_DEPTH][3];
  size_t output_size_[MAX_BUFF_DEPTH];

  int buf_depth_;
  int buf_index_;
  bool debug_;
};
}  // namespace TensorProduct
}  // namespace bgv
}  // namespace helib
}  // namespace L2
