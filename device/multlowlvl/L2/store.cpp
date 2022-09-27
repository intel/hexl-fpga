#include <L1/helib_bgv.h>

namespace L2 {
namespace helib {
namespace bgv {
// ntt1 store
void ntt1_store(std::vector<uint64_t> &c) {
  // Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif
  sycl::event e;
  sycl::buffer<uint64_t> c_buf(c);
  static sycl::queue q_ntt1_store(device_selector,
                                  property::queue::enable_profiling{});
  e = L1::helib::bgv::ntt1_store(q_ntt1_store, c_buf);

  // wait completed
  e.wait();

  double start = e.get_profiling_info<info::event_profiling::command_start>();
  double end = e.get_profiling_info<info::event_profiling::command_end>();

  // convert from nanoseconds to ms
  double kernel_time = (double)(end - start) * 1e-6;
  std::cout << "Store time: " << std::fixed << kernel_time << " ms"
            << std::endl;
}
}  // namespace bgv
}  // namespace helib
}  // namespace L2
