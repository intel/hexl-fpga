#include <L1/helib_bgv.h>

namespace L2 {
namespace helib {
namespace bgv {
// scale load
void intt1_load(std::vector<uint64_t> &c,
                std::vector<uint8_t> &prime_index_set) {
  // Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif
  sycl::event e;
  auto c_buf = new sycl::buffer<uint64_t>(c.size());
  auto prime_index_set_buf = new sycl::buffer<uint8_t>(prime_index_set.size());
  static sycl::queue q_intt1_load(device_selector);

  // copy c
  q_intt1_load.submit([&](handler &h) {
    auto c_accessor =
        c_buf->template get_access<access::mode::discard_write>(h);
    h.copy(c.data(), c_accessor);
  });

  // copy prime_index_set
  e = q_intt1_load.submit([&](handler &h) {
    auto prime_index_set_accessor =
        prime_index_set_buf->template get_access<access::mode::discard_write>(
            h);
    h.copy(prime_index_set.data(), prime_index_set_accessor);
  });

  // wait the copy done
  e.wait();

  e = L1::helib::bgv::intt1_load(q_intt1_load, *c_buf, *prime_index_set_buf);
}

// ntt1 load prime index
void ntt1_load_prime_index(std::vector<uint8_t> &prime_index_set) {
  // Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif
  sycl::event e;
  sycl::buffer<uint8_t> *prime_index_set_buf =
      new sycl::buffer(prime_index_set);
  static sycl::queue q_ntt1_prime_index(device_selector);
  std::cout << "Launching ntt1_load_prime_index" << std::endl;
  e = L1::helib::bgv::ntt1_load_prime_index(q_ntt1_prime_index,
                                            *prime_index_set_buf);
}
}  // namespace bgv
}  // namespace helib
}  // namespace L2
