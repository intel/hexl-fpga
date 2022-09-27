#include <CL/sycl.hpp>
#include <L1/helib_bgv.h>
#include <L1/keySwitchDigits.h>
#include <L2/utils.h>
#include <L2/ntt.hpp>
#include <NTL/ZZ.h>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace L2 {
namespace helib {
namespace bgv {

void LaunchKeySwitchDigitsNTT(const std::vector<uint64_t> &primes) {
  assert(primes.size() > 0);
  launch_ntt<L1::helib::bgv::keySwitchDigits_ntt_t, COEFF_COUNT>(
      L1::helib::bgv::GetKeySwitchDigitsNTT(), primes);
}

/*event keySwitchDigits(sycl::queue &q, sycl::buffer<ulong4> &primes,
                      sycl::buffer<ulong> &wa, sycl::buffer<ulong> &wb)
                      */
void KeySwitchDigits(uint32_t coeff_count, const std::vector<uint64_t> &primes,
                     const std::vector<uint8_t> &primes_index,
                     const std::vector<uint64_t> &input,
                     const std::vector<uint64_t> &wa,
                     const std::vector<uint64_t> &wb, std::vector<uint64_t> &c1,
                     std::vector<uint64_t> &c2) {
  // the primes for keyswitch digits, the prime index should start from zero as
  // the keys starts from 0
  std::vector<sycl::ulong4> primes_keyswitch_digits;

  assert(primes_index.size() > 0);
  assert(input.size() > 0);

  // pre-computing r and k for primes
  for (size_t i = 0; i < primes.size(); i++) {
    ulong4 tmp;
    auto prime = primes[i];
    auto prime_index = primes_index[i] - primes_index[0];
    tmp.s0() = prime;
    tmp.s1() = prime_index;
    tmp.s2() = precompute_modulus_r(prime);
    tmp.s3() = precompute_modulus_k(prime);
    primes_keyswitch_digits.push_back(tmp);
  }

  // Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif

  // create buffer, the scopt of buffer should be equal or larger than the
  // kernels
  // TODO: free those buffers when the kernel is finished
  // Is it possible to create a larger engough buffer and always use this
  // buffer in the following kernels?
  auto primes_keyswitch_digits_buf =
      new buffer<sycl::ulong4>(primes_keyswitch_digits.size());
  auto _input = new buffer<sycl::ulong>(input.size());
  auto _wa = new buffer<sycl::ulong>(wa.size());
  auto _wb = new buffer<sycl::ulong>(wb.size());
  auto primes_index_buf = new buffer<uint8_t>(primes_index.size());

  sycl::event e;
  static sycl::queue q(device_selector);

  // copy the input buffers explicitly
  queue_copy(q, primes_keyswitch_digits, primes_keyswitch_digits_buf);
  queue_copy(q, input, _input);
  queue_copy(q, wa, _wa);
  queue_copy(q, wb, _wb);
  queue_copy(q, primes_index, primes_index_buf);

  // launch breakIntoDigits
  L1::helib::bgv::keySwitchDigits(q, *primes_keyswitch_digits_buf, *_wa, *_wb);

  // queue for load and store
  sycl::queue q_load(device_selector);
  sycl::queue q_store_c1(device_selector);
  sycl::queue q_store_c2(device_selector);

  // launch load and store
  std::cout << "Launch keySwitchDigits_load" << std::endl;
  L1::helib::bgv::keySwitchDigits_load(q_load, *_input, *primes_index_buf);

  sycl::buffer _c1(c1);
  sycl::buffer _c2(c2);
  std::cout << "Launch keySwitchDigits_store_c1" << std::endl;
  sycl::event e1 = L1::helib::bgv::keySwitchDigits_store_c1(q_store_c1, _c1);
  std::cout << "Launch keySwitchDigits_store_c2" << std::endl;
  sycl::event e2 = L1::helib::bgv::keySwitchDigits_store_c2(q_store_c2, _c2);

  // wait result
  e1.wait();
  e2.wait();
}
}  // namespace bgv
}  // namespace helib
}  // namespace L2
