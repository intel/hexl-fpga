#include "utils.h"
#include <CL/sycl.hpp>
#include "../L1/helib_bgv.h"
#include <NTL/ZZ.h>
#include <hexl/hexl.hpp>
#include <hexl/ntt/ntt.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace L2 {
namespace helib {
namespace bgv {
template <class intt_t>
void launch_intt_config_tf(intt_t &intt, unsigned long degree,
                           const std::vector<uint64_t> &primes) {
  static std::vector<uint64_t> invRootOfUnityPowers;

  int VEC = intt.get_VEC();
  for (long prime : primes) {
    // create a HEXL ntt instance to get the twiddle factors
    ::intel::hexl::NTT ntt_hexl(degree, prime);

    // intt needs to remove the first ununsed element as the first VEC of the
    // twiddle factors needs all the elements, while ntt only use just one
    // elements so that it can be shifted out without removing the first unused
    // elements
    auto tmp = ntt_hexl.GetInvRootOfUnityPowers();

    // ignore the first one
    for (int i = 0; i < VEC - 1; i++) {
      auto val = tmp[tmp.size() - VEC + 1 + i];
      invRootOfUnityPowers.push_back(val);
      invRootOfUnityPowers.push_back(get_y_barret(val, prime));
    }
    // append the prime as the size of the twiddle factors should be N
    invRootOfUnityPowers.push_back(prime);
    // no y berrett needed
    invRootOfUnityPowers.push_back(prime);

    for (long i = 0; i < tmp.size() / VEC; i++) {
      invRootOfUnityPowers.push_back(tmp[1 + i * VEC]);
    }

    // The last group: 8,9,10,11,12,13,14,15,4,5,6,7,2,3,1,0
    // w^N/2, w^N/4, w^N/8, w^N/16 at index 1,2,4,8 -> 14,12,8,0
  }
#if DEBUG
  for (int i = 0; i < 16; i++) {
    std::cout << invRootOfUnityPowers[i] << ", " << std::endl;
  }
#endif
  assert(invRootOfUnityPowers.size() ==
         (degree / VEC + VEC * 2) * primes.size());

#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif
  static sycl::queue q_intt_config_tf(device_selector);
  std::cout << "launching intt_config_tf" << std::endl;
  intt.config_tf(q_intt_config_tf, invRootOfUnityPowers);
}  // namespace bgv

template <class intt_t>
void launch_compute_inverse(intt_t &intt, unsigned long degree,
                            const std::vector<uint64_t> &primes) {
  // generate intt configurations
  static std::vector<sycl::ulong4> intt_configs;
  for (uint64_t prime : primes) {
    sycl::ulong4 intt_config;
    intt_config[0] = prime;
    __int128 a = 1;
    unsigned long k = precompute_modulus_k(prime);
    unsigned long r = (a << (2 * k)) / prime;
    intt_config[1] = r;
    // compute N^{-1} mod prime
    intt_config[2] = ::intel::hexl::InverseMod(degree, prime);
    intt_config[3] = k;
    // std::cout << "r = " << r << ", k = " << k << std::endl;
    intt_configs.push_back(intt_config);
  }

  assert(intt_configs.size() == primes.size());

#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif
  static sycl::queue q_compute_inverse(device_selector);
  std::cout << "launching q_compute_inverse " << std::flush;
  intt.compute_inverse(q_compute_inverse, intt_configs);
  std::cout << "Done" << std::endl;
}

template <class intt_t>
void launch_intt(intt_t &intt, const std::vector<uint64_t> &primes,
                 uint64_t degree) {
  // Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif

  sycl::event e;
  static bool b_initialized = false;

  // launch the always running kernels
  if (!b_initialized) {
    b_initialized = true;

    // config the twiddle factor factory kernel
    launch_intt_config_tf(intt, degree, primes);

    static sycl::queue q_intt_read(device_selector);
    std::cout << "launching intt_read" << std::endl;
    intt.read(q_intt_read);

    static sycl::queue q_intt_write(device_selector);
    std::cout << "launching intt_write" << std::endl;
    intt.write(q_intt_write);

    static sycl::queue q_intt_norm(device_selector);
    std::cout << "launching intt_norm" << std::endl;
    intt.norm(q_intt_norm);

    launch_compute_inverse(intt, degree, primes);
  }
}
}  // namespace bgv
}  // namespace helib
}  // namespace L2
