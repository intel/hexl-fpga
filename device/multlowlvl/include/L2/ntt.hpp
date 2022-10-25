#pragma once
#include "utils.h"
#include <CL/sycl.hpp>
#include "../L1/helib_bgv.h"
#include <NTL/ZZ.h>
#include <cinttypes>
#include <hexl/hexl.hpp>
#include <hexl/ntt/ntt.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <vector>

namespace L2 {
namespace helib {
namespace bgv {

using namespace L1::helib::bgv;

template <class ntt_t, int id>
void launch_ntt_config_tf(ntt_t &ntt, unsigned long degree,
                          const std::vector<uint64_t> &primes) {
  // twiddle factors should be statis as the kernel need to access it
  static std::vector<uint64_t> rootOfUnityPowers;
  for (long prime : primes) {
    // create a HEXL ntt instance to get the twiddle factors
    ::intel::hexl::NTT ntt_hexl(degree, prime);

    auto tfdata = ntt_hexl.GetRootOfUnityPowers();

    // push w^N/2, w^N/4, w^N/8, w^N/16 at index 1,2,4,8
    rootOfUnityPowers.push_back(tfdata[1]);
    rootOfUnityPowers.push_back(get_y_barret(tfdata[1], prime));
    rootOfUnityPowers.push_back(tfdata[2]);
    rootOfUnityPowers.push_back(get_y_barret(tfdata[2], prime));
    rootOfUnityPowers.push_back(tfdata[4]);
    rootOfUnityPowers.push_back(get_y_barret(tfdata[4], prime));
    rootOfUnityPowers.push_back(tfdata[8]);
    rootOfUnityPowers.push_back(get_y_barret(tfdata[8], prime));

    // ntt doesn't need to remove the first element as
    // the first VEC of NTT operation only relies on just one element
    // so that the first un-used element can be shifted out
    rootOfUnityPowers.push_back(prime);
    for (uint64_t i = 1; i < tfdata.size() / ntt.get_VEC(); i++) {
      rootOfUnityPowers.push_back(tfdata[i * ntt.get_VEC()]);
    }
  }
  assert(rootOfUnityPowers.size() ==
         (degree / ntt.get_VEC() + 8) * primes.size());

#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif
  static sycl::queue q_ntt_config_tf(device_selector);
  std::cout << "launching ntt_config_tf" << std::endl;
  ntt.config_tf(q_ntt_config_tf, rootOfUnityPowers);
}

template <class ntt_t, int id>
void launch_compute_forward(ntt_t &ntt, unsigned long degree,
                            const std::vector<uint64_t> primes) {
  assert(primes.size() > 0);
  // generate intt configurations
  static std::vector<sycl::ulong4> ntt_configs;
  for (uint64_t prime : primes) {
    sycl::ulong4 config;
    config[0] = prime;
    __int128 a = 1;
    unsigned long k = precompute_modulus_k(prime);
    unsigned long r = (a << (2 * k)) / prime;
    config[1] = r;
    // compute N^{-1} mod prime
    config[2] = ::intel::hexl::InverseMod(degree, prime);
    config[3] = k;
    // std::cout << "r = " << r << ", k = " << k << std::endl;
    ntt_configs.push_back(config);
  }

  assert(ntt_configs.size() == primes.size());

#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif
  static sycl::queue q_ntt_compute_forward(device_selector);
  std::cout << "launching ntt_compute_forward" << std::endl;
  ntt.compute_forward(q_ntt_compute_forward, ntt_configs);
  //std::cout << "launching ntt_compute_forward done!" << std::endl;
}

template <class ntt_t, int id>
void launch_ntt(ntt_t &ntt, const std::vector<uint64_t> &primes,
                uint64_t degree) {
  // Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif

  sycl::event e;
  static bool b_initialized = false;
  //std::cout << "ntt old b_initialized address: " << &b_initialized << std::endl;

  // launch the always running kernels
  if (!b_initialized) {
    b_initialized = true;

    // config the twiddle factor factory kernel
    launch_ntt_config_tf<ntt_t, id>(ntt, degree, primes);

    static sycl::queue q_ntt_read(device_selector);
    std::cout << "launching ntt_read" << std::endl;
    ntt.read(q_ntt_read);

    static sycl::queue q_ntt_write(device_selector);
    std::cout << "launching ntt_write" << std::endl;
    ntt.write(q_ntt_write);

    launch_compute_forward<ntt_t, id>(ntt, degree, primes);
  }
}

/**************************************************************************************/

template <int id>
void launch_ntt_config_tf(NTT_Method &ntt, unsigned long degree,
                          const std::vector<uint64_t> &primes) {
  // twiddle factors should be statis as the kernel need to access it
  static std::vector<uint64_t> rootOfUnityPowers;
  for (long prime : primes) {
    // create a HEXL ntt instance to get the twiddle factors
    ::intel::hexl::NTT ntt_hexl(degree, prime);

    auto tfdata = ntt_hexl.GetRootOfUnityPowers();

    // push w^N/2, w^N/4, w^N/8, w^N/16 at index 1,2,4,8
    rootOfUnityPowers.push_back(tfdata[1]);
    rootOfUnityPowers.push_back(get_y_barret(tfdata[1], prime));
    rootOfUnityPowers.push_back(tfdata[2]);
    rootOfUnityPowers.push_back(get_y_barret(tfdata[2], prime));
    rootOfUnityPowers.push_back(tfdata[4]);
    rootOfUnityPowers.push_back(get_y_barret(tfdata[4], prime));
    rootOfUnityPowers.push_back(tfdata[8]);
    rootOfUnityPowers.push_back(get_y_barret(tfdata[8], prime));

    // ntt doesn't need to remove the first element as
    // the first VEC of NTT operation only relies on just one element
    // so that the first un-used element can be shifted out
    rootOfUnityPowers.push_back(prime);
    for (uint64_t i = 1; i < tfdata.size() / ntt.get_VEC(); i++) {
      rootOfUnityPowers.push_back(tfdata[i * ntt.get_VEC()]);
    }
  }
  assert(rootOfUnityPowers.size() ==
         (degree / ntt.get_VEC() + 8) * primes.size());

#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif
  static sycl::queue q_ntt_config_tf(device_selector);
  std::cout << "launching ntt_config_tf" << std::endl;
  ntt.config_tf(q_ntt_config_tf, rootOfUnityPowers);
}

template <int id>
void launch_compute_forward(NTT_Method &ntt, unsigned long degree,
                            const std::vector<uint64_t> primes) {
  assert(primes.size() > 0);
  // generate intt configurations
  static std::vector<sycl::ulong4> ntt_configs;
  for (uint64_t prime : primes) {
    sycl::ulong4 config;
    config[0] = prime;
    __int128 a = 1;
    unsigned long k = precompute_modulus_k(prime);
    unsigned long r = (a << (2 * k)) / prime;
    config[1] = r;
    // compute N^{-1} mod prime
    config[2] = ::intel::hexl::InverseMod(degree, prime);
    config[3] = k;
    // std::cout << "r = " << r << ", k = " << k << std::endl;
    ntt_configs.push_back(config);
  }

  assert(ntt_configs.size() == primes.size());

#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif
  static sycl::queue q_ntt_compute_forward(device_selector);
  std::cout << "launching ntt_compute_forward" << std::endl;
  ntt.compute_forward(q_ntt_compute_forward, ntt_configs);
  //std::cout << "launching ntt_compute_forward done!" << std::endl;
}

template <int id>
void launch_ntt(NTT_Method &ntt, const std::vector<uint64_t> &primes,
                uint64_t degree) {
  // Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif

  sycl::event e;
  static bool b_initialized = false;
  //std::cout << "ntt new b_initialized address: " << &b_initialized << std::endl;

  // launch the always running kernels
  if (!b_initialized) {
    b_initialized = true;

    // config the twiddle factor factory kernel
    launch_ntt_config_tf<id>(ntt, degree, primes);

    static sycl::queue q_ntt_read(device_selector);
    std::cout << "launching ntt_read" << std::endl;
    ntt.read(q_ntt_read);

    static sycl::queue q_ntt_write(device_selector);
    std::cout << "launching ntt_write" << std::endl;
    ntt.write(q_ntt_write);

    launch_compute_forward<id>(ntt, degree, primes);
  }
}





}  // namespace bgv
}  // namespace helib
}  // namespace L2
