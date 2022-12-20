#pragma once
//#include "utils.h"
#include <CL/sycl.hpp>
#include <NTL/ZZ.h>
#include <hexl/hexl.hpp>
#include <hexl/ntt/ntt.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace L1 {
static unsigned precompute_modulus_k(unsigned long modulus) {
  unsigned k;
  for (int i = 64; i > 0; i--) {
    if ((1UL << i) >= modulus) k = i;
  }

  return k;
}

static unsigned long precompute_modulus_r(unsigned long modulus) {
  __int128 a = 1;
  unsigned long k = precompute_modulus_k(modulus);
  unsigned long r = (a << (2 * k)) / modulus;
  return r;
}

static uint64_t get_y_barret(uint64_t y, uint64_t p) {
  __int128 a = y;
  a = a << 64;
  a = a / p;
  return (uint64_t)a;
}

static sycl::ulong2 mulmod_y_ext(uint64_t y, uint64_t p) {
  if (y == 0) return {0, 0};
  return {y, get_y_barret(y, p)};
}

template <class intt_t>
void launch_intt_config_tf(intt_t &intt, unsigned long degree,
                           const std::vector<uint64_t> &primes, int flag) {
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
  // std::cout << "launching intt_config_tf" << std::endl;
  intt.config_tf(q_intt_config_tf, invRootOfUnityPowers, flag);
}  // namespace bgv

template <class intt_t>
void launch_compute_inverse(intt_t &intt, unsigned long degree,
                            const std::vector<uint64_t> &primes, int flag) {
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
  // std::cout << "launching q_compute_inverse " << std::flush;
  intt.compute_inverse(q_compute_inverse, intt_configs, flag);
  // std::cout << "Done" << std::endl;
}

template <class intt_t>
void launch_intt(intt_t &intt, const std::vector<uint64_t> &primes,
                 uint64_t degree, int flag = 0xff) {
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
    launch_intt_config_tf(intt, degree, primes, flag);

    static sycl::queue q_intt_read(device_selector);
    // std::cout << "launching intt_read" << std::endl;
    intt.read(q_intt_read);

    static sycl::queue q_intt_write(device_selector);
    // std::cout << "launching intt_write" << std::endl;
    intt.write(q_intt_write);

    static sycl::queue q_intt_norm(device_selector);
    // std::cout << "launching intt_norm" << std::endl;
    intt.norm(q_intt_norm);

    launch_compute_inverse(intt, degree, primes, flag);
  }
}

template <class ntt_t>
void launch_ntt_config_tf(ntt_t &ntt, unsigned long degree,
                          const std::vector<uint64_t> &primes, int flag) {
  // twiddle factors should be statis as the kernel need to access it
  static std::vector<uint64_t> rootOfUnityPowers;
  for (long prime : primes) {
    // create a HEXL ntt instance to get the twiddle factors
    ::intel::hexl::NTT ntt_hexl(degree, prime);

    auto tfdata = ntt_hexl.GetRootOfUnityPowers();

    // pattern of twiddle factors
    // 0, 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, 1/16, 9/16, 5/16, 13/16, 3/16,
    // 11/16, 7/16, 15/16, 1/32, 17/32, 9/32, 25/32, 5/32, 21/32, 13/32,
    // 29/32, 3/32, 19/32, 11/32, 27/32, 7/32, 23/32, 15/32, 31/32, 1/64,
    // 33/64, 17/64, 49/64, 9/64, 41/64, 25/64, 57/64, 5/64, 37/64, 21/64,
    // 53/64, 13/64, 45/64, 29/64, 61/64, 3/64, 35/64, 19/64, 51/64, 11/64,
    // 43/64, 27/64, 59/64, 7/64, 39/64, 23/64, 55/64, 15/64, 47/64, 31/64,
    // 63/64

    // the first element is the prime
    rootOfUnityPowers.push_back(prime);
    rootOfUnityPowers.push_back(0);
    for (int i = 1; i < ntt.get_VEC(); i++) {
      rootOfUnityPowers.push_back(tfdata[i]);
      rootOfUnityPowers.push_back(get_y_barret(tfdata[i], prime));
    }

    for (uint64_t i = 0; i < tfdata.size() / ntt.get_VEC(); i++) {
      rootOfUnityPowers.push_back(tfdata[i * ntt.get_VEC()]);
    }
  }
  assert(rootOfUnityPowers.size() ==
         (degree / ntt.get_VEC() + ntt.get_VEC() * 2) * primes.size());

#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif
  static sycl::queue q_ntt_config_tf(device_selector);
  // std::cout << "launching ntt_config_tf" << std::endl;
  ntt.config_tf(q_ntt_config_tf, rootOfUnityPowers, flag);
}

template <class ntt_t>
void launch_compute_forward(ntt_t &ntt, unsigned long degree,
                            const std::vector<uint64_t> primes, int flag) {
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
  // std::cout << "launching ntt_compute_forward" << std::endl;
  ntt.compute_forward(q_ntt_compute_forward, ntt_configs, flag);
  // std::cout << "launching ntt_compute_forward done!" << std::endl;
}

template <class ntt_t>
void launch_ntt(ntt_t &ntt, const std::vector<uint64_t> &primes,
                uint64_t degree, int flag = 0xff) {
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
    launch_ntt_config_tf(ntt, degree, primes, flag);

    static sycl::queue q_ntt_read(device_selector);
    // std::cout << "launching ntt_read" << std::endl;
    ntt.read(q_ntt_read);

    static sycl::queue q_ntt_write(device_selector);
    // std::cout << "launching ntt_write" << std::endl;
    ntt.write(q_ntt_write);

    launch_compute_forward(ntt, degree, primes, flag);
  }
}
}  // namespace L1
