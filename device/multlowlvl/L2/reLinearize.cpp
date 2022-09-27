#include <algorithm>
#include <CL/sycl.hpp>
#include <L1/helib_bgv.h>
#include <L1/reLinearize.h>
#include <L2/utils.h>
#include <L2/ntt.hpp>
#include <NTL/ZZ.h>
#include <sycl/ext/intel/fpga_extensions.hpp>

namespace L2 {
namespace helib {
namespace bgv {
void C2DropSmall(std::vector<uint64_t> &c2, uint32_t coeff_count,
                 std::vector<uint64_t> &pi, std::vector<uint64_t> &qj,
                 uint64_t plainText) {
  size_t P, Q, I;
  std::vector<sycl::ulong2> scale_param_set;
  std::vector<uint64_t> empty_vec;
  PreComputeScaleParamSet<true, false>(pi, qj, plainText, empty_vec, P, Q, I,
                                       scale_param_set);

  // Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif

  static sycl::queue q_load(device_selector);
  static sycl::queue q_scale(device_selector);

  auto c2_buff = new sycl::buffer<uint64_t>(c2.size());
  auto scale_param_set_buf = new buffer<sycl::ulong2>(scale_param_set.size());

  queue_copy(q_scale, scale_param_set, scale_param_set_buf);
  queue_copy(q_load, c2, c2_buff);

  L1::helib::bgv::C2DropSmallLoad(q_load, *c2_buff);
  L1::helib::bgv::C2DropSmall(q_scale, coeff_count, *scale_param_set_buf, P, Q,
                              I, plainText);
}

void BreakIntoDigits(uint32_t coeff_count, std::vector<uint64_t> &pi,
                     std::vector<uint64_t> &all_primes,
                     std::vector<uint64_t> &output, unsigned num_digit_primes,
                     unsigned num_special_primes) {
  std::vector<sycl::ulong4> pstar_inv;
  std::vector<sycl::ulong2> pstar_qj;
  std::vector<sycl::ulong> P_qj;
  std::vector<sycl::ulong2> P_inv;

  /* packed parameters including all possible of number of pi primes */
  // FORMAT:
  // pi and pi recip - all normal primes and special primes
  // pstar_inv and pstar_inv_recip - all normal primes
  // P_qj - num_digit_primes (normal_primes/2) + special primes
  // P_inv: P (prod of the first prime mod digit 2) - num_digit_primes
  std::vector<sycl::ulong2> packed_precomuted_params;

  unsigned num_normal_primes = pi.size() - num_special_primes;
  // num_digit_primes is the maxmimum num of every digit primes
  unsigned num_digit1_primes = std::min(num_normal_primes, num_digit_primes);
  unsigned num_digit2_primes = num_normal_primes - num_digit1_primes;
  std::cout << "num_digit1_primes = " << num_digit1_primes
            << ", num_digit2_primes = " << num_digit2_primes << std::endl;

  // compute prod of digit 1 primes
  NTL::ZZ P1(1);
  for (int i = 0; i < num_digit1_primes; i++) {
    P1 *= pi[i];
  }

  // compute prod of digit 2 primes
  NTL::ZZ P2(1);
  for (int i = 0; i < num_digit2_primes; i++) {
    P2 *= pi[i + num_digit1_primes];
  }

  // generate digit1 and digit2 primes
  std::vector<uint64_t> digit1_primes;
  std::vector<uint64_t> digit2_primes;
  for (int i = 0; i < num_digit1_primes; i++) {
    digit1_primes.push_back(pi[i]);
  }

  for (int i = num_digit1_primes; i < num_digit1_primes + num_digit2_primes;
       i++) {
    digit2_primes.push_back(pi[i]);
  }

  assert(digit1_primes.size() == num_digit1_primes);
  assert(digit2_primes.size() == num_digit2_primes);

  // gererate digit1 qj primes
  std::vector<uint64_t> digit1_qj_primes;
  std::vector<uint64_t> digit2_qj_primes;
  for (int i = 0; i < num_digit2_primes + num_special_primes; i++) {
    digit1_qj_primes.push_back(pi[i + num_digit1_primes]);
  }

  // gererate digit2 qj primes
  for (int i = 0; i < num_digit1_primes; i++) {
    digit2_qj_primes.push_back(pi[i]);
  }
  for (int i = 0; i < num_special_primes; i++) {
    digit2_qj_primes.push_back(pi[i + num_normal_primes]);
  }

  unsigned num_digit1_qj_primes = digit1_qj_primes.size();
  unsigned num_digit2_qj_primes = digit2_qj_primes.size();

  // pstar_inv has all the primes
  for (int i = 0; i < num_digit1_primes; i++) {
    ulong p_star_inv_i = NTL::InvMod(NTL::rem(P1 / pi[i], pi[i]), pi[i]);
    auto tmp = mulmod_y_ext(p_star_inv_i, pi[i]);
    auto tmp2 = mulmod_y_ext(NTL::InvMod(p_star_inv_i, pi[i]), pi[i]);

    pstar_inv.push_back({tmp.s0(), tmp.s1(), tmp2.s0(), tmp2.s1()});
  }

  for (int i = num_digit1_primes; i < num_digit1_primes + num_digit2_primes;
       i++) {
    ulong p_star_inv_i = NTL::InvMod(NTL::rem(P2 / pi[i], pi[i]), pi[i]);
    auto tmp = mulmod_y_ext(p_star_inv_i, pi[i]);
    auto tmp2 = mulmod_y_ext(NTL::InvMod(p_star_inv_i, pi[i]), pi[i]);

    pstar_inv.push_back({tmp.s0(), tmp.s1(), tmp2.s0(), tmp2.s1()});
  }

  // compute pstar_qj
  for (int i = 0; i < MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES / 2; i++) {
    for (int j = 0; j < MAX_NORMAL_PRIMES / 2; j++) {
      // P1 mod qj
      auto P1_qj = i < num_digit1_qj_primes && j < num_digit1_primes
                       ? NTL::rem(P1 / digit1_primes[j], digit1_qj_primes[i])
                       : 0;
      // std::cout << "P1_qj = " << P1_qj << std::endl;
      pstar_qj.push_back(mulmod_y_ext(P1_qj, digit1_qj_primes[i]));
    }
    for (int j = 0; j < MAX_NORMAL_PRIMES / 2; j++) {
      // P2 mod qj
      auto P2_qj = i < num_digit2_qj_primes && j < num_digit2_primes
                       ? NTL::rem(P2 / digit2_primes[j], digit2_qj_primes[i])
                       : 0;
      pstar_qj.push_back(mulmod_y_ext(P2_qj, digit2_qj_primes[i]));
    }
  }

  // comput P_qj
  for (int i = 0; i < MAX_SPECIAL_PRIMES + MAX_NORMAL_PRIMES / 2; i++) {
    P_qj.push_back(i < num_digit1_qj_primes ? NTL::rem(P1, digit1_qj_primes[i])
                                            : 0);
    P_qj.push_back(i < num_digit2_qj_primes ? NTL::rem(P2, digit2_qj_primes[i])
                                            : 0);
  }

  // compute P_inv
  // P is the prod of digit 1 primes
  // inv on qj of digit 2 primes
  for (int i = 0; i < num_digit2_primes; i++) {
    auto qj = pi[i + num_digit1_primes];
    P_inv.push_back(mulmod_y_ext(NTL::InvMod(NTL::rem(P1, qj), qj), qj));
  }

  // compute pi_recip
  std::vector<ulong2> pi_with_recip;
  for (int i = 0; i < pi.size(); i++) {
    double pi_recip = (double)1 / pi[i];
    pi_with_recip.push_back({pi[i], *(ulong *)&pi_recip});
  }

  // packing now
  // pi and pi recip - all normal primes and special primes
  for (size_t i = 0;
       i < num_digit1_primes + num_digit2_primes + num_special_primes; i++) {
    if (i < pi_with_recip.size())
      packed_precomuted_params.push_back(pi_with_recip[i]);
    else
      packed_precomuted_params.push_back({0, 0});
  }

  // pstar_inv and pstar_inv_recip - all normal primes
  for (size_t i = 0; i < num_digit1_primes + num_digit2_primes; i++) {
    if (i < pstar_inv.size()) {
      auto tmp = pstar_inv[i];
      packed_precomuted_params.push_back({tmp.s0(), tmp.s1()});
    } else {
      packed_precomuted_params.push_back({0, 0});
    }
  }

  // pstar_inv_recip - all normal primes
  for (size_t i = 0; i < num_digit1_primes + num_digit2_primes; i++) {
    if (i < pstar_inv.size()) {
      auto tmp = pstar_inv[i];
      packed_precomuted_params.push_back({tmp.s2(), tmp.s3()});
    } else {
      packed_precomuted_params.push_back({0, 0});
    }
  }

  // P_qj - std::max(num_digit1_primes, num_digit2_primes) + num_special_primes
  for (size_t i = 0;
       i < std::max(num_digit1_primes, num_digit2_primes) + num_special_primes;
       i++) {
    if (i < (P_qj.size() / 2)) {
      // pack two elements in one ulong2
      packed_precomuted_params.push_back({P_qj[i * 2], P_qj[i * 2 + 1]});
    } else {
      packed_precomuted_params.push_back({0, 0});
    }
  }

  // P_inv: P (prod of the first prime mod digit 2) - num_digit2_primes
  for (size_t i = 0; i < num_digit2_primes; i++) {
    packed_precomuted_params.push_back(P_inv[i]);
  }

  for (int i = 0; i < num_digit2_primes + num_special_primes; i++) {
    for (int j = 0; j < num_digit1_primes; j++) {
      // P1 mod qj
      auto P1_qj = NTL::rem(P1 / digit1_primes[j], digit1_qj_primes[i]);
      // std::cout << "P1_qj = " << P1_qj << std::endl;
      packed_precomuted_params.push_back(
          mulmod_y_ext(P1_qj, digit1_qj_primes[i]));
    }
  }

  for (int i = 0; i < num_digit1_primes + num_special_primes; i++) {
    for (int j = 0; j < num_digit2_primes; j++) {
      auto P2_qj = NTL::rem(P2 / digit2_primes[j], digit2_qj_primes[i]);
      packed_precomuted_params.push_back(
          mulmod_y_ext(P2_qj, digit2_qj_primes[i]));
    }
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
  assert(pi.size() > 0);
  auto pi_buf = new buffer<sycl::ulong2>(pi.size());
  assert(pstar_inv.size() > 0);
  auto pstar_inv_buf = new buffer<sycl::ulong4>(pstar_inv.size());
  assert(pstar_qj.size() > 0);
  auto pstar_qj_buf = new buffer<sycl::ulong2>(pstar_qj.size());
  assert(P_qj.size() > 0);
  auto P_qj_buf = new buffer<sycl::ulong>(P_qj.size());
  assert(P_inv.size() > 0);
  auto P_inv_buf = new buffer<ulong2>(P_inv.size());

  auto packed_precomuted_params_buf =
      new buffer<ulong2>(packed_precomuted_params.size());

  sycl::event e;
  static sycl::queue q(device_selector);

  queue_copy(q, packed_precomuted_params, packed_precomuted_params_buf);

  // launch breakIntoDigits
  std::cout << "Launch BreakIntoDigits" << std::endl;
  L1::helib::bgv::BreakIntoDigits(q, *packed_precomuted_params_buf,
                                  num_digit1_primes, num_digit2_primes,
                                  num_special_primes);

  // queue for load and store
  sycl::queue q_store(device_selector);

  std::cout << "Launch store" << std::endl;
  sycl::buffer output_buf(output);
  e = L1::helib::bgv::BreakIntoDigits_store(q_store, output_buf);
  e.wait();
}

void LaunchReLinearizeC01NTT(const std::vector<uint64_t> &primes) {
  assert(primes.size() > 0);
  launch_ntt(L1::helib::bgv::GetReLinearizeC01NTT(), primes, COEFF_COUNT);
}

void LaunchReLinearizeC01LoadPrimeIndex(
    const std::vector<uint8_t> &primes_index) {
  // Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif
  static sycl::queue q(device_selector);
  L1::helib::bgv::ReLinearizeC01LoadPrimeIndex(
      q, sycl::uchar2{primes_index[0], primes_index[primes_index.size() - 1]});
}

void C0DropSmall(std::vector<uint64_t> &input, uint32_t coeff_count,
                 std::vector<uint64_t> &pi, std::vector<uint64_t> &qj,
                 std::vector<uint64_t> &all_primes,
                 std::vector<uint8_t> &qj_primes_index,
                 std::vector<uint64_t> &special_primes, uint64_t plainText,
                 std::vector<uint64_t> &output) {
  size_t P, Q, I;
  std::vector<sycl::ulong2> scale_param_set;
  PreComputeScaleParamSet<true, true>(pi, qj, plainText, special_primes, P, Q,
                                      I, scale_param_set);

  // Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif

  auto scale_param_set_buf = new buffer<sycl::ulong2>(scale_param_set.size());

  sycl::event e;
  auto prop_list = property_list{property::queue::enable_profiling()};
  static sycl::queue q_scale(device_selector, prop_list);

  queue_copy(q_scale, scale_param_set, scale_param_set_buf);

  static sycl::queue q_load(device_selector);

  // launch load
  auto input_buf = new sycl::buffer<uint64_t>(input.size());
  e = q_load.submit([&](handler &h) {
    h.copy(input.data(),
           input_buf->template get_access<access::mode::discard_write>(h));
  });
  e.wait();

  // launch NTT
  LaunchReLinearizeC01NTT(all_primes);

  // launch NTT prime index Loader
  LaunchReLinearizeC01LoadPrimeIndex(qj_primes_index);

  L1::helib::bgv::C0DropSmallLoad(q_load, *input_buf);
  std::cout << "Launch C0DropSmall" << std::endl;
  e = L1::helib::bgv::C0DropSmall(q_scale, coeff_count, *scale_param_set_buf, P,
                                  Q, I, plainText);

  static sycl::queue q_store(device_selector);
  sycl::buffer<uint64_t> output_buf(output);

  e = L1::helib::bgv::C0Store(q_store, output_buf);
  e.wait();
  std::cout << "Launch C0DropSmall Done!" << std::endl;
}

}  // namespace bgv
}  // namespace helib
}  // namespace L2
