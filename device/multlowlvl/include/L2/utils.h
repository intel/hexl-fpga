#pragma once
#include <CL/sycl.hpp>
#include <NTL/ZZ.h>
#include <iomanip>


namespace L2 {
namespace helib {
namespace bgv {

typedef __int128 uint128_t;

/**
 * @brief a optimized barrett multiplication
 * https://www.nayuki.io/page/barrett-reduction-algorithm
 *
 * @param modulus
 * @return unsigned
 */
unsigned precompute_modulus_k(unsigned long modulus);

/**
 * @brief precompute_modulus_r
 *
 * @param modulus
 * @return unsigned long
 */
unsigned long precompute_modulus_r(unsigned long modulus);

/**
 * @brief an optimized multiplication from Microsoft
 *
 * @param y
 * @param p
 * @return uint64_t
 */
uint64_t get_y_barret(uint64_t y, uint64_t p);

/**
 * @brief mulmod_y_ext
 *
 * @param y
 * @param p
 * @return sycl::ulong2
 */
sycl::ulong2 mulmod_y_ext(uint64_t y, uint64_t p);


/**************************************/
/*** put implementation here **********/

unsigned precompute_modulus_k(unsigned long modulus) {
  unsigned k;
  for (int i = 64; i > 0; i--) {
    if ((1UL << i) >= modulus) k = i;
  }

  return k;
}

unsigned long precompute_modulus_r(unsigned long modulus) {
  __int128 a = 1;
  unsigned long k = precompute_modulus_k(modulus);
  unsigned long r = (a << (2 * k)) / modulus;
  return r;
}

uint64_t get_y_barret(uint64_t y, uint64_t p) {
  uint128_t a = y;
  a = a << 64;
  a = a / p;
  return (uint64_t)a;
}

sycl::ulong2 mulmod_y_ext(uint64_t y, uint64_t p) {
  if (y == 0) return {0, 0};
  return {y, get_y_barret(y, p)};
}






/**
 * @brief a helper function to perform a blocking copy from a (vector) to b
 * (sycl::buffer*)
 *
 * @tparam Ta
 * @tparam Tb
 * @param q
 * @param a
 * @param b
 */
template <class Ta, class Tb>
void queue_copy(sycl::queue &q, Ta &a, Tb &b) {
  q.submit([&](sycl::handler &h) {
    // copy
    h.copy(a.data(),
           b->template get_access<sycl::access::mode::discard_write>(h));
  });
  q.wait();
}

template <class Ta, class Tb>
sycl::event queue_copy_async(sycl::queue &q, Ta &a, Tb &b) {
  auto event = q.submit([&](sycl::handler &h) {
    // copy
    h.copy(a.data(),
           b->template get_access<sycl::access::mode::discard_write>(h));
  });
  return std::move(event);
}

template <class Ta, class Tb>
sycl::event queue_copy_async2(sycl::queue &q, Ta &a, Tb &b) {
  auto event = q.submit([&](sycl::handler &h) {
    // copy
    h.copy(a, b->template get_access<sycl::access::mode::discard_write>(h));
  });
  return std::move(event);
}

/**
 * @brief Print Event Time
 *
 * @param e
 * @param tag
 */
void PrintEventTime(sycl::event &e, const char *tag);

void PrintEventTime(sycl::event &e, const char *tag) {
  auto start_time =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end_time =
      e.get_profiling_info<sycl::info::event_profiling::command_end>();
  std::cout << tag << " execution time: " << std::fixed << std::setprecision(3)
            << ((double)(end_time - start_time)) / 1000000.0 << "ms\n";
}

class Timer {
 public:
  Timer(const std::string name) : stopped_(false) {
    this->name_ = name;
    this->start_point_ = std::chrono::high_resolution_clock::now();
  }

  void stop() {
    auto end_point = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_point -
                                                                  start_point_);
    std::cout << name_ << " takes " << time_span.count() * 1000 << "ms"
              << std::endl;
    stopped_ = true;
  }
  ~Timer() {
    if (!stopped_) stop();
  }

 private:
  std::string name_;
  std::chrono::high_resolution_clock::time_point start_point_;
  bool stopped_;
};

/**
 * @brief PreComputeScaleParamSet
 *
 * @tparam added_primes_at_end
 * @tparam add_special_primes
 * @param pi
 * @param qj
 * @param plainText
 * @param special_primes
 * @param P
 * @param Q
 * @param I
 * @param scale_param_set
 */
template <bool added_primes_at_end, bool add_special_primes>
void PreComputeScaleParamSet(std::vector<uint64_t> &pi,
                             std::vector<uint64_t> &qj,
                             std::vector<uint8_t> &qj_prime_index,
                             uint64_t plainText,
                             std::vector<uint64_t> &special_primes, size_t &P,
                             size_t &Q, size_t &I,
                             std::vector<sycl::ulong2> &scale_param_set) {
  Timer timer("PreComputeScaleParamSet");
  std::vector<sycl::ulong2> pt;
  std::vector<sycl::ulong2> P_qj;
  std::vector<sycl::ulong2> pstar_inv;
  std::vector<sycl::ulong2> pstar_qj;

  // num of target primes that may add primes
  Q = qj.size();
  // num of drop primes
  P = 0;
  uint64_t t = plainText;

  // compute the num of drop primes
  for (auto prime : pi) {
    P += (std::find(qj.begin(), qj.end(), prime) == qj.end());
  }

  NTL::ZZ diffProd(1);
  for (int i = 0; i < P; i++) {
    diffProd *= pi[i];
  }

  // I is the added primes
  I = Q - (pi.size() - P);
  NTL::ZZ IProd(1);

  // configure the position of added primes
  if (added_primes_at_end) {
    for (int i = Q - I; i < Q; i++) {
      IProd *= qj[i];
    }
  } else {
    for (int i = 0; i < I; i++) {
      IProd *= qj[i];
    }
  }

  NTL::ZZ prodInv(NTL::InvMod(rem(diffProd, t), t));
  for (int i = 0; i < P; i++) {
    ulong pt_i = NTL::rem(prodInv * diffProd / pi[i], t);
    pt.push_back(mulmod_y_ext(pt_i, t));
    ulong p_star_inv_i = NTL::InvMod(NTL::rem(diffProd / pi[i], pi[i]), pi[i]);
    p_star_inv_i = NTL::rem(IProd * p_star_inv_i, pi[i]);
    pstar_inv.push_back(mulmod_y_ext(p_star_inv_i, pi[i]));
  }

  for (int i = 0; i < Q; i++) {
    ulong P_inv_qj_i = NTL::InvMod(NTL::rem(diffProd, qj[i]), qj[i]);

    for (int j = 0; j < P; j++) {
      pstar_qj.push_back(
          mulmod_y_ext(NTL::rem(diffProd / pi[j] * P_inv_qj_i, qj[i]), qj[i]));
    }
    P_inv_qj_i = NTL::rem(IProd * P_inv_qj_i, qj[i]);
    P_qj.push_back(mulmod_y_ext(P_inv_qj_i, qj[i]));
  }

  //  Prod of special primes mod qj
  if (add_special_primes) {
    NTL::ZZ prod_special_primes(1);
    for (int i = 0; i < special_primes.size(); i++) {
      prod_special_primes *= special_primes[i];
    }

    for (int i = 0; i < Q; i++) {
      P_qj.push_back(mulmod_y_ext(NTL::rem(prod_special_primes, qj[i]), qj[i]));
    }
  }

  // packing
  // pi - P
  for (size_t i = 0; i < P; i++) {
    double tmp = 1;
    tmp /= pi[i];
    scale_param_set.push_back({pi[i], *(ulong *)&tmp});
  }

  // qj - Q
  for (size_t i = 0; i < Q; i++) {
    scale_param_set.push_back({qj[i], qj_prime_index[i]});
  }

  // pt - P
  for (size_t i = 0; i < P; i++) {
    scale_param_set.push_back(pt[i]);
  }

  // pstar_inv - P
  for (size_t i = 0; i < P; i++) {
    scale_param_set.push_back(pstar_inv[i]);
  }

  // P_qj - Q
  for (size_t i = 0; i < Q; i++) {
    scale_param_set.push_back(P_qj[i]);
  }

  // R_qj - Q if applicable
  if (add_special_primes) {
    for (size_t i = 0; i < Q; i++) {
      scale_param_set.push_back(P_qj[Q + i]);
    }
  }

  // pstar_qj - Q*P
  for (size_t i = 0; i < Q; i++) {
    for (size_t j = 0; j < P; j++) {
      scale_param_set.push_back(pstar_qj[i * P + j]);
    }
  }
}
}  // namespace bgv
}  // namespace helib
}  // namespace L2
