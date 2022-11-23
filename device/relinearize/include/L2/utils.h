#pragma once
#include <CL/sycl.hpp>
#include <NTL/ZZ.h>

namespace L2 {
namespace helib {
namespace bgv {
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

template <class Ta, class Tb>
void queue_copy(sycl::queue &q, Ta &src, Tb &dst, size_t size = 0) {
  if (size == 0) size = std::min(src.size(), dst.size());
  q.submit([&](sycl::handler &h) {
    // copy
    h.copy(src.data(),
           dst.template get_access<sycl::access::mode::discard_write>(
               h, sycl::range(size)));
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

class Timer {
 public:
  Timer(const std::string name, bool debug = false, size_t num_bytes = 0)
      : stopped_(false) {
    this->name_ = name;
    this->start_point_ = std::chrono::high_resolution_clock::now();
    this->num_bytes_ = num_bytes;
    this->debug_ = debug;
  }

  void stop() {
    auto end_point = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_point -
                                                                  start_point_);
    if (debug_) {
      std::cout << name_ << " takes " << time_span.count() * 1000 << "ms";
      if (num_bytes_ != 0) {
        std::cout << ", num_bytes = " << num_bytes_ << ", throughput = "
                  << num_bytes_ / time_span.count() / 1024 / 1024 << "MB/s";
      }
      std::cout << std::endl;
    }
    stopped_ = true;
  }
  ~Timer() {
    if (!stopped_) stop();
  }

 private:
  std::string name_;
  std::chrono::high_resolution_clock::time_point start_point_;
  bool stopped_;
  size_t num_bytes_;
  bool debug_;
};
}  // namespace bgv
}  // namespace helib
}  // namespace L2
