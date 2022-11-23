#include <L2/utils.h>
#include <iomanip>

namespace L2 {
namespace helib {
namespace bgv {
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
  __int128 a = y;
  a = a << 64;
  a = a / p;
  return (uint64_t)a;
}

sycl::ulong2 mulmod_y_ext(uint64_t y, uint64_t p) {
  if (y == 0) return {0, 0};
  return {y, get_y_barret(y, p)};
}

void PrintEventTime(sycl::event &e, const char *tag) {
  auto start_time =
      e.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto end_time =
      e.get_profiling_info<sycl::info::event_profiling::command_end>();
  std::cout << tag << " execution time: " << std::fixed << std::setprecision(3)
            << ((double)(end_time - start_time)) / 1000000.0 << "ms\n";
}
}  // namespace bgv
}  // namespace helib
}  // namespace L2
