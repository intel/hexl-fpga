#include <L1/helib_bgv.h>
#include <L2/ntt.hpp>

namespace L2 {
namespace helib {
namespace bgv {
void ntt1(const std::vector<uint64_t> &primes) {
  launch_ntt<L1::helib::bgv::ntt1_t, COEFF_COUNT>(L1::helib::bgv::get_ntt1(),
                                                  primes);
}
}  // namespace bgv
}  // namespace helib
}  // namespace L2
