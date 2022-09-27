#include <L1/helib_bgv.h>
#include <L2/intt.hpp>

namespace L2 {
namespace helib {
namespace bgv {
void intt1(const std::vector<uint64_t> &primes) {
  launch_intt<L1::helib::bgv::intt1_t, COEFF_COUNT>(L1::helib::bgv::get_intt1(),
                                                    primes);
}
}  // namespace bgv
}  // namespace helib
}  // namespace L2
