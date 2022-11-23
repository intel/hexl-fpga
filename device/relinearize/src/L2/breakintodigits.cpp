#include <L2/breakintodigits-impl.hpp>

using namespace sycl;

namespace L2 {
namespace helib {
namespace bgv {
namespace BreakIntoDigits {
void Init(std::vector<uint64_t> &all_primes) {
  BreakIntoDigitsImpl &impl = BreakIntoDigitsImpl::GetInstance();
  impl.Init(all_primes, 1, 2);
}

void BreakIntoDigits(std::vector<uint64_t> &input,
                     std::vector<uint64_t> &output, std::vector<uint64_t> &pi,
                     std::vector<unsigned> num_designed_digits_primes,
                     unsigned num_special_primes) {
  BreakIntoDigitsImpl &impl = BreakIntoDigitsImpl::GetInstance();
  impl.Enqueue(input, pi, num_designed_digits_primes, num_special_primes,
               output);
}

void Wait() {
  BreakIntoDigitsImpl &impl = BreakIntoDigitsImpl::GetInstance();
  impl.ProcessLeftOutput();
}
}  // namespace BreakIntoDigits
}  // namespace bgv
}  // namespace helib
}  // namespace L2
