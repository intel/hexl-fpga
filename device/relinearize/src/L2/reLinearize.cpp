#include <L2/breakintodigits-impl.hpp>
#include <L2/keyswitchdigits-impl.hpp>

#include <execinfo.h>

using namespace sycl;

namespace L2 {
namespace helib {
namespace bgv {
namespace Relinearize {

void Init(std::vector<uint64_t> &all_primes, const std::vector<uint64_t> &keys1,
          const std::vector<uint64_t> &keys2,
          const std::vector<uint64_t> &keys3,
          const std::vector<uint64_t> &keys4) {
  auto &impl_breakintodigits =
      BreakIntoDigits::BreakIntoDigitsImpl::GetInstance();
  impl_breakintodigits.Init(all_primes, 1, 2);
  auto &impl_keyswitch_digits =
      KeySwitchDigits::KeySwitchDigitsImpl::GetInstance();
  impl_keyswitch_digits.Init(all_primes, keys1, keys2, keys3, keys4, 2, 3, 4);
}

void Relinearize(std::vector<uint64_t> &input, std::vector<uint64_t> &pi,
                 std::vector<unsigned> num_designed_digits_primes,
                 unsigned num_special_primes,
                 const std::vector<uint8_t> &primes_index,
                 std::vector<uint64_t> &output) {
  auto &impl_breakinto_digits =
      BreakIntoDigits::BreakIntoDigitsImpl::GetInstance();
  auto &impl_keyswitch_digits =
      KeySwitchDigits::KeySwitchDigitsImpl::GetInstance();

  auto e = impl_breakinto_digits.Enqueue(input, pi, num_designed_digits_primes,
                                         num_special_primes);
  auto &last_output_buff = impl_breakinto_digits.GetLastOutputBuff();
  auto buf_size = primes_index.size() * COEFF_COUNT;
  ulong4 digits_offset{0, buf_size, buf_size * 2, buf_size * 3};
  assert(output.size() == buf_size * 2);
  impl_keyswitch_digits.Enqueue(primes_index, last_output_buff, digits_offset,
                                output, e);
}

void Wait() {
  auto &impl_breakinto_digits =
      BreakIntoDigits::BreakIntoDigitsImpl::GetInstance();
  auto &impl_keyswitch_digits =
      KeySwitchDigits::KeySwitchDigitsImpl::GetInstance();
  // impl_breakinto_digits.ProcessLeftOutput();
  impl_keyswitch_digits.ProcessLeftOutput();
}
}  // namespace Relinearize
}  // namespace bgv
}  // namespace helib
}  // namespace L2
