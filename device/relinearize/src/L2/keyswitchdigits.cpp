#include <L2/keyswitchdigits-impl.hpp>
namespace L2 {
namespace helib {
namespace bgv {
namespace KeySwitchDigits {
void Init(const std::vector<uint64_t> &primes,
          const std::vector<uint64_t> &keys1,
          const std::vector<uint64_t> &keys2,
          const std::vector<uint64_t> &keys3,
          const std::vector<uint64_t> &keys4) {
  KeySwitchDigitsImpl &impl = KeySwitchDigitsImpl::GetInstance();
  impl.Init(primes, keys1, keys2, keys3, keys4, 1, 2, 3);
}

void KeySwitchDigits(const std::vector<uint8_t> &primes_index,
                     const std::vector<uint64_t> &digits,
                     std::vector<uint64_t> &output) {
  KeySwitchDigitsImpl &impl = KeySwitchDigitsImpl::GetInstance();
  sycl::event e;
  uint64_t num_primes = primes_index.size();
  uint64_t digit_size = num_primes * COEFF_COUNT;
  assert(output.size() == digit_size * 2);
  assert(digit_size * 4 == digits.size());
  sycl::ulong4 digits_offset{0, digit_size, digit_size * 2, digit_size * 3};
  impl.Enqueue(primes_index, digits, digits_offset, output, e);
}

void Wait() {
  KeySwitchDigitsImpl &impl = KeySwitchDigitsImpl::GetInstance();
  impl.ProcessLeftOutput();
}
}  // namespace KeySwitchDigits
}  // namespace bgv
}  // namespace helib
}  // namespace L2
