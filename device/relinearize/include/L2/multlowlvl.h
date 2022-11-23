#pragma once

namespace L2 {
namespace helib {
namespace bgv {
namespace MultLowLvl {
void init(std::vector<uint64_t> primes);

void MultLowLvl(uint64_t plainText, std::vector<uint64_t> &a,
                std::vector<uint8_t> &a_primes_index, std::vector<uint64_t> &b,
                std::vector<uint8_t> &b_primes_index, std::vector<uint64_t> &c0,
                std::vector<uint64_t> &c1, std::vector<uint64_t> &c2,
                std::vector<uint8_t> &output_primes_index);

void wait();
}  // namespace MultLowLvl
}  // namespace bgv
}  // namespace helib
}  // namespace L2
