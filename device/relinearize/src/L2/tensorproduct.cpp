#include <L2/tensorproduct-impl.hpp>
namespace L2 {
namespace helib {
namespace bgv {
namespace TensorProduct {
void Init(const std::vector<uint64_t> &primes, uint32_t input_mem_channel,
          uint32_t output_mem_channel) {
  TensorProductImpl &impl = TensorProductImpl::GetInstance();
  impl.Init(primes, input_mem_channel, output_mem_channel);
}

void TensorProduct(const std::vector<uint8_t> &primes_index,
                   const std::vector<uint64_t> &inputs,
                   std::vector<uint64_t> &c0, std::vector<uint64_t> &c1,
                   std::vector<uint64_t> &c2) {
  TensorProductImpl &impl = TensorProductImpl::GetInstance();
  // no dependency
  sycl::event e;
  uint64_t num_primes = primes_index.size();
  uint64_t size = num_primes * COEFF_COUNT;
  sycl::ulong4 inputs_offset{0, size, size * 2, size * 3};
  impl.Enqueue(primes_index, inputs, inputs_offset, c0, c1, c2, e);
}

void Wait() {
  TensorProductImpl &impl = TensorProductImpl::GetInstance();
  impl.ProcessLeftOutput();
}
}  // namespace TensorProduct
}  // namespace bgv
}  // namespace helib
}  // namespace L2
