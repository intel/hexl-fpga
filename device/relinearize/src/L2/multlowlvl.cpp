#include <L2/bringtoset.h>
#include <L2/tensorproduct.h>
#include <L2/bringtoset-impl.h>
#include <L2/tensorproduct-impl.h>

namespace L2 {
namespace helib {
namespace bgv {
namespace MultLowLvl {

using namespace L2::helib::bgv::BringToSet;
using namespace L2::helib::bgv::TensorProduct;

void init(std::vector<uint64_t> primes) {
  L2::helib::bgv::BringToSet::init(primes, 1, 2);
  L2::helib::bgv::TensorProduct::init(primes, 2, 3);
}

void MultLowLvl(uint64_t plainText, std::vector<uint64_t> &a,
                std::vector<uint8_t> &a_primes_index, std::vector<uint64_t> &b,
                std::vector<uint8_t> &b_primes_index, std::vector<uint64_t> &c0,
                std::vector<uint64_t> &c1, std::vector<uint64_t> &c2,
                std::vector<uint8_t> &output_primes_index) {
  auto &impl = BringToSetImpl::GetInstance();
  sycl::event e = impl.perform(plainText, a, a_primes_index, b, b_primes_index,
                               output_primes_index);
  auto &buf_output = impl.GetLastOutputBuffer();

  unsigned sub_buf_size = c0.size();

  auto &tensor_product_impl = TensorProductImpl::GetInstance();
  tensor_product_impl.perform(
      output_primes_index, buf_output, buf_output, buf_output, buf_output, 0,
      sub_buf_size, sub_buf_size * 2, sub_buf_size * 3, c0, c1, c2, e);
}

void wait() {
  L2::helib::bgv::BringToSet::wait();
  L2::helib::bgv::TensorProduct::wait();
}
}  // namespace MultLowLvl
}  // namespace bgv
}  // namespace helib
}  // namespace L2
