#pragma once
#include <CL/sycl.hpp>

#define MAX_SCALE_PARAM_SET_LEN 1024
#define MAX_NUM_PRIMES 32
#define BUFF_DEPTH 4
#define SYNC_MODE 0

namespace L2 {
namespace helib {
namespace bgv {
namespace BringToSet {
class BringToSetImpl {
 public:
  static BringToSetImpl &GetInstance();

  BringToSetImpl();

  void init(std::vector<uint64_t> &_primes, uint32_t input_mem_channel = 1,
            uint32_t output_mem_channel = 2);

  void PreComputeScaleParamSet(
      std::vector<uint64_t> &pi, std::vector<uint64_t> &qj,
      std::vector<uint8_t> &qj_prime_index, uint64_t plainText,
      std::vector<uint64_t> &special_primes, size_t &P, size_t &Q, size_t &I,
      std::vector<sycl::ulong2> &scale_param_set, bool added_primes_at_end,
      bool add_special_primes);

  void PreCompute(std::vector<uint8_t> &pi_primes_index,
                  std::vector<uint8_t> &qj_prime_index, uint64_t plainText,
                  std::vector<uint8_t> &pi_reorder_primes_index,
                  std::vector<sycl::ulong2> &scale_param_set, size_t &P,
                  size_t &Q, size_t &I);

  void LaunchLoadKernel(sycl::buffer<uint64_t> &buf_input,
                        sycl::buffer<uint8_t> &buf_primes_index,
                        std::vector<uint64_t> &input,
                        std::vector<uint8_t> &primes_index);

  void load(std::vector<uint64_t> &data, std::vector<uint8_t> &primes_index,
            sycl::buffer<uint64_t> &buf_data,
            sycl::buffer<uint8_t> &buf_primes_index,
            sycl::buffer<sycl::ulong2> &buf_precomputed_params,
            std::vector<uint8_t> &output_primes_index, uint64_t plainText);

  /**
   * @brief Get the Last Output Buffer object
   *
   * @return sycl::buffer<uint64_t>&
   */
  sycl::buffer<uint64_t> &GetLastOutputBuffer();

  /**
   * @brief Get the Last Output Buffer Index
   *
   * @return int
   */
  int GetLastOutputBufferIndex();

  /**
   * @brief perform and read the output to c
   *
   * @param plainText
   * @param a
   * @param a_primes_index
   * @param b
   * @param b_primes_index
   * @param c
   * @param output_primes_index
   */
  void perform(uint64_t plainText, std::vector<uint64_t> &a,
               std::vector<uint8_t> &a_primes_index, std::vector<uint64_t> &b,
               std::vector<uint8_t> &b_primes_index, std::vector<uint64_t> &c,
               std::vector<uint8_t> &output_primes_index);

  /**
   * @brief Perform without reading the output buffer
   *
   * @param plainText
   * @param a
   * @param a_primes_index
   * @param b
   * @param b_primes_index
   * @param output_primes_index
   * @return sycl::event
   */
  sycl::event perform(uint64_t plainText, std::vector<uint64_t> &a,
                      std::vector<uint8_t> &a_primes_index,
                      std::vector<uint64_t> &b,
                      std::vector<uint8_t> &b_primes_index,
                      std::vector<uint8_t> &output_primes_index);

  /**
   * @brief wait the all enqueued opertions done
   *
   */
  void wait();

 private:
  std::vector<uint64_t> primes;
  sycl::queue *q_scale;
  sycl::queue *q_load;
  sycl::queue *q_store;
  sycl::queue *q_load_data;
  sycl::queue *q_store_data;
  bool inited;
  sycl::buffer<sycl::ulong2> *buf_scale_param_set;
  // the actual tranfered buffer
  sycl::buffer<sycl::ulong2> *sub_buf_scale_param_set;

  sycl::buffer<uint64_t> *buf_output_[BUFF_DEPTH];
  sycl::buffer<sycl::ulong2> *buf_a_precomputed_params_[BUFF_DEPTH];
  sycl::buffer<sycl::ulong2> *buf_b_precomputed_params_[BUFF_DEPTH];
  sycl::buffer<uint64_t> *buf_a_[BUFF_DEPTH];
  sycl::buffer<uint64_t> *buf_b_[BUFF_DEPTH];
  sycl::buffer<uint8_t> *buf_a_primes_index_[BUFF_DEPTH];
  sycl::buffer<uint8_t> *buf_b_primes_index_[BUFF_DEPTH];

  sycl::event events[BUFF_DEPTH];

  size_t buf_index;
};
}  // namespace BringToSet
}  // namespace bgv
}  // namespace helib
}  // namespace L2
