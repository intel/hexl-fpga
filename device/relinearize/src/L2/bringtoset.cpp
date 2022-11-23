#include <algorithm>
#include <CL/sycl.hpp>
#include <NTL/ZZ.h>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <L1/bringtoset.h>
#include <L2/utils.h>
#include <L2/bringtoset-impl.h>

using namespace sycl;

namespace L2 {
namespace helib {
namespace bgv {
namespace BringToSet {
BringToSetImpl &BringToSetImpl::GetInstance() {
  static BringToSetImpl ins;
  return ins;
};

BringToSetImpl::BringToSetImpl() { inited = false; }

void BringToSetImpl::init(std::vector<uint64_t> &_primes,
                          uint32_t input_mem_channel,
                          uint32_t output_mem_channel) {
  if (inited) return;
    // Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif

  primes = _primes;

  auto prop_list = property_list{property::queue::enable_profiling()};
  q_scale = new sycl::queue(device_selector, prop_list);
  q_load = new sycl::queue(device_selector, prop_list);
  q_store = new sycl::queue(device_selector, prop_list);
  q_load_data = new sycl::queue(device_selector, prop_list);
  // the queue to read data back from fpga
  q_store_data = new sycl::queue(device_selector, prop_list);

  const property_list input_mem_channel_prop = {
      sycl::property::buffer::mem_channel{input_mem_channel}};
  const property_list output_mem_channel_prop = {
      sycl::property::buffer::mem_channel{output_mem_channel}};

  for (size_t i = 0; i < BUFF_DEPTH; i++) {
    buf_output_[i] = new sycl::buffer<uint64_t>(
        COEFF_COUNT * 4 * MAX_NUM_PRIMES, output_mem_channel_prop);
    buf_output_[i]->set_write_back(false);
    buf_a_precomputed_params_[i] = new sycl::buffer<ulong2>(
        MAX_SCALE_PARAM_SET_LEN, input_mem_channel_prop);
    buf_b_precomputed_params_[i] = new sycl::buffer<ulong2>(
        MAX_SCALE_PARAM_SET_LEN,
        {sycl::property::buffer::mem_channel{input_mem_channel}});
    // the size of buffer should be doubled as each ciphertext has two parts
    buf_a_[i] = new sycl::buffer<uint64_t>(MAX_NUM_PRIMES * 2 * COEFF_COUNT,
                                           input_mem_channel_prop);
    buf_b_[i] = new sycl::buffer<uint64_t>(MAX_NUM_PRIMES * 2 * COEFF_COUNT,
                                           input_mem_channel_prop);
    buf_a_primes_index_[i] =
        new sycl::buffer<uint8_t>(MAX_NUM_PRIMES * 2, input_mem_channel_prop);
    buf_b_primes_index_[i] =
        new sycl::buffer<uint8_t>(MAX_NUM_PRIMES * 2, input_mem_channel_prop);
  }

  buf_index = 0;
  // launch NTT and intt
  L1::BringToSet::ntt(primes, COEFF_COUNT, 0xff);
  L1::BringToSet::intt(primes, COEFF_COUNT, 0b111);
}

void BringToSetImpl::PreComputeScaleParamSet(
    std::vector<uint64_t> &pi, std::vector<uint64_t> &qj,
    std::vector<uint8_t> &qj_prime_index, uint64_t plainText,
    std::vector<uint64_t> &special_primes, size_t &P, size_t &Q, size_t &I,
    std::vector<sycl::ulong2> &scale_param_set, bool added_primes_at_end,
    bool add_special_primes) {
  Timer timer("PreComputeScaleParamSet");
  std::vector<sycl::ulong2> pt;
  std::vector<sycl::ulong2> P_qj;
  std::vector<sycl::ulong2> pstar_inv;
  std::vector<sycl::ulong2> pstar_qj;

  // num of target primes that may add primes
  Q = qj.size();
  // num of drop primes
  P = 0;
  uint64_t t = plainText;

  // compute the num of drop primes
  for (auto prime : pi) {
    P += (std::find(qj.begin(), qj.end(), prime) == qj.end());
  }

  NTL::ZZ diffProd(1);
  for (int i = 0; i < P; i++) {
    diffProd *= pi[i];
  }

  // I is the added primes
  I = Q - (pi.size() - P);
  NTL::ZZ IProd(1);

  // configure the position of added primes
  if (added_primes_at_end) {
    for (int i = Q - I; i < Q; i++) {
      IProd *= qj[i];
    }
  } else {
    for (int i = 0; i < I; i++) {
      IProd *= qj[i];
    }
  }

  NTL::ZZ prodInv(NTL::InvMod(rem(diffProd, t), t));
  for (int i = 0; i < P; i++) {
    ulong pt_i = NTL::rem(prodInv * diffProd / pi[i], t);
    pt.push_back(mulmod_y_ext(pt_i, t));
    ulong p_star_inv_i = NTL::InvMod(NTL::rem(diffProd / pi[i], pi[i]), pi[i]);
    p_star_inv_i = NTL::rem(IProd * p_star_inv_i, pi[i]);
    pstar_inv.push_back(mulmod_y_ext(p_star_inv_i, pi[i]));
  }

  for (int i = 0; i < Q; i++) {
    ulong P_inv_qj_i = NTL::InvMod(NTL::rem(diffProd, qj[i]), qj[i]);

    for (int j = 0; j < P; j++) {
      pstar_qj.push_back(
          mulmod_y_ext(NTL::rem(diffProd / pi[j] * P_inv_qj_i, qj[i]), qj[i]));
    }
    P_inv_qj_i = NTL::rem(IProd * P_inv_qj_i, qj[i]);
    P_qj.push_back(mulmod_y_ext(P_inv_qj_i, qj[i]));
  }

  //  Prod of special primes mod qj
  if (add_special_primes) {
    NTL::ZZ prod_special_primes(1);
    for (int i = 0; i < special_primes.size(); i++) {
      prod_special_primes *= special_primes[i];
    }

    for (int i = 0; i < Q; i++) {
      P_qj.push_back(mulmod_y_ext(NTL::rem(prod_special_primes, qj[i]), qj[i]));
    }
  }

  // packing
  // pi - P
  for (size_t i = 0; i < P; i++) {
    double tmp = 1;
    tmp /= pi[i];
    scale_param_set.push_back({pi[i], *(ulong *)&tmp});
  }

  // qj - Q
  for (size_t i = 0; i < Q; i++) {
    scale_param_set.push_back(
        {qj[i], qj_prime_index.size() > 0 ? qj_prime_index[i] : 0});
  }

  // pt - P
  for (size_t i = 0; i < P; i++) {
    scale_param_set.push_back(pt[i]);
  }

  // pstar_inv - P
  for (size_t i = 0; i < P; i++) {
    scale_param_set.push_back(pstar_inv[i]);
  }

  // P_qj - Q
  for (size_t i = 0; i < Q; i++) {
    scale_param_set.push_back(P_qj[i]);
  }

  // R_qj - Q if applicable
  if (add_special_primes) {
    for (size_t i = 0; i < Q; i++) {
      scale_param_set.push_back(P_qj[Q + i]);
    }
  }

  // pstar_qj - Q*P
  for (size_t i = 0; i < Q; i++) {
    for (size_t j = 0; j < P; j++) {
      scale_param_set.push_back(pstar_qj[i * P + j]);
    }
  }
}

void BringToSetImpl::PreCompute(std::vector<uint8_t> &pi_primes_index,
                                std::vector<uint8_t> &qj_prime_index,
                                uint64_t plainText,
                                std::vector<uint8_t> &pi_reorder_primes_index,
                                std::vector<sycl::ulong2> &scale_param_set,
                                size_t &P, size_t &Q, size_t &I) {
  // the source primes
  std::vector<uint64_t> pi;
  // the target primes
  std::vector<uint64_t> qj;
  // compute the num of dropped primes
  int num_dropped_primes = 0;
  for (auto prime_index : pi_primes_index) {
    num_dropped_primes +=
        (std::find(qj_prime_index.begin(), qj_prime_index.end(), prime_index) ==
         qj_prime_index.end());
  }

  // reoder pi, put the dropped primes to the beginning
  for (size_t i = pi_primes_index.size() - num_dropped_primes;
       i < pi_primes_index.size(); i++) {
    pi_reorder_primes_index.push_back(pi_primes_index[i]);
    pi.push_back(primes[pi_primes_index[i]]);
  }
  for (size_t i = 0; i < pi_primes_index.size() - num_dropped_primes; i++) {
    pi_reorder_primes_index.push_back(pi_primes_index[i]);
    pi.push_back(primes[pi_primes_index[i]]);
  }

  // fill qj
  for (size_t i = 0; i < qj_prime_index.size(); i++) {
    qj.push_back(primes[qj_prime_index[i]]);
  }

  assert(pi_reorder_primes_index.size() == pi_primes_index.size());

  std::vector<uint64_t> empty_vec;
  PreComputeScaleParamSet(pi, qj, qj_prime_index, plainText, empty_vec, P, Q, I,
                          scale_param_set, false, false);
}

void BringToSetImpl::LaunchLoadKernel(sycl::buffer<uint64_t> &buf_input,
                                      sycl::buffer<uint8_t> &buf_primes_index,
                                      std::vector<uint64_t> &input,
                                      std::vector<uint8_t> &primes_index) {
  assert(primes_index.size() <= buf_primes_index.size());
  Timer timer("load", input.size() * sizeof(uint64_t));

  queue_copy(*q_load_data, input, buf_input);
  timer.stop();
  queue_copy(*q_load_data, primes_index, buf_primes_index);

  L1::BringToSet::load(*q_load, buf_input, buf_primes_index,
                       primes_index.size());
}

void BringToSetImpl::load(std::vector<uint64_t> &data,
                          std::vector<uint8_t> &primes_index,
                          sycl::buffer<uint64_t> &buf_data,
                          sycl::buffer<uint8_t> &buf_primes_index,
                          sycl::buffer<ulong2> &buf_precomputed_params,
                          std::vector<uint8_t> &output_primes_index,
                          uint64_t plainText) {
  std::vector<sycl::ulong2> scale_param_set;
  std::vector<uint8_t> pi_reorder_primes_index;
  size_t P, Q, I;

  PreCompute(primes_index, output_primes_index, plainText,
             pi_reorder_primes_index, scale_param_set, P, Q, I);

  // double primes index
  auto len = pi_reorder_primes_index.size();
  for (size_t i = 0; i < len; i++) {
    pi_reorder_primes_index.push_back(pi_reorder_primes_index[i]);
  }

  // launch load
  LaunchLoadKernel(buf_data, buf_primes_index, data, pi_reorder_primes_index);

  queue_copy(*q_load_data, scale_param_set, buf_precomputed_params);
  L1::BringToSet::kernel(*q_scale, COEFF_COUNT, buf_precomputed_params, P, Q, I,
                         plainText);
}

sycl::buffer<uint64_t> &BringToSetImpl::GetLastOutputBuffer() {
  return *buf_output_[(BUFF_DEPTH + buf_index - 1) % BUFF_DEPTH];
}

int BringToSetImpl::GetLastOutputBufferIndex() {
  return (BUFF_DEPTH + buf_index - 1) % BUFF_DEPTH;
}

void BringToSetImpl::perform(uint64_t plainText, std::vector<uint64_t> &a,
                             std::vector<uint8_t> &a_primes_index,
                             std::vector<uint64_t> &b,
                             std::vector<uint8_t> &b_primes_index,
                             std::vector<uint64_t> &c,
                             std::vector<uint8_t> &output_primes_index) {
  Timer timer("BringToSet::perform (load, kernel launching and store data)");
  auto e = perform(plainText, a, a_primes_index, b, b_primes_index,
                   output_primes_index);

  auto &buf_output = GetLastOutputBuffer();

  Timer timer_store("BringToSet::store", c.size() * sizeof(uint64_t));

  events[GetLastOutputBufferIndex()] =
      q_store_data->submit([&](sycl::handler &h) {
        h.depends_on(e);
        h.copy(buf_output.template get_access<sycl::access::mode::read>(
                   h, sycl::range<1>(c.size())),
               c.data());
      });

#if SYNC_MODE
  q_store_data->wait();
  timer_store.stop();
#endif
}

sycl::event BringToSetImpl::perform(uint64_t plainText,
                                    std::vector<uint64_t> &a,
                                    std::vector<uint8_t> &a_primes_index,
                                    std::vector<uint64_t> &b,
                                    std::vector<uint8_t> &b_primes_index,
                                    std::vector<uint8_t> &output_primes_index) {
  Timer timer(
      "BringToSet::perform (load and kernel launching - without store data)");
  // make sure this buffer is free
  events[buf_index].wait();

  auto &buf_output = *buf_output_[buf_index];
  auto &buf_a_precomputed_params = *buf_a_precomputed_params_[buf_index];
  auto &buf_b_precomputed_params = *buf_b_precomputed_params_[buf_index];
  auto &buf_a = *buf_a_[buf_index];
  auto &buf_b = *buf_b_[buf_index];
  auto &buf_a_primes_index = *buf_a_primes_index_[buf_index];
  auto &buf_b_primes_index = *buf_b_primes_index_[buf_index];

  Timer t_load("BringToSet::load");
  load(a, a_primes_index, buf_a, buf_a_primes_index, buf_a_precomputed_params,
       output_primes_index, plainText);
  load(b, b_primes_index, buf_b, buf_b_primes_index, buf_b_precomputed_params,
       output_primes_index, plainText);
  t_load.stop();

#if SYNC_MODE
  Timer t_store("BringToSet::kernel");
#endif
  // launch the store kernel
  unsigned output_size = output_primes_index.size() * 4 * COEFF_COUNT;
  auto e = L1::BringToSet::store(*q_store, buf_output, output_size);

#if SYNC_MODE
  e.wait();
  t_store.stop();
  PrintEventTime(e, "L1::BringToSet");
#endif

  buf_index = (buf_index + 1) % BUFF_DEPTH;
  return e;
}

void BringToSetImpl::wait() { q_store_data->wait(); }

void BringToSet(uint64_t plainText, std::vector<uint64_t> &a,
                std::vector<uint8_t> &a_primes_index, std::vector<uint64_t> &b,
                std::vector<uint8_t> &b_primes_index, std::vector<uint64_t> &c,
                std::vector<uint8_t> &output_primes_index) {
  Timer timer("BringToSet");
  auto &impl = BringToSetImpl::GetInstance();
  impl.perform(plainText, a, a_primes_index, b, b_primes_index, c,
               output_primes_index);
}

void init(std::vector<uint64_t> primes, uint32_t input_mem_channel,
          uint32_t output_mem_channel) {
  auto &impl = BringToSetImpl::GetInstance();
  impl.init(primes, input_mem_channel, output_mem_channel);
}

void wait() {
  auto &impl = BringToSetImpl::GetInstance();
  impl.wait();
}
}  // namespace BringToSet
}  // namespace bgv
}  // namespace helib
}  // namespace L2
