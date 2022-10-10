// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <sycl/ext/intel/fpga_extensions.hpp>

#include "dynamic_loading_kernel_IF.hpp"

static MultLowLvlDynaimcIF* g_multlowlvl = new MultLowLvlDynaimcIF("multlowlvl.so");

/**
 * runtime functions, comes from src/L2/multLowLvl.cpp file.
 */

void LaunchINTT1(std::vector<uint64_t> &primes) {
  // launch iNTT
  launch_intt(g_multlowlvl->GetINTT1(), primes, COEFF_COUNT);
}

void LaunchINTT2(std::vector<uint64_t> &primes) {
  // launch iNTT
  launch_intt(g_multlowlvl->GetINTT1(), primes, COEFF_COUNT);
}

struct Context {
  std::vector<uint64_t> primes;
  sycl::queue *q_scale1;
  sycl::queue *q_scale2;
  sycl::queue *q_load_data_copy;
  sycl::queue *q_load1;
  sycl::queue *q_load2;
  sycl::queue *q_tensor_product;
  sycl::queue *q_tensor_product_store0;
  sycl::queue *q_tensor_product_store12;
  sycl::queue *q_tensor_product_memcpy;
};

struct Context &GetContext() {
  static struct Context context;
  return context;
};

void Init(std::vector<uint64_t> &primes) {
  // Create queue, get platform and device
#if defined(FPGA_EMULATOR)
  sycl::ext::intel::fpga_emulator_selector device_selector;
#else
  sycl::ext::intel::fpga_selector device_selector;
#endif

  struct Context &ctxt = GetContext();

  ctxt.primes = primes;

  auto prop_list = property_list{property::queue::enable_profiling()};
  ctxt.q_scale1 = new sycl::queue(device_selector, prop_list);
  ctxt.q_scale2 = new sycl::queue(device_selector, prop_list);
  ctxt.q_load_data_copy = new sycl::queue(device_selector, prop_list);
  ctxt.q_load1 = new sycl::queue(device_selector, prop_list);
  ctxt.q_load2 = new sycl::queue(device_selector, prop_list);
  ctxt.q_tensor_product = new sycl::queue(device_selector, prop_list);
  ctxt.q_tensor_product_store0 = new sycl::queue(device_selector, prop_list);
  ctxt.q_tensor_product_store12 = new sycl::queue(device_selector, prop_list);
  ctxt.q_tensor_product_memcpy = new sycl::queue(device_selector, prop_list);

  LaunchINTT1(primes);
  LaunchINTT2(primes);

  // launch NTT
  // Todo: expose GetTensorProductNTT1 like GetINTT1/GetINTT2
  launch_ntt(L1::helib::bgv::GetTensorProductNTT1(), primes, COEFF_COUNT);
  launch_ntt(L1::helib::bgv::GetTensorProductNTT2(), primes, COEFF_COUNT);
}


template <int engine>
void LaunchBringToSet(std::vector<uint8_t> &pi_primes_index,
                      std::vector<uint8_t> &qj_prime_index, uint64_t plainText,
                      std::vector<uint8_t> &pi_reorder_primes_index) {
  struct Context &ctxt = GetContext();

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
    pi.push_back(ctxt.primes[pi_primes_index[i]]);
  }
  for (size_t i = 0; i < pi_primes_index.size() - num_dropped_primes; i++) {
    pi_reorder_primes_index.push_back(pi_primes_index[i]);
    pi.push_back(ctxt.primes[pi_primes_index[i]]);
  }

  // fill qj
  for (size_t i = 0; i < qj_prime_index.size(); i++) {
    qj.push_back(ctxt.primes[qj_prime_index[i]]);
  }

  assert(pi_reorder_primes_index.size() == pi_primes_index.size());

  size_t P, Q, I;
  std::vector<sycl::ulong2> scale_param_set;
  std::vector<uint64_t> empty_vec;
  PreComputeScaleParamSet<false, false>(pi, qj, qj_prime_index, plainText,
                                        empty_vec, P, Q, I, scale_param_set);

  auto scale_param_set_buf = new buffer<sycl::ulong2>(scale_param_set.size());

  assert(engine == 1 || engine == 2);
  if (engine == 1) {
    Timer timer("g_multlowlvl->BringToSet");
    queue_copy(*ctxt.q_scale1, scale_param_set, scale_param_set_buf);
    g_multlowlvl->BringToSet(*ctxt.q_scale1, COEFF_COUNT,
                               *scale_param_set_buf, P, Q, I, plainText);
  } else {
    Timer timer("g_multlowlvl->BringToSet2");
    queue_copy(*ctxt.q_scale2, scale_param_set, scale_param_set_buf);
    g_multlowlvl->BringToSet2(*ctxt.q_scale2, COEFF_COUNT,
                                *scale_param_set_buf, P, Q, I, plainText);
  }
}

void LaunchBringToSet1(std::vector<uint8_t> &pi_primes_index,
                       std::vector<uint8_t> &qj_prime_index, uint64_t plainText,
                       std::vector<uint8_t> &pi_reorder_primes_index) {
  Timer timer("LaunchBringToSet1");
  LaunchBringToSet<1>(pi_primes_index, qj_prime_index, plainText,
                      pi_reorder_primes_index);
}

void LaunchBringToSet2(std::vector<uint8_t> &pi_primes_index,
                       std::vector<uint8_t> &qj_prime_index, uint64_t plainText,
                       std::vector<uint8_t> &pi_reorder_primes_index) {
  Timer timer("LaunchBringToSet2");
  LaunchBringToSet<2>(pi_primes_index, qj_prime_index, plainText,
                      pi_reorder_primes_index);
}

void TensorProduct(std::vector<uint64_t> &primes,
                   std::vector<uint8_t> &primes_index) {
  Timer timer("TensorProduct");
  std::vector<ulong4> primes_mulmod;
  // primes have all primes including the small primes and special primes
  for (auto prime_index : primes_index) {
    auto prime = primes[prime_index];
    primes_mulmod.push_back(
        {prime, precompute_modulus_r(prime), precompute_modulus_k(prime), 0});
  }

  struct Context &ctxt = GetContext();
  sycl::queue &q = *ctxt.q_tensor_product;

  auto primes_mulmod_buf = new buffer<ulong4>(primes_mulmod.size());
  queue_copy(q, primes_mulmod, primes_mulmod_buf);

  // launch TensorProduct
  g_multlowlvl->TensorProduct(q, *primes_mulmod_buf);
}

template <int engine>
void Load(std::vector<uint64_t> &input, std::vector<uint8_t> &primes_index) {
  auto input_buff = new sycl::buffer<uint64_t>(input.size());
  auto primes_index_buf = new sycl::buffer<uint8_t>(primes_index.size());

  // do not static
  Context &ctxt = GetContext();

  queue_copy_async(*ctxt.q_load_data_copy, input, input_buff);
  auto copy_event =
      queue_copy_async(*ctxt.q_load_data_copy, primes_index, primes_index_buf);

  assert(engine == 1 || engine == 2);
  if (engine == 1) {
    g_multlowlvl->BringToSetLoad(*ctxt.q_load1, copy_event, *input_buff,
                                   *primes_index_buf);
  } else {
    g_multlowlvl->BringToSetLoad2(*ctxt.q_load2, copy_event, *input_buff,
                                    *primes_index_buf);
  }
}


void Load1(std::vector<uint64_t> &input, std::vector<uint8_t> &primes_index) {
  Timer timer("Load1");
  Load<1>(input, primes_index);
}

void Load2(std::vector<uint64_t> &input, std::vector<uint8_t> &primes_index) {
  Timer timer("Load2");
  Load<2>(input, primes_index);
}


void Store(std::vector<uint64_t> &output1, std::vector<uint64_t> &output2,
           std::vector<uint64_t> &output3, size_t BATCH = 1) {
  Timer timer("Store");
  struct Context &ctxt = GetContext();
  sycl::queue &q_tensor_product_store0 = *ctxt.q_tensor_product_store0;
  sycl::queue &q_tensor_product_store12 = *ctxt.q_tensor_product_store12;
  sycl::queue &q_tensor_product_memcpy = *ctxt.q_tensor_product_memcpy;

  Timer timer1("Buffer");
  sycl::buffer<uint64_t> output1_buf(output1.size());
  timer1.stop();
  auto kernel_store0_event =
      g_multlowlvl->TensorProductStore0(q_tensor_product_store0, output1_buf);

  auto copy0_event = q_tensor_product_memcpy.submit([&](sycl::handler &h) {
    // copy
    h.depends_on(kernel_store0_event);
    h.copy(output1_buf.template get_access<sycl::access::mode::read>(h),
           output1.data());
  });
  assert(output2.size() == output3.size());

  if (BATCH == 0) BATCH = output2.size() / COEFF_COUNT;
  size_t batch_size = COEFF_COUNT * BATCH;

  sycl::buffer<uint64_t> output2_buf[] = {sycl::buffer<uint64_t>(batch_size),
                                          sycl::buffer<uint64_t>(batch_size)};
  sycl::buffer<uint64_t> output3_buf[] = {sycl::buffer<uint64_t>(batch_size),
                                          sycl::buffer<uint64_t>(batch_size)};

  sycl::event kernel_event[2];

  Timer timer_launch_kernels("LaunchKernels");
  kernel_event[0] = g_multlowlvl->TensorProductStore12(
      q_tensor_product_store12, output2_buf[0], output3_buf[0]);

  size_t iters = (output2.size() - 1) / batch_size + 1;
  for (size_t i = 1; i < iters + 1; i++) {
    size_t j = i % 2;
    size_t last = (i - 1) % 2;
    if (i < iters) {
      kernel_event[j] = g_multlowlvl->TensorProductStore12(
          q_tensor_product_store12, output2_buf[j], output3_buf[j]);
    }

    q_tensor_product_memcpy.submit([&](sycl::handler &h) {
      // copy
      h.depends_on(kernel_event[last]);
      h.copy(output2_buf[last].template get_access<sycl::access::mode::read>(h),
             output2.data() + (i - 1) * batch_size);
    });
    q_tensor_product_memcpy.submit([&](sycl::handler &h) {
      // copy
      h.copy(output3_buf[last].template get_access<sycl::access::mode::read>(h),
             output3.data() + (i - 1) * batch_size);
    });
    q_tensor_product_memcpy.wait();
  }

  timer_launch_kernels.stop();

  q_tensor_product_memcpy.wait();

  // kernel_store12_event.wait();
  // PrintEventTime(kernel_store12_event, "Store Kernel");
}


// c0 = a0 * b0, c1 = a0 * b1 + a1 * b0, c2 = a1 * b1
void MultLowLvl(std::vector<uint64_t> &a0, std::vector<uint64_t> &a1,
                std::vector<uint8_t> &a_primes_index, std::vector<uint64_t> &b0,
                std::vector<uint64_t> &b1, std::vector<uint8_t> &b_primes_index,
                uint64_t plainText, std::vector<uint64_t> &c0,
                std::vector<uint64_t> &c1, std::vector<uint64_t> &c2,
                std::vector<uint8_t> &output_primes_index) {
  struct Context &ctxt = GetContext();
  std::vector<uint8_t> pi_reorder_primes_index1;
  std::vector<uint8_t> pi_reorder_primes_index2;
  
  LaunchBringToSet1(a_primes_index, output_primes_index,
                                    plainText, pi_reorder_primes_index1);
  LaunchBringToSet2(b_primes_index, output_primes_index,
                                    plainText, pi_reorder_primes_index2);

  // load part 1
  Load1(a0, pi_reorder_primes_index1);
  Load2(b0, pi_reorder_primes_index2);

  TensorProduct(ctxt.primes, output_primes_index);

  // load part 2
  Load1(a1, pi_reorder_primes_index1);
  Load2(b1, pi_reorder_primes_index2);

  // launch store
  Store(c0, c1, c2, 0);
}



