
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <CL/sycl.hpp>
#include "keyswitch/autorun_kernel_instances.hpp"
#include "keyswitch/load.hpp"
#include "keyswitch/store.hpp"

class keyswitch_load_kernel;
class keyswitch_store_kernel;

extern "C" {

sycl::event load(sycl::queue& q, sycl::event* inDepsEv,
                 sycl::buffer<uint64_t>& t_target_iter_ptr, moduli_t moduli,
                 uint64_t coeff_count, uint64_t decomp_modulus_size,
                 uint64_t num_batch, invn_t inv_n, unsigned rmem) {
    auto event = load<keyswitch_load_kernel>(
        q, inDepsEv, t_target_iter_ptr, moduli, coeff_count,
        decomp_modulus_size, num_batch, inv_n, rmem);
    return event;
}

sycl::event store(sycl::queue& q, sycl::event* inDepsEv,
                  sycl::buffer<sycl::ulong2>& dp_results, uint64_t num_batch,
                  uint64_t coeff_count, uint64_t decomp_modulus_size,
                  moduli_t moduli, unsigned rmem, unsigned wmem) {
    auto event = store<keyswitch_store_kernel>(
        q, inDepsEv, dp_results, num_batch, coeff_count, decomp_modulus_size,
        moduli, rmem, wmem);
    return event;
}

void launchConfigurableKernels(sycl::queue& q,
                               sycl::buffer<uint64_t>* buff_twiddles,
                               unsigned coeff_count,
                               bool load_twiddle_factors) {
    initDispatchTwiddleFactors(q, buff_twiddles, coeff_count,
                               load_twiddle_factors);
}
void launchStoreSwitchKeys(sycl::queue& q,
                           sycl::buffer<uint256_t>& buff_k_switch_keys1,
                           sycl::buffer<uint256_t>& buff_k_switch_keys2,
                           sycl::buffer<uint256_t>& buff_k_switch_keys3,
                           int batch_size) {
    initBroadCastKeys(q, buff_k_switch_keys1, buff_k_switch_keys2,
                      buff_k_switch_keys3, batch_size);
}

void launchAllAutoRunKernels(sycl::queue& q) {
    initTwiddleGenerator(q);
    initInnt1Kernels(q);
    initINTTBroadcast(q);
    init_intt1_redu(q);
    _ntt1_instance_generator(q);
    ntt2_forward(q);
    ntt1_backward(q);
    _dyadmutl_instance_generator(q);
    _ntt2_instance_generator(q);
    _ms_instance_generator(q);
    _intt2_instance_generator(q);
    _intt2_forward_instance_generator(q);
    _intt2_redu_instance_generator(q);
}

}  // end of extern "C"
