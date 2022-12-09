
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __AUTORUN_KERNEL_INSTANCES_HPP_
#define __AUTORUN_KERNEL_INSTANCES_HPP_
// INFO: This header will define the connections of all the kernels and top
// level for instantiating auto run kernels INFO: include header file that
// defines all the FIFOs for streaming connections INFO: of kernels

#include "pipes_keyswitch.hpp"

// INFO :: Define connections for INTT kernel first kernel in keyswitch

#include "dyadmult.hpp"
#include "intt1.hpp"
#include "intt1_forward.hpp"
#include "intt1_redu.hpp"
#include "intt2_core.hpp"
#include "intt2_forward.hpp"
#include "intt2_redu.hpp"
#include "ms.hpp"
#include "ntt1.hpp"
#include "ntt2.hpp"
#include "twiddle_generator.hpp"
#include "params.hpp"

#define DEF_NAME_CLASS(kernel, id) \
    template <int iid = id>        \
    class _##kernel_kernelNameClass;

template <int instanceID = 0>
class dispatch_twiddle_factors_kernelNameClass;
template <int instanceID = 0>
class dispatch_intt1_twiddle_factor_kernelNameClass;
template <int instanceID = 0>
class dispatch_ntt1_twiddle_factor_kernelNameClass;
template <int instanceID = 0>
class dispatch_intt2_twiddle_factor_kernelNameClass;
template <int instanceID = 0>
class dispatch_ntt2_twiddle_factor_kernelNameClass;

inline void initDispatchTwiddleFactors(sycl::queue& q,
                                       sycl::buffer<uint64_t>* buff_twiddles,
                                       unsigned coeff_count,
                                       unsigned num_moduli) {
    dispatch_twiddle_factors<dispatch_twiddle_factors_kernelNameClass<0>,
                             ch_intt1_decomp_size, ch_ntt2_decomp_size,
                             ch_twiddle_factor_rep, ch_intt1_twiddle_factor_rep,
                             ch_intt2_twiddle_factor_rep>(
        q, buff_twiddles, coeff_count, num_moduli);
}

inline void initTwiddleGenerator(sycl::queue& q) {
    dispatch_intt1_twiddle_factor<
        dispatch_intt1_twiddle_factor_kernelNameClass<0>,
        ch_intt1_twiddle_factor_rep, ch_intt1_twiddle_factor>(q);

    dispatch_ntt1_twiddle_factor<
        dispatch_ntt1_twiddle_factor_kernelNameClass<0>, ch_twiddle_factor_rep,
        ch_twiddle_factor>(q);

    dispatch_intt2_twiddle_factor<
        dispatch_intt2_twiddle_factor_kernelNameClass<0>,
        ch_intt2_twiddle_factor_rep, ch_intt2_twiddle_factor>(q);

    dispatch_ntt2_twiddle_factor<
        dispatch_ntt2_twiddle_factor_kernelNameClass<0>, ch_twiddle_factor_rep,
        ch_twiddle_factor>(q);
}

// intt1.cl
inline void initInnt1Kernels(sycl::queue& q) {
    _intt_1<NUM_CORES, 0, ch_intt_modulus, ch_intt_elements, ch_normalize,
            ch_intt1_twiddle_factor, ch_intt_elements_in,
            ch_intt_elements_out_inter, ch_intt_elements_out>(q);
}

// INFO: Recursive function to generate _intt_broadcast kernels
template <int iid, int coreNum>
class _intt_broadcast_kernelNameClass;

template <int tp_numCores>
void _intt_broadcast_generator(sycl::queue& q) {
    Unroller<0, tp_numCores>::Step([&](auto core) {
        _intt_broadcast<_intt_broadcast_kernelNameClass<0, core>,
                        ch_intt_elements_out, ch_intt_elements_out_rep, core>(
            q);
    });
}

inline void initINTTBroadcast(sycl::queue& q) {
    _intt_broadcast_generator<NUM_CORES>(q);
}

// INFO: _intt1_redu.cl
template <int instanceID = 0>
class intt1_redu_kernelNameClass;
inline void init_intt1_redu(sycl::queue& q) {
    intt1_redu<intt1_redu_kernelNameClass<0>, ch_intt_redu_params,
               ch_ntt_modulus, ch_intt_elements_out_rep, ch_ntt_elements_in>(q);
}

// INFO:: ntt1 instance generator ntt1.cl
inline void _ntt1_instance_generator(sycl::queue& q) {
    NTT_INS_generator<ch_ntt_modulus, ch_ntt_elements, ch_twiddle_factor,
                      NUM_CORES, MAX_RNS_MODULUS_SIZE>(q);
}

// INFOL ntt2_forward kernel generator ntt1.cl
class _ntt2_forward_only_kernelNameClass;
inline void ntt2_forward(sycl::queue& q) {
    ntt2_forward<_ntt2_forward_only_kernelNameClass, ch_ntt_elements,
                 ch_ntt_elements_out, NUM_CORES>(q);
}

class _ntt1_backward_only_kernelNameClass;
inline void ntt1_backward(sycl::queue& q) {
    ntt1_backward<_ntt1_backward_only_kernelNameClass, ch_ntt_elements_in,
                  ch_ntt_elements, NUM_CORES>(q);
}

// INFO:: ntt2.cl ntt2 instance generator
inline void _ntt2_instance_generator(sycl::queue& q) {
    NTT2_INS_generator<ch_ntt_modulus, ch_ntt_elements, ch_twiddle_factor,
                       ch_ntt_elements_in, ch_ntt_elements_out, NUM_CORES,
                       MAX_RNS_MODULUS_SIZE, MAX_RNS_MODULUS_SIZE + 1>(q);
}
template <unsigned int iid>
class broadcast_keys_kernelNameClass;

inline void initBroadCastKeys(sycl::queue& q,
                              sycl::buffer<uint256_t>& buff_k_switch_keys1,
                              sycl::buffer<uint256_t>& buff_k_switch_keys2,
                              sycl::buffer<uint256_t>& buff_k_switch_keys3,
                              int batch_size) {
    broadcast_keys<broadcast_keys_kernelNameClass<0>, ch_keyswitch_params,
                   ch_dyadmult_keys, NUM_CORES, MAX_RNS_MODULUS_SIZE>(
        q, buff_k_switch_keys1, buff_k_switch_keys2, buff_k_switch_keys3,
        batch_size);
}
template <unsigned int iid, unsigned int coreid, unsigned int ins_id>
class _dyadmult_kernelNameClass;
template <unsigned int iid, unsigned int coreid, unsigned int ins_id>
class _dyadmult_last_stage_kernelNameClass;
inline void _dyadmutl_instance_generator(sycl::queue& q) {
    Unroller<0, NUM_CORES>::Step([&](auto core) {
        Unroller<0, MAX_RNS_MODULUS_SIZE - 1>::Step([&](auto iid) {
            _dyadmult<_dyadmult_kernelNameClass<0, core, iid>, core, iid,
                      MAX_KEY_MODULUS_SIZE, MAX_KEY_COMPONENT_SIZE,
                      ch_dyadmult_params, ch_ntt_elements_out, ch_dyadmult_keys,
                      ch_intt_modulus, ch_t_poly_prod_iter, NUM_CORES,
                      MAX_RNS_MODULUS_SIZE>(q);
        });
    });

    Unroller<0, NUM_CORES>::Step([&](auto core) {
        Unroller<MAX_RNS_MODULUS_SIZE - 1, MAX_RNS_MODULUS_SIZE>::Step(
            [&](auto iid) {
                _dyadmult_last_stage<
                    _dyadmult_last_stage_kernelNameClass<0, core, iid>, core,
                    iid, MAX_KEY_MODULUS_SIZE, MAX_KEY_COMPONENT_SIZE,
                    ch_dyadmult_params, ch_ntt_elements_out, ch_dyadmult_keys,
                    ch_intt_modulus, ch_t_poly_prod_iter_last, NUM_CORES,
                    MAX_RNS_MODULUS_SIZE>(q);
            });
    });
}
template <unsigned int iid, unsigned int coreid, unsigned int ins_id>
class _ms_kernelNameClass;
void inline _ms_instance_generator(sycl::queue& q) {
    Unroller<0, NUM_CORES>::Step([&](auto coreNum) {
        Unroller<0, MAX_KEY_COMPONENT_SIZE>::Step([&](auto iid) {
            _ms<_ms_kernelNameClass<0, coreNum, iid>, coreNum, iid,
                MAX_KEY_MODULUS_SIZE, ch_ms_params, ch_t_poly_prod_iter,
                ch_ntt_elements_out, ch_result>(q);
        });
    });
}
template <unsigned int iid, unsigned int coreid, unsigned int ins_id>
class _intt_internal2_kernelNameClass;
template <unsigned int iid, unsigned int coreid, unsigned int ins_id>
class _intt_backward2_kernelNameClass;
template <unsigned int iid, unsigned int coreid, unsigned int ins_id>
class _intt_forward2_kernelNameClass;
template <unsigned int iid, unsigned int coreid, unsigned int ins_id>
class _intt_normalize2_kernelNameClass;

inline void _intt2_instance_generator(sycl::queue& q) {
    // INFO : Loop unroll limits from intt2.cl
    Unroller<0, NUM_CORES>::Step([&](auto coreNum) {
        Unroller<1, 3>::Step([&](auto iid) {
            using pipeNum1 =
                typename ch_intt_modulus::template PipeAt<coreNum, iid>;
            using pipeNum2 =
                typename ch_intt2_elements::template PipeAt<coreNum,
                                                            (iid - 1) * 2>;
            using pipeNum3 =
                typename ch_intt2_elements::template PipeAt<coreNum,
                                                            (iid - 1) * 2 + 1>;
            using pipeNum4 =
                typename ch_normalize::template PipeAt<coreNum, iid>;
            using pipeNum5 =
                typename ch_intt2_twiddle_factor::template PipeAt<coreNum,
                                                                  iid - 1>;
            using pipeNum6 =
                typename ch_intt_elements_in::template PipeAt<coreNum, iid>;
            using pipeNum7 =
                typename ch_intt_elements_out_inter::template PipeAt<coreNum,
                                                                     iid>;
            using pipeNum8 =
                typename ch_intt_elements_out::template PipeAt<coreNum, iid>;

            _intt_internal2<_intt_internal2_kernelNameClass<0, coreNum, iid>, 2,
                            pipeNum1, pipeNum2, pipeNum3, pipeNum4, pipeNum5>(
                q);

            _intt_backward2<_intt_backward2_kernelNameClass<0, coreNum, iid>,
                            pipeNum6, pipeNum2>(q);

            _intt_forward2<_intt_forward2_kernelNameClass<0, coreNum, iid>,
                           pipeNum7, pipeNum3>(q);

            _intt_normalize2<_intt_normalize2_kernelNameClass<0, coreNum, iid>,
                             pipeNum7, pipeNum8, pipeNum4>(q);
        });
    });
}
template <unsigned int iid, unsigned int coreid, unsigned int ins_id>
class _intt2_forward_kernelNameClass;
inline void _intt2_forward_instance_generator(sycl::queue& q) {
    // INFO : Loop unroll limits from intt2_forward.cl
    Unroller<0, NUM_CORES>::Step([&](auto coreNum) {
        Unroller<0, MAX_KEY_COMPONENT_SIZE>::Step([&](auto iid) {
            _intt2_forward<_intt2_forward_kernelNameClass<0, coreNum, iid>,
                           coreNum, iid, ch_t_poly_prod_iter_last,
                           ch_intt_elements_in>(q);
        });
    });
}

template <unsigned int iid, unsigned int coreid, unsigned int ins_id>
class _intt2_redu_kernelNameClass;

inline void _intt2_redu_instance_generator(sycl::queue& q) {
    // INFO : Loop unroll limits from intt2_redu.cl
    Unroller<0, NUM_CORES>::Step([&](auto coreNum) {
        Unroller<0, MAX_KEY_COMPONENT_SIZE>::Step([&](auto iid) {
            _intt2_redu<_intt2_redu_kernelNameClass<0, coreNum, iid>, coreNum,
                        iid, MAX_KEY_MODULUS_SIZE, ch_intt2_redu_params,
                        ch_intt_elements_out, ch_ntt_modulus,
                        ch_ntt_elements_in>(q);
        });
    });
}

#endif
