// *****************************************************************************
// INTEL CONFIDENTIAL
// Copyright 2021 Intel Corporation
//
// This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may
// not use, modify, copy, publish, distribute, disclose or transmit this
// software or the related documents without Intel's prior written permission.
// *****************************************************************************

#include <seal/seal.h>
#include <cmath>
#include <cassert>

#include "seal_ckks_context.h"
#include "seal_ckks_executor.h"
#include "seal_omp_utils.h"
#include "blobs.h"
#include "fpga_seal_ckks_executor.h"
#include "hexl-fpga.h"

using namespace intel::he::heseal;

FPGA_SealCKKSExecutor::FPGA_SealCKKSExecutor(
    std::shared_ptr<ConfigurationBlob>& cblob)
    : SealCKKSExecutor(cblob->context()),
      cblob_(cblob),
      context_(cblob->context()),
      encoder_(cblob->context()->encoder()),
      encryptor_(cblob->context()->encryptor()),
      evaluator_(cblob->context()->evaluator()),
      galois_keys_(cblob->context()->galois_keys()),
      relin_keys_(cblob->context()->relin_keys()) {}

std::vector<seal::Ciphertext> FPGA_SealCKKSExecutor::evaluateLR(
    std::vector<std::vector<seal::Ciphertext>>& cipher_inputs) {
    auto cipher_weights = cblob_->weights();
    auto cipher_bias = cblob_->bias();
    auto weights_size = cblob_->weight_size();
    auto sigmoid_degree = cblob_->sigmoid_degree();

    return evaluateLogisticRegression(cipher_weights, cipher_inputs,
                                      cipher_bias, weights_size,
                                      sigmoid_degree);
}

std::vector<seal::Ciphertext> FPGA_SealCKKSExecutor::matMul(
    const std::vector<std::vector<seal::Ciphertext>>& A,
    const std::vector<std::vector<seal::Ciphertext>>& B_T, size_t count) {
    char* run = getenv("RUN_CHOICE");
    if (run && (0 == strtol(run, NULL, 10))) {
        return SealCKKSExecutor::matMul(A, B_T, count);
    } else {
        size_t m = A.size();
        size_t n = A[0].size();
        size_t r = B_T.size();

        std::vector<seal::Ciphertext> retval(m * r);

        assert(count > 0);
        size_t slot_count = encoder_->slot_count();
        size_t chunk_count =
            (count % slot_count == 0) ? slot_count : (count % slot_count);
        auto max_steps =
            (1 << seal::util::get_significant_bit_count(chunk_count - 1));

        std::vector<std::vector<std::vector<seal::Ciphertext>>> AB(
            m, std::vector<std::vector<seal::Ciphertext>>(
                   r, std::vector<seal::Ciphertext>(n)));

#pragma omp parallel for collapse(3) \
    num_threads(OMPUtilitiesS::getThreadsAtLevel())
        for (size_t A_r = 0; A_r < m; ++A_r) {
            for (size_t B_T_r = 0; B_T_r < r; ++B_T_r) {
                for (size_t i = 0; i < n; ++i) {
                    evaluator_->multiply(A[A_r][i], B_T[B_T_r][i],
                                         AB[A_r][B_T_r][i]);
                }
            }
        }

        intel::hexl::set_worksize_KeySwitch(m * r * n);
        for (size_t A_r = 0; A_r < m; ++A_r) {
            for (size_t B_T_r = 0; B_T_r < r; ++B_T_r) {
                for (size_t i = 0; i < n; ++i) {
                    evaluator_->relinearize_inplace(AB[A_r][B_T_r][i],
                                                    relin_keys_);
                }
            }
        }
        intel::hexl::KeySwitchCompleted();

#pragma omp parallel for collapse(3) \
    num_threads(OMPUtilitiesS::getThreadsAtLevel())
        for (size_t A_r = 0; A_r < m; ++A_r) {
            for (size_t B_T_r = 0; B_T_r < r; ++B_T_r) {
                for (size_t i = 0; i < n; ++i) {
                    evaluator_->rescale_to_next_inplace(AB[A_r][B_T_r][i]);
                }
            }
        }

        std::vector<std::vector<std::vector<seal::Ciphertext>>> rotated(
            m, std::vector<std::vector<seal::Ciphertext>>(
                   r, std::vector<seal::Ciphertext>(n)));

        for (int steps = 1; steps < max_steps; steps <<= 1) {
            intel::hexl::set_worksize_KeySwitch(m * r * n);
            for (size_t A_r = 0; A_r < m; ++A_r) {
                for (size_t B_T_r = 0; B_T_r < r; ++B_T_r) {
                    for (size_t i = 0; i < n; ++i) {
                        evaluator_->rotate_vector(AB[A_r][B_T_r][i], steps,
                                                  galois_keys_,
                                                  rotated[A_r][B_T_r][i]);
                    }
                }
            }
            intel::hexl::KeySwitchCompleted();

#pragma omp parallel for collapse(3) \
    num_threads(OMPUtilitiesS::getThreadsAtLevel())
            for (size_t A_r = 0; A_r < m; ++A_r) {
                for (size_t B_T_r = 0; B_T_r < r; ++B_T_r) {
                    for (size_t i = 0; i < n; ++i) {
                        evaluator_->add_inplace(AB[A_r][B_T_r][i],
                                                rotated[A_r][B_T_r][i]);
                    }
                }
            }
        }

#pragma omp parallel for
        for (size_t i = 0; i < m * r; i++) {
            encryptor_->encrypt_zero(retval[i]);
            retval[i].scale() = context_->scale();
        }
#pragma omp parallel for collapse(3) \
    num_threads(OMPUtilitiesS::getThreadsAtLevel())
        for (size_t A_r = 0; A_r < m; ++A_r) {
            for (size_t B_T_r = 0; B_T_r < r; ++B_T_r) {
                for (size_t i = 0; i < n; ++i) {
                    matchLevel(&retval[A_r * r + B_T_r], &AB[A_r][B_T_r][i]);
                    retval[A_r * r + B_T_r].scale() = AB[A_r][B_T_r][i].scale();
                    evaluator_->add_inplace(retval[A_r * r + B_T_r],
                                            AB[A_r][B_T_r][i]);
                }
            }
        }
        return retval;
    }
}

std::vector<seal::Ciphertext> FPGA_SealCKKSExecutor::collapse(
    const std::vector<seal::Ciphertext>& ciphers) {
    std::vector<seal::Ciphertext> retval;
    size_t slot_count = encoder_->slot_count();
    size_t total_chunks = ciphers.size() / slot_count +
                          (ciphers.size() % slot_count == 0 ? 0 : 1);
    retval.resize(total_chunks);
    seal::Plaintext plain;
    encoder_->encode(0.0, context_->scale(), plain);
    for (size_t i = 0; i < retval.size(); ++i)
        encryptor_->encrypt(plain, retval[i]);
    std::vector<double> identity;
    size_t cipher_i = 0;
    for (size_t chunk_i = 0; chunk_i < total_chunks; ++chunk_i) {
        seal::Ciphertext& retval_chunk = retval[chunk_i];
        identity.resize(
            (chunk_i + 1 == total_chunks ? ciphers.size() % slot_count
                                         : slot_count),
            0.0);
        for (size_t i = 0; i < identity.size(); ++i) {
            const seal::Ciphertext& cipher = ciphers[cipher_i++];
            if (i > 0) identity[i - 1] = 0.0;
            identity[i] = 1.0;
            encoder_->encode(identity, context_->scale(), plain);
            seal::Ciphertext tmp;
            evaluator_->rotate_vector(cipher, -static_cast<int>(i),
                                      galois_keys_, tmp);
            evaluator_->mod_switch_to_inplace(plain, tmp.parms_id());
            evaluator_->multiply_plain_inplace(tmp, plain);
            evaluator_->relinearize_inplace(tmp, relin_keys_);
            evaluator_->rescale_to_next_inplace(tmp);
            matchLevel(&retval_chunk, &tmp);
            tmp.scale() = context_->scale();
            retval_chunk.scale() = context_->scale();
            evaluator_->add_inplace(retval_chunk, tmp);
        }
        identity.back() = 0.0;
    }
    return retval;
}

std::vector<seal::Ciphertext> FPGA_SealCKKSExecutor::evaluatePolynomial(
    const std::vector<seal::Ciphertext>& inputs,
    const gsl::span<const double>& coefficients) {
    if (coefficients.empty())
        throw std::invalid_argument("coefficients cannot be empty");

    std::vector<seal::Plaintext> plain_coeff(coefficients.size());
    for (size_t coeff_i = 0; coeff_i < coefficients.size(); ++coeff_i)
        encoder_->encode(coefficients[coeff_i], context_->scale(),
                         plain_coeff[coeff_i]);

    std::vector<seal::Ciphertext> retval(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        seal::Ciphertext cipher_input_chunk = inputs[i];
        seal::Ciphertext& cipher_result_chunk = retval[i];

        auto it = plain_coeff.rbegin();
        encryptor_->encrypt(*it, cipher_result_chunk);
        for (++it; it != plain_coeff.rend(); ++it) {
            matchLevel(&cipher_input_chunk, &cipher_result_chunk);

            evaluator_->multiply_inplace(cipher_result_chunk,
                                         cipher_input_chunk);
            evaluator_->relinearize_inplace(cipher_result_chunk, relin_keys_);
            evaluator_->rescale_to_next_inplace(cipher_result_chunk);
            auto result_parms_id = cipher_result_chunk.parms_id();
            evaluator_->mod_switch_to_inplace(*it, result_parms_id);

            cipher_result_chunk.scale() = context_->scale();
            evaluator_->add_plain_inplace(cipher_result_chunk, *it);
        }
    }
    return retval;
}
