// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include <gsl/span>
#include <chrono>
#include "seal_ckks_executor.h"
#include "seal_ckks_context.h"
#include "seal_omp_utils.h"

namespace intel {
namespace he {
namespace heseal {

const double SealCKKSExecutor::sigmoid_coeff_3[] = {0.5, 0.15012, 0.0,
                                                    -0.001593008};
const double SealCKKSExecutor::sigmoid_coeff_5[] = {
    0.5, 0.19131, 0.0, -0.0045963, 0.0, 0.000041233};
const double SealCKKSExecutor::sigmoid_coeff_7[] = {
    0.5, 0.21687, 0.0, -0.008191543, 0.0, 0.000165833, 0.0, 0.000001196};

SealCKKSExecutor::SealCKKSExecutor(const seal::EncryptionParameters& params,
                                   double scale,
                                   const seal::PublicKey& public_key,
                                   const seal::RelinKeys& relin_keys,
                                   const seal::GaloisKeys& galois_keys) {
    if (params.scheme() != seal::scheme_type::ckks)
        throw std::invalid_argument("Only CKKS scheme supported.");
    m_scale = scale;
    m_public_key = public_key;
    m_relin_keys = relin_keys;
    m_galois_keys = galois_keys;
    m_pcontext = std::make_shared<seal::SEALContext>(params);
    m_pevaluator = std::make_shared<seal::Evaluator>(*m_pcontext);
    m_pencoder = std::make_shared<seal::CKKSEncoder>(*m_pcontext);
    m_pencryptor = std::make_shared<seal::Encryptor>(*m_pcontext, m_public_key);
}

SealCKKSExecutor::~SealCKKSExecutor() {
    m_pencryptor.reset();
    m_pencoder.reset();
    m_pevaluator.reset();
}

std::vector<seal::Ciphertext> SealCKKSExecutor::add(
    const std::vector<seal::Ciphertext>& A,
    const std::vector<seal::Ciphertext>& B) {
    std::vector<seal::Ciphertext> retval;
    if (A.size() != B.size())
        throw std::invalid_argument("A.size() != B.size()");
    retval.resize(A.size());
    for (size_t i = 0; i < retval.size(); ++i) {
        m_pevaluator->add(A[i], B[i], retval[i]);
    }
    return retval;
}

seal::Ciphertext SealCKKSExecutor::accumulate_internal(
    const seal::Ciphertext& cipher_input, std::size_t count) {
    seal::Ciphertext retval;
    if (count > 0) {
        retval = cipher_input;
        auto max_steps = (1 << seal::util::get_significant_bit_count(count));
        for (int steps = 1; steps < max_steps; steps <<= 1) {
            seal::Ciphertext rotated;
            m_pevaluator->rotate_vector(retval, steps, m_galois_keys, rotated,
                                        seal::MemoryPoolHandle::ThreadLocal());
            m_pevaluator->add_inplace(retval, rotated);
        }
    } else {
        m_pencryptor->encrypt_zero(retval);
        retval.scale() = cipher_input.scale();
    }

    return retval;
}

seal::Ciphertext SealCKKSExecutor::accumulate(
    const std::vector<seal::Ciphertext>& V, std::size_t count) {
    seal::Ciphertext retval;
    m_pencryptor->encrypt_zero(retval);

    if (count > 0) {
        std::size_t slot_count = m_pencoder->slot_count();
        for (std::size_t i = 0; i < V.size(); ++i) {
            std::size_t chunk_count =
                i + 1 < V.size() ? slot_count : count % slot_count;
            seal::Ciphertext chunk_retval =
                accumulate_internal(V[i], chunk_count);
            matchLevel(&retval, &chunk_retval);
            retval.scale() = chunk_retval.scale();
            m_pevaluator->add_inplace(retval, chunk_retval);
        }
    }

    return retval;
}

seal::Ciphertext SealCKKSExecutor::dot(const std::vector<seal::Ciphertext>& A,
                                       const std::vector<seal::Ciphertext>& B,
                                       size_t count) {
    seal::Ciphertext retval;

    if (count > 0) {
        std::vector<seal::Ciphertext> AB(A.size());
        for (size_t i = 0; i < AB.size(); ++i) {
            m_pevaluator->multiply(A[i], B[i], AB[i]);
            m_pevaluator->relinearize_inplace(AB[i], m_relin_keys);
            m_pevaluator->rescale_to_next_inplace(AB[i]);
        }
        retval = accumulate(AB, count);
    } else {
        m_pencryptor->encrypt_zero(retval);
        retval.scale() = m_scale;
    }
    return retval;
}

std::vector<seal::Ciphertext> SealCKKSExecutor::matMul(
    const std::vector<std::vector<seal::Ciphertext>>& A,
    const std::vector<std::vector<seal::Ciphertext>>& B_T, size_t cols) {
    std::vector<seal::Ciphertext> retval(A.size() * B_T.size());
#pragma omp parallel for collapse(2) \
    num_threads(OMPUtilitiesS::getThreadsAtLevel())
    for (size_t A_r = 0; A_r < A.size(); ++A_r) {
        for (size_t B_T_r = 0; B_T_r < B_T.size(); ++B_T_r) {
            retval[A_r * B_T.size() + B_T_r] = dot(A[A_r], B_T[B_T_r], cols);
        }
    }
    return retval;
}

std::vector<seal::Ciphertext> SealCKKSExecutor::evaluatePolynomial(
    const std::vector<seal::Ciphertext>& inputs,
    const gsl::span<const double>& coefficients) {
    if (coefficients.empty())
        throw std::invalid_argument("coefficients cannot be empty");

    std::vector<seal::Plaintext> plain_coeff(coefficients.size());
    for (size_t coeff_i = 0; coeff_i < coefficients.size(); ++coeff_i)
        m_pencoder->encode(coefficients[coeff_i], m_scale,
                           plain_coeff[coeff_i]);

    std::vector<seal::Ciphertext> retval(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        seal::Ciphertext cipher_input_chunk = inputs[i];
        seal::Ciphertext& cipher_result_chunk = retval[i];

        auto it = plain_coeff.rbegin();
        m_pencryptor->encrypt(*it, cipher_result_chunk);
        for (++it; it != plain_coeff.rend(); ++it) {
            matchLevel(&cipher_input_chunk, &cipher_result_chunk);

            m_pevaluator->multiply_inplace(cipher_result_chunk,
                                           cipher_input_chunk);
            m_pevaluator->relinearize_inplace(cipher_result_chunk,
                                              m_relin_keys);
            m_pevaluator->rescale_to_next_inplace(cipher_result_chunk);
            auto result_parms_id = cipher_result_chunk.parms_id();
            m_pevaluator->mod_switch_to_inplace(*it, result_parms_id);

            cipher_result_chunk.scale() = m_scale;
            m_pevaluator->add_plain_inplace(cipher_result_chunk, *it);
        }
    }
    return retval;
}

std::vector<seal::Ciphertext> SealCKKSExecutor::collapse(
    const std::vector<seal::Ciphertext>& ciphers) {
    std::vector<seal::Ciphertext> retval;
    size_t slot_count = m_pencoder->slot_count();
    size_t total_chunks = ciphers.size() / slot_count +
                          (ciphers.size() % slot_count == 0 ? 0 : 1);
    retval.resize(total_chunks);
    seal::Plaintext plain;
    m_pencoder->encode(0.0, m_scale, plain);
    for (size_t i = 0; i < retval.size(); ++i)
        m_pencryptor->encrypt(plain, retval[i]);
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
            m_pencoder->encode(identity, m_scale, plain);
            seal::Ciphertext tmp;
            m_pevaluator->rotate_vector(cipher, -static_cast<int>(i),
                                        m_galois_keys, tmp);
            m_pevaluator->mod_switch_to_inplace(plain, tmp.parms_id());
            m_pevaluator->multiply_plain_inplace(tmp, plain);
            m_pevaluator->relinearize_inplace(tmp, m_relin_keys);
            m_pevaluator->rescale_to_next_inplace(tmp);
            matchLevel(&retval_chunk, &tmp);
            tmp.scale() = m_scale;
            retval_chunk.scale() = m_scale;
            m_pevaluator->add_inplace(retval_chunk, tmp);
        }
        identity.back() = 0.0;
    }
    return retval;
}

std::vector<seal::Ciphertext> SealCKKSExecutor::evaluateLinearRegression(
    std::vector<seal::Ciphertext>& weights,
    std::vector<std::vector<seal::Ciphertext>>& inputs, seal::Ciphertext& bias,
    size_t weights_count) {
    std::vector<std::vector<seal::Ciphertext>> weights_copy{weights};
    std::vector<seal::Ciphertext> retval =
        collapse(matMul(weights_copy, inputs, weights_count));
    weights_copy.clear();
    for (size_t i = 0; i < retval.size(); ++i) {
        matchLevel(&retval[i], &bias);
        bias.scale() = m_scale;
        retval[i].scale() = m_scale;
        m_pevaluator->add_inplace(retval[i], bias);
    }
    return retval;
}

std::vector<seal::Ciphertext> SealCKKSExecutor::evaluateLogisticRegression(
    std::vector<seal::Ciphertext>& weights,
    std::vector<std::vector<seal::Ciphertext>>& inputs, seal::Ciphertext& bias,
    size_t weights_count, unsigned int sigmoid_degree) {
    std::vector<seal::Ciphertext> retval =
        evaluateLinearRegression(weights, inputs, bias, weights_count);

    switch (sigmoid_degree) {
    case 5:
        retval = sigmoid<5>(retval);
        break;
    case 7:
        retval = sigmoid<7>(retval);
        break;
    default:
        retval = sigmoid<3>(retval);
        break;
    }
    return retval;
}

void SealCKKSExecutor::matchLevel(seal::Ciphertext* a,
                                  seal::Ciphertext* b) const {
    int a_level = getLevel(*a);
    int b_level = getLevel(*b);
    if (a_level > b_level)
        m_pevaluator->mod_switch_to_inplace(*a, b->parms_id());
    else if (a_level < b_level)
        m_pevaluator->mod_switch_to_inplace(*b, a->parms_id());
}

}  // namespace heseal
}  // namespace he
}  // namespace intel
