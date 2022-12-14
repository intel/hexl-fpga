// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <seal/seal.h>
#include <gsl/span>

#include <memory>
#include <utility>
#include <vector>

#include "seal_ckks_context.h"

namespace intel {
namespace he {
namespace heseal {

class SealCKKSExecutor {
public:
    explicit SealCKKSExecutor(
        const seal::EncryptionParameters& params, double scale,
        const seal::PublicKey& public_key,
        const seal::RelinKeys& relin_keys = seal::RelinKeys(),
        const seal::GaloisKeys& galois_keys = seal::GaloisKeys());

    SealCKKSExecutor(
        const std::shared_ptr<intel::he::heseal::SealCKKSContext>& context)
        : m_scale(context->scale()),
          m_public_key(context->public_key()),
          m_relin_keys(context->relin_keys()),
          m_galois_keys(context->galois_keys()),
          m_pcontext(context->context()),
          m_pevaluator(context->evaluator()),
          m_pencoder(context->encoder()),
          m_pencryptor(context->encryptor()) {}

    ~SealCKKSExecutor();

    // Adds two ciphertext vectors element-wise
    std::vector<seal::Ciphertext> add(const std::vector<seal::Ciphertext>& A,
                                      const std::vector<seal::Ciphertext>& B);

    // Sums together all the ciphertexts in v.
    // Assumes each ciphertext stores batch_size scalars
    // @param batch_size Number of scalars stored in each ciphertext
    // @return A ciphertext storing the sum in the first slot
    seal::Ciphertext accumulate(const std::vector<seal::Ciphertext>& v,
                                size_t count);

    // Performs dot product of A and B
    // @param A vector of ciphertexts
    // @param B vector of ciphertext
    // @param batch_size Number of scalars stored in each ciphertext
    // @returns A ciphertext containing the dot product in the first slot
    seal::Ciphertext dot(const std::vector<seal::Ciphertext>& A,
                         const std::vector<seal::Ciphertext>& B,
                         size_t batch_size);

    virtual std::vector<seal::Ciphertext> matMul(
        const std::vector<std::vector<seal::Ciphertext>>& A,
        const std::vector<std::vector<seal::Ciphertext>>& B_T, size_t cols);

    virtual std::vector<seal::Ciphertext> evaluatePolynomial(
        const std::vector<seal::Ciphertext>& inputs,
        const gsl::span<const double>& coefficients);

    virtual std::vector<seal::Ciphertext> collapse(
        const std::vector<seal::Ciphertext>& ciphers);

    template <unsigned int degree = 3>
    std::vector<seal::Ciphertext> sigmoid(
        const std::vector<seal::Ciphertext>& inputs);

    std::vector<seal::Ciphertext> evaluateLinearRegression(
        std::vector<seal::Ciphertext>& weights,
        std::vector<std::vector<seal::Ciphertext>>& inputs,
        seal::Ciphertext& bias, size_t weights_count);

    std::vector<seal::Ciphertext> evaluateLogisticRegression(
        std::vector<seal::Ciphertext>& weights,
        std::vector<std::vector<seal::Ciphertext>>& inputs,
        seal::Ciphertext& bias, size_t weights_count,
        unsigned int sigmoid_degree = 3);

    //  Modulus switches a and b such that their levels match
    void matchLevel(seal::Ciphertext* a, seal::Ciphertext* b) const;

    // Returns the level of the ciphertext
    size_t getLevel(const seal::Ciphertext& cipher) const {
        return m_pcontext->get_context_data(cipher.parms_id())->chain_index();
    }

    std::shared_ptr<seal::Evaluator> getEvaluator() const {
        return m_pevaluator;
    }

    // Returns index of coordinate [i,j] of [dim1 x dim2] matrix stored in
    // column-major format
    static size_t colMajorIndex(size_t dim1, size_t dim2, size_t i, size_t j) {
        if (i >= dim1) {
            std::stringstream ss;
            ss << i << " too large for dim1 (" << dim1 << ")";
            throw std::runtime_error(ss.str());
        }
        if (j >= dim2) {
            std::stringstream ss;
            ss << j << " too large for dim2 (" << dim2 << ")";
            throw std::runtime_error(ss.str());
        }
        return i + j * dim1;
    }

    std::shared_ptr<seal::SEALContext> context() { return m_pcontext; }

private:
    static const double sigmoid_coeff_3[];
    static const double sigmoid_coeff_5[];
    static const double sigmoid_coeff_7[];

    seal::Ciphertext accumulate_internal(const seal::Ciphertext& v,
                                         size_t count);
    std::shared_ptr<seal::SEALContext> m_pcontext;
    std::shared_ptr<seal::Evaluator> m_pevaluator;
    std::shared_ptr<seal::Encryptor> m_pencryptor;
    std::shared_ptr<seal::CKKSEncoder> m_pencoder;
    seal::EncryptionParameters m_params;
    seal::PublicKey m_public_key;
    seal::RelinKeys m_relin_keys;
    seal::GaloisKeys m_galois_keys;
    double m_scale;
};

template <>
inline std::vector<seal::Ciphertext> SealCKKSExecutor::sigmoid<3>(
    const std::vector<seal::Ciphertext>& inputs) {
    return evaluatePolynomial(inputs, gsl::span(sigmoid_coeff_3, 4));
}

template <>
inline std::vector<seal::Ciphertext> SealCKKSExecutor::sigmoid<5>(
    const std::vector<seal::Ciphertext>& inputs) {
    return evaluatePolynomial(inputs, gsl::span(sigmoid_coeff_5, 6));
}

template <>
inline std::vector<seal::Ciphertext> SealCKKSExecutor::sigmoid<7>(
    const std::vector<seal::Ciphertext>& inputs) {
    return evaluatePolynomial(inputs, gsl::span(sigmoid_coeff_7, 8));
}

}  // namespace heseal
}  // namespace he
}  // namespace intel
