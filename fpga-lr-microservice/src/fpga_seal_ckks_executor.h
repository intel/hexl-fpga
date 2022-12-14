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

#pragma once

#include <vector>
#include <memory>

#include "seal/seal.h"
#include "seal_ckks_context.h"
#include "seal_ckks_executor.h"
#include "blobs.h"

class FPGA_SealCKKSExecutor : public intel::he::heseal::SealCKKSExecutor {
private:
    std::shared_ptr<ConfigurationBlob> cblob_;
    std::shared_ptr<intel::he::heseal::SealCKKSContext> context_;
    std::shared_ptr<seal::CKKSEncoder> encoder_;
    std::shared_ptr<seal::Encryptor> encryptor_;
    std::shared_ptr<seal::Evaluator> evaluator_;
    const seal::RelinKeys& relin_keys_;
    const seal::GaloisKeys& galois_keys_;

public:
    explicit FPGA_SealCKKSExecutor(std::shared_ptr<ConfigurationBlob>& cblob);
    ~FPGA_SealCKKSExecutor() = default;

    std::vector<seal::Ciphertext> evaluateLR(
        std::vector<std::vector<seal::Ciphertext>>& inputs);

    std::vector<seal::Ciphertext> matMul(
        const std::vector<std::vector<seal::Ciphertext>>& A,
        const std::vector<std::vector<seal::Ciphertext>>& B_T, size_t cols);

    std::vector<seal::Ciphertext> collapse(
        const std::vector<seal::Ciphertext>& ciphers);

    std::vector<seal::Ciphertext> evaluatePolynomial(
        const std::vector<seal::Ciphertext>& inputs,
        const gsl::span<const double>& coefficients);
};
