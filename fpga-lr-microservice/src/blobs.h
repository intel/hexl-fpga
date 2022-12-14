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
#include <string>
#include <sstream>
#include <memory>

#include "seal/seal.h"
#include "seal_ckks_context.h"

class Blob {
public:
    Blob() = default;
    Blob(size_t bsize, char* buffer) {
        stream_.write(reinterpret_cast<const char*>(buffer), bsize);
    }
    virtual ~Blob() {}

    size_t size() const { return stream_.str().size(); }
    char* data() {
        size_t csize = stream_.str().size();
        data_.resize(csize);
        stream_.read(reinterpret_cast<char*>(data_.data()), csize);
        return data_.data();
    }

    virtual void save() = 0;
    virtual void restore(
        const std::shared_ptr<intel::he::heseal::SealCKKSContext>&
            other_context = nullptr) = 0;

protected:
    static const seal::compr_mode_type comprMode_{seal::compr_mode_type::zstd};
    std::vector<char> data_;
    std::stringstream stream_;
};

class ConfigurationBlob : public Blob {
public:
    explicit ConfigurationBlob(
        std::shared_ptr<intel::he::heseal::SealCKKSContext>& context,
        std::string dataset);
    explicit ConfigurationBlob(size_t bsize, char* buffer)
        : Blob(bsize, buffer) {}

    uint32_t sigmoid_degree() const { return sigmoid_degree_; }
    std::vector<seal::Ciphertext>& weights() { return cipher_weights_; }
    seal::Ciphertext& bias() { return cipher_bias_; }
    uint32_t weight_size() const { return weight_size_; }

    void save() override;
    void restore(const std::shared_ptr<intel::he::heseal::SealCKKSContext>&
                     other_context) override;

    std::shared_ptr<intel::he::heseal::SealCKKSContext> context() {
        return context_server_;
    }

private:
    std::shared_ptr<intel::he::heseal::SealCKKSContext> context_;
    uint32_t sigmoid_degree_;

    std::vector<seal::Ciphertext> cipher_weights_;
    seal::Ciphertext cipher_bias_;
    uint32_t weight_size_;
    std::shared_ptr<intel::he::heseal::SealCKKSContext> context_server_;
};

class InputBlob : public Blob {
public:
    explicit InputBlob(std::vector<std::vector<seal::Ciphertext>>& inputs);
    explicit InputBlob(size_t bsize, char* buffer) : Blob(bsize, buffer) {}

    std::vector<std::vector<seal::Ciphertext>>& inputs() {
        return blob_inputs_;
    }

    void save() override;
    void restore(const std::shared_ptr<intel::he::heseal::SealCKKSContext>&
                     other_context) override;

private:
    std::vector<std::vector<seal::Ciphertext>> blob_inputs_;
};

class InputBlobs {
public:
    explicit InputBlobs(
        std::shared_ptr<intel::he::heseal::SealCKKSContext>& context,
        std::string dataset, uint32_t batch_size = 10);

    uint32_t size() const { return input_blobs_.size(); }
    std::shared_ptr<InputBlob> getBlob(uint32_t idx) const {
        return input_blobs_[idx];
    }

private:
    std::shared_ptr<intel::he::heseal::SealCKKSContext> context_;
    std::vector<std::shared_ptr<InputBlob>> input_blobs_;
};

class OutputBlob : public Blob {
public:
    explicit OutputBlob(std::vector<seal::Ciphertext>& outputs);
    explicit OutputBlob(size_t bsize, char* buffer) : Blob(bsize, buffer) {}

    std::vector<seal::Ciphertext>& outputs() { return cipher_outputs_; }

    void save() override;
    void restore(const std::shared_ptr<intel::he::heseal::SealCKKSContext>&
                     other_context) override;

private:
    std::vector<seal::Ciphertext> cipher_outputs_;
};
