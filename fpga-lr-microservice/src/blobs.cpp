// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include "gsl/span"
#include "seal/seal.h"
#include "seal_ckks_context.h"
#include "blobs.h"
#include "data_loader.h"

ConfigurationBlob::ConfigurationBlob(
    std::shared_ptr<intel::he::heseal::SealCKKSContext>& context,
    std::string dataset)
    : Blob(), sigmoid_degree_(3), context_(context) {
    std::vector<double> raw_weights = weightsLoaderCSV(dataset);
    std::vector<double> weights;
    double bias;
    splitWeights(raw_weights, weights, bias);
    weight_size_ = weights.size();
    std::cout << "    Encode/Encrypt model data" << std::endl;
    std::cout << "        Weight dimension: " << weight_size_ << std::endl;

    auto t_started = std::chrono::high_resolution_clock::now();
    cipher_weights_ =
        context->encryptVector(gsl::span(weights.data(), weight_size_),
                               context->encoder()->slot_count());

    seal::Plaintext plain_bias;
    context->encoder()->encode(bias, context->scale(), plain_bias);
    context->encryptor()->encrypt(plain_bias, cipher_bias_);
}

void ConfigurationBlob::save() {
    context_->parms().save(stream_, comprMode_);
    double scale = context_->scale();
    stream_.write(reinterpret_cast<const char*>(&scale), sizeof(double));
    context_->public_key().save(stream_, comprMode_);
    context_->relin_keys().save(stream_, comprMode_);
    context_->galois_keys().save(stream_, comprMode_);

    uint32_t cipher_weights_size =
        static_cast<uint32_t>(cipher_weights_.size());
    stream_.write(reinterpret_cast<const char*>(&cipher_weights_size),
                  sizeof(uint32_t));
    for (const auto& weight : cipher_weights_) {
        weight.save(stream_, comprMode_);
    }
    cipher_bias_.save(stream_, comprMode_);

    stream_.write(reinterpret_cast<const char*>(&weight_size_),
                  sizeof(uint32_t));
    stream_.write(reinterpret_cast<const char*>(&sigmoid_degree_),
                  sizeof(uint32_t));
}

void ConfigurationBlob::restore(
    const std::shared_ptr<intel::he::heseal::SealCKKSContext>& other_context) {
    seal::sec_level_type sec_level = seal::sec_level_type::tc128;
    seal::EncryptionParameters parms{seal::scheme_type::ckks};
    parms.load(stream_);
    auto seal_context =
        std::make_shared<seal::SEALContext>(parms, true, sec_level);
    double scale;
    stream_.read(reinterpret_cast<char*>(&scale), sizeof(double));
    seal::PublicKey public_key;
    public_key.load(*seal_context, stream_);
    seal::RelinKeys relin_keys;
    relin_keys.load(*seal_context, stream_);
    seal::GaloisKeys galois_keys;
    galois_keys.load(*seal_context, stream_);
    uint32_t cipher_weight_size;
    stream_.read(reinterpret_cast<char*>(&cipher_weight_size),
                 sizeof(uint32_t));
    cipher_weights_.clear();
    seal::Ciphertext weight;
    for (uint32_t i = 0; i < cipher_weight_size; ++i) {
        weight.load(*seal_context, stream_);
        cipher_weights_.push_back(weight);
    }
    cipher_bias_.load(*seal_context, stream_);
    stream_.read(reinterpret_cast<char*>(&weight_size_), sizeof(uint32_t));
    stream_.read(reinterpret_cast<char*>(&sigmoid_degree_), sizeof(uint32_t));

    context_server_ = std::make_shared<intel::he::heseal::SealCKKSContext>(
        seal_context, scale, public_key, relin_keys, galois_keys);
}

InputBlobs::InputBlobs(
    std::shared_ptr<intel::he::heseal::SealCKKSContext>& context,
    std::string dataset, uint32_t batch_size)
    : context_(context) {
    std::vector<std::vector<double>> raw_data =
        dataLoader(dataset, DataMode::TEST);
    std::vector<std::vector<double>> inputs;
    std::vector<double> target;
    splitData(raw_data, inputs, target);

    std::cout << "    Encode/Encrypt input data" << std::endl;
    std::cout << "        input_size: " << inputs.size() << std::endl;
    std::cout << "        chunk_size: " << batch_size << std::endl;

    std::vector<std::vector<seal::Ciphertext>> cipher_inputs(inputs.size());

    auto t_started = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < inputs.size(); ++i) {
        cipher_inputs[i] = context->encryptVector(
            gsl::span(inputs[i].data(), inputs[i].size()),
            context->encoder()->slot_count());
    }
    auto t_done = std::chrono::high_resolution_clock::now();
    std::cout << "        Elapsed: "
              << static_cast<double>(
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         t_done - t_started)
                         .count()) /
                     1000.0
              << " seconds" << std::endl;

    uint32_t chunks = (inputs.size() + batch_size - 1) / batch_size;
    for (uint32_t i = 0; i < chunks; ++i) {
        auto first = i * batch_size;
        auto last = (((i + 1) * batch_size) < inputs.size())
                        ? ((i + 1) * batch_size)
                        : inputs.size();
        std::vector<std::vector<seal::Ciphertext>> blob_data;
        std::copy(cipher_inputs.cbegin() + first, cipher_inputs.cbegin() + last,
                  std::back_inserter(blob_data));
        auto input_blob = std::make_shared<InputBlob>(blob_data);
        input_blobs_.push_back(input_blob);
    }
}

InputBlob::InputBlob(std::vector<std::vector<seal::Ciphertext>>& inputs)
    : Blob(), blob_inputs_(inputs) {}

void InputBlob::save() {
    size_t input_size = blob_inputs_.size();
    stream_.write(reinterpret_cast<const char*>(&input_size), sizeof(size_t));
    for (size_t i = 0; i < input_size; ++i) {
        for (const auto& input : blob_inputs_[i]) {
            input.save(stream_, comprMode_);
        }
    }
}

void InputBlob::restore(
    const std::shared_ptr<intel::he::heseal::SealCKKSContext>& other_context) {
    size_t input_size;
    stream_.read(reinterpret_cast<char*>(&input_size), sizeof(size_t));
    blob_inputs_.clear();
    for (size_t i = 0; i < input_size; ++i) {
        std::vector<seal::Ciphertext> cinputs;
        seal::Ciphertext ci;
        ci.load(*(other_context->context()), stream_);
        cinputs.push_back(ci);
        blob_inputs_.push_back(cinputs);
    }
}

OutputBlob::OutputBlob(std::vector<seal::Ciphertext>& outputs)
    : Blob(), cipher_outputs_(outputs) {}

void OutputBlob::save() {
    uint32_t output_count = cipher_outputs_.size();
    stream_.write(reinterpret_cast<const char*>(&output_count),
                  sizeof(uint32_t));

    for (const auto& output : cipher_outputs_) {
        output.save(stream_, comprMode_);
    }
}

void OutputBlob::restore(
    const std::shared_ptr<intel::he::heseal::SealCKKSContext>& other_context) {
    uint32_t output_count;
    stream_.read(reinterpret_cast<char*>(&output_count), sizeof(uint32_t));
    cipher_outputs_.clear();
    for (uint32_t i = 0; i < output_count; ++i) {
        seal::Ciphertext co;
        co.load(*(other_context->context()), stream_);
        cipher_outputs_.push_back(co);
    }
}
