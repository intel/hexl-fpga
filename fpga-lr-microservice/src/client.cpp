// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <queue>
#include <memory>
#include <cassert>
#include "seal/seal.h"
#include "blobs.h"
#include "seal_ckks_context.h"
#include "client.h"

void Client::init() {
    input_idx_ = 0;
    chunk_size_ = 1;

    poly_modulus_degree_ = 16384;
    coeff_.resize(0);
    scale_ = 1.0;
    sec_lvl_ = seal::sec_level_type::tc128;
}

Client::Client() { init(); }

Client::Client(std::shared_ptr<Buffer>& buffer, std::string dataset,
               size_t batch_size, const size_t poly_modulus_degree,
               const std::vector<int>& coeff_mod_bit_sizes,
               const unsigned scale_bit_size,
               const seal::sec_level_type sec_level)
    : Party(buffer), dataset_(dataset) {
    init();
    chunk_size_ = batch_size;
    poly_modulus_degree_ = poly_modulus_degree;
    coeff_ = coeff_mod_bit_sizes;
    scale_ = static_cast<double>(1UL << scale_bit_size);
    sec_lvl_ = sec_level;
    context_ = std::make_shared<intel::he::heseal::SealCKKSContext>(
        poly_modulus_degree_, coeff_, scale_, sec_lvl_, true, true);
    context_->printParameters();
}

int Client::put_dataset_filename(const char* input_filename_base) {
    dataset_ = std::string(input_filename_base);
    return 0;
}

int Client::put_context_data(size_t poly_modulus_degree,
                             const int* coeff_mod_bit_sizes,
                             size_t sz_coeff_mod_bits_sizes,
                             size_t scale_bit_size, size_t sec_level,
                             size_t client_chunk_size) {
    chunk_size_ = client_chunk_size;
    poly_modulus_degree_ = poly_modulus_degree;
    coeff_.insert(coeff_.begin(), coeff_mod_bit_sizes,
                  coeff_mod_bit_sizes + sz_coeff_mod_bits_sizes);
    scale_ = static_cast<double>(1UL << scale_bit_size);

    switch (sec_level) {
    case 0:
        sec_lvl_ = seal::sec_level_type::none;
        break;
    case 128:
        sec_lvl_ = seal::sec_level_type::tc128;
        break;
    case 192:
        sec_lvl_ = seal::sec_level_type::tc192;
        break;
    case 256:
        sec_lvl_ = seal::sec_level_type::tc256;
        break;
    default:
        std::cout << "ERROR: sec_level must be one of {0, 128, 192, 256}.\n";
        return 1;
    }

    return 0;
}

int Client::get_config_buffer(char** config_buffer, size_t* sz_config_buffer) {
    context_ = std::make_shared<intel::he::heseal::SealCKKSContext>(
        poly_modulus_degree_, coeff_, scale_, sec_lvl_, true, true);
    context_->printParameters();

    config_blob_ = std::make_shared<ConfigurationBlob>(context_, dataset_);
    config_blob_->save();

    *config_buffer = config_blob_->data();
    *sz_config_buffer = config_blob_->size();

    return 0;
}

int Client::get_total_num_chunks() {
    if (!input_blobs_) {
        input_blobs_ =
            std::make_shared<InputBlobs>(context_, dataset_, chunk_size_);
    }
    return input_blobs_->size();
}

int Client::get_encrypted_chunk_data(char** encrypted_data,
                                     size_t* sz_encrypted_data,
                                     size_t chunk_index) {
    assert((chunk_index >= 0) && (chunk_index < input_blobs_->size()));
    current_input_blob_ = input_blobs_->getBlob(chunk_index);
    if (!current_input_blob_) {
        return 1;
    }
    current_input_blob_->save();

    *encrypted_data = const_cast<char*>(current_input_blob_->data());
    *sz_encrypted_data = current_input_blob_->size();

    return 0;
}

int Client::get_encrypted_result_buffer(char** encrypted_result,
                                        size_t sz_encrypted_result) {
    encrypted_result_.resize(sz_encrypted_result);
    *encrypted_result = encrypted_result_.data();
    return 0;
}

int Client::get_decrypted_decoded_result(int** result, size_t* sz_result,
                                         char* encrypted_result,
                                         size_t sz_encrypted_result) {
    auto oblob =
        std::make_shared<OutputBlob>(sz_encrypted_result, encrypted_result);
    result_ = decryptDecode(oblob);
    *result = result_.data();
    *sz_result = result_.size();
    return 0;
}

bool Client::setup() {
    // Todo:
    // client constructs setup request
    // server answers with fpga context setup OK or not.

    std::cout << "[Client] starting the service ..." << std::endl;

    // prepare inputs data from disk to memory
    collectInputs();
    return true;
}

void Client::teardown() {
    // Todo:
    // client constructs close request
    // server closes the service.
    std::cout << "[Client] closing the service ..." << std::endl;
}

void Client::configure() {
    std::cout << "[Client] sending configuration data ..." << std::endl;
    auto blob = std::make_shared<ConfigurationBlob>(context_, dataset_);
    blob->save();
    buffer_->queue[Buffer::ConfigQ].push(blob);
}

void Client::collectInputs() {
    std::cout << "[Client] preparing input data ..." << std::endl;
    input_blobs_ =
        std::make_shared<InputBlobs>(context_, dataset_, chunk_size_);
}

bool Client::moreInputs() { return (input_idx_ < input_blobs_->size()); }

void Client::process() {
    std::cout << "[Client] sending input data ... chunk " << input_idx_
              << std::endl;
    auto input_blob = input_blobs_->getBlob(input_idx_);
    assert(input_blob);
    input_blob->save();
    buffer_->queue[Buffer::InputQ].push(input_blob);
    input_idx_++;

    while (!buffer_->queue[Buffer::OutputQ].empty()) {
        auto blob = buffer_->queue[Buffer::OutputQ].front();
        buffer_->queue[Buffer::OutputQ].pop();
        std::shared_ptr<OutputBlob> oblob =
            std::dynamic_pointer_cast<OutputBlob>(blob);
        result_ = decryptDecode(oblob);
    }
}

std::vector<int> Client::decryptDecode(std::shared_ptr<OutputBlob>& oblob) {
    oblob->restore(context_);
    std::vector<seal::Ciphertext> cipher_retval = oblob->outputs();

    std::vector<seal::Plaintext> pt_retval =
        context_->decryptVector(cipher_retval);
    std::vector<double> retval = context_->decodeVector(pt_retval);
    std::vector<int> int_retval;
    std::transform(
        retval.begin(), retval.end(), std::back_inserter(int_retval),
        [](const double& val) { return static_cast<int>(val + 0.5); });

    return int_retval;
}
