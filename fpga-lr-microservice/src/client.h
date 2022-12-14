// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "seal/seal.h"
#include "blobs.h"
#include "buffer.h"
#include "seal_ckks_context.h"

class Client : public Party {
public:
    Client();
    explicit Client(std::shared_ptr<Buffer>& buffer, std::string dataset,
                    size_t batch_size, const size_t poly_modulus_degree,
                    const std::vector<int>& coeff_mod_bit_sizes,
                    const unsigned scale_bit_size,
                    const seal::sec_level_type sec_level);
    ~Client() { teardown(); }

    bool setup() override;
    void configure() override;
    void process() override;
    void teardown() override;
    bool moreInputs();

    // C APIs
    int put_dataset_filename(const char* input_filename_base);
    int put_context_data(size_t poly_modulus_degree,
                         const int* coeff_mod_bit_sizes,
                         size_t sz_coeff_mod_bits_sizes, size_t scale_bit_size,
                         size_t sec_level, size_t client_chunk_size);

    int get_config_buffer(char** config_buffer, size_t* sz_config_buffer);
    int get_total_num_chunks();  // total num of client chunks
    int get_encrypted_chunk_data(char** encrypted_data,
                                 size_t* sz_encrypted_data, size_t chunk_index);
    int get_decrypted_decoded_result(int** result, size_t* sz_result,
                                     char* encrypted_result,
                                     size_t sz_encrypted_result);

    int get_encrypted_result_buffer(char** encrypted_result,
                                    size_t sz_encrypted_result);
    // C APIs end

private:
    void init();
    void collectInputs();
    std::vector<int> decryptDecode(std::shared_ptr<OutputBlob>& oblob);

    std::string dataset_;
    size_t input_idx_;
    size_t chunk_size_;
    size_t poly_modulus_degree_;
    std::vector<int> coeff_;
    double scale_;
    seal::sec_level_type sec_lvl_;
    std::shared_ptr<intel::he::heseal::SealCKKSContext> context_;
    std::shared_ptr<InputBlobs> input_blobs_;
    std::shared_ptr<InputBlob> current_input_blob_;
    std::vector<char> encrypted_result_;
    std::vector<int> result_;
    std::shared_ptr<ConfigurationBlob> config_blob_;
};
