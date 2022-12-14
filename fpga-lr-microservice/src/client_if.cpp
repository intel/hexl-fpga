// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "client_if.h"
#include "client.h"

static Client client;

int put_dataset_filename(const char* input_filename_base) {
    return client.put_dataset_filename(input_filename_base);
}

int put_context_data(size_t poly_modulus_degree, const int* coeff_mod_bit_sizes,
                     size_t sz_coeff_mod_bits, size_t scale_bit_size,
                     size_t sec_lvl, size_t client_chunk_size) {
    return client.put_context_data(poly_modulus_degree, coeff_mod_bit_sizes,
                                   sz_coeff_mod_bits, scale_bit_size, sec_lvl,
                                   client_chunk_size);
}

int get_config_buffer(char** config_buffer, size_t* sz_config_buffer) {
    std::cout << "[Client] preparing configuration data ..." << std::endl;
    return client.get_config_buffer(config_buffer, sz_config_buffer);
}

size_t get_total_num_chunks() {
    std::cout << "[Client] preparing input data ..." << std::endl;
    return client.get_total_num_chunks();
}

int get_encrypted_chunk_data(char** encrypted_data, size_t* sz_encrypted_data,
                             size_t chunk_index) {
    std::cout << "[Client] sending input data ... chunk " << chunk_index
              << std::endl;
    return client.get_encrypted_chunk_data(encrypted_data, sz_encrypted_data,
                                           chunk_index);
}

int get_encrypted_result_buffer(char** encrypted_data,
                                size_t sz_encrypted_data) {
    return client.get_encrypted_result_buffer(encrypted_data,
                                              sz_encrypted_data);
}

int get_decrypted_decoded_result(int** result, size_t* sz_result,
                                 char* encrypted_result,
                                 size_t sz_encrypted_result) {
    std::cout << "[Client] decryting/decoding result ..." << std::endl;
    return client.get_decrypted_decoded_result(
        result, sz_result, encrypted_result, sz_encrypted_result);
}
