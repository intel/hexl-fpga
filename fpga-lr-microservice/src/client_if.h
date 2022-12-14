// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
int put_dataset_filename(const char* input_filename_base);
int put_context_data(size_t poly_modulus_degree, const int* coeff_mod_bit_sizes,
                     size_t sz_coeff_mod_bits_sizes, size_t scale_bit_size,
                     size_t sec_lvl, size_t client_chunk_size);

int get_config_buffer(char** config_buffer, size_t* sz_config_buffer);
size_t get_total_num_chunks();  // total num of client chunks
int get_encrypted_chunk_data(char** encrypted_data, size_t* sz_encrypted_data,
                             size_t chunk_index);
int get_encrypted_result_buffer(char** encrypted_data,
                                size_t sz_encrypted_data);
int get_decrypted_decoded_result(int** result, size_t* sz_result,
                                 char* encrypted_result,
                                 size_t sz_encrypted_result);
#ifdef __cplusplus
}
#endif
