// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <memory>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cstring>

#include "gflags/gflags.h"
#include "client_if.h"
#include "server_if.h"

DEFINE_string(data, "homecreditdefaultrisk",
              "Dataset name to test - choose one of heartdisease, "
              "homecreditdefaultrisk, breastcancer");
DEFINE_uint32(
    chunk_size, 10,
    "The batch size of input data in a chunk, which is a unit for processing");
DEFINE_uint32(poly_modulus_degree, 16384,
              "Degree of the polynomial modulus. Must be a power of 2 between "
              "1024 and 16384.");
DEFINE_string(
    coeff_mod_bit_sizes, "52",
    "Cefficient modulus. Comma-separated list of bit-lengths of the primes to "
    "be generated."
    "Values must be between 1 and 52. The list length can't be more than 7.");
DEFINE_uint32(
    scale_bit_size, 0,
    "Bit-length for the scaling parameter, which defines encoding precision."
    "Scale will be set as 2^scale_bit_size."
    "Must be between 1 and 52."
    "The default (0) is valid only for benchmark mode and sets it to the "
    "square root of the "
    "last prime of the coefficient modulus.");
DEFINE_uint32(security_lvl, 128, "Security level. One of {0, 128, 192, 256}.");
DEFINE_uint32(ncards, 0,
              "ncards. One of {0, 1}. Default is 0, which runs on CPU.");

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << "LR inference with dataset: " << FLAGS_data << std::endl;
    if (FLAGS_poly_modulus_degree < 1024 || FLAGS_poly_modulus_degree > 16384 ||
        FLAGS_poly_modulus_degree & (FLAGS_poly_modulus_degree - 1) == 0) {
        std::cout << "ERROR: poly_modulus_degree must be a power of 2 between "
                     "1024 and 16384.\n";
        return EXIT_FAILURE;
    }
    std::vector<int> coeff_mod_bit_sizes;
    std::stringstream ss(FLAGS_coeff_mod_bit_sizes);
    for (int i; ss >> i;) {
        coeff_mod_bit_sizes.push_back(i);
        if (ss.peek() == ',') {
            ss.ignore();
        }
    }
    if (coeff_mod_bit_sizes.size() == 0) {
        std::cout << "ERROR: coeff_mod_bit_sizes must contain at least one "
                     "element.\n";
        return EXIT_FAILURE;
    }
    for (int val : coeff_mod_bit_sizes) {
        if (val < 0 || val > 52) {
            std::cout << "ERROR: coeff_mod_bit_sizes values must be between 1 "
                         "and 52.\n";
            return EXIT_FAILURE;
        }
    }
    if ((FLAGS_scale_bit_size == 0) || FLAGS_scale_bit_size > 52) {
        std::cout << "ERROR: scale_bit_size must be between 1 and 52.\n";
        return EXIT_FAILURE;
    }

    if (!(FLAGS_ncards == 0 || FLAGS_ncards == 1)) {
        std::cout << "ERROR: ncards must be 0 for CPU and 1 for single FPGA.\n";
        return EXIT_FAILURE;
    }

    put_dataset_filename(FLAGS_data.c_str());
    put_context_data(FLAGS_poly_modulus_degree, coeff_mod_bit_sizes.data(),
                     coeff_mod_bit_sizes.size(), FLAGS_scale_bit_size,
                     FLAGS_security_lvl, FLAGS_chunk_size);

    char* config_buffer = nullptr;
    size_t sz_config_buffer;
    get_config_buffer(&config_buffer, &sz_config_buffer);

    char* bitstream_dir = nullptr;
    char* env = getenv("FPGA_BITSTREAM");
    if (env) {
        bitstream_dir = env;
    }
    char* kernel = nullptr;
    env = getenv("FPGA_KERNEL");
    if (env) {
        kernel = env;
    }
    size_t batch_size = 4;
    env = getenv("BATCH_SIZE_KEYSWITCH");
    if (env) {
        batch_size = atoi(env);
    }
    size_t ncards = FLAGS_ncards;
    configure(config_buffer, sz_config_buffer, bitstream_dir, kernel,
              batch_size, ncards);

    int sum = 0;
    size_t num_chunks = get_total_num_chunks();
    for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        char* encrypted_input = nullptr;
        size_t sz_encrypted_input;
        get_encrypted_chunk_data(&encrypted_input, &sz_encrypted_input,
                                 chunk_idx);

        // server side
        char* encrypted_input_server = nullptr;
        get_encrypted_input_buffer(&encrypted_input_server, sz_encrypted_input);

        // use copy to simulate the network transfer [client -> server]
        memcpy(encrypted_input_server, encrypted_input, sz_encrypted_input);

        char* encrypted_result = nullptr;
        size_t sz_encrypted_result;
        process(&encrypted_result, &sz_encrypted_result, encrypted_input_server,
                sz_encrypted_input, "Client info");

        // client side
        char* encrypted_result_client = nullptr;
        get_encrypted_result_buffer(&encrypted_result_client,
                                    sz_encrypted_result);

        // use copy to simulate the network transfer [server -> client]
        memcpy(encrypted_result_client, encrypted_result, sz_encrypted_result);

        int* result = nullptr;
        size_t sz_result;
        get_decrypted_decoded_result(
            &result, &sz_result, encrypted_result_client, sz_encrypted_result);
        for (int i = 0; i < sz_result; ++i) {
            sum += result[i];
        }
    }

    return 0;
}
