// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "seal/seal.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>
#include <cassert>
#include "gflags/gflags.h"
#include "fpga_context.h"

bool check_results(const std::vector<double>& output,
                   const std::vector<double>& expected, double precision) {
    if (output.size() != expected.size()) {
        std::cout << "ERROR: Functionally incorrect: Input and ouput vectors "
                     "have different sizes.\n\n";
        return false;
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(output[i] - expected[i]) >= precision) {
            std::cout << "expected[" << i << "]=" << expected[i] << " output["
                      << i << "]=" << output[i] << '\n';
            std::cout << "ERROR: Functionally incorrect, values differ between "
                         "expected and output.\n\n";
            return false;
        }
    }

    return true;
}

std::vector<double> calculate_expected(const std::vector<double>& input) {
    std::vector<double> expected(input.size());

    std::transform(input.begin(), input.end(), expected.begin(),
                   [](double d) { return d * d; });
    std::rotate(expected.begin(), expected.begin() + 1, expected.end());

    return expected;
}

void print_parameters(const seal::SEALContext& context, const double scale) {
    auto& context_data = *context.key_context_data();

    std::cout << "\n/\n";
    std::cout << "| Parameters :\n";

    std::string scheme_name;
    switch (context_data.parms().scheme()) {
    case seal::scheme_type::ckks:
        scheme_name = "CKKS";
        break;
    default:
        throw std::invalid_argument("Unsupported scheme.");
        break;
    }
    std::cout << "|   scheme: " << scheme_name << '\n';
    std::cout << "|   poly_modulus_degree: "
              << context_data.parms().poly_modulus_degree() << '\n';

    std::cout << "|   coeff_modulus size: ";
    std::cout << context_data.total_coeff_modulus_bit_count() << " (";
    auto coeff_modulus = context_data.parms().coeff_modulus();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    for (std::size_t i = 0; i < coeff_modulus_size - 1; i++) {
        std::cout << coeff_modulus[i].bit_count() << " + ";
    }
    std::cout << coeff_modulus.back().bit_count();
    std::cout << ") bits (" << coeff_modulus.size() << " elements)\n";

    std::cout << "|   scale: " << static_cast<long>(scale) << '\n';
    std::cout << "|   security_lvl: "
              << static_cast<int>(context_data.qualifiers().sec_level) << '\n';
    std::cout << "\\" << std::endl;
}

void run_internal(const seal::SEALContext& context, const double scale,
                  double data_bound, const unsigned test_loops,
                  const double test_precision) {
    assert(context.using_keyswitching());

    print_parameters(context, scale);

    if (data_bound == 0) {
        auto& context_data = *context.key_context_data();
        auto coeff_modulus = context_data.parms().coeff_modulus();
        std::vector<int> coeff_mod_bit_sizes;
        for (auto&& mod : coeff_modulus) {
            coeff_mod_bit_sizes.push_back(mod.bit_count());
        }
        auto data_bound_bit_size =
            *std::min_element(coeff_mod_bit_sizes.cbegin(),
                              coeff_mod_bit_sizes.cend()) /
            2;
        data_bound = static_cast<double>(1L << data_bound_bit_size);
    }
    std::uniform_real_distribution<double> distr(-data_bound, data_bound);

    seal::KeyGenerator keygen(context);
    auto secret_key = keygen.secret_key();
    seal::PublicKey public_key;
    keygen.create_public_key(public_key);
    seal::RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    seal::GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);

    seal::CKKSEncoder encoder(context);
    seal::Encryptor encryptor(context, public_key);
    seal::Evaluator evaluator(context);
    seal::Decryptor decryptor(context, secret_key);

    std::random_device rd;
    std::mt19937 gen(rd());

    for (unsigned loop_count = 0; loop_count < test_loops; ++loop_count) {
        std::vector<double> input_clear(encoder.slot_count());
        // Fill input vector with random data
        for (size_t i = 0; i < input_clear.size(); i++) {
            input_clear[i] = static_cast<double>(distr(gen));
        }

        // [Encoding]
        seal::Plaintext input_plain;
        encoder.encode(input_clear, scale, input_plain);

        // [Encryption]
        seal::Ciphertext encrypted(context);
        encryptor.encrypt(input_plain, encrypted);

        // [Multiply]
        evaluator.multiply_inplace(encrypted, encrypted);

        // [Relinearize]
        evaluator.relinearize_inplace(encrypted, relin_keys);

        // [Rescale]
        evaluator.rescale_to_next_inplace(encrypted);

        // [Rotate Vector]
        evaluator.rotate_vector_inplace(encrypted, 1, gal_keys);

        // [Decryption]
        seal::Plaintext output_plain;
        decryptor.decrypt(encrypted, output_plain);

        // [Decoding]
        std::vector<double> output_clear;
        encoder.decode(output_plain, output_clear);

        std::vector<double> expected = calculate_expected(input_clear);

        if (check_results(output_clear, expected, test_precision)) {
            std::cout << "SUCCESS: Test passed... " << loop_count << std::endl;
        } else {
            std::cout << "FAIL: Test failed... " << loop_count << std::endl;
        }
    }
}

void run(const size_t poly_modulus_degree,
         const std::vector<int>& coeff_mod_bit_sizes,
         const unsigned scale_bit_size, const seal::sec_level_type sec_lvl,
         const double data_bound, const unsigned test_loops,
         const double test_precision) {
    seal::EncryptionParameters params(seal::scheme_type::ckks);
    params.set_poly_modulus_degree(poly_modulus_degree);
    params.set_coeff_modulus(
        seal::CoeffModulus::Create(poly_modulus_degree, coeff_mod_bit_sizes));

    double scale = static_cast<double>(1UL << scale_bit_size);

    seal::SEALContext context(params, true, sec_lvl);

    run_internal(context, scale, data_bound, test_loops, test_precision);
}

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
DEFINE_double(data_bound, 0,
              "Limit for the random data generated for the test input vector."
              "Symmetric in the positive and negative axes."
              "The default (0) sets it to a power of two, where the power is "
              "the minimum of "
              "coeff_mod_bit_sizes, divided by two.");
DEFINE_uint32(test_loops, 1,
              "Amount of times to run the test. Must be between 1 and 10000.");
DEFINE_double(test_precision, 0.00005,
              "Precision for verifying test results. Default is 0.00005");

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

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

    seal::sec_level_type sec_lvl;
    switch (FLAGS_security_lvl) {
    case 0:
        sec_lvl = seal::sec_level_type::none;
        break;
    case 128:
        sec_lvl = seal::sec_level_type::tc128;
        break;
    case 192:
        sec_lvl = seal::sec_level_type::tc192;
        break;
    case 256:
        sec_lvl = seal::sec_level_type::tc256;
        break;
    default:
        std::cout << "ERROR: security_lvl must be one of {0, 128, 192, 256}.\n";
        return EXIT_FAILURE;
    }
    if (FLAGS_data_bound < 0) {
        std::cout << "ERROR: data_bound can't be negative.\n";
        return EXIT_FAILURE;
    }

#ifdef HEXL_FPGA
    fpga_context context;
#endif

    run(FLAGS_poly_modulus_degree, coeff_mod_bit_sizes, FLAGS_scale_bit_size,
        sec_lvl, FLAGS_data_bound, FLAGS_test_loops, FLAGS_test_precision);

    return 0;
}
