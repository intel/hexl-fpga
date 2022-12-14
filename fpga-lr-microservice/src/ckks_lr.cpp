// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "seal/seal.h"
#include "gflags/gflags.h"
#include "buffer.h"
#include "client.h"
#include "server.h"
#include "service.h"

using namespace intel;
using namespace intel::he;
using namespace intel::he::heseal;

DEFINE_string(data, "homecreditdefaultrisk",
              "Dataset name to test - choose one of heartdisease, "
              "homecreditdefaultrisk, sklearn_breastcancer");
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

    auto buffer = std::make_shared<Buffer>();
    Client client(buffer, FLAGS_data, FLAGS_chunk_size,
                  FLAGS_poly_modulus_degree, coeff_mod_bit_sizes,
                  FLAGS_scale_bit_size, sec_lvl);
    Server server(buffer);

    Service service(client, server);

    bool succ = service.setup();
    if (succ) {
        service.configure();
        service.process();
        service.teardown();
    } else {
        std::cerr
            << "Setup service failed due to unavailable resources, try later"
            << std::endl;
    }

    return 0;
}
