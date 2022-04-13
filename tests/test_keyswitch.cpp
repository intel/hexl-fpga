// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <glob.h>
#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <vector>

#include "hexl-fpga.h"

static uint32_t get_n() {
    char* env = getenv("N");
    // the valid values are 1024, 2048, 4096, 8192, 16384
    // the default is 16384
    uint32_t val = 16384;
    if (env) {
        val = strtol(env, NULL, 10);
        assert((val == 1024) || (val == 2048) || (val == 4096) ||
               (val == 8192) || (val == 16384));
    }
    return val;
}

static uint32_t n_size = get_n();

struct KeySwitchTestVector {
    explicit KeySwitchTestVector(const char* json_filename);

    size_t coeff_count;
    size_t decomp_modulus_size;
    size_t key_modulus_size;
    size_t rns_modulus_size;
    size_t key_component_count;

    std::vector<uint64_t> moduli;
    std::vector<uint64_t> modswitch_factors;
    std::vector<uint64_t> twiddle_factors;
    std::vector<const uint64_t*> key_vectors;

    std::vector<uint64_t> t_target_iter_ptr;
    std::vector<uint64_t> input;
    std::vector<uint64_t> expected_output;

    std::vector<std::vector<uint64_t>> vectors;

private:
    std::ifstream json_file;
    nlohmann::json js;
};

KeySwitchTestVector::KeySwitchTestVector(const char* json_filename)
    : json_file(json_filename) {
    json_file >> js;

    coeff_count = js["coeff_count"].get<uint64_t>();
    decomp_modulus_size = js["decomp_modulus_size"].get<uint64_t>();
    key_modulus_size = js["key_modulus_size"].get<uint64_t>();
    rns_modulus_size = js["rns_modulus_size"].get<uint64_t>();
    key_component_count = js["key_component_count"].get<uint64_t>();

    moduli = js["moduli"].get<std::vector<uint64_t>>();
    modswitch_factors = js["modswitch_factors"].get<std::vector<uint64_t>>();

    bool twiddles = js.contains("inv_root_of_unity_powers");
    twiddles &= js.contains("precon64_inv_root_of_unity_powers");
    twiddles &= js.contains("root_of_unity_powers");
    twiddles &= js.contains("precon64_root_of_unity_powers");

    if (twiddles) {
        for (uint64_t k = 0; k < key_modulus_size; k++) {
            for (uint64_t i = 0; i < coeff_count; i++) {
                twiddle_factors.push_back(js["inv_root_of_unity_powers"][k][i]);
            }
            for (uint64_t i = 0; i < coeff_count; i++) {
                twiddle_factors.push_back(
                    js["precon64_inv_root_of_unity_powers"][k][i]);
            }
            for (uint64_t i = 0; i < coeff_count; i++) {
                twiddle_factors.push_back(js["root_of_unity_powers"][k][i]);
            }
            for (uint64_t i = 0; i < coeff_count; i++) {
                twiddle_factors.push_back(
                    js["precon64_root_of_unity_powers"][k][i]);
            }
        }
    }

    for (uint64_t k = 0; k < decomp_modulus_size; k++) {
        std::vector<uint64_t> key_vector;
        for (uint64_t i = 0; i < 2 * key_modulus_size * coeff_count; i++) {
            key_vector.push_back(js["key_vector"][k][i]);
        }
        vectors.push_back(key_vector);
        key_vectors.push_back(const_cast<uint64_t*>(&vectors[k].data()[0]));
    }

    t_target_iter_ptr = js["t_target_iter_ptr"].get<std::vector<uint64_t>>();
    for (uint64_t k = 0; k < decomp_modulus_size * coeff_count; k++) {
        input.push_back(js["input"][k]);
        input.push_back(js["input"][k + decomp_modulus_size * coeff_count]);
        expected_output.push_back(js["expected_output"][k]);
        expected_output.push_back(
            js["expected_output"][k + decomp_modulus_size * coeff_count]);
    }
}

std::vector<std::string> glob(const char* pattern) {
    glob_t glob_result = {0};

    ::glob(pattern, GLOB_TILDE, NULL, &glob_result);

    std::vector<std::string> filenames(
        glob_result.gl_pathv, glob_result.gl_pathv + glob_result.gl_pathc);

    globfree(&glob_result);

    return filenames;
}

void test_KeySwitch(const std::vector<std::string>& files) {
    std::vector<KeySwitchTestVector> test_vectors;

    for (size_t i = 0; i < files.size(); i++) {
        std::cout << "Constructing Test Vector " << i << " from File ... "
                  << files[i] << std::endl;
        test_vectors.push_back(KeySwitchTestVector(files[i].c_str()));
    }

    size_t test_vector_size = test_vectors.size();
    assert(test_vector_size > 0);

    intel::hexl::set_worksize_KeySwitch(test_vector_size);
    for (size_t i = 0; i < test_vector_size; i++) {
        intel::hexl::KeySwitch(
            test_vectors[i].input.data(),
            test_vectors[i].t_target_iter_ptr.data(),
            test_vectors[0].coeff_count, test_vectors[0].decomp_modulus_size,
            test_vectors[0].key_modulus_size, test_vectors[0].rns_modulus_size,
            test_vectors[0].key_component_count, test_vectors[0].moduli.data(),
            test_vectors[0].key_vectors.data(),
            test_vectors[0].modswitch_factors.data(),
            test_vectors[0].twiddle_factors.data());
    }
    intel::hexl::KeySwitchCompleted();

    for (size_t i = 0; i < files.size(); i++) {
        ASSERT_EQ(test_vectors[i].input, test_vectors[i].expected_output);
    }
}

TEST(KeySwitch, batch_6_7_7_2) {
    const char* fname = getenv("KEYSWITCH_DATA_DIR");
    if (!fname) {
        std::cerr << "set env KEYSWITCH_DATA_DIR to the test vector dir"
                  << std::endl;
        exit(1);
    }

    std::vector<std::string> files;

    for (size_t n = 0; n < 2; n++) {
        std::string test_file = "/" + std::to_string(n_size) + "_6_7_7_2_*";
        std::string test_fullname = fname + test_file + ".json";
        std::vector<std::string> filesx = glob(test_fullname.c_str());

        for (size_t i = 0; i < filesx.size(); i++) {
            files.push_back(filesx[i]);
        }
    }

    test_KeySwitch(files);
}

TEST(KeySwitch, batch_5_7_6_2_2) {
    const char* fname = getenv("KEYSWITCH_DATA_DIR");
    if (!fname) {
        std::cerr << "set env KEYSWITCH_DATA_DIR to the test vector dir"
                  << std::endl;
        exit(1);
    }

    std::vector<std::string> files;

    for (size_t n = 0; n < 2; n++) {
        std::string test_file = "/" + std::to_string(n_size) + "_5_7_6_2_*";
        std::string test_fullname = fname + test_file + ".json";
        std::vector<std::string> filesx = glob(test_fullname.c_str());

        for (size_t i = 0; i < filesx.size(); i++) {
            files.push_back(filesx[i]);
        }
    }

    test_KeySwitch(files);
}
