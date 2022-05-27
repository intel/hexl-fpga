// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <glob.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <benchmark/benchmark.h>

#include "hexl-fpga.h"

static uint32_t get_iter() {
    char* env = getenv("ITER");
    uint32_t val = 40;
    if (env) {
        val = strtol(env, NULL, 10);
    }
    return val;
}

static uint32_t n_iter = get_iter();

struct KeySwitchTestVector {
    explicit KeySwitchTestVector(const char* json_filename);

    size_t coeff_count;
    size_t decomp_modulus_size;
    size_t key_modulus_size;
    size_t rns_modulus_size;
    size_t key_component_count;

    std::vector<uint64_t> moduli;
    std::vector<uint64_t> modswitch_factors;
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

    for (uint64_t k = 0; k < decomp_modulus_size; k++) {
        std::vector<uint64_t> key_vector;
        for (uint64_t i = 0; i < 2 * key_modulus_size * coeff_count; i++) {
            key_vector.push_back(js["key_vector"][k][i]);
        }
        vectors.push_back(key_vector);
        key_vectors.push_back(const_cast<uint64_t*>(&vectors[k].data()[0]));
    }

    t_target_iter_ptr = js["t_target_iter_ptr"].get<std::vector<uint64_t>>();
    input = js["input"].get<std::vector<uint64_t>>();
}

class keyswitch : public benchmark::Fixture {
public:
    std::vector<std::string> glob(const char* pattern);
    void setup_keyswitch(const std::vector<std::string>& files);
    void bench_keyswitch();

    enum { ITERATIONS = 40 };

private:
    std::vector<KeySwitchTestVector> test_vectors_;
    size_t test_vector_size_;
};

std::vector<std::string> keyswitch::glob(const char* pattern) {
    glob_t glob_result = {0};

    ::glob(pattern, GLOB_TILDE, NULL, &glob_result);

    std::vector<std::string> filenames(
        glob_result.gl_pathv, glob_result.gl_pathv + glob_result.gl_pathc);

    globfree(&glob_result);

    return filenames;
}

void keyswitch::setup_keyswitch(const std::vector<std::string>& files) {
    for (size_t i = 0; i < files.size(); i++) {
        std::cout << "Constructing Test Vector " << i << " from File ... "
                  << files[i] << std::endl;
        test_vectors_.push_back(KeySwitchTestVector(files[i].c_str()));
    }

    test_vector_size_ = test_vectors_.size();
    assert(test_vector_size_ > 0);
}

void keyswitch::bench_keyswitch() {
    intel::hexl::set_worksize_KeySwitch(test_vector_size_ * n_iter);
    for (size_t n = 0; n < n_iter; n++) {
        for (size_t i = 0; i < test_vector_size_; i++) {
            intel::hexl::KeySwitch(test_vectors_[i].input.data(),
                                   test_vectors_[i].t_target_iter_ptr.data(),
                                   test_vectors_[0].coeff_count,
                                   test_vectors_[0].decomp_modulus_size,
                                   test_vectors_[0].key_modulus_size,
                                   test_vectors_[0].rns_modulus_size,
                                   test_vectors_[0].key_component_count,
                                   test_vectors_[0].moduli.data(),
                                   test_vectors_[0].key_vectors.data(),
                                   test_vectors_[0].modswitch_factors.data());
        }
    }
    intel::hexl::KeySwitchCompleted();
}

BENCHMARK_F(keyswitch, 16384_6_7_7_2)
(benchmark::State& state) {
    const char* fname = getenv("KEYSWITCH_DATA_DIR");
    if (!fname) {
        std::cerr << "set env KEYSWITCH_DATA_DIR to the test vector dir"
                  << std::endl;
        exit(1);
    }

    std::string test_file = "/16384_6_7_7_2_*";
    std::string test_fullname = fname + test_file + ".json";
    std::vector<std::string> filesx = glob(test_fullname.c_str());

    std::vector<std::string> files;
    for (size_t i = 0; i < filesx.size(); i++) {
        files.push_back(filesx[i]);
    }

    setup_keyswitch(files);

    // warm up the FPGA kernels specially the twiddle factor dispatching kernel
    bench_keyswitch();

    for (auto st : state) {
        bench_keyswitch();
    }
}
