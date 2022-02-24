// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <vector>

#include <glob.h>
#include "gtest/gtest.h"

#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <string>

#include "hexl-fpga.h"

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

class dyadic_multiply_keyswitch_test {
public:
    explicit dyadic_multiply_keyswitch_test(uint64_t num_dyadic_multiply,
                                            uint64_t num_moduli,
                                            uint64_t coeff_count,
                                            const std::string& test_fullname);
    dyadic_multiply_keyswitch_test(const dyadic_multiply_keyswitch_test& test) =
        delete;

    void test_dyadic_multiply_keyswitch();
    void check_results();

    class dyadic_multiply_test {
    public:
        explicit dyadic_multiply_test(uint64_t n_dyadic_multiply,
                                      uint64_t n_moduli, uint64_t coeff);
        dyadic_multiply_test(const dyadic_multiply_test& test) = delete;

        void test_dyadic_multiply();
        void check_results();

    private:
        uint64_t num_dyadic_multiply;
        uint64_t num_moduli;
        uint64_t coeff_count;
        std::vector<uint64_t> moduli;
        std::vector<uint64_t> op1;
        std::vector<uint64_t> op2;
        std::vector<uint64_t> exp_out;
        std::vector<uint64_t> out;
    };

    class keyswitch_test {
    public:
        explicit keyswitch_test(const std::string& test_fullname);
        keyswitch_test(const keyswitch_test& test) = delete;

        void test_keyswitch();
        void check_results();

    private:
        std::vector<std::string> glob(const char* pattern);
        std::vector<KeySwitchTestVector> test_vectors;
    };

private:
    dyadic_multiply_test dyadic_multiply;
    keyswitch_test keyswitch;
};

dyadic_multiply_keyswitch_test::dyadic_multiply_keyswitch_test(
    uint64_t num_dyadic_multiply, uint64_t num_moduli, uint64_t coeff_count,
    const std::string& test_fullname)
    : dyadic_multiply(num_dyadic_multiply, num_moduli, coeff_count),
      keyswitch(test_fullname) {}

void dyadic_multiply_keyswitch_test::test_dyadic_multiply_keyswitch() {
    dyadic_multiply.test_dyadic_multiply();
    keyswitch.test_keyswitch();
}

void dyadic_multiply_keyswitch_test::check_results() {
    dyadic_multiply.check_results();
    keyswitch.check_results();
}

dyadic_multiply_keyswitch_test::dyadic_multiply_test::dyadic_multiply_test(
    uint64_t n_dyadic_multiply, uint64_t n_moduli, uint64_t coeff)
    : num_dyadic_multiply(n_dyadic_multiply),
      num_moduli(n_moduli),
      coeff_count(coeff) {
    for (uint64_t b = 0; b < num_dyadic_multiply; b++) {
        for (uint64_t m = 0; m < num_moduli; m++) {
            moduli.push_back((b + m + 1) * 10);
        }

        for (uint64_t m = 0; m < num_moduli; m++) {
            for (uint64_t i = 0; i < coeff_count; i++) {
                op1.push_back(b + i + 1 + m * coeff_count);
                op2.push_back(b + i + 2 + m * coeff_count);
            }
        }

        for (uint64_t m = 0; m < num_moduli; m++) {
            for (uint64_t i = 0; i < coeff_count; i++) {
                op1.push_back(b + i + 11 + m * coeff_count);
                op2.push_back(b + i + 22 + m * coeff_count);
            }
        }

        for (uint64_t m = 0; m < num_moduli; m++) {
            uint64_t poly0_offset =
                b * num_moduli * coeff_count * 2 + m * coeff_count;
            uint64_t m_offset = b * num_moduli + m;
            for (uint64_t i = 0; i < coeff_count; i++) {
                uint64_t a = op1[poly0_offset + i] * op2[poly0_offset + i];
                exp_out.push_back(a % moduli[m_offset]);
            }
        }
        for (uint64_t m = 0; m < num_moduli; m++) {
            uint64_t poly0_offset =
                b * num_moduli * coeff_count * 2 + m * coeff_count;
            uint64_t poly1_offset = b * num_moduli * coeff_count * 2 +
                                    (m + num_moduli) * coeff_count;
            uint64_t m_offset = b * num_moduli + m;
            for (uint64_t i = 0; i < coeff_count; i++) {
                uint64_t x = op1[poly0_offset + i] * op2[poly1_offset + i];
                uint64_t y = op1[poly1_offset + i] * op2[poly0_offset + i];
                exp_out.push_back((x + y) % moduli[m_offset]);
            }
        }
        for (uint64_t m = 0; m < num_moduli; m++) {
            uint64_t poly1_offset = b * num_moduli * coeff_count * 2 +
                                    (m + num_moduli) * coeff_count;
            uint64_t m_offset = b * num_moduli + m;
            for (uint64_t i = 0; i < coeff_count; i++) {
                uint64_t x = op1[poly1_offset + i] * op2[poly1_offset + i];
                exp_out.push_back(x % moduli[m_offset]);
            }
        }
    }
    out.resize(num_dyadic_multiply * 3 * num_moduli * coeff_count, 0);
}

void dyadic_multiply_keyswitch_test::dyadic_multiply_test::
    test_dyadic_multiply() {
    intel::hexl::set_worksize_DyadicMultiply(num_dyadic_multiply);
    for (uint64_t n = 0; n < num_dyadic_multiply; n++) {
        uint64_t* pout = &out[0] + n * num_moduli * coeff_count * 3;
        uint64_t* pop1 = &op1[0] + n * num_moduli * coeff_count * 2;
        uint64_t* pop2 = &op2[0] + n * num_moduli * coeff_count * 2;
        uint64_t* pmoduli = &moduli[0] + n * num_moduli;
        intel::hexl::DyadicMultiply(pout, pop1, pop2, coeff_count, pmoduli,
                                    num_moduli);
    }
    intel::hexl::DyadicMultiplyCompleted();
}

void dyadic_multiply_keyswitch_test::dyadic_multiply_test::check_results() {
    ASSERT_EQ(out, exp_out);
}

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
    input = js["input"].get<std::vector<uint64_t>>();
    expected_output = js["expected_output"].get<std::vector<uint64_t>>();
}

std::vector<std::string> dyadic_multiply_keyswitch_test::keyswitch_test::glob(
    const char* pattern) {
    glob_t glob_result = {0};

    ::glob(pattern, GLOB_TILDE, NULL, &glob_result);

    std::vector<std::string> filenames(
        glob_result.gl_pathv, glob_result.gl_pathv + glob_result.gl_pathc);

    globfree(&glob_result);

    return filenames;
}

dyadic_multiply_keyswitch_test::keyswitch_test::keyswitch_test(
    const std::string& test_fullname) {
    std::vector<std::string> test_vector_files;
    for (size_t n = 0; n < 2; n++) {
        std::vector<std::string> filesx = glob(test_fullname.c_str());

        for (size_t i = 0; i < filesx.size(); i++) {
            test_vector_files.push_back(filesx[i]);
        }
    }
    for (size_t i = 0; i < test_vector_files.size(); i++) {
        std::cout << "Constructing Test Vector " << i << " from File ... "
                  << test_vector_files[i] << std::endl;
        test_vectors.push_back(
            KeySwitchTestVector(test_vector_files[i].c_str()));
    }
}

void dyadic_multiply_keyswitch_test::keyswitch_test::test_keyswitch() {
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
}

void dyadic_multiply_keyswitch_test::keyswitch_test::check_results() {
    size_t test_vector_size = test_vectors.size();
    assert(test_vector_size > 0);
    for (size_t i = 0; i < test_vector_size; i++) {
        ASSERT_EQ(test_vectors[i].input, test_vectors[i].expected_output);
    }
}

TEST(dyadic_multiply_keyswitch_test, batch_16384_6_7_7_2) {
    uint64_t coeff_count = 16384;
    uint64_t num_moduli = 6;
    uint64_t num_dyadic_multiply = 16;

    const char* fname = getenv("KEYSWITCH_DATA_DIR");
    if (!fname) {
        std::cerr << "set env(KEYSWITCH_DATA_DIR) to the test vector dir"
                  << std::endl;
        exit(1);
    }
    std::string test_file = "/16384_6_7_7_2_*";
    std::string test_fullname = fname + test_file + ".json";

    dyadic_multiply_keyswitch_test test(coeff_count / 2, num_moduli,
                                        num_dyadic_multiply, test_fullname);
    test.test_dyadic_multiply_keyswitch();
    test.check_results();
}
