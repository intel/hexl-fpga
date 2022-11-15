// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//#include <L2/helib_bgv.h>
#include <chrono>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "hexl-fpga.h"

//#include "multLowLvl_runtime.hpp"

using json = nlohmann::json;

template <class T>
static void print_vector(const char *tag, std::vector<T> data,
                         size_t max_size = 32) {
  std::cout << tag << ": ";
  for (size_t i = 0; i < std::min(data.size(), max_size); i++) {
    // cout can't print unsigned char, so cast it
    std::cout << +data[i] << ", ";
  }
  std::cout << std::endl;
}

class DataLoader {
 public:
  DataLoader(const json &js, const char *in, const char *out) {
    auto &js_input = js[in];
    auto &js_output = js[out];

    size_t num_primes = js_input["primes"].size();
    auto coeff_count = js_input["data"][0].size();
    std::cout << "coeff_count = " << coeff_count << std::endl;

    all_primes = js["all_primes"].get<std::vector<uint64_t>>();
    std::vector<uint64_t> pi = js_input["primes"].get<std::vector<uint64_t>>();
    auto qj = js_output["primes"].get<std::vector<uint64_t>>();

    primes_index = js_input["index_set"].get<std::vector<uint8_t>>();
    auto qj_prime_index = js_output["index_set"].get<std::vector<uint8_t>>();

    print_vector("pi", pi, pi.size());
    print_vector("qj", qj, qj.size());

    print_vector("primes_index", primes_index, primes_index.size());
    print_vector("qj_prime_index", qj_prime_index, qj_prime_index.size());

    // compute the num of dropped primes
    int num_dropped_primes = 0;
    for (auto prime : pi) {
      num_dropped_primes +=
          (std::find(qj.begin(), qj.end(), prime) == qj.end());
    }

    // reoder pi, put the dropped primes to the beginning
    for (size_t i = pi.size() - num_dropped_primes; i < pi.size(); i++) {
      for (auto val : js_input["data"][i].get<std::vector<uint64_t>>()) {
        input.push_back(val);
      }
    }
    for (size_t i = 0; i < pi.size() - num_dropped_primes; i++) {
      for (auto val : js_input["data"][i].get<std::vector<uint64_t>>()) {
        input.push_back(val);
      }
    }

    t = 65537;
  }

  std::vector<uint64_t> all_primes;
  std::vector<uint8_t> primes_index;
  std::vector<uint64_t> input;
  uint64_t t;
};

class TensorProductLoader {
 public:
  TensorProductLoader(const json &js) {
    auto primes = js["all_primes"].get<std::vector<uint64_t>>();
    primes_index = js["tensorProduct_result_part0"]["index_set"]
                       .get<std::vector<uint8_t>>();
    size_t num_primes = primes_index.size();

    print_vector("primes", primes);
    print_vector("primes_index", primes_index);

    for (size_t i = 0; i < num_primes; i++) {
      for (auto val : js["tensorProduct_result_part0"]["data"][i]
                          .get<std::vector<uint64_t>>())
        expected_output1.push_back(val);
      for (auto val : js["tensorProduct_result_part1"]["data"][i]
                          .get<std::vector<uint64_t>>())
        expected_output2.push_back(val);
      for (auto val : js["tensorProduct_result_part2"]["data"][i]
                          .get<std::vector<uint64_t>>())
        expected_output3.push_back(val);
    }
  }
  std::vector<uint8_t> primes_index;
  std::vector<uint64_t> expected_output1;
  std::vector<uint64_t> expected_output2;
  std::vector<uint64_t> expected_output3;
};

TEST(HELIBBGV, MultLowLvl) {
  std::ifstream jsonfile("multLowLvl.json");
  json js;
  jsonfile >> js;

  DataLoader data_loader1(js, "c0_0_ntt", "tensorProduct_result_part0");
  DataLoader data_loader2(js, "c0_1_ntt", "tensorProduct_result_part0");
  DataLoader data_loader3(js, "c1_0_ntt", "tensorProduct_result_part0");
  DataLoader data_loader4(js, "c1_1_ntt", "tensorProduct_result_part0");

  TensorProductLoader tensor_product_loader(js);

  auto expected_output1 = tensor_product_loader.expected_output1;
  auto expected_output2 = tensor_product_loader.expected_output2;
  auto expected_output3 = tensor_product_loader.expected_output3;

  std::vector<uint64_t> output1(expected_output1.size());
  std::vector<uint64_t> output2(expected_output2.size());
  std::vector<uint64_t> output3(expected_output3.size());

  std::cout << "Init" << std::endl;
  //Init(data_loader1.all_primes);

  //L2::helib::bgv::Timer timer("MultLowLvl");
  // MultLowLvl(data_loader1.input, data_loader2.input,
  //                            data_loader1.primes_index, data_loader3.input,
  //                            data_loader4.input, data_loader3.primes_index,
  //                            data_loader1.t, output1, output2, output3,
  //                            tensor_product_loader.primes_index);
  intel::hexl::set_worksize_MultLowLvl(1);
  std::cout << "plainText: " << data_loader1.t << std::endl;
  intel::hexl::MultLowLvl(data_loader1.input.data(), data_loader2.input.data(),
                             data_loader1.primes_index.size(), data_loader1.primes_index.data(),
                             data_loader3.input.data(), data_loader4.input.data(), 
                             data_loader3.primes_index.size(), data_loader3.primes_index.data(),
                             data_loader1.t, 65536,
                             output1.data(), output2.data(), output3.data(),
                             tensor_product_loader.primes_index.size(), tensor_product_loader.primes_index.data(),
                             data_loader1.all_primes.data(), data_loader1.all_primes.size());
  intel::hexl::MultLowLvlCompleted();
  //timer.stop();

  for (int i = 0; i < expected_output1.size(); i++) {
    ASSERT_EQ(output1[i], expected_output1[i]) << "output 1 at " << i;
  }
  for (int i = 0; i < expected_output2.size(); i++) {
    ASSERT_EQ(output2[i], expected_output2[i]) << "output 2 at " << i;
  }
  for (int i = 0; i < expected_output3.size(); i++) {
    ASSERT_EQ(output3[i], expected_output3[i]) << "output 3 at " << i;
  }
}
