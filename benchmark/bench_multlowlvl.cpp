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


class multlowlvl: public benchmark::Fixture {
public:
    void load_multlowlvl_data(std::string&);
    void bench_multlowlvl();

private:
    std::vector<DataLoader> data_loader_vec;
    std::vector<TensorProductLoader> tensor_product_loader_vec;
    std::vector<std::vector<uint64_t>> output_vec;
};


void multlowlvl::load_multlowlvl_data(std::string& multlowlvl_json) {
    std::ifstream jsonfile(multlowlvl_json.c_str());
    json js;
    jsonfile >> js;
    data_loader_vec.emplace_back(DataLoader(js, "c0_0_ntt", "tensorProduct_result_part0"));
    data_loader_vec.emplace_back(DataLoader(js, "c0_1_ntt", "tensorProduct_result_part0"));
    data_loader_vec.emplace_back(DataLoader(js, "c1_0_ntt", "tensorProduct_result_part0"));
    data_loader_vec.emplace_back(DataLoader(js, "c1_1_ntt", "tensorProduct_result_part0"));

    tensor_product_loader_vec.emplace_back(TensorProductLoader(js));

    auto expected_output1 = tensor_product_loader_vec[0].expected_output1;
    auto expected_output2 = tensor_product_loader_vec[0].expected_output2;
    auto expected_output3 = tensor_product_loader_vec[0].expected_output3;

    output_vec.resize(3);
    output_vec[0].resize(expected_output1.size());
    output_vec[1].resize(expected_output2.size());
    output_vec[2].resize(expected_output3.size());
}


void multlowlvl::bench_multlowlvl() {
    intel::hexl::set_worksize_MultLowLvl(1);
    intel::hexl::MultLowLvl(data_loader_vec[0].input.data(), data_loader_vec[1].input.data(),
                                data_loader_vec[0].primes_index.size(), data_loader_vec[0].primes_index.data(),
                                data_loader_vec[2].input.data(), data_loader_vec[3].input.data(), 
                                data_loader_vec[2].primes_index.size(), data_loader_vec[2].primes_index.data(),
                                data_loader_vec[0].t, 65536,
                                output_vec[0].data(), output_vec[1].data(), output_vec[2].data(),
                                tensor_product_loader_vec[0].primes_index.size(), tensor_product_loader_vec[0].primes_index.data(),
                                data_loader_vec[0].all_primes.size(), data_loader_vec[0].all_primes.data());
    intel::hexl::MultLowLvlCompleted();
}

BENCHMARK_F(multlowlvl, fpga_multlowlvl_65536)(benchmark::State& state) {
    
    std::string multlowlovl_input("multLowLvl.json");
    load_multlowlvl_data(multlowlovl_input);
    for (auto st :state) {
        bench_multlowlvl();
    }
}
