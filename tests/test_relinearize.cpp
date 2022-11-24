// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "hexl-fpga.h"


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

TEST(HELIBBGV, Relinearize) {
  std::ifstream jsonfile("reLinearize.json");

  json js;
  jsonfile >> js;
  std::vector<uint64_t> c2;

  size_t num_primes = js["c2_after_drop_small_ntt"]["primes"].size();
  auto all_primes = js["all_primes"].get<std::vector<uint64_t>>();
  print_vector("all_primes", all_primes, all_primes.size());
  auto prime_index_set =
      js["c2_after_drop_small_ntt"]["index_set"].get<std::vector<uint8_t>>();
  print_vector("prime_index_set", prime_index_set, prime_index_set.size());

  for (size_t i = 0; i < num_primes; i++) {
    for (auto val :
         js["c2_after_drop_small_ntt"]["data"][i].get<std::vector<uint64_t>>())
      c2.push_back(val);
  }

  size_t coeff_count = js["c2_after_drop_small_ntt"]["data"][0].size();

  std::vector<uint64_t> pi =
      js["c2_after_drop_small_ntt"]["primes"].get<std::vector<uint64_t>>();
  std::vector<uint64_t> qj =
      js["c2_after_drop_small_ntt"]["primes"].get<std::vector<uint64_t>>();

  print_vector("pi", pi, pi.size());
  print_vector("qj", qj, qj.size());

  uint64_t t = 65537;

  std::vector<uint64_t> primes =
      js["digits"][0]["primes"].get<std::vector<uint64_t>>();

  std::vector<uint64_t> keys[4];
  // interleaved in every prime
  size_t num_key_primes = js["wa"][0]["data"].size();
#if DEBUG
  std::cout << "num_key_primes = " << num_key_primes << std::endl;
#endif
  for (int j = 0; j < 4; j++)
    for (size_t i = 0; i < num_key_primes; i++) {
      std::vector<uint64_t> wa =
          js["wa"][j]["data"][i].get<std::vector<uint64_t>>();
      std::vector<uint64_t> wb =
          js["wb"][j]["data"][i].get<std::vector<uint64_t>>();

      for (int n = 0; n < wa.size(); n++) {
        keys[j].push_back(wa[n]);
        keys[j].push_back(wb[n]);
      }
    }

  // put init function into intel::hex::reLinearize();
  // L2::helib::bgv::Relinearize::Init(all_primes, keys[0], keys[1], keys[2],
  //                                   keys[3]);

  // Prepare the expected c0 and c1
  std::vector<uint64_t> expected_output;

  size_t num_c_primes = js["keyswitched_c1"][0]["data"].size();
  for (size_t i = 0; i < num_c_primes; i++) {
    auto c1_0 = js["keyswitched_c1"][0]["data"][i].get<std::vector<uint64_t>>();
    auto c1_1 = js["keyswitched_c1"][1]["data"][i].get<std::vector<uint64_t>>();
    auto c1_2 = js["keyswitched_c1"][2]["data"][i].get<std::vector<uint64_t>>();
    auto c1_3 = js["keyswitched_c1"][3]["data"][i].get<std::vector<uint64_t>>();
    for (size_t j = 0; j < c1_0.size(); j++) {
      expected_output.push_back((c1_0[j] + c1_1[j] + c1_2[j] + c1_3[j]) %
                                primes[i]);
    }
  }

  for (size_t i = 0; i < num_c_primes; i++) {
    auto c2_0 = js["keyswitched_c2"][0]["data"][i].get<std::vector<uint64_t>>();
    auto c2_1 = js["keyswitched_c2"][1]["data"][i].get<std::vector<uint64_t>>();
    auto c2_2 = js["keyswitched_c2"][2]["data"][i].get<std::vector<uint64_t>>();
    auto c2_3 = js["keyswitched_c2"][3]["data"][i].get<std::vector<uint64_t>>();
    for (size_t j = 0; j < c2_0.size(); j++) {
      expected_output.push_back((c2_0[j] + c2_1[j] + c2_2[j] + c2_3[j]) %
                                primes[i]);
    }
  }

  // launch BreakIntoDigits
  {
    unsigned coeff_count = js["c2_after_drop_small_ntt"]["data"][0].size();
#if DEBUG
    std::cout << "coeff_count = " << coeff_count << std::endl;
#endif
    // pi should include the normal (Not ALL, only current) primes and special
    // primes
    std::vector<uint64_t> pi =
        js["expected_digits"][0]["primes"].get<std::vector<uint64_t>>();

    std::vector<unsigned> num_designed_digits_primes{4, 4, 4, 4};
    unsigned num_special_primes = 4;
    unsigned num_digits = 4;

    std::vector<uint8_t> primes_index =
        js["digits"][0]["index_set"].get<std::vector<uint8_t>>();

    // normal primes set
    print_vector("primes_index", primes_index, primes_index.size());

    // Prepare the output memory
    std::vector<uint64_t> output(primes_index.size() * coeff_count * 2);

    // warm up
    // L2::helib::bgv::Relinearize::Relinearize(c2, pi, num_designed_digits_primes,
    //                                          num_special_primes, primes_index,
    //                                          output);
    // L2::helib::bgv::Relinearize::Wait();

    intel::hexl::Relinearize(all_primes.data(), all_primes.size(), 
        keys[0].data(), keys[1].data(), 
        keys[2].data(), keys[3].data(), keys[0].size(),
        c2.data(), c2.size(), pi.data(), pi.size(), 
        num_designed_digits_primes.data(),
        num_designed_digits_primes.size(),
        num_special_primes, primes_index.data(), primes_index.size(),
        output.data(), output.size());
    
    intel::hexl::RelinearizeCompleted();

#if BENCH
    const int test_times = getenv("TIMES") ? atoi(getenv("TIMES")) : 1;

    auto start = std::chrono::steady_clock::now();
    for (int times = 0; times < test_times; times++) {
      L2::helib::bgv::Relinearize::Relinearize(
          c2, pi, num_designed_digits_primes, num_special_primes, primes_index,
          output);
    }
    L2::helib::bgv::Relinearize::Wait();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Average process time: "
              << elapsed_seconds.count() / test_times * 1000 << "ms\n";
#endif
    // compare the result
    ASSERT_EQ(expected_output.size(), output.size());
    for (int i = 0; i < expected_output.size(); i++) {
      ASSERT_EQ(output[i], expected_output[i]) << "at " << i;
    }
  }
}

