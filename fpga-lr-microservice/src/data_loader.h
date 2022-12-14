// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

inline bool file_exists(const std::string& fn) {
    struct stat buffer;
    return (stat(fn.c_str(), &buffer) == 0);
}

enum class CSVState { UnquotedField, QuotedField, QuotedQuote };

std::vector<std::string> readCSVRow(const std::string& row);
std::vector<std::vector<std::string>> readCSV(std::istream& in);

enum class DataMode { TRAIN, TEST };

std::vector<std::vector<double>> dataLoader(std::string dataset_name,
                                            DataMode mode);
enum class WeightType { W, V };

std::vector<double> weightsLoaderCSV(std::string dataset_name,
                                     WeightType wtype = WeightType::W);

void splitWeights(const std::vector<double>& rawweights,
                  std::vector<double>& out_weight, double& out_bias);

void splitData(const std::vector<std::vector<double>>& rawdata,
               std::vector<std::vector<double>>& out_data,
               std::vector<double>& out_target);

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
    out << "{";
    size_t last = v.size() - 1;
    for (size_t i = 0; i < v.size(); ++i) {
        out << v[i];
        if (i != last) out << ", ";
    }
    out << "}";
    return out;
}

std::vector<std::vector<double>> transpose(
    std::vector<std::vector<double>>& data);
