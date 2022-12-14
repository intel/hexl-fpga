// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <seal/seal.h>

#include <algorithm>
#include <string>
#include <memory>
#include <utility>
#include <vector>
#include "seal_ckks_context.h"
#include <gsl/span>

namespace intel {
namespace he {
namespace heseal {

void SealCKKSContext::printParameters() {
    auto& context_data = *(m_seal_context->key_context_data());

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
    std::cout << "|   scale: " << static_cast<long>(m_scale) << '\n';
    std::cout << "|   security_lvl: "
              << static_cast<int>(context_data.qualifiers().sec_level) << '\n';
    std::cout << "\\" << std::endl << std::endl;
}

std::vector<seal::Plaintext> SealCKKSContext::encodeVector(
    const gsl::span<const double>& values, size_t batch_size) {
    size_t total_chunks =
        values.size() / batch_size + (values.size() % batch_size == 0 ? 0 : 1);
    size_t last_chunk_size = values.size() % batch_size == 0
                                 ? batch_size
                                 : values.size() % batch_size;

    std::vector<seal::Plaintext> ret(total_chunks);
#pragma omp parallel for
    for (size_t i = 0; i < total_chunks; ++i) {
        size_t actual_chunk_size =
            (i == total_chunks - 1) ? last_chunk_size : batch_size;
        gsl::span data_chunk(&values[i * batch_size], actual_chunk_size);
        m_encoder->encode(data_chunk, m_scale, ret[i]);
    }
    return ret;
}

std::vector<seal::Plaintext> SealCKKSContext::encodeVector(
    const gsl::span<const double>& v) {
    std::size_t slot_count = m_encoder->slot_count();
    std::size_t total_chunks =
        v.size() / slot_count + (v.size() % slot_count == 0 ? 0 : 1);
    gsl::span<const double> data = v;
    std::vector<seal::Plaintext> retval;
    retval.reserve(total_chunks);
    while (!data.empty()) {
        std::size_t actual_chunk_size =
            (data.size() > slot_count ? slot_count : data.size());
        gsl::span data_chunk = data.first(actual_chunk_size);
        data = data.last(data.size() - actual_chunk_size);
        seal::Plaintext plain;
        m_encoder->encode(data_chunk, m_scale, plain);
        retval.emplace_back(std::move(plain));
    }
    return retval;
}

std::vector<double> SealCKKSContext::decodeVector(
    const std::vector<seal::Plaintext>& plain, size_t batch_size) {
    std::vector<double> ret(plain.size() * batch_size);
#pragma omp parallel for
    for (size_t i = 0; i < plain.size(); ++i) {
        std::vector<double> tmp;
        m_encoder->decode(plain[i], tmp);
        std::copy(tmp.begin(), tmp.begin() + batch_size,
                  ret.begin() + i * batch_size);
    }
    return ret;
}

std::vector<double> SealCKKSContext::decodeVector(
    const std::vector<seal::Plaintext>& plain) {
    std::size_t slot_count = m_encoder->slot_count();
    std::vector<double> ret(plain.size() * slot_count);
#pragma omp parallel for
    for (size_t i = 0; i < plain.size(); ++i) {
        std::vector<double> tmp;
        m_encoder->decode(plain[i], tmp);
        std::size_t min_size = std::min(slot_count, tmp.size());
        std::copy(tmp.begin(), tmp.begin() + min_size,
                  ret.begin() + i * slot_count);
    }
    return ret;
}

std::vector<seal::Ciphertext> SealCKKSContext::encryptVector(
    const std::vector<seal::Plaintext>& plain) {
    std::vector<seal::Ciphertext> ret(plain.size());
#pragma omp parallel for
    for (size_t i = 0; i < plain.size(); ++i)
        m_encryptor->encrypt(plain[i], ret[i]);
    return ret;
}

std::vector<seal::Plaintext> SealCKKSContext::decryptVector(
    const std::vector<seal::Ciphertext>& cipher) {
    std::vector<seal::Plaintext> ret(cipher.size());
#pragma omp parallel for
    for (size_t i = 0; i < cipher.size(); ++i)
        m_decryptor->decrypt(cipher[i], ret[i]);
    return ret;
}

}  // namespace heseal
}  // namespace he
}  // namespace intel
