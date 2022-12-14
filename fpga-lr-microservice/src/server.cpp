// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <memory>
#include <cassert>
#include <cstdlib>
#include "seal/seal.h"
#include "blobs.h"
#include "seal_ckks_context.h"
#include "service.h"
#include "hexl-fpga.h"
#include "fpga_seal_ckks_executor.h"
#include "server.h"

struct FPGAGasket {
    FPGAGasket() {
        intel::hexl::acquire_FPGA_resources();
        intel::hexl::set_worksize_KeySwitch(1);
    }

    ~FPGAGasket() { intel::hexl::release_FPGA_resources(); }
};

class FPGAContext {
public:
    static FPGAGasket& fpgaContext() {
        static FPGAGasket gasket;
        return gasket;
    }

private:
    FPGAContext() {}
    ~FPGAContext() {}
    FPGAContext(const FPGAContext&) = delete;
    FPGAContext& operator=(const FPGAContext&) = delete;
};

bool Server::setup() {
    FPGAGasket& fpgaGasket = FPGAContext::fpgaContext();
    return true;
}

void Server::teardown() {
    std::cout << "[Server] closing the service ..." << std::endl;
}

int Server::configureInternal(std::shared_ptr<ConfigurationBlob>& cblob) {
    cblob->restore(nullptr);
    context_ = cblob->context();
    executor_ = std::make_unique<FPGA_SealCKKSExecutor>(cblob);
    return 0;
}

void Server::configure() {
    std::cout << "[Server] configuring the service ..." << std::endl;
    assert(!buffer_->queue[Buffer::ConfigQ].empty());

    std::shared_ptr<Blob> blob = buffer_->queue[Buffer::ConfigQ].front();
    buffer_->queue[Buffer::ConfigQ].pop();

    auto cblob = std::dynamic_pointer_cast<ConfigurationBlob>(blob);
    configureInternal(cblob);
}

int Server::configure(char* config_buffer, size_t config_buffer_size,
                      char* bitstream_dir, char* kernel, size_t batch_size,
                      size_t ncards) {
    if (ncards == 0) {
        std::cout << "[Server] configuring server CPU" << std::endl;
        setenv("RUN_CHOICE", "0", 1);
    } else {
        std::cout << "[Server] configuring server FPGA" << std::endl;
        setenv("RUN_CHOICE", "2", 1);
        setenv("FPGA_BITSTREAM", bitstream_dir, 1);
        setenv("KERNEL", kernel, 1);
    }
    setenv("BATCH_SIZE_KEYSWITCH", std::to_string(batch_size).c_str(), 1);
    setup();

    auto cblob =
        std::make_shared<ConfigurationBlob>(config_buffer_size, config_buffer);
    configureInternal(cblob);
    return 0;
}

int Server::get_encrypted_input_buffer(char** server_buffer,
                                       size_t in_buffer_size) {
    server_buffer_.resize(in_buffer_size);
    *server_buffer = server_buffer_.data();
    return 0;
}

void Server::processBlob() {
    std::cout << "[Server] processing LR inference ..." << std::endl;
    current_input_blob_->restore(context_);
    std::vector<seal::Ciphertext> cipher_retval;

    auto t_started = std::chrono::high_resolution_clock::now();
    cipher_retval = executor_->evaluateLR(current_input_blob_->inputs());
    auto t_done = std::chrono::high_resolution_clock::now();
    double elapsed = static_cast<double>(
                         std::chrono::duration_cast<std::chrono::milliseconds>(
                             t_done - t_started)
                             .count()) /
                     1000.0;
    std::cout << "    Elapsed: " << elapsed << " seconds" << std::endl;
    current_output_blob_ = std::make_shared<OutputBlob>(cipher_retval);
    current_output_blob_->save();
}

void Server::process() {
    while (!buffer_->queue[Buffer::InputQ].empty()) {
        std::shared_ptr<Blob> blob = buffer_->queue[Buffer::InputQ].front();
        buffer_->queue[Buffer::InputQ].pop();
        current_input_blob_ = std::dynamic_pointer_cast<InputBlob>(blob);
        processBlob();
        buffer_->queue[Buffer::OutputQ].push(current_output_blob_);
    }
}

int Server::process(char** out_cstr, size_t* osz_ptr, char* in_buffer,
                    size_t in_buffer_size, const char* info_str) {
    current_input_blob_ =
        std::make_shared<InputBlob>(in_buffer_size, in_buffer);

    processBlob();

    *out_cstr = current_output_blob_->data();
    *osz_ptr = current_output_blob_->size();

    return 0;
}
