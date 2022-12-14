// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <memory>
#include "seal/seal.h"
#include "blobs.h"
#include "buffer.h"
#include "seal_ckks_context.h"
#include "fpga_seal_ckks_executor.h"

class Server : public Party {
public:
    Server() : Party() {}
    explicit Server(std::shared_ptr<Buffer>& buffer) : Party(buffer) {}
    ~Server() { teardown(); }

    bool setup() override;
    void configure() override;
    void teardown() override;
    void process() override;

    // config_buffer: context(params, keys) + model(weights, bias)
    // cofig_buffer_size: config_buffer size
    // bitstream_dir: path to bitstream
    // kernel: kernel name
    // ncards=0 default to CPU run.
    // ncards=1 run with a single FPGA.
    // return 0 as configuration success
    //        otherwise as fail.
    int configure(char* config_buffer, size_t cofig_buffer_size,
                  char* bitstream_dir, char* kernel, size_t batch_size,
                  size_t ncards);
    // out_cstr: output response data
    // osze_ptr: output size, this can vary.
    // in_buffer: pointer to the input buffer with a batch of input data (chunk)
    // in_buffer_size: input buffer size
    // chunk is defined as the minimal size of input data that server will
    // process. info_str: is for passing debugging or booking info.
    int process(char** out_cstr, size_t* osz_ptr, char* in_buffer,
                size_t in_buffer_size, const char* info_str);
    int get_encrypted_input_buffer(char** server_buffer, size_t in_buffer_size);

private:
    int configureInternal(std::shared_ptr<ConfigurationBlob>& cblob);
    void processBlob();

    std::vector<char> server_buffer_;
    std::shared_ptr<InputBlob> current_input_blob_;
    std::shared_ptr<OutputBlob> current_output_blob_;
    std::unique_ptr<FPGA_SealCKKSExecutor> executor_;
    std::shared_ptr<intel::he::heseal::SealCKKSContext> context_;
};
