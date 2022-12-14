// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "server_if.h"
#include "server.h"

static Server server;

int configure(char* config_buffer, size_t bsize, char* bitstream_dir,
              char* kernel, size_t batch_size, size_t ncards) {
    return server.configure(config_buffer, bsize, bitstream_dir, kernel,
                            batch_size, ncards);
}

int get_encrypted_input_buffer(char** server_buffer, size_t in_buffer_size) {
    return server.get_encrypted_input_buffer(server_buffer, in_buffer_size);
}

int process(char** out_cstr, size_t* osz_ptr, char* in_buffer,
            size_t in_buffer_size, const char* info_str) {
    return server.process(out_cstr, osz_ptr, in_buffer, in_buffer_size,
                          info_str);
}
