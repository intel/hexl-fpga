// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "stack_trace.h"

#include <cxxabi.h>
#include <execinfo.h>
#include <unistd.h>

#include <iostream>
#include <list>
#include <memory>
#include <string>

namespace intel {
namespace hexl {
namespace fpga {

class StackTrace_Impl : public StackTrace {
public:
    enum { BUFFER_SIZE = 4096 };
    StackTrace_Impl() {
        int n_frames = backtrace(buffer, BUFFER_SIZE);
        char** frames = backtrace_symbols(buffer, n_frames);
        if (!frames) {
            std::cerr << "backtrace dump failed" << std::endl;
            exit(1);
        }

        size_t funcnamesize = 256;
        char* funcname = (char*)malloc(funcnamesize);

        // skip the first 2, they are related to StackTrace itself
        for (int i = 2; i < n_frames; i++) {
            char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

            for (char* p = frames[i]; *p; ++p) {
                if (*p == '(') {
                    begin_name = p;
                } else if (*p == '+') {
                    begin_offset = p;
                } else if (*p == ')' && begin_offset) {
                    end_offset = p;
                    break;
                }
            }

            if (begin_name && begin_offset && end_offset &&
                begin_name < begin_offset) {
                *begin_name++ = '\0';
                *begin_offset++ = '\0';
                *end_offset = '\0';

                int status;
                char* real_name = abi::__cxa_demangle(begin_name, funcname,
                                                      &funcnamesize, &status);
                if (status == 0) {
                    frames_.push_back(std::string(real_name));
                } else {
                    frames_.push_back(std::string(begin_name));
                }
            } else {
                frames_.push_back(std::string(frames[i]));
            }
        }

        free(funcname);
        free(frames);
    }

    void dump(std::ostream& os) {
        for (const auto& frame : frames_) {
            std::cout << "    " << frame << std::endl;
        }
    }

private:
    std::list<std::string> frames_;
    void* buffer[BUFFER_SIZE];
};

StackTrace* StackTrace::stack() { return new StackTrace_Impl; }

}  // namespace fpga
}  // namespace hexl
}  // namespace intel
