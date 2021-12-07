// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __FPGA_ASSERT_H__
#define __FPGA_ASSERT_H__

#define FPGA_ASSERT_1_ARGS(condition) FPGA_ASSERT_INT(condition, 0)
#define FPGA_ASSERT_2_ARGS(condition, message) \
    FPGA_ASSERT_INT(condition, message)

#define GET_ARG(arg1, arg2, func, ...) func
#define FPGA_ASSERT_MACRO_CHOOSER(...) \
    GET_ARG(__VA_ARGS__, FPGA_ASSERT_2_ARGS, FPGA_ASSERT_1_ARGS)

#define FPGA_ASSERT(...) FPGA_ASSERT_MACRO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)

#ifdef FPGA_DEBUG
#include <iostream>

#include "stack_trace.h"

#define FPGA_ASSERT_INT(condition, message)                             \
    {                                                                   \
        if (!(condition)) {                                             \
            std::cerr << "Assertion: '" << #condition << "' failed at " \
                      << __FILE__ << ":" << __LINE__ << std::endl;      \
            if (message) {                                              \
                std::cerr << "    Error: " << #message << std::endl;    \
            }                                                           \
            intel::hexl::fpga::StackTrace* stack =                      \
                intel::hexl::fpga::StackTrace::stack();                 \
            stack->dump(std::cerr);                                     \
            abort();                                                    \
            exit(1);                                                    \
        }                                                               \
    }

#else

#define FPGA_ASSERT_INT(condition, message) \
    {}

#endif

#endif
