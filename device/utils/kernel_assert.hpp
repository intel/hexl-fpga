// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __KERNEL_ASSERT_HPP_
#define __KERNEL_ASSERT_HPP_

#include <CL/sycl.hpp>
#include "dpc_common.hpp"

#ifdef EMULATOR
// Macro for emulating printf like behavior in kernels
#define kprintf(format, ...)                                       \
    {                                                              \
        static const CL_CONSTANT char _format[] = format;          \
        ext::oneapi::experimental::printf(_format, ##__VA_ARGS__); \
        \                                                          \
    }
#else
#define kprintf(format, ...)
#endif

#ifdef EMULATOR
#define ASSERT(cond, message, ...)              \
    if (!(cond)) {                              \
        kprintf("%s#%d: ", __FILE__, __LINE__); \
        kprintf(message, ##__VA_ARGS__);        \
    }
#else
#define ASSERT(cond, message, ...)
#endif

#endif  // __KERNEL_ASSERT_HPP_
