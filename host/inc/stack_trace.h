// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __STACK_TRACE_H__
#define __STACK_TRACE_H__

#include <iostream>

namespace intel {
namespace hexl {
namespace fpga {
/// @brief
/// Class StackTrace
/// Allows the investigation of the traces
/// @function dump
/// Dumps the traces
/// @param[in] os vector of traces
///
class StackTrace {
public:
    static StackTrace* stack();
    virtual ~StackTrace() = default;

    virtual void dump(std::ostream& os) = 0;

protected:
    StackTrace() {}
};

}  // namespace fpga
}  // namespace hexl
}  // namespace intel

#endif
