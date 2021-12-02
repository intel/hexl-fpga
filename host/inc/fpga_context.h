// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __FPGA_CONTEXT_H__
#define __FPGA_CONTEXT_H__

#include <cstdint>

namespace intel {
namespace hexl {
namespace fpga {
/// @brief
/// @function acquire_FPGA_resources
/// Called at the beginning of the workload to acquire the usage of an FPGA
///
void acquire_FPGA_resources();
/// @brief
/// @function release_FPGA_resources
/// Called at the end of the workload to release the FPGA
///
void release_FPGA_resources();

}  // namespace fpga
}  // namespace hexl
}  // namespace intel

#endif
