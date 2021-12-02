// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "fpga_context.h"

#include <iostream>

#include "fpga.h"
#include "hexl-fpga.h"

namespace intel {
namespace hexl {
namespace fpga {

void acquire_FPGA_resources() { attach_fpga_pooling(); }

void release_FPGA_resources() { detach_fpga_pooling(); }

}  // namespace fpga
}  // namespace hexl
}  // namespace intel
