// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef HEXL_FPGA

#include "hexl-fpga.h"

class fpga_context {
public:
    ~fpga_context() { intel::hexl::release_FPGA_resources(); }
    fpga_context() {
        intel::hexl::acquire_FPGA_resources();
        intel::hexl::set_worksize_KeySwitch(1);
    }
};

#endif
