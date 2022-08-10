// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
// class for FPGA context setup, performs setup and teardown
#include "hexl-fpga.h"

class fpga_context : public ::testing::Environment {
public:
    virtual ~fpga_context() {}
    virtual void SetUp() { intel::hexl::acquire_FPGA_resources(); }

    virtual void TearDown() { intel::hexl::release_FPGA_resources(); }
};
