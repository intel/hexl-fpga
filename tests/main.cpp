// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "fpga_context.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    ::testing::AddGlobalTestEnvironment(new fpga_context);

    int rc = RUN_ALL_TESTS();
    return rc;
}
