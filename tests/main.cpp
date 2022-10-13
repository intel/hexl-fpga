// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

//#include "fpga_context.h"

int main(int argc, char** argv) {
    // relay the command line arguments to google test e.g. : --test_repeat=2
    ::testing::InitGoogleTest(&argc, argv);

    // setup and tear down done on global level for all test
    // it can also be done based on test level by passing a test object
    //::testing::AddGlobalTestEnvironment(new fpga_context);

    // Lauch all the tests ...
    int rc = RUN_ALL_TESTS();

    return rc;
}
