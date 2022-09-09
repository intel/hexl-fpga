// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include "hexl-fpga.h"

int main(int argc, char** argv) {
    intel::hexl::acquire_FPGA_resources();

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    intel::hexl::release_FPGA_resources();
    return 0;
}
