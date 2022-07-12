// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#ifndef FPGA_USE_INTEL_HEXL
#include "hexl-fpga.h"
#endif

int main(int argc, char** argv) {
#ifndef FPGA_USE_INTEL_HEXL
    intel::hexl::acquire_FPGA_resources();
#endif
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

#ifndef FPGA_USE_INTEL_HEXL
    intel::hexl::release_FPGA_resources();
#endif
    return 0;
}
