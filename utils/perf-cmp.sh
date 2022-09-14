# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

workdir=$PWD
results=${workdir}/results
mkdir -p ${results}
mkdir -p testdata

setup_testdata() {
    pushd testdata

    # downloading bitstream libkeyswitch.so
    wget https://github.com/intel/hexl-fpga/releases/download/v2.0-rc1/libkeyswitch.so
    shared_lib=${PWD}/libkeyswitch.so
    export FPGA_BITSTREAM=${shared_lib}
    export FPGA_KERNEL=KEYSWITCH

    # downloading test vectors testdata.zip
    wget https://github.com/intel/hexl-fpga/releases/download/v1.1/testdata.zip
    unzip testdata.zip
    export KEYSWITCH_DATA_DIR=${PWD}/testdata

    popd
}

build() {
    rm -rf build
    mkdir build

    cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=./hexl-fpga-install \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=dpcpp \
    -DCMAKE_C_COMPILER=gcc \
    -DENABLE_TESTS=ON \
    -DENABLE_BENCHMARK=ON \
    -DFPGA_USE_INTEL_HEXL=ON \
    -DFPGA_BUILD_INTEL_HEXL=ON

    cmake --build build -j
}

run_cpu() {
    iter=$1
    export ITER=${iter}
    RUN_CHOICE=0 ./bench_keyswitch > ${results}/cpu.iter-${iter}.log
}

run_fpga() {
    iter=$1
    batch=$2
    export ITER=${iter}
    export BATCH_SIZE_KEYSWITCH=${batch}
    RUN_CHOICE=2 ./bench_keyswitch  > ${results}/fpga.iter-${iter}.batch-${batch}.log
}

build

setup_testdata

pushd build/benchmark
for i in 1 1024 4096 16384
do
    run_cpu $i
done
popd

# Initializing FPGA to use non-USM BSP
aocl initialize acl0 pac_s10
pushd build/benchmark
batch=128
for i in 1 1024 4096 16384
do
    run_fpga $i ${batch}
done
popd
