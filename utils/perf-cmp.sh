# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

workdir=$PWD
results=${workdir}/results
mkdir -p ${results}
mkdir -p testdata

setup_testdata() {
    pushd testdata

    # downloading bitstream libkeyswitch_hw.so
    wget https://github.com/intel/hexl-fpga/releases/download/v1.1/libkeyswitch_hw.so
    shared_lib=${PWD}/libkeyswitch_hw.so
    export FPGA_BITSTREAM_PATH=${PWD}/
    export FPGA_KERNEL=KEYSWITCH

    # downloading test vectors testdata.zip
    wget https://github.com/intel/hexl-fpga/releases/download/v1.1/testdata.zip
    unzip testdata.zip
    export KEYSWITCH_DATA_DIR=${PWD}/testdata

    popd
}

build_cpu() {
    rm -rf build
    mkdir build

    cd build
    cmake .. \
    -DCMAKE_INSTALL_PREFIX=./hexl-fpga-install \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=dpcpp \
    -DCMAKE_C_COMPILER=gcc \
    -DENABLE_TESTS=ON \
    -DENABLE_BENCHMARK=ON \
    -DFPGA_USE_INTEL_HEXL=ON \
    -DFPGA_BUILD_INTEL_HEXL=ON

    make -j
    cd ..
}

build_fpga() {
    rm -rf build-fpga
    mkdir build-fpga

    cd build-fpga
    cmake .. \
    -DCMAKE_INSTALL_PREFIX=./hexl-fpga-install \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=dpcpp \
    -DCMAKE_C_COMPILER=gcc \
    -DENABLE_TESTS=ON \
    -DENABLE_BENCHMARK=ON \
    -DFPGA_USE_INTEL_HEXL=OFF \
    -DFPGA_BUILD_INTEL_HEXL=OFF

    make -j
    cd ..
}

run_cpu() {
    iter=$1
    export ITER=${iter}
    ./bench_keyswitch > ${results}/cpu.iter-${iter}.log
}

run_fpga() {
    iter=$1
    batch=$2
    export ITER=${iter}
    export BATCH_SIZE_KEYSWITCH=${batch}
    ./bench_keyswitch  > ${results}/fpga.iter-${iter}.batch-${batch}.log
}

build_cpu
build_fpga

setup_testdata

pushd build/benchmark
for i in 1 1024 4096 16384
do
    run_cpu $i
done
popd

# Initializing FPGA to use non-USM BSP
aocl initialize acl0 pac_s10
pushd build-fpga/benchmark
batch=128
for i in 1 1024 4096 16384
do
    run_fpga $i ${batch}
done
popd
