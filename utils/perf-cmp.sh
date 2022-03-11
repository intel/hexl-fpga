#!/usr/bin/env bash

workdir=$PWD
results=${workdir}/results
mkdir -p ${results}

testv=/disk1/hexl-fpga-data/test-vectors/16384_6_7_7_2_0.json
mkdir -p ${workdir}/test-vectors
cp -f ${testv} ${workdir}/test-vectors/
export KEYSWITCH_DATA_DIR=${workdir}/test-vectors

build_cpu() {
    rm -rf build
    mkdir build

    cd build
    cmake .. \
    -DCMAKE_INSTALL_PREFIX=./hexl-fpga-install \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=g++ \
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
    -DCMAKE_CXX_COMPILER=g++ \
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

pushd build/benchmark
for i in 1 1024 4096 16384
do
    run_cpu $i
done
popd

aocx=/disk1/hexl-fpga-data/bitstreams/keyswitch.aocx
export FPGA_BITSTREAM=${aocx}
export FPGA_KERNEL=KEYSWITCH
aocl program acl0 ${aocx}
pushd build-fpga/benchmark
batch=128
for i in 1 1024 4096 16384
do
    run_fpga $i ${batch}
done
popd
