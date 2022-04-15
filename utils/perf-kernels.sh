#!/usr/bin/env bash

testv=/disk1/hexl-fpga-data/test-vectors/16384_6_7_7_2_0.json
bitstreams_folder="/disk1/hexl-fpga-data/bitstreams"

keyswitch_aocx=${bitstreams_folder}/keyswitch.aocx
ntt_aocx=${bitstreams_folder}/fwd_ntt.aocx
intt_aocx=${bitstreams_folder}/inv_ntt.aocx

workdir=$PWD

mkdir -p ${workdir}/test-vectors
cp -f ${testv} ${workdir}/test-vectors/
export KEYSWITCH_DATA_DIR=${workdir}/test-vectors

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

run_keyswitch() {
    export FPGA_BITSTREAM=${keyswitch_aocx}
    export FPGA_KERNEL=KEYSWITCH
    aocl program acl0 ${keyswitch_aocx}
    iter=$1
    batch=$2
    export ITER=${iter}
    export BATCH_SIZE_KEYSWITCH=${batch}
    export FPGA_DEBUG=1
    ./bench_keyswitch
}

run_ntt() {
    export FPGA_BITSTREAM=${ntt_aocx}
    export FPGA_KERNEL=NTT
    aocl program acl0 ${ntt_aocx}
    iter=$1
    batch=$2
    export ITER=${iter}
    export BATCH_SIZE_NTT=${batch}
    export FPGA_DEBUG=1
    ./bench_fwd_ntt
}

run_intt() {
    export FPGA_BITSTREAM=${intt_aocx}
    export FPGA_KERNEL=INTT
    aocl program acl0 ${intt_aocx}
    iter=$1
    batch=$2
    export ITER=${iter}
    export BATCH_SIZE_INTT=${batch}
    export FPGA_DEBUG=1
    ./bench_inv_ntt
}

build_fpga

pushd build-fpga/benchmark
batch=128
iter=128
run_keyswitch ${iter} ${batch}
run_ntt ${iter} ${batch}
run_intt ${iter} ${batch}
popd

