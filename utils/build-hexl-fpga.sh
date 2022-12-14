#!/usr/bin/env bash

install=$1
build_type=$2
compiler=$3

rm -rf build

cmake -S . -B build \
-DCMAKE_INSTALL_PREFIX=${install} \
-DCMAKE_PREFIX_PATH=${install}/lib/cmake \
-DCMAKE_INSTALL_LIBDIR=lib \
-DCMAKE_CXX_COMPILER=${compiler} \
-DCMAKE_BUILD_TYPE=${build_type} \
-DFPGA_USE_INTEL_HEXL=ON \
-DFPGA_BUILD_INTEL_HEXL=OFF \
-DENABLE_BENCHMARK=OFF \
-DENABLE_TESTS=OFF

cmake --build build -j
cmake --build build --target emulation
cmake --install build

#install pre-built hardware bitstreams
#cp -f /disk1/hexl-fpga-data-oneapi/bitstreams/*.so ${install}/fpga/
#cp -f /disk1/hexl-fpga-data/bitstreams/*.aocx ${install}/fpga/
