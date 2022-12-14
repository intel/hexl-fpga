#!/usr/bin/env bash

install=$1
build_type=$2
compiler=$3

rm -rf build

cmake -S . -B build \
-DCMAKE_INSTALL_PREFIX=${install} \
-DCMAKE_INSTALL_LIBDIR=lib \
-DCMAKE_CXX_COMPILER=${compiler} \
-DCMAKE_BUILD_TYPE=${build_type} \
-DHEXL_SHARED_LIB=ON \
-DHEXL_TESTING=OFF \
-DHEXL_BENCHMARK=OFF \
-DHEXL_EXPERIMENTAL=ON \
-DHEXL_FPGA_COMPATIBILITY=2

cmake --build build --target hexl -j
cmake --install build
