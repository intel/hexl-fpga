#!/usr/bin/env bash

install=$1
build_type=$2
compiler=$3

git apply hexl-fpga-BRIDGE-seal-4.0.0.patch

rm -rf build

cmake -S . -B build \
-DCMAKE_PREFIX_PATH="${install}/lib/cmake;\\${install}/share/cmake" \
-DCMAKE_INSTALL_PREFIX=${install} \
-DCMAKE_INSTALL_LIBDIR=lib \
-DCMAKE_CXX_COMPILER=${compiler} \
-DCMAKE_BUILD_TYPE=${build_type} \
-DBUILD_SHARED_LIBS=ON \
-DSEAL_BUILD_DEPS=OFF \
-DSEAL_USE_ZLIB=OFF \
-DSEAL_USE_ZSTD=ON \
-DSEAL_USE_MSGSL=ON \
-DSEAL_USE_INTEL_HEXL=ON \
-DSEAL_USE_INTEL_HEXL_FPGA=ON \
-DHEXL_EXPERIMENTAL=ON

cmake --build build -j
cmake --install build
