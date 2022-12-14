#!/usr/bin/env bash

install=$1
build_type=$2
compiler=$3

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
-DSEAL_USE_MSGSL=ON

cmake --build build -j
cmake --install build
