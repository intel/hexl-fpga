#!/usr/bin/env bash

install=$1
build_type=$2
compiler=$3

rm -rf build

cmake -S . -B build \
-DCMAKE_INSTALL_PREFIX=${install} \
-DCMAKE_INSTALL_LIBDIR=lib \
-DCMAKE_INSTALL_INCLUDEDIR=include \
-DCMAKE_CXX_COMPILER=${compiler} \
-DCMAKE_BUILD_TYPE=${build_type} \
-DZSTD_BUILD_PROGRAMS=OFF \
-DZSTD_BUILD_SHARED=OFF \
-DZLIB_BUILD_STATIC=ON \
-DZSTD_BUILD_TESTS=OFF \
-DZSTD_MULTITHREAD_SUPPORT=OFF

cmake --build build -j
cmake --install build
