#!/usr/bin/env bash

install=$1

rm -rf build

cmake -S . -B build \
-DCMAKE_INSTALL_PREFIX=${install} \
-DCMAKE_INSTALL_LIBDIR=lib \
-DCMAKE_INSTALL_INCLUDEDIR=include

cmake --install build
