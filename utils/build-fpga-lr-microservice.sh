#!/usr/bin/env bash

install=$1
build_type=$2

install_server=${install}/server

# service
pushd service
    rm -rf ../build/service

    cmake -S . -B ../build/service \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DCMAKE_PREFIX_PATH="${install_server}/share/cmake;\\${install_server}/lib/cmake"

    cmake --build ../build/service -j
    ln -s ${install_server}/fpga/libkeyswitch.so ../build/service
popd
