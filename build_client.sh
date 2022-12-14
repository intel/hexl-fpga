#!/usr/bin/env bash

project_dir=${PWD}
install=${project_dir}/install
install_client=${install}/client
build_type=Release
client_compiler=g++

rm -rf ${install}/client tools/client

mkdir -p tools/client

pushd tools/client
    compiler=${client_compiler}
    rm -rf gsl-3.1.0
    git clone -b v3.1.0 https://github.com/microsoft/GSL.git gsl-3.1.0
    pushd gsl-3.1.0
        ln -s ${project_dir}/utils/build-gsl.sh .
        build-gsl.sh ${install_client}
    popd

    rm -rf zstd-1.5.2
    git clone -b v1.5.2 https://github.com/facebook/zstd.git zstd-1.5.2
    pushd zstd-1.5.2/build/cmake
        ln -s ${project_dir}/utils/build-zstd.sh .
        build-zstd.sh ${install_client} Release ${compiler}
    popd

    rm -rf seal-4.0.0
    git clone -b v4.0.0 https://github.com/microsoft/SEAL.git seal-4.0.0
    pushd seal-4.0.0
        ln -s ${project_dir}/utils/build-seal-client.sh .
        build-seal-client.sh ${install_client} ${build_type} ${compiler}
    popd
popd # client


# client APIs
install_service=${install}/service
mkdir -p ${install_service}
pushd ${project_dir}/fpga-lr-microservice/client
    rm -rf ../build/client
    cmake -S . -B ../build/client \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DCMAKE_INSTALL_PREFIX=${install_service}/client \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_INSTALL_INCLUDEDIR=include \
    -DCMAKE_PREFIX_PATH="${install_client}/share/cmake;\\${install_client}/lib/cmake"

    cmake --build ../build/client -j
    cmake --install ../build/client
popd # client APIs
