#!/usr/bin/env bash

project_dir=${PWD}
install=${project_dir}/install
install_server=${install}/server
build_type=Release
server_compiler=dpcpp

rm -rf ${install}/server tools/server

mkdir -p tools/server

pushd tools/server
    compiler=${server_compiler}
    rm -rf hexl-1.2.4
    git clone -b v1.2.4 https://github.com/intel/hexl.git hexl-1.2.4
    pushd hexl-1.2.4
        ln -s ${project_dir}/utils/build-hexl.sh .
        build-hexl.sh ${install_server} ${build_type} ${compiler}
    popd

    rm -rf hexl-fpga-development
    git clone -b development https://github.com/intel/hexl-fpga.git hexl-fpga-development
    pushd hexl-fpga-development
        ln -s ${project_dir}/utils/build-hexl-fpga.sh .
        build-hexl-fpga.sh ${install_server} ${build_type} ${compiler}
    popd

    rm -rf gsl-3.1.0
    git clone -b v3.1.0 https://github.com/microsoft/GSL.git gsl-3.1.0
    pushd gsl-3.1.0
        ln -s ${project_dir}/utils/build-gsl.sh .
        build-gsl.sh ${install_server}
    popd

    rm -rf zstd-1.5.2
    git clone -b v1.5.2 https://github.com/facebook/zstd.git zstd-1.5.2
    pushd zstd-1.5.2/build/cmake
        ln -s ${project_dir}/utils/build-zstd.sh .
        build-zstd.sh ${install_server} Release ${compiler}
    popd

    rm -rf seal-4.0.0
    git clone -b v4.0.0 https://github.com/microsoft/SEAL.git seal-4.0.0
    pushd seal-4.0.0
        ln -s ${project_dir}/hexl-fpga-BRIDGE-seal-4.0.0.patch .
        ln -s ${project_dir}/utils/build-seal-server.sh
        build-seal-server.sh ${install_server} ${build_type} ${compiler}
    popd
popd # sever

# server APIs
install_service=${install}/service
pushd ${project_dir}/fpga-lr-microservice/server
    rm -rf ../build/server
    cmake -S . -B ../build/server \
    -DCMAKE_BUILD_TYPE=${build_type} \
    -DCMAKE_INSTALL_PREFIX=${install_service}/server \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_INSTALL_INCLUDEDIR=include \
    -DCMAKE_PREFIX_PATH="${install_server}/share/cmake;\\${install_server}/lib/cmake"

    cmake --build ../build/server -j
    cmake --install ../build/server
popd # server APIs
