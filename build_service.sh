#!/usr/bin/env bash

project_dir=${PWD}
install=${project_dir}/install
build_type=Release

pushd fpga-lr-microservice
    rm -f build-fpga-lr-microservice.sh
    ln -s ${project_dir}/utils/build-fpga-lr-microservice.sh .
    build-fpga-lr-microservice.sh ${install} ${build_type}
popd
