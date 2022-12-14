#! /usr/bin/env bash

pushd fpga-lr-microservice/build/service
RUN_CHOICE=$1 make fpga_lr_microservice
RUN_CHOICE=$1 make test_service
popd
