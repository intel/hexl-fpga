# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

target=$1
cmake_src_dir=$2
cmake_bin_dir=$3

spath=$(dirname $0)
. ${spath}/config.sh

compile() {
    device=$1
    kernel=$2
    src_dir=$3
    bin_dir=$4
    others=${@:5}

    export TMPDIR=${bin_dir}/device/${kernel}
    mkdir -p ${TMPDIR}

    dpcpp ${device} \
        -fPIC -O3 -DNDEBUG -fintelfpga -shared -qactypes \
        -Wno-ignored-attributes -Wno-return-type-c-linkage -Wno-unknown-pragmas \
        ${others} \
        -o lib${kernel}.so \
        ${src_dir}/device/${kernel}.cpp \
        ${cmake_src_dir}/device/multlowlvl/src/L1/multLowLvl.cpp \
        ${cmake_src_dir}/device/multlowlvl/src/L1/tensorProduct.cpp 

    rm -rf ${TMPDIR}
    export TMPDIR=
}

for kernel in ${kernels}
do
    echo "Compiling bitstream for ${kernel}"
    configs="config_${kernel}"
    compile ${target} ${kernel} ${cmake_src_dir} ${cmake_bin_dir} ${!configs} ${fpga_args}
done
