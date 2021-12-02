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

    aoc ${device} \
        ${src_dir}/device/${kernel}.cl \
        -I ${INTELFPGAOCLSDKROOT}/include/kernel_headers \
        -I ${src_dir}/device/lib/hls \
        -L ${bin_dir}/device/lib/hls -l ip.aoclib \
        -o ${kernel} \
        ${others}
}

for kernel in ${kernels}
do
    echo "Compiling bitstream for ${kernel}"
    configs="config_${kernel}"
    compile ${target} ${kernel} ${cmake_src_dir} ${cmake_bin_dir} ${!configs} ${fpga_args}
done
