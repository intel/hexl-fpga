# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

set -eo pipefail

spath=$(dirname $0)
. ${spath}/bitstream_dir.sh
if [[ -z ${RUN_CHOICE} ]] || [[ ${RUN_CHOICE} -eq 2 ]]
then
    aocl initialize acl0 pac_s10
fi
########################################
# FPGA run with individual bitstream
########################################

echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libkeyswitch.so FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=1"
FPGA_BITSTREAM=${bitstream_dir}/libkeyswitch.so ITER=256 N=16384 FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=1 ./bench_keyswitch
echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libkeyswitch.so N=16384 FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=16"
FPGA_BITSTREAM=${bitstream_dir}/libkeyswitch.so ITER=256 N=16384 FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=16 ./bench_keyswitch
echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libkeyswitch.so N=16384 FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=128"
FPGA_BITSTREAM=${bitstream_dir}/libkeyswitch.so ITER=256 N=16384 FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=128 ./bench_keyswitch
