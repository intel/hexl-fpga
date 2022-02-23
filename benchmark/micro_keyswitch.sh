# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

set -eo pipefail

spath=$(dirname $0)
. ${spath}/bitstream_dir.sh

########################################
# FPGA run with individual bitstream
########################################
if [[ -z ${RUN_CHOICE} ]] || [[ ${RUN_CHOICE} -eq 2 ]]
then
    aocl program acl0 ${bitstream_dir}/keyswitch.aocx
fi

echo ""
echo "N=16384 FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=1"
ITER=256 N=16384 FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=1 ./bench_keyswitch
echo ""
echo "N=16384 FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=16"
ITER=256 N=16384 FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=16 ./bench_keyswitch
echo ""
echo "N=16384 FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=128"
ITER=256 N=16384 FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=128 ./bench_keyswitch
