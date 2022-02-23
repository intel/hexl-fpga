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
echo "FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH"
# default N=16384 BATCH_SIZE_KEYSWITCH=1
FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH ./test_keyswitch
echo ""
echo "N=16384 FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=1"
N=16384 FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=1 ./test_keyswitch
echo ""
echo "N=16384 FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=2"
N=16384 FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=2 ./test_keyswitch
echo ""
echo "N=8192 FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=1"
N=8192 FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=1 ./test_keyswitch
echo ""
echo "N=8192 FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=2"
N=8192 FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=2 ./test_keyswitch
