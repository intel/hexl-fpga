# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

#set -eo pipefail

spath=$(dirname $0)
. ${spath}/bitstream_dir.sh

########################################
# FPGA run with individual bitstream
########################################

echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libmultlowlvl.so FPGA_KERNEL=MULTLOWLVL BATCH_SIZE_MULTLOWLVL=1"
FPGA_BITSTREAM=${bitstream_dir}/libmultlowlvl.so FPGA_KERNEL=MULTLOWLVL BATCH_SIZE_MULTLOWLVL=1 ./bench_multlowlvl
