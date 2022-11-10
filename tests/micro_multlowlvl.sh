# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

spath=$(dirname $0)
. ${spath}/bitstream_dir.sh

########################################
# FPGA run with individual bitstream
########################################

echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libmultlowlvl.so FPGA_KERNEL=MULTLOWLVL"
# batch 1 (default)
FPGA_BITSTREAM=${bitstream_dir}/libmultlowlvl.so FPGA_KERNEL=MULTLOWLVL ./test_multlowlvl

