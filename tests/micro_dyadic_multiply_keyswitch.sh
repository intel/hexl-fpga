# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

set -eo pipefail

spath=$(dirname $0)
. ${spath}/bitstream_dir.sh

if [[ -z ${RUN_CHOICE} ]] || [[ ${RUN_CHOICE} -eq 2 ]]
then
    aocl initialize acl0 pac_s10_usm
fi
########################################
# FPGA run with integrated bitstream
########################################

echo ""
# batch 1 (default)
echo "FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply_keyswitch.so FPGA_KERNEL=DYADIC_MULTIPLY_KEYSWITCH BATCH_SIZE_DYADIC_MULTIPLY=1 BATCH_SIZE_KEYSWITCH=1"
FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply_keyswitch.so FPGA_KERNEL=DYADIC_MULTIPLY_KEYSWITCH ./test_dyadic_multiply_keyswitch
# batch 2
echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply_keyswitch.so FPGA_KERNEL=DYADIC_MULTIPLY_KEYSWITCH BATCH_SIZE_DYADIC_MULTIPLY=2 BATCH_SIZE_KEYSWITCH=2"
FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply_keyswitch.so FPGA_KERNEL=DYADIC_MULTIPLY_KEYSWITCH BATCH_SIZE_DYADIC_MULTIPLY=2 BATCH_SIZE_KEYSWITCH=2 ./test_dyadic_multiply_keyswitch
