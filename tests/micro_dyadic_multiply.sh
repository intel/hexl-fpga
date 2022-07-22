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
# FPGA run with individual bitstream
########################################

echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply.so FPGA_KERNEL=DYADIC_MULTIPLY BATCH_SIZE_DYADIC_MULTIPLY=1"
# batch 1 (default)
FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply.so FPGA_KERNEL=DYADIC_MULTIPLY BATCH_SIZE_DYADIC_MULTIPLY=1 ./test_dyadic_multiply
echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply.so FPGA_KERNEL=DYADIC_MULTIPLY BATCH_SIZE_DYADIC_MULTIPLY=8"
# batch 8
FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply.so FPGA_KERNEL=DYADIC_MULTIPLY BATCH_SIZE_DYADIC_MULTIPLY=8 ./test_dyadic_multiply
