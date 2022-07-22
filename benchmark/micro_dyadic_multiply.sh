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
echo "FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply.so FPGA_KERNEL=DYADIC_MULTIPLY BATCH_SIZE_DYADIC_MULTIPLY = 1"
FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply.so FPGA_KERNEL=DYADIC_MULTIPLY BATCH_SIZE_DYADIC_MULTIPLY=1 ./bench_dyadic_multiply
echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply.so FPGA_KERNEL=DYADIC_MULTIPLY BATCH_SIZE_DYADIC_MULTIPLY = 4"
FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply.so FPGA_KERNEL=DYADIC_MULTIPLY BATCH_SIZE_DYADIC_MULTIPLY=4 ./bench_dyadic_multiply
echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply.so FPGA_KERNEL=DYADIC_MULTIPLY BATCH_SIZE_DYADIC_MULTIPLY = 8"
FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply.so FPGA_KERNEL=DYADIC_MULTIPLY BATCH_SIZE_DYADIC_MULTIPLY=8 ./bench_dyadic_multiply
echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply.so FPGA_KERNEL=DYADIC_MULTIPLY BATCH_SIZE_DYADIC_MULTIPLY = 16"
FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply.so FPGA_KERNEL=DYADIC_MULTIPLY BATCH_SIZE_DYADIC_MULTIPLY=16 ./bench_dyadic_multiply
echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply.so FPGA_KERNEL=DYADIC_MULTIPLY BATCH_SIZE_DYADIC_MULTIPLY = 32"
FPGA_BITSTREAM=${bitstream_dir}/libdyadic_multiply.so FPGA_KERNEL=DYADIC_MULTIPLY BATCH_SIZE_DYADIC_MULTIPLY=32 ./bench_dyadic_multiply
