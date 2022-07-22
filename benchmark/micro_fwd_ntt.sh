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
echo "FPGA_BITSTREAM=${bitstream_dir}/libfwd_ntt.so FPGA_KERNEL=NTT BATCH_SIZE_NTT = 1"
FPGA_BITSTREAM=${bitstream_dir}/libfwd_ntt.so FPGA_KERNEL=NTT BATCH_SIZE_NTT=1 ./bench_fwd_ntt
echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libfwd_ntt.so FPGA_KERNEL=NTT BATCH_SIZE_NTT = 8"
FPGA_BITSTREAM=${bitstream_dir}/libfwd_ntt.so FPGA_KERNEL=NTT BATCH_SIZE_NTT=8 ./bench_fwd_ntt
echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libfwd_ntt.so FPGA_KERNEL=NTT BATCH_SIZE_NTT = 32"
FPGA_BITSTREAM=${bitstream_dir}/libfwd_ntt.so FPGA_KERNEL=NTT BATCH_SIZE_NTT=32 ./bench_fwd_ntt
echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libfwd_ntt.so FPGA_KERNEL=NTT BATCH_SIZE_NTT = 128"
FPGA_BITSTREAM=${bitstream_dir}/libfwd_ntt.so FPGA_KERNEL=NTT BATCH_SIZE_NTT=128 ./bench_fwd_ntt
echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libfwd_ntt.so FPGA_KERNEL=NTT BATCH_SIZE_NTT = 512"
FPGA_BITSTREAM=${bitstream_dir}/libfwd_ntt.so FPGA_KERNEL=NTT BATCH_SIZE_NTT=512 ./bench_fwd_ntt
