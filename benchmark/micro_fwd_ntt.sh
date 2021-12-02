# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

set -eo pipefail

# FPGA run with individual bitstream
if [[ -n ${RUN_CHOICE} ]] && [[ ${RUN_CHOICE} -eq 2 ]]
then
    aocl program acl0 fwd_ntt.aocx
fi

echo ""
echo "FPGA_KERNEL=NTT BATCH_SIZE_NTT = 1"
FPGA_KERNEL=NTT BATCH_SIZE_NTT=1 ./bench_fwd_ntt
echo ""
echo "FPGA_KERNEL=NTT BATCH_SIZE_NTT = 8"
FPGA_KERNEL=NTT BATCH_SIZE_NTT=8 ./bench_fwd_ntt
echo ""
echo "FPGA_KERNEL=NTT BATCH_SIZE_NTT = 32"
FPGA_KERNEL=NTT BATCH_SIZE_NTT=32 ./bench_fwd_ntt
echo ""
echo "FPGA_KERNEL=NTT BATCH_SIZE_NTT = 128"
FPGA_KERNEL=NTT BATCH_SIZE_NTT=128 ./bench_fwd_ntt
echo ""
echo "FPGA_KERNEL=NTT BATCH_SIZE_NTT = 512"
FPGA_KERNEL=NTT BATCH_SIZE_NTT=512 ./bench_fwd_ntt
