# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

set -eo pipefail

# FPGA run with individual bitstream
if [[ -n ${RUN_CHOICE} ]] && [[ ${RUN_CHOICE} -eq 2 ]]
then
    aocl program acl0 inv_ntt.aocx
fi

echo ""
echo "FPGA_KERNEL=INTT BATCH_SIZE_INTT = 1"
FPGA_KERNEL=INTT BATCH_SIZE_INTT=1 ./bench_inv_ntt
echo ""
echo "FPGA_KERNEL=INTT BATCH_SIZE_INTT = 8"
FPGA_KERNEL=INTT BATCH_SIZE_INTT=8 ./bench_inv_ntt
echo ""
echo "FPGA_KERNEL=INTT BATCH_SIZE_INTT = 32"
FPGA_KERNEL=INTT BATCH_SIZE_INTT=32 ./bench_inv_ntt
echo ""
echo "FPGA_KERNEL=INTT BATCH_SIZE_INTT = 128"
FPGA_KERNEL=INTT BATCH_SIZE_INTT=128 ./bench_inv_ntt
echo ""
echo "FPGA_KERNEL=INTT BATCH_SIZE_INTT = 512"
FPGA_KERNEL=INTT BATCH_SIZE_INTT=512 ./bench_inv_ntt
