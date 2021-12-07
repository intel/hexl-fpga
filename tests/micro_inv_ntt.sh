# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

set -eo pipefail

########################################
# FPGA run with individual bitstream
########################################
if [[ -n ${RUN_CHOICE} ]] && [[ ${RUN_CHOICE} -eq 2 ]]
then
    aocl program acl0 inv_ntt.aocx
fi

echo ""
echo "FPGA_KERNEL=INTT"
# batch 1 (default)
FPGA_KERNEL=INTT ./test_inv_ntt
echo ""
echo "FPGA_KERNEL=INTT BATCH_SIZE_INTT=8"
# batch 8
FPGA_KERNEL=INTT BATCH_SIZE_INTT=8 ./test_inv_ntt

########################################
# FPGA run with integrated bitstream
########################################
if [[ -n ${RUN_CHOICE} ]] && [[ ${RUN_CHOICE} -eq 2 ]]
then
    aocl program acl0 hexl_fpga.aocx
fi

echo ""
# batch 1 (default)
./test_inv_ntt
echo ""
echo "FPGA_KERNEL=INTEGRATED BATCH_SIZE_INTT=8"
# batch 8
FPGA_KERNEL=INTEGRATED BATCH_SIZE_INTT=8 ./test_inv_ntt
