# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

set -eo pipefail

########################################
# FPGA run with individual bitstream
########################################
if [[ -n ${RUN_CHOICE} ]] && [[ ${RUN_CHOICE} -eq 2 ]]
then
    aocl program acl0 dyadic_multiply.aocx
fi

echo ""
echo "FPGA_KERNEL=DYADIC_MULTIPLY"
# batch 1 (default)
FPGA_KERNEL=DYADIC_MULTIPLY ./test_dyadic_multiply
echo ""
echo "FPGA_KERNEL=DYADIC_MULTIPLY BATCH_SIZE_DYADIC_MULTIPLY=8"
# batch 8
FPGA_KERNEL=DYADIC_MULTIPLY BATCH_SIZE_DYADIC_MULTIPLY=8 ./test_dyadic_multiply

########################################
# FPGA run with integrated bitstream
########################################
if [[ -n ${RUN_CHOICE} ]] && [[ ${RUN_CHOICE} -eq 2 ]]
then
    aocl program acl0 hexl_fpga.aocx
fi

echo ""
# batch 1 (default)
./test_dyadic_multiply
# batch 8
echo ""
echo "FPGA_KERNEL=INTEGRATED BATCH_SIZE_DYADIC_MULTIPLY=8"
FPGA_KERNEL=INTEGRATED BATCH_SIZE_DYADIC_MULTIPLY=8 ./test_dyadic_multiply
