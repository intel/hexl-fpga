# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

set -eo pipefail

########################################
# FPGA run with individual bitstream
########################################
if [[ -n ${RUN_CHOICE} ]] && [[ ${RUN_CHOICE} -eq 2 ]]
then
    aocl program acl0 fwd_ntt.aocx
fi

echo ""
echo "FPGA_KERNEL=NTT"
# batch 1 (default)
FPGA_KERNEL=NTT ./test_fwd_ntt
echo ""
echo "FPGA_KERNEL=NTT BATCH_SIZE_NTT=8"
# batch 8
FPGA_KERNEL=NTT BATCH_SIZE_NTT=8 ./test_fwd_ntt

########################################
# FPGA run with integrated bitstream
########################################
if [[ -n ${RUN_CHOICE} ]] && [[ ${RUN_CHOICE} -eq 2 ]]
then
    aocl program acl0 hexl_fpga.aocx
fi

echo ""
# batch 1 (default)
./test_fwd_ntt
echo ""
echo "FPGA_KERNEL=INTEGRATED BATCH_SIZE_NTT=8"
# batch 8
FPGA_KERNEL=INTEGRATED BATCH_SIZE_NTT=8 ./test_fwd_ntt
