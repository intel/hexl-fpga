# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

set -eo pipefail

spath=$(dirname $0)
. ${spath}/bitstream_dir.sh

########################################
# FPGA run with individual bitstream
########################################
if [[ -z ${RUN_CHOICE} ]] || [[ ${RUN_CHOICE} -eq 2 ]]
then
    aocl program acl0 ${bitstream_dir}/inv_ntt.aocx
fi

echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/inv_ntt.aocx FPGA_KERNEL=INTT"
# batch 1 (default)
FPGA_BITSTREAM=${bitstream_dir}/inv_ntt.aocx FPGA_KERNEL=INTT ./test_inv_ntt
echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/inv_ntt.aocx FPGA_KERNEL=INTT BATCH_SIZE_INTT=8"
# batch 8
FPGA_BITSTREAM=${bitstream_dir}/inv_ntt.aocx FPGA_KERNEL=INTT BATCH_SIZE_INTT=8 ./test_inv_ntt
