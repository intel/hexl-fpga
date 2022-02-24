# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

set -eo pipefail

spath=$(dirname $0)
. ${spath}/bitstream_dir.sh

########################################
# FPGA run with integrated bitstream
########################################
if [[ -z ${RUN_CHOICE} ]] || [[ ${RUN_CHOICE} -eq 2 ]]
then
    aocl program acl0 ${bitstream_dir}/dyadic_multiply_keyswitch.aocx
fi

echo ""
# batch 1 (default)
FPGA_BITSTREAM=${bitstream_dir}/dyadic_multiply_keyswitch.aocx ./test_dyadic_multiply_keyswitch
# batch 2
echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/dyadic_multiply_keyswitch.aocx FPGA_KERNEL=DYADIC_MULTIPLY_KEYSWITCH BATCH_SIZE_DYADIC_MULTIPLY=2 BATCH_SIZE_KEYSWITCH=2 "
FPGA_BITSTREAM=${bitstream_dir}/dyadic_multiply_keyswitch.aocx FPGA_KERNEL=DYADIC_MULTIPLY_KEYSWITCH BATCH_SIZE_DYADIC_MULTIPLY=2 BATCH_SIZE_KEYSWITCH=2 ./test_dyadic_multiply_keyswitch
