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
echo "FPGA_BITSTREAM=${bitstream_dir}/libfwd_ntt.so FPGA_KERNEL=NTT"
# batch 1 (default)
FPGA_BITSTREAM=${bitstream_dir}/libfwd_ntt.so FPGA_KERNEL=NTT ./test_fwd_ntt
echo ""
echo "FPGA_BITSTREAM=${bitstream_dir}/libfwd_ntt.so FPGA_KERNEL=NTT BATCH_SIZE_NTT=8"
# batch 8
FPGA_BITSTREAM=${bitstream_dir}/libfwd_ntt.so FPGA_KERNEL=NTT BATCH_SIZE_NTT=8 ./test_fwd_ntt
