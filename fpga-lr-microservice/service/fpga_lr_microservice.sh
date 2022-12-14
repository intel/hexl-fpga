# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

set -eo pipefail

spath=$(dirname $0)
. ${spath}/bitstream_dir.sh

if [[ -z ${RUN_CHOICE} ]] || [[ ${RUN_CHOICE} -eq 2 ]]
then
    aocl initialize acl0 pac_s10
fi

########################################
# FPGA run with individual bitstream
########################################

export FPGA_BITSTREAM=${bitstream_dir}/libkeyswitch.so
export FPGA_KERNEL=KEYSWITCH
export BATCH_SIZE_KEYSWITCH=4
ckks_lr -data=kaggle_homecreditdefaultrisk -chunk_size=40 -poly_modulus_degree=16384 -coeff_mod_bit_sizes=52,52,52,50,50,45,27 -scale_bit_size=48 -security_lvl=128
