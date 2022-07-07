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
    aocl program acl0 ${bitstream_dir}/keyswitch.aocx
fi

FPGA_BITSTREAM=${bitstream_dir}/keyswitch.aocx FPGA_KERNEL=KEYSWITCH BATCH_SIZE_KEYSWITCH=1 ./keyswitch-example -test_loops=3 -poly_modulus_degree=16384 -coeff_mod_bit_sizes=52,30,30,40,27,27,27 -scale_bit_size=52 -security_lvl=128
