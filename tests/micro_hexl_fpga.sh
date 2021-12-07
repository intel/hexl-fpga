# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

set -eo pipefail

########################################
# FPGA run with integrated bitstream
########################################
if [[ -n ${RUN_CHOICE} ]] && [[ ${RUN_CHOICE} -eq 2 ]]
then
    aocl program acl0 hexl_fpga.aocx
fi

./test_hexl_fpga
