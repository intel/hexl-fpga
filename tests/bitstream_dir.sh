# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

set -eo pipefail

bitstream_dir="."
if [[ ! -z "${FPGA_BITSTREAM_DIR}" ]];
then
    bitstream_dir=${FPGA_BITSTREAM_DIR}
fi
