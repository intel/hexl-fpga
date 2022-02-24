# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

kernels=""
kernels+=" dyadic_multiply"
kernels+=" fwd_ntt"
kernels+=" inv_ntt"
kernels+=" keyswitch"
kernels+=" dyadic_multiply_keyswitch"

config_dyadic_multiply=""
config_dyadic_multiply+=" -board=pac_s10_usm"
config_dyadic_multiply+=" -clock=400MHz"

config_fwd_ntt="  -DFPGA_NTT_SIZE=16384"
config_fwd_ntt+=" -DNUM_NTT_COMPUTE_UNITS=1"
config_fwd_ntt+=" -DVEC=8"
config_fwd_ntt+=" -board=pac_s10_usm"
config_fwd_ntt+=" -clock=400MHz"

config_inv_ntt="  -DFPGA_INTT_SIZE=16384"
config_inv_ntt+=" -DNUM_INTT_COMPUTE_UNITS=1"
config_inv_ntt+=" -DVEC_INTT=8"
config_inv_ntt+=" -board=pac_s10_usm"
config_inv_ntt+=" -clock=400MHz"

config_keyswitch=""
config_keyswitch+=" -DCORES=2"
config_keyswitch+=" -no-interleaving=DDR"
config_keyswitch+=" -board=pac_s10"
config_keyswitch+=" -clock=240MHz"

config_dyadic_multiply_keyswitch=${config_dyadic_multiply}
config_dyadic_multiply_keyswitch+=" -DCORES=1"

fpga_args=""
fpga_args+=" -bsp-flow=flat"
fpga_args+=" -seed=789045"
fpga_args+=" -hyper-optimized-handshaking=off"
fpga_args+=" -opt-arg=-nocaching"
fpga_args+=" -dont-error-if-large-area-est"
