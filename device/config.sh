# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

kernels=""
#kernels+=" dyadic_multiply"
#kernels+=" fwd_ntt"
#kernels+=" inv_ntt"
#kernels+=" keyswitch"
#kernels+=" dyadic_multiply_keyswitch"
kernels+=" multlowlvl"

config_dyadic_multiply=""
config_dyadic_multiply+=" -Xsboard=intel_s10sx_pac:pac_s10_usm"
config_dyadic_multiply+=" -Xsclock=400MHz"

config_fwd_ntt="  -DFPGA_NTT_SIZE=16384"
config_fwd_ntt+=" -DNUM_NTT_COMPUTE_UNITS=1"
config_fwd_ntt+=" -DVEC=8"
config_fwd_ntt+=" -Xsboard=intel_s10sx_pac:pac_s10_usm"
config_fwd_ntt+=" -Xsclock=400MHz"

config_inv_ntt="  -DFPGA_INTT_SIZE=16384"
config_inv_ntt+=" -DNUM_INTT_COMPUTE_UNITS=1"
config_inv_ntt+=" -DVEC_INTT=8"
config_inv_ntt+=" -Xsboard=intel_s10sx_pac:pac_s10_usm"
config_inv_ntt+=" -Xsclock=360MHz"

config_keyswitch=""
config_keyswitch+=" -DCORES=1"
config_keyswitch+=" -Xsno-interleaving=DDR"
config_keyswitch+=" -Xshyper-optimized-handshaking=off"
config_keyswitch+=" -Xsboard=intel_s10sx_pac:pac_s10"
config_keyswitch+=" -Xsclock=240MHz"

config_dyadic_multiply_keyswitch=${config_dyadic_multiply}
config_dyadic_multiply_keyswitch+=" -DCORES=1"

config_multlowlvl=""
config_multlowlvl+=" -Xshyper-optimized-handshaking=off"
#config_multlowlvl+=" -Xstiming-failure-mode=ignore"
#config_multlowlvl+=" -Xsopt-arg=-nocaching"
config_multlowlvl+=" -Xsboard=de10_agilex:B2E2_8GBx4"
config_multlowlvl+=" -Xsclock=520MHz"


fpga_args=""
fpga_args+=" -Xsbsp-flow=flat"
fpga_args+=" -Xsseed=789045"
