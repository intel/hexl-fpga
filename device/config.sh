# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

kernels=""
kernels+=" dyadic_multiply"
kernels+=" fwd_ntt"
kernels+=" inv_ntt"
kernels+=" hexl_fpga"
kernels+=" keyswitch"

config_dyadic_multiply=""

config_fwd_ntt="  -DFPGA_NTT_SIZE=16384"
config_fwd_ntt+=" -DNUM_NTT_COMPUTE_UNITS=1"
config_fwd_ntt+=" -DVEC=8"

config_inv_ntt="  -DFPGA_INTT_SIZE=16384"
config_inv_ntt+=" -DNUM_INTT_COMPUTE_UNITS=1"
config_inv_ntt+=" -DVEC_INTT=8"

config_hexl_fpga=${config_dyadic_multiply}
config_hexl_fpga+=${config_fwd_ntt}
config_hexl_fpga+=${config_inv_ntt}

config_keyswitch="-no-interleaving=default"

fpga_args=""
fpga_args+=" -bsp-flow=flat"
fpga_args+=" -clock=400MHz"
fpga_args+=" -seed=789045"
fpga_args+=" -hyper-optimized-handshaking=off"
fpga_args+=" -opt-arg=-nocaching"
