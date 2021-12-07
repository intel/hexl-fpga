# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

file(GLOB HLS_SOURCE mod_ops.cpp)
set(HLS_OBJECT mod_ops.aoco)
set(HLS_LIBNAME ip.aoclib)

add_custom_command(
    OUTPUT ${HLS_LIBNAME}
    COMMENT "Generating library of IPs created by HLS"
    COMMAND fpga_crossgen ${HLS_SOURCE} --target ocl
    COMMAND fpga_libtool --target ocl --create ${HLS_LIBNAME} ${HLS_OBJECT}
    DEPENDS ${HLS_SOURCE}
)

add_custom_target(hls DEPENDS ${HLS_LIBNAME})
