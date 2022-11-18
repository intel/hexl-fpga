# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# include_directories(multlowlvl/include)
# include_directories(multlowlvl/src)
# include_directories(multlowlvl)

add_custom_target(emulation
    COMMENT "Compiling for emulation"
    COMMAND bash ${CMAKE_SOURCE_DIR}/device/compile.sh -DFPGA_EMULATOR ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}
)

add_custom_target(fpga
    COMMENT "Compiling FPGA bitstream"
    COMMAND bash ${CMAKE_SOURCE_DIR}/device/compile.sh -Xshardware ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}
)

install(FILES
    ${CMAKE_BINARY_DIR}/device/libdyadic_multiply.so
    ${CMAKE_BINARY_DIR}/device/libfwd_ntt.so
    ${CMAKE_BINARY_DIR}/device/libinv_ntt.so
    ${CMAKE_BINARY_DIR}/device/libkeyswitch.so
    ${CMAKE_BINARY_DIR}/device/libdyadic_multiply_keyswitch.so
    DESTINATION fpga
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
