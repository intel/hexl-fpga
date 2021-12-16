# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(lib/hls)

add_custom_target(emulation
    COMMENT "Compiling for emulation"
    COMMAND ${CMAKE_SOURCE_DIR}/device/compile.sh -march=emulator ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}
)
add_dependencies(emulation hls)

add_custom_target(fpga
    COMMENT "Compiling FPGA bitstream"
    COMMAND ${CMAKE_SOURCE_DIR}/device/compile.sh -board=pac_s10_usm ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR}
)
add_dependencies(fpga hls)

install(FILES
    ${CMAKE_BINARY_DIR}/device/dyadic_multiply.aocx
    ${CMAKE_BINARY_DIR}/device/fwd_ntt.aocx
    ${CMAKE_BINARY_DIR}/device/inv_ntt.aocx
    ${CMAKE_BINARY_DIR}/device/hexl_fpga.aocx
    ${CMAKE_BINARY_DIR}/device/keyswitch.aocx
    DESTINATION fpga
    PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
