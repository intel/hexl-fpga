# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.5.1)
project(examples)

set(HEXL_FPGA_HINT_DIR ${CMAKE_PREFIX_PATH})

message(STATUS "CMAKE_PREFIX_INSTALL: ${CMAKE_PREFIX_PATH}")

find_package(hexl-fpga
    HINTS ${HEXL_FPGA_HINT_DIR}
    REQUIRED)
if (NOT TARGET hexl-fpga::hexl-fpga)
    message(STATUS "Intel HE Acceleration Library for FPGAs: not found")
else()
    message(STATUS "Intel HE Acceleration Library for FPGAs: found")
endif()

add_executable(example_dyadic_multiply example_dyadic_multiply.cpp examples.cpp)

target_link_libraries(example_dyadic_multiply PRIVATE hexl-fpga::hexl-fpga)

if(EMULATION_LIB)
    add_custom_command(
        TARGET example_dyadic_multiply POST_BUILD
        COMMAND rm -f libdyadic_multiply_keyswitch.so
        COMMAND ln -s ${HEXL_FPGA_HINT_DIR}../../emu/libdyadic_multiply_keyswitch.so .
        VERBATIM
    )
else()
    add_custom_command(
        TARGET example_dyadic_multiply POST_BUILD
        COMMAND rm -f libdyadic_multiply_keyswitch.so
        COMMAND ln -s ${HEXL_FPGA_HINT_DIR}../../fpga/libdyadic_multiply_keyswitch.so .
        VERBATIM
    )
endif()

add_custom_target(examples
    COMMAND ./example_dyadic_multiply DEPENDS example_dyadic_multiply
)
