# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

project(seal_bridge_test VERSION 4.0.0 LANGUAGES CXX)

include(cmake/gflags.cmake)

find_package(hexl-fpga REQUIRED)
find_package(HEXL 1.2.4 REQUIRED)
find_package(SEAL 4.0.0 REQUIRED)

add_executable(keyswitch-example)
target_sources(keyswitch-example PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/keyswitch-example.cpp
)

target_compile_options(keyswitch-example PRIVATE -DHEXL_FPGA)

target_link_libraries(keyswitch-example PRIVATE libgflags)

if(TARGET SEAL::seal_shared)
    target_link_libraries(keyswitch-example PRIVATE SEAL::seal_shared)
else()
    message(FATAL_ERROR "Cannot find target SEAL::seal or SEAL::seal_shared")
endif()

if(TARGET hexl-fpga::hexl-fpga)
    target_link_libraries(keyswitch-example PRIVATE hexl-fpga::hexl-fpga)
else()
    message(FATAL_ERROR "Cannot find target hexl-fpga")
endif()

if(TARGET HEXL::hexl)
    target_link_libraries(keyswitch-example PRIVATE HEXL::hexl)
else()
    message(FATAL_ERROR "Cannot find target hexl")
endif()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/seal_test.sh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/bitstream_dir.sh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

list(GET CMAKE_PREFIX_PATH 0 fpga)

add_custom_command(
    TARGET keyswitch-example POST_BUILD
    COMMAND rm -f libkeyswitch.so
    COMMAND ln -s ${fpga}/fpga/libkeyswitch.so .
    VERBATIM)

add_custom_target(seal_test
    COMMAND ./seal_test.sh DEPENDS keyswitch-example
)
