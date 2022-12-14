# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)
project(fpga-lr-microservice VERSION 4.0.0 LANGUAGES CXX)

include(../cmake/gflags.cmake)

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(hexl-fpga REQUIRED)
find_package(HEXL 1.2.4 REQUIRED)
find_package(SEAL 4.0.0 REQUIRED)
find_package(Microsoft.GSL CONFIG)

set(CKKS_LR_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/ckks_lr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/client.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/blobs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/data_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/seal_ckks_executor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/seal_ckks_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/seal_omp_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/fpga_seal_ckks_executor.cpp
)

add_executable(ckks_lr ${CKKS_LR_SRC})

target_include_directories(ckks_lr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

target_link_libraries(ckks_lr PRIVATE libgflags)
target_link_libraries(ckks_lr PRIVATE Threads::Threads)
target_link_libraries(ckks_lr PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(ckks_lr PRIVATE HEXL::hexl)
target_link_libraries(ckks_lr PRIVATE hexl-fpga::hexl-fpga)
target_link_libraries(ckks_lr PRIVATE SEAL::seal_shared)
target_link_libraries(ckks_lr PRIVATE Microsoft.GSL::GSL)

add_custom_command(
    TARGET ckks_lr POST_BUILD
    COMMAND rm -f datasets
    COMMAND ln -s ${CMAKE_CURRENT_SOURCE_DIR}/../datasets .
    COMMAND rm -f fpga_lr_microservice.sh
    COMMAND ln -s ${CMAKE_CURRENT_SOURCE_DIR}/fpga_lr_microservice.sh .
    COMMAND rm -f bitstream_dir.sh
    COMMAND ln -s ${CMAKE_CURRENT_SOURCE_DIR}/bitstream_dir.sh .
    VERBATIM)

add_custom_target(fpga_lr_microservice
    COMMAND ./fpga_lr_microservice.sh DEPENDS ckks_lr
)

set(CLIENT_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/client.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/client_if.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/blobs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/data_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/seal_ckks_context.cpp
)

set(SERVER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/server_if.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/blobs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/seal_ckks_executor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/seal_ckks_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/seal_omp_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/fpga_seal_ckks_executor.cpp
)

set(IF_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/test_service_if.cpp
)

add_executable(test_service_if ${IF_SRC} ${CLIENT_SRCS} ${SERVER_SRCS})

target_include_directories(test_service_if PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

target_link_libraries(test_service_if PRIVATE libgflags)
target_link_libraries(test_service_if PRIVATE Threads::Threads)
target_link_libraries(test_service_if PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(test_service_if PRIVATE HEXL::hexl)
target_link_libraries(test_service_if PRIVATE hexl-fpga::hexl-fpga)
target_link_libraries(test_service_if PRIVATE SEAL::seal_shared)
target_link_libraries(test_service_if PRIVATE Microsoft.GSL::GSL)

add_custom_command(
    TARGET test_service_if POST_BUILD
    COMMAND rm -f test_service_if.sh
    COMMAND ln -s ${CMAKE_CURRENT_SOURCE_DIR}/test_service_if.sh .
    VERBATIM)

add_custom_target(test_service
    COMMAND ./test_service_if.sh DEPENDS test_service_if
)
