# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)
project(fpga-lr-microservice-server VERSION 4.0.0 LANGUAGES CXX)

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(hexl-fpga REQUIRED)
find_package(HEXL 1.2.4 REQUIRED)
find_package(SEAL 4.0.0 REQUIRED)
find_package(Microsoft.GSL CONFIG)

set(SERVER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/server_if.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/blobs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/seal_ckks_executor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/seal_ckks_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/seal_omp_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/fpga_seal_ckks_executor.cpp
)

add_library(fhe_microservice_server SHARED ${SERVER_SRCS})

set_property(TARGET fhe_microservice_server PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(fhe_microservice_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

target_link_libraries(fhe_microservice_server PRIVATE Threads::Threads)
target_link_libraries(fhe_microservice_server PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(fhe_microservice_server PRIVATE HEXL::hexl)
target_link_libraries(fhe_microservice_server PRIVATE hexl-fpga::hexl-fpga)
target_link_libraries(fhe_microservice_server PRIVATE SEAL::seal_shared)
target_link_libraries(fhe_microservice_server PRIVATE Microsoft.GSL::GSL)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../src/server_if.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)

install(TARGETS fhe_microservice_server
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
