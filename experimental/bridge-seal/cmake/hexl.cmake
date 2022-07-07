# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

include(ExternalProject)

set(HEXL_GIT_REPO_URL https://github.com/intel/hexl.git)
set(HEXL_GIT_LABEL v1.2.4)

ExternalProject_Add(
    ext_hexl
    PREFIX ext_hexl
    GIT_REPOSITORY ${HEXL_GIT_REPO_URL}
    GIT_TAG ${HEXL_GIT_LABEL}
    CMAKE_ARGS
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DHEXL_SHARED_LIB=ON
        -DHEXL_BENCHMARK=OFF
        -DHEXL_TESTING=OFF
        -DHEXL_EXPERIMENTAL=ON
        -DHEXL_FPGA_COMPATIBILITY=2
    EXCLUDE_FROM_ALL TRUE
    UPDATE_COMMAND "")

ExternalProject_Get_Property(ext_hexl SOURCE_DIR BINARY_DIR)

set(INTEL_HEXL_HINT_DIR ${CMAKE_INSTALL_PREFIX}/lib/cmake)
