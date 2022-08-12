# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

include(ExternalProject)

set(SEAL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext_seal)
set(SEAL_SRC_DIR ${SEAL_PREFIX}/src/ext_seal/)

set(SEAL_REPO_URL https://github.com/microsoft/SEAL.git)
set(SEAL_GIT_TAG v4.0.0)

ExternalProject_Add(
  ext_seal
  GIT_REPOSITORY ${SEAL_REPO_URL}
  GIT_TAG ${SEAL_GIT_TAG}
  PREFIX ${SEAL_PREFIX}
  INSTALL_DIR ${SEAL_PREFIX}
  LIST_SEPARATOR |
  CMAKE_ARGS ${BENCHMARK_FORWARD_CMAKE_ARGS}
    -DSEAL_BUILD_DEPS=OFF
    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_LIBDIR=lib
    -DCMAKE_INSTALL_INCLUDEDIR=include
    -DSEAL_USE_INTEL_HEXL=ON
    -DSEAL_USE_INTEL_HEXL_FPGA=ON
    -DBUILD_SHARED_LIBS=ON
    -DSEAL_USE_ZSTD=OFF
    -DSEAL_USE_ZLIB=OFF
    -DSEAL_USE_MSGSL=OFF
    -DSEAL_BUILD_EXAMPLES=OFF
  PATCH_COMMAND git apply ${CMAKE_SOURCE_DIR}/patches/hexl-fpga-BRIDGE-seal-4.0.0.patch
  UPDATE_COMMAND ""
  DEPENDS ext_hexl)

ExternalProject_Get_Property(ext_seal SOURCE_DIR BINARY_DIR)

add_library(libseal INTERFACE)
add_dependencies(libseal ext_seal)
target_include_directories(libseal INTERFACE ${SEAL_PREFIX}/include/SEAL-4.0)
target_link_libraries(libseal INTERFACE ${SEAL_PREFIX}/lib/libseal.so.4.0)
