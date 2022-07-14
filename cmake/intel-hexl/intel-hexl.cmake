# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

include(ExternalProject)

set(INTEL_HEXL_GIT_REPO_URL https://github.com/intel/hexl.git)
set(INTEL_HEXL_GIT_LABEL v1.2.4)

ExternalProject_Add(
    ext_intel_hexl
    PREFIX ext_intel_hexl
    GIT_REPOSITORY ${INTEL_HEXL_GIT_REPO_URL}
    GIT_TAG ${INTEL_HEXL_GIT_LABEL}
    CMAKE_ARGS
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DHEXL_SHARED_LIB=ON
        -DHEXL_EXPERIMENTAL=ON
        -DHEXL_DEBUG=OFF
        -DHEXL_BENCHMARK=OFF
        -DHEXL_COVERAGE=OFF
        -DHEXL_TESTING=OFF
        -DCMAKE_INSTALL_INCLUDEDIR=include
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/hexl-install
    EXCLUDE_FROM_ALL TRUE
    # Skip updates
    UPDATE_COMMAND "")

ExternalProject_Get_Property(ext_intel_hexl SOURCE_DIR BINARY_DIR)

add_library(libhexl INTERFACE)
add_dependencies(libhexl ext_intel_hexl)
target_include_directories(libhexl INTERFACE ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
target_link_libraries(libhexl INTERFACE ${CMAKE_INSTALL_PREFIX}/${CAMKE_INSTALL_LIBDIR}/libhexl-fpga.so)
