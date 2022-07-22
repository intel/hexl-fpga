# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.5.1)
project(hexl-fpga)
set(hexl-fpga_VERSION 2.0)

set(CMAKE_CXX_COMPILER dpcpp)
set(FPGA_SRC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

option(ENABLE_FPGA_DEBUG "Enable FPGA debugging info" OFF)
message(STATUS "Using Compiler for C++ : ${CMAKE_CXX_COMPILER}")
message(STATUS "ENABLE_FPGA_DEBUG: ${ENABLE_FPGA_DEBUG}")
option(ENABLE_TESTS "Enable Tests" OFF)
message(STATUS "ENABLE_TESTS: ${ENABLE_TESTS}")
option(ENABLE_BENCHMARK "Enable Benchmark" OFF)
message(STATUS "ENABLE_BENCHMARK: ${ENABLE_BENCHMARK}")
option(ENABLE_DOCS "Enable Documentation" OFF)
message(STATUS "ENABLE_DOCS: ${ENABLE_DOCS}")

option(FPGA_USE_INTEL_HEXL "Use Intel HEXL" OFF)
message(STATUS "FPGA_USE_INTEL_HEXL: ${FPGA_USE_INTEL_HEXL}")
option(FPGA_BUILD_INTEL_HEXL "Build INTEL HEXL" OFF)
message(STATUS "FPGA_BUILD_INTEL_HEXL: ${FPGA_BUILD_INTEL_HEXL}")

if (LINUX)
    include(GNUInstallDirs)
else()
    set(CMAKE_INSTALL_INCLUDEDIR "include")
    set(CMAKE_INSTALL_LIBDIR "lib")
endif()

if(FPGA_USE_INTEL_HEXL)
    if(FPGA_BUILD_INTEL_HEXL)
        message(STATUS "Intel HEXL: download ...")
        include(cmake/intel-hexl/intel-hexl.cmake)
    else()
        find_package(HEXL 1.2.4)
        if (NOT TARGET HEXL::hexl)
            message(FATAL_ERROR "Intel HEXL: not found")
        endif()
    endif()
endif()


add_subdirectory(host)
add_subdirectory(device)

if(ENABLE_DOCS)
    find_package(Doxygen)
    option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" ${DOXYGEN_FOUND})
    if(BUILD_DOCUMENTATION)
       if(NOT DOXYGEN_FOUND)
          message(FATAL_ERROR "Doxygen was not found (Required)")
       else()
          add_subdirectory(doc)
       endif()
    endif()
endif()

if (ENABLE_TESTS)
    add_subdirectory(cmake/gtest)
    add_subdirectory(tests)
endif (ENABLE_TESTS)

if (ENABLE_BENCHMARK)
    add_subdirectory(cmake/gbenchmark)
    add_subdirectory(benchmark)
endif (ENABLE_BENCHMARK)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/hexl-fpga/hexl-fpgaConfigVersion.cmake"
    VERSION ${hexl-fpga_VERSION}
    COMPATIBILITY AnyNewerVersion
)

export(EXPORT hexl-fpgaTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/hexl-fpga/hexl-fpgaTargets.cmake"
    NAMESPACE hexl-fpga::
)
configure_file(cmake/hexl-fpga/hexl-fpgaConfig.cmake
  "${CMAKE_CURRENT_BINARY_DIR}/hexl-fpga/hexl-fpgaConfig.cmake"
)

set(ConfigPackageLocation ${CMAKE_INSTALL_LIBDIR}/cmake/hexl-fpga)
install (
    EXPORT hexl-fpgaTargets
    FILE hexl-fpgaTargets.cmake
    NAMESPACE hexl-fpga::
    DESTINATION ${ConfigPackageLocation}
)

install(
  FILES
    cmake/hexl-fpga/hexl-fpgaConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/hexl-fpga/hexl-fpgaConfigVersion.cmake"
  DESTINATION
    ${ConfigPackageLocation}
  COMPONENT
    Devel
)
if(ENABLE_DOCS)
install(DIRECTORY ${DOXYGEN_DOC_DIR}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/doc)
endif()
