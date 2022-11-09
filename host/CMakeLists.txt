# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.5.1)

if(DEFINED CHECK_BUILD_HOST)
    if(UNIX)
        # Direct CMake to use dpcpp rather than the default C++ compiler/linker
        set(CMAKE_CXX_COMPILER dpcpp)
    else() # Windows
        # Force CMake to use dpcpp rather than the default C++ compiler/linker
        # (needed on Windows only)
        include (CMakeForceCompiler)
        CMAKE_FORCE_CXX_COMPILER (dpcpp IntelDPCPP)
        include (Platform/Windows-Clang)
    endif()
endif()

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE STRING INTERNAL FORCE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build Type: Debug or Release")
set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -Wall")
set(CMAKE_C_FLAGS_DEBUG "-g3 -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3 -funroll-loops -Wall ")
set(CMAKE_C_FLAGS_RELEASE "-march=native -O3 -funroll-loops -Wall ")

set(KERNELS_SRCS
    ${FPGA_SRC_ROOT_DIR}/host/src/multlowlvl.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/dyadic_multiply.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/dl_kernel_interfaces.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/intt.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/ntt.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/keyswitch.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/twiddle-factors.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/number_theory_util.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/fpga_int.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/fpga.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/fpga_context.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/hexl-fpga.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/stack_trace.cpp
)

include(GenerateExportHeader)
add_library(hexl-fpga SHARED ${KERNELS_SRCS})

set_property(TARGET hexl-fpga PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_options(hexl-fpga PRIVATE -fintelfpga -qactypes -Xsv -fPIE -fPIC -fstack-protector -Wformat -Wformat-security)
target_compile_options(hexl-fpga PRIVATE -Wno-address-of-packed-member -Wno-strict-aliasing -Wno-deprecated-declarations)

if(ENABLE_FPGA_DEBUG)
    target_compile_options(hexl-fpga PRIVATE -DFPGA_DEBUG)
endif(ENABLE_FPGA_DEBUG)

if(ENABLE_PROF_KS)
    target_compile_options(hexl-fpga PRIVATE -D__DEBUG_KS_RUNTIME)
endif(ENABLE_PROF_KS)

if(FPGA_USE_INTEL_HEXL)
    target_compile_options(hexl-fpga PRIVATE -DFPGA_USE_INTEL_HEXL)
    if(FPGA_BUILD_INTEL_HEXL)
        add_dependencies(hexl-fpga ext_intel_hexl)
        target_include_directories(hexl-fpga PRIVATE ${CMAKE_BINARY_DIR}/hexl-install/${CMAKE_INSTALL_INCLUDEDIR})
        target_link_directories(hexl-fpga PRIVATE ${CMAKE_BINARY_DIR}/hexl-install/${CMAKE_INSTALL_LIBDIR})
        target_link_libraries(hexl-fpga PRIVATE ${CMAKE_BINARY_DIR}/hexl-install/${CMAKE_INSTALL_LIBDIR}/libhexl.so)
    else()
        get_target_property(
            HEXL_INCLUDE_DIR
            HEXL::hexl
            INTERFACE_INCLUDE_DIRECTORIES)
        target_include_directories(hexl-fpga PRIVATE ${HEXL_INCLUDE_DIR})
        target_link_libraries(hexl-fpga PRIVATE HEXL::hexl)
    endif()
endif(FPGA_USE_INTEL_HEXL)

list(APPEND incPath "${FPGA_SRC_ROOT_DIR}/host/inc")
target_include_directories(hexl-fpga PUBLIC "$<BUILD_INTERFACE:${incPath}>")
target_include_directories(hexl-fpga INTERFACE "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")

target_link_libraries(hexl-fpga PUBLIC rt)
target_link_libraries(hexl-fpga PUBLIC pthread)

install(TARGETS hexl-fpga EXPORT hexl-fpgaTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES inc/hexl-fpga.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
