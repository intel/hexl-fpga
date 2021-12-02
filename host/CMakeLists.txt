# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.5.1)

set(COMMON_SRCS
    ${FPGA_SRC_ROOT_DIR}/host/common/src/AOCLUtils/opencl.cpp
    ${FPGA_SRC_ROOT_DIR}/host/common/src/AOCLUtils/options.cpp
)

set(KERNELS_SRCS
    ${COMMON_SRCS}
    ${FPGA_SRC_ROOT_DIR}/host/src/dyadic_multiply.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/intt.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/ntt.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/fpga_int.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/fpga.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/fpga_context.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/hexl-fpga.cpp
    ${FPGA_SRC_ROOT_DIR}/host/src/stack_trace.cpp
)

include(GenerateExportHeader)
add_library(hexl-fpga SHARED ${COMMON_SRCS} ${KERNELS_SRCS})

set_property(TARGET hexl-fpga PROPERTY POSITION_INDEPENDENT_CODE ON)

target_compile_options(hexl-fpga PRIVATE -fPIE -fPIC -pie -fstack-protector -Wformat -Wformat-security)
target_compile_options(hexl-fpga PRIVATE -z noexecstack -Wl,-z,relro,-z,now -Wl,-Bsymbolic)

if(ENABLE_FPGA_DEBUG)
    target_compile_options(hexl-fpga PRIVATE -DFPGA_DEBUG)
endif(ENABLE_FPGA_DEBUG)

list(APPEND incPath
    "${OPENCL_ROOT_PATH}/host/include"
    "${FPGA_SRC_ROOT_DIR}/host/common/inc"
    "${FPGA_SRC_ROOT_DIR}/host/inc"
)
target_include_directories(hexl-fpga PUBLIC "$<BUILD_INTERFACE:${incPath}>")
target_include_directories(hexl-fpga INTERFACE "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")

target_link_libraries(hexl-fpga PUBLIC ${OPENCL_ROOT_PATH}/linux64/lib/libOpenCL.so)
target_link_libraries(hexl-fpga PUBLIC rt)
target_link_libraries(hexl-fpga PUBLIC pthread)

install(TARGETS hexl-fpga EXPORT hexl-fpgaTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES inc/hexl-fpga.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ)
