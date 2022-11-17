# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.5.1)
include(../cmake/json/json.cmake)


set(CMAKE_CXX_COMPILER dpcpp)

function(test_function kernel)

    set (KERNEL ${kernel})
    set (SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_utils/ntt.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/test_${KERNEL}.cpp
    )

    add_executable(test_${KERNEL} ${SRC})

    target_compile_options(test_${KERNEL} PRIVATE -gdwarf-4 -g3 -O0 -fPIE -fPIC -fstack-protector -Wformat -Wformat-security)
    target_include_directories(test_${KERNEL} PRIVATE $<BUILD_INTERFACE:${CMAKE_INSTALL_PREFIX}/inc>)
    target_include_directories(test_${KERNEL} PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
    target_link_directories(test_${KERNEL} PUBLIC ${CMAKE_INSTALL_PREFIX}/lib)
    target_link_directories(test_${KERNEL} PUBLIC ${CMAKE_BINARY_DIR}/hexl-install/lib)
    target_link_libraries(test_${KERNEL} PUBLIC hexl-fpga)
    target_link_libraries(test_${KERNEL} PRIVATE gtest)
    target_link_libraries(test_${KERNEL} PRIVATE nlohmann_json::nlohmann_json)

    if(ENABLE_FPGA_DEBUG)
        target_compile_options(test_${KERNEL} PRIVATE -DFPGA_DEBUG)
    endif(ENABLE_FPGA_DEBUG)

    add_custom_command(
        TARGET test_${KERNEL} POST_BUILD
        COMMAND rm -f lib${KERNEL}.so
        COMMAND ln -s ${CMAKE_INSTALL_PREFIX}/fpga/lib${KERNEL}.so .
        VERBATIM
    )

    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/micro_${KERNEL}.sh
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/bitstream_dir.sh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

test_function(dyadic_multiply)
test_function(fwd_ntt)
test_function(inv_ntt)
test_function(keyswitch)
test_function(dyadic_multiply_keyswitch)
test_function(multlowlvl)

add_custom_target(tests
    COMMAND bash ./micro_dyadic_multiply.sh DEPENDS test_dyadic_multiply
    COMMAND bash ./micro_fwd_ntt.sh DEPENDS test_fwd_ntt
    COMMAND bash ./micro_inv_ntt.sh DEPENDS test_inv_ntt
    COMMAND bash ./micro_keyswitch.sh DEPENDS test_keyswitch
    COMMAND bash ./micro_dyadic_multiply_keyswitch.sh DEPENDS test_dyadic_multiply_keyswitch
    COMMAND bash ./micro_multlowlvl.sh DEPENDS test_multlowlvl
)

add_custom_target(run_test_multlowlvl
    COMMAND bash ./micro_multlowlvl.sh DEPENDS test_multlowlvl
)
add_custom_target(run_test_keyswitch
    COMMAND bash ./micro_keyswitch.sh DEPENDS test_keyswitch
)
add_custom_target(run_test_fwd_ntt
    COMMAND bash ./micro_fwd_ntt.sh DEPENDS test_fwd_ntt
)
add_custom_target(run_test_inv_ntt
    COMMAND bash ./micro_inv_ntt.sh DEPENDS test_inv_ntt
)
add_custom_target(run_test_dyadic_multiply
    COMMAND bash ./micro_dyadic_multiply.sh DEPENDS test_dyadic_multiply
)
add_custom_target(run_test_dyadic_multiply_keyswitch
    COMMAND bash ./micro_dyadic_multiply_keyswitch.sh DEPENDS test_dyadic_multiply_keyswitch
)
