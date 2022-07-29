# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.5.1)

function(bench_function kernel)
    set (KERNEL ${kernel})
    set (SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/bench_${KERNEL}.cpp
    )

    add_executable(bench_${KERNEL} ${SRC})

    target_compile_options(bench_${KERNEL} PRIVATE -fPIE -fPIC -fstack-protector -Wformat -Wformat-security)
    target_include_directories(bench_${KERNEL} PRIVATE $<BUILD_INTERFACE:${CMAKE_INSTALL_PREFIX}/inc>)
    target_link_directories(bench_${KERNEL} PUBLIC ${CMAKE_INSTALL_PREFIX}/lib)
    target_link_directories(bench_${KERNEL} PUBLIC ${CMAKE_BINARY_DIR}/hexl-install/lib)
    target_link_libraries(bench_${KERNEL} PUBLIC hexl-fpga)
    target_link_libraries(bench_${KERNEL} PRIVATE benchmark::benchmark)
    target_link_libraries(bench_${KERNEL} PRIVATE nlohmann_json::nlohmann_json)

    if(FPGA_USE_INTEL_HEXL)
        target_compile_options(bench_${KERNEL} PRIVATE -DFPGA_USE_INTEL_HEXL)
        if(FPGA_BUILD_INTEL_HEXL)
            add_dependencies(bench_${KERNEL} ext_intel_hexl)
            target_include_directories(bench_${KERNEL} PRIVATE ${CMAKE_BINARY_DIR}/hexl-install/${CMAKE_INSTALL_INCLUDEDIR})
            target_link_directories(bench_${KERNEL} PRIVATE ${CMAKE_BINARY_DIR}/hexl-install/${CMAKE_INSTALL_LIBDIR})
            target_link_libraries(bench_${KERNEL} PRIVATE ${CMAKE_BINARY_DIR}/hexl-install/${CMAKE_INSTALL_LIBDIR}/libhexl.so)
        else()
            get_target_property(
                HEXL_INCLUDE_DIR
                HEXL::hexl
                INTERFACE_INCLUDE_DIRECTORIES)
            target_include_directories(bench_${KERNEL} PRIVATE ${HEXL_INCLUDE_DIR})
            target_link_libraries(bench_${KERNEL} PRIVATE HEXL::hexl)
        endif()
    endif(FPGA_USE_INTEL_HEXL)

    if (EMULATION_LIB)
        add_custom_command(
            TARGET bench_${KERNEL} POST_BUILD
            COMMAND rm -f lib${KERNEL}.so
            COMMAND ln -s ${CMAKE_INSTALL_PREFIX}/emu/lib${KERNEL}.so .
            VERBATIM
        )
    else()
        add_custom_command(
            TARGET bench_${KERNEL} POST_BUILD
            COMMAND rm -f lib${KERNEL}.so
            COMMAND ln -s ${CMAKE_INSTALL_PREFIX}/fpga/lib${KERNEL}.so .
            VERBATIM
        )
    endif()

    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/micro_${KERNEL}.sh
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/bitstream_dir.sh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

bench_function(keyswitch)
bench_function(dyadic_multiply)
bench_function(fwd_ntt)
bench_function(inv_ntt)

add_custom_target(bench
    COMMAND ./micro_dyadic_multiply.sh DEPENDS bench_dyadic_multiply
    COMMAND ./micro_fwd_ntt.sh DEPENDS bench_fwd_ntt
    COMMAND ./micro_inv_ntt.sh DEPENDS bench_inv_ntt
    COMMAND ./micro_keyswitch.sh DEPENDS bench_keyswitch
)

add_custom_target(run_bench_ntt
    COMMAND ./micro_fwd_ntt.sh DEPENDS bench_fwd_ntt
)
add_custom_target(run_bench_intt
    COMMAND ./micro_inv_ntt.sh DEPENDS bench_inv_ntt
)
add_custom_target(run_bench_keyswitch
    COMMAND ./micro_keyswitch.sh DEPENDS bench_keyswitch
)
add_custom_target(run_bench_dyadicmult
    COMMAND ./micro_dyadic_multiply.sh DEPENDS bench_dyadic_multiply
)
