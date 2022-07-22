if(DEFINED CHECK_BUILD_DEVICE)
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
cmake_minimum_required (VERSION 3.4)

project(PROJ_CHECK_BUILD_DEVICE)
endif()

set(DPCPP_VERBOSE "-v" )

## Add different macro to different kernel compilation.
set(config_dyadic_multiply )
set(config_fwd_ntt -DNUM_NTT_COMPUTE_UNITS=1 -DVEC=8 -DFPGA_NTT_SIZE=16384)
set(config_inv_ntt -DNUM_INTT_COMPUTE_UNITS=1 -DVEC_INTT=8 -DFPGA_INTT_SIZE=16384)
set(config_keyswitch -DCORES=1)
set(config_dyadic_multiply_keyswitch -DCORES=1)

set(FMAX_dyadic_multiply 400MHz)
set(FMAX_fwd_ntt 400MHz)
set(FMAX_inv_ntt 360MHz)
set(FMAX_keyswitch 240MHz)
set(FMAX_dyadic_multiply_keyswitch 240MHz)

set(BOARD_dyadic_multiply intel_s10sx_pac:pac_s10_usm)
set(BOARD_fwd_ntt intel_s10sx_pac:pac_s10_usm)
set(BOARD_inv_ntt intel_s10sx_pac:pac_s10_usm)
set(BOARD_keyswitch intel_s10sx_pac:pac_s10)
set(BOARD_dyadic_multiply_keyswitch intel_s10sx_pac:pac_s10)

set(HW_SEED 745873)

set(GLOBAL_MEM_NAME "DDR")
set(DISABLE_INTERLEAVING off)
set(ENABLE_FLAT_BSP on)
set(ENABLE_PROFILING off)
set(ENABLE_SHARED_PERF_COUNTER on)
set(HIGH_EFFORT_COMPILE off)
set(ENABLE_HYPER_OPTIMIZED_HANDSHAKING off)
set(ENABLE_NO_CACHING on)

if (ENABLE_SHARED_PERF_COUNTER)
    set(SHARED_PERF_COUNTER "-Xsprofile-shared-counters")
else()
    set(SHARED_PERF_COUNTER "")
endif()

if(DISABLE_INTERLEAVING)
    set(INTERLEAVING_CONTROL  "-Xsno-interleaving=${GLOBAL_MEM_NAME}" )
    message(STATUS "Disabling default memory interleaving using dpcpp switch : ${INTERLEAVING_CONTROL}")
else()
    message(STATUS "Using default interleaving scheme ...")
    set(INTERLEAVING_CONTROL  "")
endif()

if(ENABLE_FLAT_BSP)
  set(BSP_CHOICE_CONTROL "-Xsbsp-flow=flat")
else()
  set(BSP_CHOICE_CONTROL "")
endif()

if(ENABLE_PROFILING)
	set(PROFILING_CONTROL "-Xsprofile")
else()
	set(PROFILING_CONTROL "")
endif()

if(HIGH_EFFORT_COMPILE)
  set(HIGH_EFFORT_COMPILE_CONTROL "-Xshigh-effort")
else()
  set(HIGH_EFFORT_COMPILE_CONTROL "")
endif()

if (ENABLE_HYPER_OPTIMIZED_HANDSHAKING)
    set(HYPER_OPTIMIZED_HANDSHAKING_SET "-Xshyper-optimized-handshaking=off")
else()
    set(HYPER_OPTIMIZED_HANDSHAKING_SET "")
endif()

if (ENABLE_NO_CACHING)
    set(NOCACHING "-Xsopt-arg=-nocaching")
else()
    set(NOCACHING "")
endif()

set(AC_TYPES_FLAGS "-qactypes")

###############################################################################
### FPGA Emulation compile flow
###############################################################################

option(EMULATION_LIB "Enable FPGA emulation library build flow" OFF)
message(STATUS "EMULATION_LIB: ${EMULATION_LIB}")

if(EMULATION_LIB)

set(EMU_COMPILE_FLAGS -fPIC -fintelfpga -c ${AC_TYPES_FLAGS})
set(EMU_LINK_FLAGS -fintelfpga)

function(build_standalone_emu_shared_lib kernel)
    set(KERNEL ${kernel})
    set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}.cpp)

    message(STATUS "${KERNEL} configurations: ${config_${KERNEL}}")

    add_library(${KERNEL} EXCLUDE_FROM_ALL SHARED ${SRC})
    target_compile_options(${KERNEL} PRIVATE ${EMU_COMPILE_FLAGS} ${config_${kernel}})
    target_link_options(${KERNEL} PRIVATE ${EMU_LINK_FLAGS})

endfunction()

build_standalone_emu_shared_lib(dyadic_multiply)
build_standalone_emu_shared_lib(fwd_ntt)
build_standalone_emu_shared_lib(inv_ntt)
build_standalone_emu_shared_lib(keyswitch)
build_standalone_emu_shared_lib(dyadic_multiply_keyswitch)

add_custom_target(fwd_emu DEPENDS fwd_ntt)
add_custom_target(inv_emu DEPENDS inv_ntt)
add_custom_target(dyadicmult_emu DEPENDS dyadic_multiply)
add_custom_target(ks_emu DEPENDS keyswitch)
add_custom_target(dyadicmult_ks_emu DEPENDS dyadic_multiply_keyswitch)


add_custom_target(emulation
    COMMENT "Compiling for emulation"
    DEPENDS dyadic_multiply_keyswitch dyadic_multiply fwd_ntt inv_ntt keyswitch
)

install(TARGETS dyadic_multiply fwd_ntt inv_ntt keyswitch dyadic_multiply_keyswitch
    LIBRARY
    DESTINATION emu)

else()

##############################################################
# Build FPGA Hardware shared library
##############################################################

set(HW_COMPILE_FLAGS -fPIC -fintelfpga -c ${AC_TYPES_FLAGS} )
set(HW_LINK_FLAGS  -fintelfpga -shared -Xshardware ${AC_TYPES_FLAGS}
    ${INTERLEAVING_CONTROL} ${BSP_CHOICE_CONTROL} ${HIGH_EFFORT_COMPILE_CONTROL} ${PROFILING_CONTROL} ${HYPER_OPTIMIZED_HANDSHAKING_SET} ${NOCACHING})

message(STATUS "Build Hardware shared lib image Compile Flags: ${HW_COMPILE_FLAGS}")
message(STATUS "Build Hardware shared lib image Link Flags: ${HW_LINK_FLAGS}")


function(build_standalone_hw_shared_lib kernel)
    set(KERNEL ${kernel})
    set(SRC ${KERNEL}.cpp)

    add_library(${KERNEL} EXCLUDE_FROM_ALL SHARED ${SRC})
    target_compile_options(${KERNEL} PRIVATE ${HW_COMPILE_FLAGS} ${config_${KERNEL}} )
    target_link_options(${KERNEL} PRIVATE ${HW_LINK_FLAGS} -Xsboard=${BOARD_${KERNEL}} -Xsclock=${FMAX_${KERNEL}} -Xsseed=${HW_SEED})

    message(STATUS "Build ${KERNEL} with target Fmax: ${FMAX_${KERNEL}}, FPGA board: ${BOARD_${KERNEL}}, Seed: ${HW_SEED}")
endfunction()

build_standalone_hw_shared_lib(fwd_ntt)
build_standalone_hw_shared_lib(inv_ntt)
build_standalone_hw_shared_lib(dyadic_multiply)
build_standalone_hw_shared_lib(keyswitch)
build_standalone_hw_shared_lib(dyadic_multiply_keyswitch)

add_custom_target(fwd_hw DEPENDS fwd_ntt)
add_custom_target(inv_hw DEPENDS inv_ntt)
add_custom_target(dyad_hw DEPENDS dyadic_multiply)
add_custom_target(ks_hw DEPENDS keyswitch)
add_custom_target(dyadicmult_ks_hw DEPENDS dyadic_multiply_keyswitch)

add_custom_target(fpga
    COMMENT "Compiling FPGA bitstream"
    DEPENDS dyadic_multiply fwd_ntt inv_ntt keyswitch dyadic_multiply_keyswitch
)

## since FPGA hardware bitstream build prcess takes long time, so the fpga bitstream libary installation
## should seperate from the default installation.
## run "cmake --install build --component hw_install" to install hw bitstream
install(TARGETS  dyadic_multiply fwd_ntt keyswitch dyadic_multiply_keyswitch
    LIBRARY
    DESTINATION fpga
    COMPONENT hw_install
    EXCLUDE_FROM_ALL)

endif()


###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> -fsycl-link=early kernel.cpp -o kernel_report.a
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus


set(REPORT_COMPILE_FLAGS -fPIC -fintelfpga -c ${AC_TYPES_FLAGS})
set(REPORT_LINK_FLAGS -fintelfpga -Xshardware -fsycl-link=early )

function(build_standalone_report kernel)

    set(KERNEL ${kernel})
    set(SRC ${KERNEL}.cpp)

    add_library(${KERNEL}_report EXCLUDE_FROM_ALL SHARED ${SRC})
    target_compile_options(${KERNEL}_report PRIVATE ${REPORT_COMPILE_FLAGS} ${config_${KERNEL}})
    target_link_options(${KERNEL}_report PRIVATE ${REPORT_LINK_FLAGS} -Xsboard=${BOARD_${KERNEL}})

endfunction()

build_standalone_report(fwd_ntt)
build_standalone_report(inv_ntt)
build_standalone_report(dyadic_multiply)
build_standalone_report(keyswitch)

add_custom_target(reports
    DEPENDS fwd_ntt_report inv_ntt_report dyadic_multiply_report keyswitch_report
)
