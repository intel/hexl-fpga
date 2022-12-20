project(hexl-fpga-helib)

set(TARGET_NAME helib_bgv_l1)
set(SOURCE_FILE
    bringtoset.cpp
    tensorproduct.cpp
    breakintodigits.cpp
    keyswitchdigits.cpp
    )
set(EMULATOR_TARGET ${TARGET_NAME}_emu)
set(FPGA_TARGET ${TARGET_NAME})

# FPGA board selection
if(NOT DEFINED FPGA_BOARD)
    set(FPGA_BOARD "de10_agilex:B2E2_8GBx4")
    message(STATUS "FPGA_BOARD was not specified.\
                    \nConfiguring the design to run on the default FPGA board ${FPGA_BOARD} \
                    \nPlease refer to the README for information on board selection.")
else()
    message(STATUS "Configuring the design to run on FPGA board ${FPGA_BOARD}")
endif()

set(AC_TYPES_FLAG "-qactypes")
set(COMMON_COMPILE_FLAGS "-fintelfpga -fPIC ${AC_TYPES_FLAG} ")
set(COMMON_LINK_FLAGS "-fintelfpga -fPIC ${AC_TYPES_FLAG} -lpthread ")
# A DPC++ ahead-of-time (AoT) compile processes the device code in two stages.
# 1. The "compile" stage compiles the device code to an intermediate representation (SPIR-V).
# 2. The "link" stage invokes the compiler's FPGA backend before linking.
#    For this reason, FPGA backend flags must be passed as link flags in CMake.
set(EMULATOR_COMPILE_FLAGS "${COMMON_COMPILE_FLAGS} -DFPGA_EMULATOR")
set(EMULATOR_LINK_FLAGS "${COMMON_LINK_FLAGS}")
set(HARDWARE_COMPILE_FLAGS "${COMMON_COMPILE_FLAGS}")
string(RANDOM LENGTH 3 ALPHABET 0123456789 SEED)
set(HARDWARE_LINK_FLAGS "${COMMON_LINK_FLAGS} -Xshardware -Xsno-interleaving=DDR -Xshyper-optimized-handshaking=off \
    -Xsopt-arg=-nocaching -Xsbsp-flow=flat -Xstiming-failure-mode=ignore \
    -Xsseed=${SEED} -Xsboard=${FPGA_BOARD} ${USER_HARDWARE_FLAGS}")
# use cmake -D USER_HARDWARE_FLAGS=<flags> to set extra flags for FPGA backend compilation

###############################################################################
### FPGA Emulator
###############################################################################
# To compile in a single command:
#    dpcpp -fintelfpga -DFPGA_EMULATOR host.cpp kernel.cpp -o fast_recompile.fpga_emu
# CMake executes:
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -o host.cpp.o -c host.cpp
#    [compile] dpcpp -fintelfpga -DFPGA_EMULATOR -o kernel.cpp.o -c kernel.cpp
#    [link]    dpcpp -fintelfpga host.cpp.o kernel.cpp.o -o fast_recompile.fpga_emu
add_library(${EMULATOR_TARGET} SHARED ${SOURCE_FILE})
set_target_properties(${EMULATOR_TARGET} PROPERTIES COMPILE_FLAGS "${EMULATOR_COMPILE_FLAGS}")
set_target_properties(${EMULATOR_TARGET} PROPERTIES LINK_FLAGS "${EMULATOR_LINK_FLAGS}")
add_custom_target(fpga_emu DEPENDS ${EMULATOR_TARGET})

###############################################################################
### Generate Report
###############################################################################
# To compile manually:
#   dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> -fsycl-link=early loop_unroll.cpp -o loop_unroll_report.a
set(FPGA_EARLY_IMAGE ${TARGET_NAME}_report.a)
# The compile output is not an executable, but an intermediate compilation result unique to DPC++.
add_executable(${FPGA_EARLY_IMAGE} EXCLUDE_FROM_ALL ${SOURCE_FILE})
add_custom_target(report DEPENDS ${FPGA_EARLY_IMAGE})
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_EARLY_IMAGE} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -fsycl-link=early")
# fsycl-link=early stops the compiler after RTL generation, before invoking Quartus®

###############################################################################
### FPGA Hardware
###############################################################################
# To compile in a single command:
#   dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> loop_unroll.cpp -o loop_unroll.fpga
# CMake executes:
#   [compile] dpcpp -fintelfpga -o loop_unroll.cpp.o -c loop_unroll.cpp
#   [link]    dpcpp -fintelfpga -Xshardware -Xsboard=<FPGA_BOARD> loop_unroll.cpp.o -o loop_unroll.fpga
add_library(${FPGA_TARGET} EXCLUDE_FROM_ALL SHARED ${SOURCE_FILE})
add_custom_target(fpga DEPENDS ${FPGA_TARGET})
set_target_properties(${FPGA_TARGET} PROPERTIES COMPILE_FLAGS "${HARDWARE_COMPILE_FLAGS}")
set_target_properties(${FPGA_TARGET} PROPERTIES LINK_FLAGS "${HARDWARE_LINK_FLAGS} -reuse-exe=${CMAKE_BINARY_DIR}/lib${FPGA_TARGET}.so")

# The -reuse-exe flag enables rapid recompilation of host-only code changes.
# See DPC++FPGA/GettingStarted/fast_recompile for details.

install(TARGETS ${EMULATOR_TARGET} DESTINATION lib)
install(TARGETS ${FPGA_TARGET} EXCLUDE_FROM_ALL DESTINATION lib)
