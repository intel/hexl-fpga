set(CMAKE_CXX_COMPILER dpcpp)
set(KERNEL multlowlvl_top)
set(SRC 
            ./multlowlvl/src/L1/multLowLvl.cpp
            ./multlowlvl/src/L1/tensorProduct.cpp
            ./multlowlvl_top.cpp)

set(AC_TYPES_FLAGS "-qactypes")
set(FPGA_BOARD "de10_agilex:B2E2_8GBx4")

if(DEFINED EMU_ENABLE_DEBUG)
    set(EMU_COMPILE_FLAGS -fPIC -fintelfpga -c -O0 -g ${AC_TYPES_FLAGS})
    set(EMU_LINK_FLAGS -fintelfpga -O0 -g -fPIC)
else()
    set(EMU_COMPILE_FLAGS  -fPIC -fintelfpga -c ${AC_TYPES_FLAGS} -DFPGA_EMULATOR)
    set(EMU_LINK_FLAGS -fintelfpga -O0 -fPIC)
endif()

add_library(${KERNEL}_emu  SHARED ${SRC})
target_compile_options(${KERNEL}_emu PRIVATE ${EMU_COMPILE_FLAGS} ${config_${kernel}})
target_link_options(${KERNEL}_emu PRIVATE ${EMU_LINK_FLAGS})

add_custom_target(emu DEPENDS ${KERNEL}_emu)

########################################################################################################


set(HW_COMPILE_FLAGS -fintelfpga -c -fPIC ${AC_TYPES_FLAGS} )
set(HW_LINK_FLAGS -fintelfpga -fPIC -Xshardware ${AC_TYPES_FLAGS} -Xsboard=${FPGA_BOARD} -Xsseed=789145 
    -Xshyper-optimized-handshaking=off -Xsbsp-flow=flat -Xstiming-failure-mode=ignore -Xsopt-arg=-nocaching)

add_library(${KERNEL}_hw EXCLUDE_FROM_ALL SHARED ${SRC})
target_compile_options(${KERNEL}_hw PRIVATE ${HW_COMPILE_FLAGS} )
target_link_options(${KERNEL}_hw PRIVATE ${HW_LINK_FLAGS})


add_custom_target(fpga DEPENDS ${KERNEL}_hw)