# Example Using Intel HE Acceleration Library for FPGAs in An External Application

After installing Intel HE Acceleration Library for FPGAs, you can use the library in an external application in the following way.

## Include Intel HE Acceleration Library for FPGAs package into your `CMakeLists.txt`:

```
find_package(hexl-fpga 1.1
    HINTS ${HEXL_FPGA_HINT_DIR}
    REQUIRED)
target_link_libraries(<your target> PRIVATE hexl-fpga::hexl-fpga)
```

If Intel HE Acceleration Library for FPGAs is installed globally, `HEXL_FPGA_HINT_DIR` is not needed. Otherwise, `HEXL_FPGA_HINT_DIR` should be the directory containing  `hexl-fpgaConfig.cmake`, e.g. `${CMAKE_INSTALL_PREFIX}/lib/cmake/`, where ${CMAKE_INSTALL_PREFIX} is the pre-installed directory.

## Build your application with preinstalled Intel HE Acceleration Library for FPGAs package

To build your application with Intel HE Acceleration Library for FPGAs, point ${CMAKE_PREFIX_PATH} to the pre-installed Intel HE Acceleration Library for FPGAs package.
```
mkdir -p build-examples
cd build-examples
cmake ../examples/ -DCMAKE_PREFIX_PATH=${CMAKE_INSTALL_PREFIX}/lib/cmake/ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc
```

## Compile the examples
```
make
```

The compiled executable is located in build-examples/ directory.

## Run the examples

### Run the examples in emulation mode
```
export RUN_CHOICE=1
make examples
```

### Run the examples on FPGA card
```
export RUN_CHOICE=2
make examples
```
