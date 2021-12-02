# Example Using Intel HEXL for FPGA in An External Application

After installing Intel HEXL for FPGA, you can use Intel HEXL for FPGA in an external application in the following way.

## Include Intel HEXL for FPGA package into your `CMakeLists.txt`:

```
find_package(hexl-fpga 0.1
    HINTS ${HEXL_FPGA_HINT_DIR}
    REQUIRED)
target_link_libraries(<your target> PRIVATE hexl-fpga::hexl-fpga)
```

If Intel HEXL for FPGA is installed globally, `HEXL_FPGA_HINT_DIR` is not needed. Otherwise, `HEXL_FPGA_HINT_DIR` should be the directory containing  `hexl-fpgaConfig.cmake`, e.g. `${CMAKE_INSTALL_PREFIX}/lib/cmake/`, where ${CMAKE_INSTALL_PREFIX} is the pre-installed directory.

## Build your application with preinstalled Intel HEXL for FPGA package

To build your application with the Intel HEXL for FPGA package, point ${CMAKE_PREFIX_PATH} to the pre-installed Intel HEXL for FPGA package.
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
