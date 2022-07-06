# Experimental integration of Intel HE Acceleration Library for FPGAs with Microsoft SEAL

## Disclaimer: The experimental integration provides sample code to integrate with external libraries.  It is for experiments only.  You are free to use it at your own risk.  Developers bear no liability of any form.

After installing Intel HE Acceleration Library for FPGAs, you can use the library to accelerate operations in an external library in the following way.

## Experimental integration with [Microsoft SEAL v4.0.0](https://github.com/microsoft/SEAL/tree/v4.0.0)

The integration with Microsoft SEAL is through a patch.  To build SEAL library accelerated with Intel HE Acceleration Library for FPGAs, point ${CMAKE_PREFIX_PATH} to the pre-installed Intel HE Acceleration Library for FPGAs package.
```
cmake -S . -B build-seal4.0.0-fpga -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(realpath ${PWD}/../../hexl-fpga-install)" -DCMAKE_INSTALL_PREFIX=seal-fpga-install
cmake --build build-seal4.0.0-fpga -j
```

## Compile the SEAL-based test that accelerated with Intel HE Acceleration Library for FPGAs.
```
pushd tests
cmake -S . -B build -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(realpath ${PWD}/../../../hexl-fpga-install);$(realpath ${PWD}/../seal-fpga-install)"
cmake --build build -j
popd
```

The compiled executable is located in tests/build directory.

### Run the test on a FPGA card
```
pushd tests/build

export aocx=/path-to-fpga-bitstreams/keyswitch.aocx

export FPGA_BITSTREAM=${aocx}
export FPGA_KERNEL=KEYSWITCH
export BATCH_SIZE_KEYSWITCH=16

export RUN_CHOICE=2

aocl program acl0 ${aocx}

./keyswitch-example -test_loops=3 -poly_modulus_degree=16384 -coeff_mod_bit_sizes=52,30,30,40,27,27,27 -scale_bit_size=52 -security_lvl=128

popd
```
