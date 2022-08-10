main: [![Build and Test](https://github.com/intel/hexl-fpga/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/intel/hexl-fpga/actions/workflows/ci.yml)
development: [![Build and Test](https://github.com/intel/hexl-fpga/actions/workflows/ci.yml/badge.svg?branch=development)](https://github.com/intel/hexl-fpga/actions/workflows/ci.yml)

# Intel Homomorphic Encryption (HE) Acceleration Library for FPGAs <br>
Intel:registered: HE Acceleration Library for FPGAs is an open-source library that provides an implementation of homomorphic encryption primitives on Intel FPGAs. Intel HE Acceleration Library for FPGAs targets integer arithmetic with word-sized primes, typically 40-60 bits. Intel HE Acceleration Library for FPGAs provides APIs for 64-bit unsigned integers and targets Intel FPGAs.

## Contents
- [Intel Homomorphic Encryption Acceleration Library for FPGAs](#intel-homomorphic-encryption-acceleration-library-for-fpgas)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Setting up Environment](#setting-up-environment)
  - [Building Intel HE Acceleration Library for FPGAs](#building-intel-he-acceleration-library-for-fpgas)
    - [Dependencies](#dependencies)
    - [Create Build Directory and Configure cmake Build](#create-build-directory-and-configure-cmake-build)
      - [Configuration Options](#configuration-options)
    - [Compiling Intel HE Acceleration Library for FPGAs](#compiling-intel-he-acceleration-library-for-fpgas)
      - [Compiling Device Kernels](#compiling-device-kernels)
        - [Compile Kernels for Emulation](#compile-kernels-for-emulation)
        - [Compile Kernels for Generating FPGA Bitstream](#compile-kernels-for-generating-fpga-bitstream)
      - [Compiling Host Side](#compiling-host-side)
  - [Installing Intel HE Acceleration Library for FPGAs](#installing-intel-he-acceleration-library-for-fpgas)
  - [Testing Intel HE Acceleration Library for FPGAs](#testing-intel-he-acceleration-library-for-fpgas)
    - [Download Test Data for Keyswitch Related Tests](#download-test-data-for-keyswitch-related-tests)
    - [Run Tests in Emulation Mode](#run-tests-in-emulation-mode)
    - [Run Tests on FPGA Card](#run-tests-on-fpga-card)
  - [Benchmarking Intel HE Acceleration Library for FPGAs](#benchmarking-intel-he-acceleration-library-for-fpgas)
    - [Run Benchmarks in Emulation Mode](#run-benchmarks-in-emulation-mode)
    - [Run Benchmarks on FPGA Card](#run-benchmarks-on-fpga-card)
  - [Using Intel HE Acceleration Library for FPGAs](#using-intel-he-acceleration-library-for-fpgas)
  - [Debugging](#debugging)
- [Documentation](#documentation)
    - [Doxygen](#doxygen)
- [Contributing](#contributing)
  - [Pull request acceptance criteria (Pending performance validation)](#pull-request-acceptance-criteria-pending-performance-validation)
  - [Repository layout](#repository-layout)
- [Citing Intel HE Acceleration Library for FPGAs](#citing-intel-he-acceleration-library-for-fpgas)
- [Contributors](#contributors)
- [Contact us](#contact-us)

## Introduction
Many cryptographic applications, particularly homomorphic encryption (HE), rely on integer polynomial arithmetic in a finite field. HE, which enables computation on encrypted data, typically uses polynomials with degree `N:` a power of two roughly in the range `N=[2^{10}, 2^{14}]`. The coefficients of these polynomials are in a finite field with a word-sized primes, `p`, up to `p`~62 bits. More precisely, the polynomials live in the ring `Z_p[X]/(X^N + 1)`. That is, when adding or multiplying two polynomials, each coefficient of the result is reduced by the prime modulus `p`. When multiplying two polynomials, the resulting polynomials of degree `2N` is additionally reduced by taking the remainder when dividing by `X^N+1`.

The primary bottleneck in many HE applications is polynomial-polynomial multiplication in `Z_p[X]/(X^N + 1)`. Intel HE Acceleration Library for FPGAs provides an experimental implementation of the basic primitives for accelerating polynomial multiplication.  We distribute the basic primitives as source code with open source Apache 2.0 license, which allows developers and communities to experiment with the polynomial multiplication.

Ciphertext relinearization and rotation requires the computation intensive operation of keyswitch.  In Intel HE Acceleration Library, we include efficient FPGA kernel implementation for accelerating keyswitch and support various configurations of polynomial sizes and decomposed modulus sizes.

Intel HE Acceleration Library for FPGAs implements the negacyclic number-theoretic transform (NTT) that is commonly used in polynomial multiplication.  We also provide an alternative polynomial multiplication using dyadic multiplication algorithms.  To multiply two polynomials, `p_1(x), p_2(x)` using the NTT, we perform the forward number-theoretic transform on the two input polynomials, then perform an element-wise modular multiplication, and perform the inverse number-theoretic transform on the result.

In sum, Intel HE Acceleration Library for FPGAs implements the following functions:
- Dyadic multiplication
- KeySwitch
- Forward and inverse negacyclic number-theoretic transforms (NTT)

To ensure the correctness of the functions in Intel HE Acceleration Library for FPGAs, the functions support the following configurations.  Dyadic multiplication supports the ciphertext polynomial size of 1024, 2048, 4096, 8192, 16384, and 32768.  Keyswitch supports the ciphertext polynomial size of 1024, 2048, 4096, 8192, and 16384, the decomposed modulus size of no more than seven, and all ciphertext moduli to be no more than 52 bits.  The standalone forward and inverse negacyclic number-theoretic transform functions support the ciphertext polynomial size of 16384.

For each function, the library provides an FPGA implementation using Intel(R) oneAPI.

> **_NOTE:_**  This distribution aims at allowing researchers, developers, and community access to FPGA kernel source code, to experiment with the basic primitives.

> **_NOTE:_**  This distribution provides high performance kernels for running on FPGAs, including dyadic multiplication and keyswitch.  The kernel and the host runtime support batching and streaming between the host and FPGA cards.  Intel Homomorphic Encryption Library for FPGAs enables researchers, developers and community to connect to third party homomorphic encryption libraries through host APIs to accelerate common homomorphic encryption operations.

> **_NOTE:_**  This distribution provides an experimental integrated kernel implementing the dyadic multiplication and keyswitch in one file. We also provide for convenience kernels implementing only one function stand alone. Those FPGA kernels work independently of each other, i.e. one does not require the use of another. The stand alone kernels allow testing and experimentation on a single primitive. For example, the forward NTT and the inverse NTT kernels are for experiments and functional testing only.

## Setting up Environment
To use this code, a prerequisite is to install a PCIe card Intel PAC D5005 and its software stack, named Intel Acceleration Stack, which includes Quartus Prime, Intel FPGA SDK and Intel PAC D5005 board software package. See [PREREQUISITE.md](PREREQUISITE.md) for details. If you have already installed the PCIe card and above mentioned softwares you can skip the procedure in the link given below. <br>

You can find installation instructions for the FPGA PAC D5005 board software package following this link: <br>
[ Hardware/Software Installation link ](https://www.intel.com/content/www/us/en/programmable/documentation/edj1542148561811.html)

Check that your installation is functional with the software environment by running the Hello FPGA test code as indicated in the above link. <br>

## Building Intel HE Acceleration Library for FPGAs
Building Intel HE Acceleration Library for FPGAs library requires building all the depedencies ( mostly dealt automatically by cmake scripts) and two other separate pieces:
- Host application and related dependencies.
- FPGA kernels and HLS libraries needed by the kernels.

From user point of view it is required to go through these two main steps. Without building kernels, tests, benchmark and examples cannot be launched.

### Dependencies
We have tested Intel HE Acceleration Library for FPGAs on the following operating systems:  <br>
- Centos 7.9.2009  <br>
- To check your Centos 7 version: <br>
```
cat /etc/centos-release
```

Intel HE Acceleration Library for FPGAs requires the following dependencies:

| Dependency    | Version                                      |
|---------------|----------------------------------------------|
| Centos 7      | 7.9.2009                                     |
| CMake         | 3.18.2                                       |
| Compiler      | g++ 9.1.0                                    |
| Doxygen       | 1.8.5                                        |
| Hardware      | PCIe Card PAC D5005                          | 

### Create Build Directory and Configure cmake Build
After cloning the git repository into your local area, you can use the following commands to set the install path and create a build directory. It will also create cmake cache files and make files that will be used for building host and kernels. Most of the build options described in previous section can be enabled or disabled by modifying the command given below:

```
cmake -S . -B build -DCMAKE_CXX_COMPILER=dpcpp -DCMAKE_INSTALL_PREFIX=./hexl-fpga-install -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTS=ON -DENABLE_BENCHMARK=ON -DENABLE_DOCS=ON -DENABLE_FPGA_DEBUG=ON
```

Different cmake options are provided allowing users to configure the overall build process. With these options the user can control if it is required to build tests, benchmark etc. Note that by default all options are off: the user must enable at least a few options to create a useful code. The recommended options can be found below. 
The details of these options is given in next section with default selection: <br>

#### Configuration Options
In addition to the standard CMake configuration options, Intel HE Acceleration Library for FPGAs supports several cmake options to configure the build.
For convenience, they are listed below:

| CMake option                  | Values                 |                                                                            |
| ------------------------------| ---------------------- | -------------------------------------------------------------------------- |
| ENABLE_BENCHMARK              | ON / OFF (default OFF) | Set to OFF, enable benchmark suite via Google benchmark                    |
| ENABLE_FPGA_DEBUG             | ON / OFF (default OFF) | Set to OFF, enable debug log at large runtime penalty                      |
| ENABLE_TESTS                  | ON / OFF (default OFF) | Set to OFF, enable building of unit-tests                                  |
| ENABLE_DOCS                   | ON / OFF (default OFF) | Set to OFF, enable building of documentation                               |

### Compiling Intel HE Acceleration Library for FPGAs
Compiling HE Acceleration Library for FPGAs requires two steps: compiling the C++ host code and compiling the oneAPI kernels. Start by compiling the kernels as they will be needed during the host installation.
Before proceeding to the compilations and installation, make sure that your environment variables are set according to the instructions in the Intel PACD5005 Software Package installation guide.

#### Compiling Device Kernels
The kernels can be compiled in two different modes, emulation and FPGA. The emulation mode runs the kernels on the CPU. Compiling in emulation mode takes only a few minutes. The resulting bitstream can be used to verify the functionality of kernels on the CPU. The FPGA mode builds the kernel bitstream for FGPA card. Compiling the kernels in FPGA mode can take a few hours. 


##### Compile Kernels for Emulation
To compile the device kernel for running in emulation mode: <br>
```
cmake --build build --target emulation
```

This command takes a few minutes to execute.

> **_NOTE:_**  If you are interested to run kernels in software emulation mode only then this step is enough and you can move to building the host code. If you want to run the kernels on actual FPGA board please follow the next steps for building bitstream for the FPGA card.


##### Compile Kernels for Generating FPGA Bitstream

To compile the device kernel in fpga mode: <br>
```
cmake --build build --target fpga
```
This command takes a few hours to execute.

The bitstreams will be located in the installation directory specified when calling the cmake command.(See installation below) <br>


#### Compiling Host Side  
To build the host application, tests, benchmark, and documentation (depending on the options selected above) run the following command:
```
cmake --build build 
```      

This will build the Intel HE Acceleration Library for FPGAs in the `build/host/` directory.  <br>

## Installing Intel HE Acceleration Library for FPGAs
After compiling both host side and device kernels, users need to install HE Acceleration Library for FPGAs as a standalone library. The library is used for building and running HE Acceleration Library for FPGAs tests and benchmarks, and it can also be used as a third-party library.  To install Intel HE Acceleration Library for FPGAs to the installation directory specified at configuration time:  <br>
```  
cmake --install build 
``` 

## Testing Intel HE Acceleration Library for FPGAs
To run a set of unit tests via Googletest run the following command ( for running the test you should have chosen  `-DENABLE_TESTS=ON` otherwise tests may not be enabled) (see [Configuration Options](#configuration-options)).  <br>
Make sure that the .aocx files have been installed in the install directory that was chosen during configuration. The default choice we made was "./hexl-fpga-install". <br>

### Download Test Data for Keyswitch Related Tests
For running Keyswitch related tests, users can download the attached testdata.zip in the release v1.1, unzip it to a local directory and point to it through an environment variable KEYSWITCH_DATA_DIR.
```
mkdir test_data_dir
cd test_data_dir
wget https://github.com/intel/hexl-fpga/releases/download/v1.1/testdata.zip
unzip testdata.zip
export KEYSWITCH_DATA_DIR=$PWD/testdata
```

### Run Tests in Emulation Mode
In emulation mode the kernel will run on the CPU and the user will be able to test and validate the kernel. <br>
To run in emulation mode (setting RUN_CHOICE to different values informs host code about emulation mode or FPGA run): <br>

```   
export RUN_CHOICE=1 
cmake --build build --target tests
```  

### Run Tests on FPGA Card
To run using actual FPGA card, run the following command (setting RUN_CHOICE to different values informs host code about emulation mode or FPGA run): <br>

```
export RUN_CHOICE=2 
cmake --build build --target tests
```  
The tests executables are located in `build/tests/` directory <br>

## Benchmarking Intel HE Acceleration Library for FPGAs
To run a set of benchmarks via Google benchmark, configure and build Intel HE Acceleration Library for FPGAs with `-DENABLE_BENCHMARK=ON` (see [Configuration Options](#configuration-options)).  <br>
Make sure that the .aocx files have been installed in `<chosen install directory>/bench/` directory. <br>

### Run Benchmarks in Emulation Mode
To run the benchmark in emulation mode: <br>
```
export RUN_CHOICE=1
cmake --build build --target bench
```
### Run Benchmarks on FPGA Card
To run the benchmark on the fpga, run   <br>
```   
export RUN_CHOICE=2 
cmake --build build --target bench
```
The benchmark executables are located in `build/benchmark/` directory <br>

## Using Intel HE Acceleration Library for FPGAs
The `examples` folder contains an example showing how to use Intel HE Acceleration Library for FPGAs in a third-party project. See  [examples/README.md](examples/README.md) for details.  <br>

## Debugging
For optimal performance, Intel HE Acceleration Library for FPGAs does not perform input validation. In many cases the time required for the validation would be longer than the execution of the function itself. To debug Intel HE Acceleration Library for FPGAs, configure and build Intel HE Acceleration Library for FPGAs with the option <br>
`-DCMAKE_BUILD_TYPE=Debug`
This will generate a debug version of the library that can be used to debug the execution. To enable the FPGA logs, configure the build with `-DENABLE_FPGA_DEBUG=ON` (see [Configuration Options](#configuration-options)).  <br>

 > **_NOTE:_**  Enabling `-DCMAKE_BUILD_TYPE=Debug` will result in a significant runtime overhead.  <br>

# Documentation
See [https://intel.github.io/hexl-fpga](https://intel.github.io/hexl-fpga) for Doxygen documentation. <br>

Intel HE Acceleration Library for FPGAs supports documentation via Doxygen.
To build documentation, first install `doxygen` and `graphviz`, e.g.
```bash
sudo yum install doxygen graphviz
```
Then, configure Intel HE Acceleration Library for FPGAs with `-DENABLE_DOCS=ON` (see [Configuration Options](#configuration-options)).
### Doxygen
 To build Doxygen documentation, after configuring Intel HE Acceleration Library for FPGAs with `-DENABLE_DOCS=ON`, run
```
cmake --build build --target docs
```
To view the generated Doxygen documentation, open the generated `build/doc/doxygen/html/index.html` file in a web browser.
> **_NOTE:_** After running the cmake --install build command, the documentation will also be available in: <br>
`<chosen install directory>/doc/doxygen/html/index.html`.

# Contributing

At this time, Intel HE Acceleration Library for FPGAs welcomes external contributions. To contribute to Intel HE Acceleration Library for FPGAs, see [CONTRIBUTING.md](CONTRIBUTING.md). We encourage feedback and suggestions via Github Issues as well as discussion via Github Discussions.

Please use [pre-commit](https://pre-commit.com/) to validate the formatting of the code before submitting a pull request. <br>
To install pre-commit: <br>
```
pip install --user cpplint pre-commit
```
To run pre-commit:
```
pre-commit run --all
```

## Pull request acceptance criteria (Pending performance validation)  
Pull requests will be accepted if they provide better acceleration, fix a bug or add a desirable new feature.  

Before contributing, please run
```bash
cmake --build build --target tests
```
and make sure pre-commit checks and all unit tests pass. <br>

```
pre-commit run --all
```

## Repository layout
Public headers reside in the `hexl-fpga-install/include` folder.
Private headers, e.g. those containing fpga code should not be put in this folder.


# Citing Intel HE Acceleration Library for FPGAs
To cite Intel HE Acceleration Library for FPGAs, please use the following BibTeX entry.

### Version 2.0
```tex
    @misc{IntelHEXLFPGA,
        author={Meng, Yan and Butt, Shahzad and Wang, Yong and Zhou, Yongfa and Simoni, Steven and others},
        title = {{I}ntel {Homomorphic Encryption Acceleration Library for FPGAs} (Version 2.0)},
        howpublished = {\url{https://github.com/intel/hexl-fpga}},
        month = August,
        year = 2022,
        key = {Intel HE Acceleration Library for FPGAs}  
    }
```

### Version 1.1
```tex
    @misc{IntelHEXLFPGA,
        author={Meng, Yan and Zhou, Yongfa and Butt, Shahzad and González Aragón, Tomás and Wang, Yong and others},
        title = {{I}ntel {Homomorphic Encryption Acceleration Library for FPGAs} (Version 1.1)},
        howpublished = {\url{https://github.com/intel/hexl-fpga}},
        month = March,
        year = 2022,
        key = {Intel HE Acceleration Library for FPGAs}  
    }
```

### Version 1.0
```tex 
    @misc{IntelHEXLFPGA, 
        author={Meng, Yan and de Souza, Fillipe D. M. and Butt, Shahzad and de Lassus, Hubert and González Aragón, Tomás and Zhou, Yongfa and Wang, Yong and others},
        title = {{I}ntel {Homomorphic Encryption Acceleration Library for FPGAs} (Version 1.0)},
        howpublished = {\url{https://github.com/intel/hexl-fpga}},
        month = December,
        year = 2021,
        key = {Intel HE Acceleration Library for FPGAs}
    }
```

# Contributors
The Intel contributors to this project, sorted by last name, are
  - [Paky Abu-Alam](https://www.linkedin.com/in/paky-abu-alam-89797710/)
  - [Tomas Gonzalez Aragon](https://www.linkedin.com/in/tomas-gonzalez-aragon/)
  - [Flavio Bergamaschi](https://www.linkedin.com/in/flavio-bergamaschi-1634141/)
  - [Shahzad Butt](https://www.linkedin.com/in/shahzad-ahmad-butt-4b44971b/)
  - [Hubert de Lassus](https://www.linkedin.com/in/hubert-de-lassus/)
  - [Fillipe D. M. de Souza](https://www.linkedin.com/in/fillipe-d-m-de-souza-a8281820/)
  - [Anil Goteli](https://www.linkedin.com/in/anil-goteti)
  - [Jingyi Jin](https://www.linkedin.com/in/jingyi-jin-655735/)
  - [Yan Meng](https://www.linkedin.com/in/yan-meng-5832895/) (lead)
  - [Nir Peled](https://www.linkedin.com/in/nir-peled-4a52266/)
  - [Steven Simoni](https://www.linkedin.com/in/steven-simoni-0745823)
  - [Dennis Calderon Vega](https://www.linkedin.com/in/dennis-calderon-996840a9/)
  - [Yong Wang](https://github.com/wangyon1/)
  - [Yongfa Zhou](https://www.linkedin.com/in/yongfa-zhou-16217166/)
  
# Contact us
  - he_fpga_support@intel.com
