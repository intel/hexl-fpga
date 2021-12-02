# Intel Homomorphic Encryption Acceleration Library for FPGAs <br>(Intel HEXL for FPGA)
Intel:registered: HEXL for FPGA is an open-source library that provides an implementation of homomorphic encryption primitives on Intel FPGAs. Intel HEXL for FPGA targets integer arithmetic with word-sized primes, typically 40-60 bits. Intel HEXL for FPGA provides an API for 64-bit unsigned integers and targets Intel FPGAs.

## Contents
- [Intel Homomorphic Encryption Acceleration Library for FPGAs (Intel HEXL for FPGA)](#intel-homomorphic-encryption-acceleration-library-for-fpgas-intel-hexl-for-fpga)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Setting up Environment](#setting-up-environment)
  - [Building Intel HEXL for FPGA](#building-intel-hexl-for-fpga)
    - [Dependencies](#dependencies)
    - [Create Build Directory and Configure cmake Build](#create-build-directory-and-configure-cmake-build)
      - [Configuration Options](#configuration-options)
    - [Compiling Intel HEXL for FPGA](#compiling-intel-hexl-for-fpga)
      - [Compiling Device Kernels](#compiling-device-kernels)
        - [Compile Kernels for Emulation](#compile-kernels-for-emulation)
        - [Compile Kernels for Generating FPGA Bitstream](#compile-kernels-for-generating-fpga-bitstream)
      - [Compiling Host Side](#compiling-host-side)
  - [Installing Intel HEXL for FPGA](#installing-intel-hexl-for-fpga)
  - [Testing Intel HEXL for FPGA](#testing-intel-hexl-for-fpga)
    - [Run Tests in Emulation Mode](#run-tests-in-emulation-mode)
    - [Run Tests on FPGA Card](#run-tests-on-fpga-card)
  - [Benchmarking Intel HEXL for FPGA](#benchmarking-intel-hexl-for-fpga)
    - [Run Benchmarks in Emulation Mode](#run-benchmarks-in-emulation-mode)
    - [Run Benchmarks on FPGA Card](#run-benchmarks-on-fpga-card)
  - [Using Intel HEXL for FPGA](#using-intel-hexl-for-fpga)
  - [Debugging](#debugging)
- [Documentation](#documentation)
    - [Doxygen](#doxygen)
- [Contributing](#contributing)
  - [Pull request acceptance criteria (Pending performance validation)](#pull-request-acceptance-criteria-pending-performance-validation)
  - [Repository layout](#repository-layout)
- [Citing Intel HEXL for FPGA](#citing-intel-hexl-for-fpga)
    - [Version 1.0](#version-10)
- [Contributors](#contributors)
- [Contact us](#contact-us)

## Introduction
Many cryptographic applications, particularly homomorphic encryption (HE), rely on integer polynomial arithmetic in a finite field. HE, which enables computation on encrypted data, typically uses polynomials with degree `N:` a power of two roughly in the range `N=[2^{10}, 2^{14}]`. The coefficients of these polynomials are in a finite field with a word-sized primes, `p`, up to `p`~62 bits. More precisely, the polynomials live in the ring `Z_p[X]/(X^N + 1)`. That is, when adding or multiplying two polynomials, each coefficient of the result is reduced by the prime modulus `p`. When multiplying two polynomials, the resulting polynomials of degree `2N` is additionally reduced by taking the remainder when dividing by `X^N+1`.

 The primary bottleneck in many HE applications is polynomial-polynomial multiplication in `Z_p[X]/(X^N + 1)`. Intel HEXL for FPGA provides the basic primitives that allow polynomial multiplication.   For efficient implementation, Intel HEXL for FPGA uses the negacyclic number-theoretic transform (NTT). To multiply two polynomials, `p_1(x), p_2(x)` using the NTT, we perform the forward number-theoretic transform on the two input polynomials, then perform an element-wise modular multiplication, and perform the inverse number-theoretic transform on the result.

Intel HEXL for FPGA implements the following functions:
- The forward and inverse negacyclic number-theoretic transform (NTT).
- Dyadic multiplication.

For each function, the library provides an FPGA implementation using OpenCL.

For additional functionality, see the public headers, located in `include/hexl-fpga.h`

Note: we provide an integrated kernel implementing the NTT/INTT and the dyadic multiplication in one file. We also provide for convenience kernels implementing only one function stand alone. Those FPGA kernels work independently of each other, i.e. one does not require the use of another. The stand alone kernels allow testing and experimentation on a single primitive.

## Setting up Environment
To use this code, a prerequisite is to install a PCIe card Intel PAC D5005 and its software stack, named Intel Acceleration Stack, which includes Quartus Prime, Intel FPGA SDK and Intel PAC D5005 board software package. See [PREREQUISITE.md](PREREQUISITE.md) for details. If you have already installed the PCIe card and above mentioned softwares you can skip the procedure in the link given below. <br>

You can find installation instructions for the FPGA PAC D5005 board software package following this link: <br>
[ Hardware/Software Installation link ](https://www.intel.com/content/www/us/en/programmable/documentation/edj1542148561811.html)

Check that your installation is functional with the software environment by running the Hello FPGA test code as indicated in the above link. <br>

## Building Intel HEXL for FPGA
Building Intel HEXL for FPGA library requires building all the depedencies ( mostly dealt automatically by cmake scripts) and two other separate pieces:
- Host application and related dependencies.
- FPGA kernels and HLS libraries needed by the kernels.

From user point of view it is required to go through these two main steps. Without building kernels, tests, benchmark and examples cannot be launched.

### Dependencies
We have tested Intel HEXL for FPGA on the following operating systems:  <br>
- Centos 7.9.2009  <br>
- To check your Centos 7 version: <br>
```
cat /etc/centos-release
```

Intel HEXL for FPGA requires the following dependencies:

| Dependency    | Version                                      |
|---------------|----------------------------------------------|
| Centos 7      | 7.9.2009                                     |
| CMake         | >= 3.5                                       |
| Compiler      | g++ >= 9.1                                   |
| Doxygen       | 1.8.5                                        |
| Hardware      | PCIe Card PAC D5005                          | 

### Create Build Directory and Configure cmake Build
After cloning the git repository into your local area, you can use the following commands to set the install path and create a build directory. It will also create cmake cache files and make files that will be used for building host and kernels. Most of the build options described in previous section can be enabled or disabled by modifying the command given below:

```
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=./hexl-fpga-install -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DENABLE_FPGA_DEBUG=ON -DENABLE_TESTS=ON -DENABLE_DOCS=ON -DENABLE_BENCHMARK=ON
```

Different cmake options are provided allowing users to configure the overall build process. With these options the user can control if it is required to build tests, benchmark etc. Note that by default all options are off: the user must enable at least a few options to create a useful code. The recommended options can be found below. 
The details of these options is given in next section with default selection: <br>

#### Configuration Options
In addition to the standard CMake configuration options, Intel HEXL for FPGA supports several cmake options to configure the build.
For convenience, they are listed below:

| CMake option                  | Values                 |                                                                            |
| ------------------------------| ---------------------- | -------------------------------------------------------------------------- |
| ENABLE_BENCHMARK              | ON / OFF (default OFF) | Set to OFF, enable benchmark suite via Google benchmark                   |
| ENABLE_FPGA_DEBUG             | ON / OFF (default OFF) | Set to OFF, enable debug log at large runtime penalty                      |
| ENABLE_TESTS                  | ON / OFF (default OFF) | Set to OFF, enable building of unit-tests                                  |
| ENABLE_DOCS                   | ON / OFF (default OFF) | Set to OFF, enable building of documentation                               |

### Compiling Intel HEXL for FPGA
Compiling HEXL for FPGA requires two steps: compiling the C++ host code and compiling the OpenCL kernels. Start by compiling the kernels as they will be needed during the host installation.
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

This will build the Intel HEXL for FPGA library in the `build/host/` directory.  <br>

## Installing Intel HEXL for FPGA
After compiling both host side and device kernels, users need to install HEXL for FPGA as a standalone library. The library is used for building and running HEXL for FPGA tests and benchmarks, and it can also be used as a third-party library.  To install Intel HEXL for FPGA to the installation directory specified at configuration time:  <br>
```  
cmake --install build 
``` 

## Testing Intel HEXL for FPGA
To run a set of unit tests via Googletest run the following command ( for running the test you should have chosen  `-DENABLE_TESTS=ON` otherwise tests may not be enabled) (see [Configuration Options](#configuration-options)).  <br>
Make sure that the .aocx files have been installed in the install directory that was chosen during configuration. The default choice we made was "./hexl-fpga-install". <br>

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

## Benchmarking Intel HEXL for FPGA
To run a set of benchmarks via Google benchmark, configure and build Intel HEXL for FPGA with `-DENABLE_BENCHMARK=ON` (see [Configuration Options](#configuration-options)).  <br>
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

## Using Intel HEXL for FPGA
The `examples` folder contains an example showing how to use Intel HEXL for FPGA library in a third-party project. See  [examples/README.md](examples/README.md) for details.  <br>

## Debugging
For optimal performance, Intel HEXL for FPGA does not perform input validation. In many cases the time required for the validation would be longer than the execution of the function itself. To debug Intel HEXL for FPGA, configure and build Intel HEXL for FPGA with the option <br>
`-DCMAKE_BUILD_TYPE=Debug`
This will generate a debug version of the library that can be used to debug the execution. To enable the FPGA logs, configure the build with `-DENABLE_FPGA_DEBUG=ON` (see [Configuration Options](#configuration-options)).  <br>

 > **_NOTE:_**  Enabling `-DCMAKE_BUILD_TYPE=Debug` will result in a significant runtime overhead.  <br>

# Documentation
See [https://intel.github.io/hexl-fpga](https://intel.github.io/hexl-fpga) for Doxygen documentation. <br>

Intel HEXL for FPGA supports documentation via Doxygen.
To build documentation, first install `doxygen` and `graphviz`, e.g.
```bash
sudo yum install doxygen graphviz
```
Then, configure Intel HEXL for FPGA with `-DENABLE_DOCS=ON` (see [Configuration Options](#configuration-options)).
### Doxygen
 To build Doxygen documentation, after configuring Intel HEXL for FPGA with `-DENABLE_DOCS=ON`, run
```
cmake --build build --target docs
```
To view the generated Doxygen documentation, open the generated `build/doc/doxygen/html/index.html` file in a web browser.
> **_NOTE:_** After running the cmake --install build command, the documentation will also be available in: <br>
`<chosen install directory>/doc/doxygen/html/index.html`.

# Contributing

At this time, Intel HEXL for FPGA welcomes external contributions. To contribute to Intel HEXL for FPGA, see [CONTRIBUTING.md](CONTRIBUTING.md). We encourage feedback and suggestions via Github Issues as well as discussion via Github Discussions.

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


# Citing Intel HEXL for FPGA
To cite Intel HEXL for FPGA, please use the following BibTeX entry.

### Version 1.0
```tex 
    @misc{IntelHEXLFPGA,
        author={Meng,Yan and de Souza, Fillipe and Butt, Shahzad and de Lassus, Hubert and Gonzales Aragon, Tomas and Zhou, Yongfa and Wang, Yong and others},
        title = {{I}ntel {Homomorphic Encryption Acceleration Library for FPGAs} (release 1.0)},
        howpublished = {\url{https://github.com/intel/hexl-fpga}},
        month = December,
        year = 2021,
        key = {Intel HEXL for FPGA}
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
  - [Yan Meng](https://www.linkedin.com/in/yan-meng-5832895/)
  - [Nir Peled](https://www.linkedin.com/in/nir-peled-4a52266/)
  - [Yong Wang](https://github.com/wangyon1/)
  - [Yongfa Zhou](https://www.linkedin.com/in/yongfa-zhou-16217166/)
  
# Contact us
  - he_fpga_support@intel.com
