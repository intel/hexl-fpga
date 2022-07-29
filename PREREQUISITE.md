# Prerequisite -- Environment Setup for Using Intel FPGA Acceleration Card

Intel provides detailed information about [Intel(R) oneAPI Toolkits](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#gs.4ip3tr) on its website, users can find information about how to download, install and use the Intel(R) oneAPI toolkits. The oneAPI version used in the hexl-fpga project is Intel® oneAPI Base Toolkit (version 2022.2.0). <br>

## Intel(R) oneAPI base toolkit installation

Following the [Intel(R) oneAPI toolkits installation guide for Linux* OS](https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top.html) to download and install the Intel(R) oneAPI base basekit. <br>

## Set up a system for FPGA with the Intel(R) PAC

To use Intel(R) PAC FPGA with Intel(R) oneAPI, you also need to install [Intel(R) FPGA add-on package](https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/set-up-a-system-for-fpga-with-the-intel-pac/install-fpga-add-on-for-oneapi-base-toolkit.html) and [Intel(R) PAC software stack](https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/set-up-a-system-for-fpga-with-the-intel-pac/install-intel-pac-software-stack.html).


## Initializing the FPGA Environment

Upon completion of the installation of the FPGA software stack, the next step is to initialize the environment for FPGA runtime and development. Use the below command to initialize the environment whenever you want to use the oneAPI. <br>

```
source /opt/intel/oneapi/setvars.sh
```

## Build and run sample code to test your setup

After you finish the installation, you can build and run a sample code to test your environment. You can find [Intel(R) oneAPI sample code](https://github.com/oneapi-src/oneAPI-samples) in this Github repo. [Every sample project](https://github.com/oneapi-src/oneAPI-samples/tree/master/DirectProgramming/DPC%2B%2BFPGA) has detailed steps to guide you build and run the code on Intel(R) FPGA.
