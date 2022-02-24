# Prerequisite -- Environment Setup for Using Intel FPGA Acceleration Card

## Intel Acceleration Stack

There are two types of Intel Acceleration Stack (version 2.0.1), namely, the [Acceleration Stack for Runtime](https://www.intel.com/content/www/us/en/programmable/f/download/accelerator/license-agreement-pac-d5005.html?swcode=WWW-SWD-IAS-RTE-201) and the [Acceleration Stack for Development](https://www.intel.com/content/altera-www/global/en_us/index/f/download/accelerator/pac-d5005-thank-you.html?swcode=WWW-SWD-IAS-DEV-201). The runtime stack provides a smaller footprint package for software development of runtime host application.  It includes Intel FPGA Runtime Environment (RTE) for OpenCL but does not include Intel Quartus Prime; thus, it assumes that the FPGA bitstreams are available. The development stack allows for accelerator function development using the Intel Quartus Prime Pro Edition software (required and included). Additionally, it comes with the Intel FPGA Software Development Kit (SDK) for OpenCL and the Acceleration Stack. <br>

The Intel Acceleration Stack for development (`d5005_pac_ias_2_0_1_pv_dev_installer.tar.gz`) is encouraged and required to reap the full benefits of Intel HE Acceleration Library for FPGAs, especially before attempting to build the FPGA kernels and if intended usage of Intel HE Acceleration Library for FPGAs includes development contributions. Download, read more detailed installation instructions, updates and related additional resources at [Intel Acceleration Stack link](https://www.intel.com/content/www/us/en/programmable/products/boards_and_kits/dev-kits/altera/intel-fpga-pac-d5005/getting-started.html). <br>

Note: Even though the validated operating system is RHEL 7.6, we used CentOS 7.9 without issues.<br>

## Intel Quartus Prime Pro Edition

Quartus Prime version 19.2 is installed in the Acceleration Stack installation. For Intel HE Acceleration Library for FPGAs, the installation of Quartus Prime version 20.3 is required. [Download the complete version ](https://fpgasoftware.intel.com/20.3/?edition=pro) and follow the instructions below. <br>

```
tar xvf Quartus-pro-20.3.0.158-linux-complete.tar
./setup_pro.sh
```

Use the following configuration:<br>

```
Select the components you want to install:

Quartus Prime Pro Edition [Y/n]: Y

ModelSim -- Intel FPGA Starter Edition (Free) (17169.4MB) [Y/n]: n

ModelSim -- Intel FPGA Edition (Free) (17169.4MB) [Y/n]: n

Intel High Level Synthesis Compiler (2481.9MB) [Y/n]: Y

DSP Builder Pro Edition (185.8MB) [Y/n]: n

Intel FPGA SDK for OpenCL Pro Edition (1822.6MB) [Y/n]: Y
```

Note that the installation of Intel HLS compiler is required to compile the smaller IP modules that support the larger kernels with modular addition and modular multiplication arithmetic.<br>

## Intel OneAPI

Install the oneAPI Toolkit from the following site: <br>

[Installation Guide for Intel OneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html).<br>

More information on the FPGA related components can be found on this [here](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/fpga.html#gs.6nbq2b). <br>

Instructions on direct installation via yum located at [https://software.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/yum-dnf-zypper.html](https://software.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/yum-dnf-zypper.html). <br>

```
sudo yum install intel-basekit
sudo yum install intel-oneapi-intelfpgadpcpp-custom-platforms-quartus20.3
```

## Initializing the FPGA Environment

Upon completion of the installation of the FPGA software stack, the next step is to initialize the environment for FPGA runtime and development. We provide below an example script that automates this process, in particular for the combination of software versions installed. <br>

```
# init_env.sh

export QUARTUS_HOME="/disk1/tools/intelFPGA_pro/19.2/quartus"
export OPAE_PLATFORM_ROOT="/disk1/tools/inteldevstack/d5005_ias_2_0_1_b237"

export AOCL_BOARD_PACKAGE_ROOT="/disk1/tools/inteldevstack/d5005_ias_2_0_1_b237/opencl/opencl_bsp"
if ls /dev/intel-fpga-* 1> /dev/null 2>&1; then
source $AOCL_BOARD_PACKAGE_ROOT/linux64/libexec/setup_permissions.sh >> /dev/null
fi

OPAE_PLATFORM_BIN="/disk1/tools/inteldevstack/d5005_ias_2_0_1_b237/bin"
if [[ ":${PATH}:" = *":${OPAE_PLATFORM_BIN}:"* ]] ;then
    echo "\$OPAE_PLATFORM_ROOT/bin is in PATH already"
else
    echo "Adding \$OPAE_PLATFORM_ROOT/bin to PATH"
    export PATH="${PATH}":"${OPAE_PLATFORM_BIN}"
fi

echo export AOCL_BOARD_PACKAGE_ROOT="/opt/intel/oneapi/intelfpgadpcpp/latest/board/intel_s10sx_pac"
export AOCL_BOARD_PACKAGE_ROOT="/opt/intel/oneapi/intelfpgadpcpp/latest/board/intel_s10sx_pac"
if ls /dev/intel-fpga-* 1> /dev/null 2>&1; then
   echo source $AOCL_BOARD_PACKAGE_ROOT/linux64/libexec/setup_permissions.sh
   source $AOCL_BOARD_PACKAGE_ROOT/linux64/libexec/setup_permissions.sh >> /dev/null
fi

echo export INTELFPGAOCLSDKROOT="/disk1/tools/intelFPGA_pro/20.3/hld"
export INTELFPGAOCLSDKROOT="/disk1/tools/intelFPGA_pro/20.3/hld"

# Enable Backwards Compatibility with older BSP
export ACL_ACDS_VERSION_OVERRIDE="19.2"
export QUARTUS_ROOTDIR_OVERRIDE="/disk1/tools/intelFPGA_pro/quartus_19.2.0b57/quartus"

echo export ALTERAOCLSDKROOT=$INTELFPGAOCLSDKROOT
export ALTERAOCLSDKROOT=$INTELFPGAOCLSDKROOT
export PAC_DMA_WORK_THREAD=yes

QUARTUS_BIN="/disk1/tools/intelFPGA_pro/20.3/quartus/bin"
if [[ ":${PATH}:" = *":${QUARTUS_BIN}:"* ]] ;then
    echo "\$QUARTUS_HOME/bin is in PATH already"
else
    echo "Adding \$QUARTUS_HOME/bin to PATH"
    export PATH="${QUARTUS_BIN}":"${PATH}"
fi

export AOCL_BOARD_PACKAGE_ROOT=/opt/intel/oneapi/intelfpgadpcpp/latest/board/intel_s10sx_pac
source $INTELFPGAOCLSDKROOT/init_opencl.sh >> /dev/null

aocl initialize acl0 pac_s10_usm
```

The script above needs to be modified to reflect the real paths where the installation have been placed. This initialization process requires sudo (administrator) privileges. The OneAPI `AOCL_BOARD_PACKAGE_ROOT` variable value may differ from the example, depending from the source package, which could be `AOCL_BOARD_PACKAGE_ROOT="/opt/intel/oneapi/compiler/2021.3.0/linux/lib/oclfpga/board/intel_s10sx_pac`. Try `locate intel_s10sx_pac` to find out the actual path.

### Fix Permissions on /dev Files

The PAC D5005 requires sudo access with default permissions. This needs to be modified at least once during installation.  The unmodified version of `init_env.sh` performs this step every time it is called, but that requires all users to have sudo access to source `init_env.sh`. Instead, run the permission setting script on its own after modifying `init_env.sh`.

```
$ source $AOCL_BOARD_PACKAGE_ROOT/linux64/libexec/setup_permissions.sh
Configuring locked memory setting
Configuring udev rules for intel-fpga device permission
Configuring system with 2048 2M hugepages
Finished setup_permissions.sh script. All configuration settings are persistent.
```
