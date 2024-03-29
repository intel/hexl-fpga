
// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dl_kernel_interfaces.hpp"
namespace intel {
namespace hexl {
namespace fpga {

DynamicIF::DynamicIF(const std::string& libName) : m_lib_name_(libName) {
    std::cout << "Using FPGA shared library: " << m_lib_name_ << std::endl;
    m_lib_handle_ = dlopen(m_lib_name_.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!m_lib_handle_) {
        std::cout << "Failed to open dynamic library: " << m_lib_name_
                  << std::endl;
        std::cout << "dlopen error: " << dlerror() << std::endl;
        exit(-1);
    } else {
        std::cout << "Successfully opened library: " << m_lib_name_
                  << std::endl;
    }
}
DynamicIF::~DynamicIF() {
    if (m_lib_handle_) dlclose(m_lib_handle_);
}

void* DynamicIF::loadKernel(const char* kernelName) const {
    void* temp = dlsym(m_lib_handle_, kernelName);
    if (!temp)
        std::cerr << "Cannot load symbol function " << kernelName << std::endl;
    return temp;
}
std::string DynamicIF::getLibName() const { return m_lib_name_; }
/**
 * @brief Construct a new fwd ntt interface::fwd ntt interface object
 *
 * @param lib the pathname to the shared library.
 * Note, the caller must spacify emulator or hardware shared library to use.
 */
NTTDynamicIF::NTTDynamicIF(const std::string& lib)
    : DynamicIF(lib),
      fwd_ntt(nullptr),
      ntt_input(nullptr),
      ntt_output(nullptr) {
    fwd_ntt = (void (*)(sycl::queue&))loadKernel("fwd_ntt");
    ntt_input = (sycl::event(*)(sycl::queue&, unsigned int, uint64_t*,
                                uint64_t*, uint64_t*, uint64_t*,
                                uint64_t*))loadKernel("ntt_input");
    ntt_output =
        (sycl::event(*)(sycl::queue&, int, uint64_t*))loadKernel("ntt_output");
}

INTTDynamicIF::INTTDynamicIF(std::string& lib) : DynamicIF(lib) {
    inv_ntt = (void (*)(sycl::queue&))loadKernel("inv_ntt");
    intt_output =
        (sycl::event(*)(sycl::queue&, unsigned int,
                        unsigned long* __restrict__))loadKernel("intt_output");

    intt_input =
        (sycl::event(*)(sycl::queue&, unsigned int, uint64_t* __restrict__,
                        uint64_t* __restrict__, uint64_t* __restrict__,
                        uint64_t* __restrict__, uint64_t* __restrict__,
                        uint64_t* __restrict__))loadKernel("intt_input");
}

DyadicMultDynamicIF::DyadicMultDynamicIF(std::string& lib) : DynamicIF(lib) {
    input_fifo_usm = (sycl::event(*)(
        sycl::queue&, uint64_t * __restrict__, uint64_t * __restrict__,
        uint64_t, moduli_info_t * __restrict__, uint64_t, int, uint64_t*,
        uint64_t*, uint64_t)) loadKernel("input_fifo_usm");

    output_nb_fifo_usm = (sycl::event(*)(sycl::queue&, uint64_t*, int*,
                                         int*))loadKernel("output_nb_fifo_usm");
    submit_autorun_kernels =
        (void (*)(sycl::queue&))loadKernel("submit_autorun_kernels");
}

KeySwitchDynamicIF::KeySwitchDynamicIF(std::string& lib) : DynamicIF(lib) {
    load = (sycl::event(*)(sycl::queue&, sycl::event*, sycl::buffer<uint64_t>&,
                           moduli_t, uint64_t, uint64_t, uint64_t, invn_t,
                           unsigned))loadKernel("load");

    store = (sycl::event(*)(
        sycl::queue&, sycl::event*, sycl::buffer<sycl::ulong2>&, uint64_t,
        uint64_t, uint64_t, moduli_t, unsigned, unsigned))loadKernel("store");

    launchConfigurableKernels =
        (void (*)(sycl::queue&, sycl::buffer<uint64_t>*, unsigned,
                  bool))loadKernel("launchConfigurableKernels");
    launchStoreSwitchKeys =
        (void (*)(sycl::queue&, sycl::buffer<uint256_t>&,
                  sycl::buffer<uint256_t>&, sycl::buffer<uint256_t>&,
                  int batch_size))loadKernel("launchStoreSwitchKeys");

    launchAllAutoRunKernels =
        (void (*)(sycl::queue&))loadKernel("launchAllAutoRunKernels");
}

DyadicMultKeySwitchDynamicIF::DyadicMultKeySwitchDynamicIF(std::string& lib)
    : DynamicIF(lib) {
    input_fifo_usm = (sycl::event(*)(
        sycl::queue&, uint64_t * __restrict__, uint64_t * __restrict__,
        uint64_t, moduli_info_t * __restrict__, uint64_t, int, uint64_t*,
        uint64_t*, uint64_t)) loadKernel("input_fifo_usm");

    output_nb_fifo_usm = (sycl::event(*)(sycl::queue&, uint64_t*, int*,
                                         int*))loadKernel("output_nb_fifo_usm");

    submit_autorun_kernels =
        (void (*)(sycl::queue&))loadKernel("submit_autorun_kernels");

    load = (sycl::event(*)(sycl::queue&, sycl::event*, sycl::buffer<uint64_t>&,
                           moduli_t, uint64_t, uint64_t, uint64_t, invn_t,
                           unsigned))loadKernel("load");

    store = (sycl::event(*)(
        sycl::queue&, sycl::event*, sycl::buffer<sycl::ulong2>&, uint64_t,
        uint64_t, uint64_t, moduli_t, unsigned, unsigned))loadKernel("store");

    launchConfigurableKernels =
        (void (*)(sycl::queue&, sycl::buffer<uint64_t>*, unsigned,
                  bool))loadKernel("launchConfigurableKernels");
    launchStoreSwitchKeys =
        (void (*)(sycl::queue&, sycl::buffer<uint256_t>&,
                  sycl::buffer<uint256_t>&, sycl::buffer<uint256_t>&,
                  int batch_size))loadKernel("launchStoreSwitchKeys");

    launchAllAutoRunKernels =
        (void (*)(sycl::queue&))loadKernel("launchAllAutoRunKernels");
}

}  // namespace fpga
}  // namespace hexl
}  // namespace intel
