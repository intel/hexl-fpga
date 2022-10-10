// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dynamic_loading_kernel_IF.hpp"

DynamicIF::DynamicIF(const std::string& libName) : m_lib_name_(libName) {
    std::cout << "Using FPGA shared library: " << m_lib_name_ << std::endl;
    m_lib_handle_ = dlopen(m_lib_name_.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!m_lib_handle_) {
        std::cout << "Failed to open dynamic library: " << m_lib_name_
                  << std::endl;
        std::cout << "dlopen error: " << dlerror() << std::endl;
        exit(-1);
    } else {
        std::cout << "Successfull Opened : " << m_lib_name_ << std::endl;
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

MultLowLvlDynaimcIF::MultLowLvlDynaimcIF(const std::string &lib) 
    : DynamicIF(lib),
    : BringToSetLoad(nullptr),
    : BringToSetLoad2(nullptr),
    : TensorProductStore0(nullptr),
    : TensorProductStore12(nullptr),
    : BringToSet(nullptr),
    : BringToSet2(nullptr),
    : TensorProduct(nullptr),
    : GetINTT1(nullptr),
    : GetINTT2(nullptr) {
    
    BringToSetLoad = (sycl::event (*)(sycl::queue&, sycl::event&, 
                                 sycl::buffer<uint64_t>&,
                                 sycl::buffer<uint8_t>&))loadKernel("BringToSetLoad");
    
    BringToSetLoad2 = (sycl::event (*)(sycl::queue&, sycl::event&, 
                                 sycl::buffer<uint64_t>&,
                                 sycl::buffer<uint8_t>&))loadKernel("BringToSetLoad2");

    TensorProductStore0 = (sycl::event (*)(sycl::queue&,
                                       sycl::buffer<ulong>&))loadKernel("TensorProductStore0");
    
    TensorProductStore12 = (sycl::event (*)(sycl::queue&,
                                        sycl::buffer<ulong>&,
                                        sycl::buffer<ulong>&))loadKernel("TensorProductStore12");
    
    BringToSet = (sycl::event (*)(sycl::queue&, uint32_t,
                             sycl::buffer<ulong2>&, uint32_t,
                             uint32_t, uint, uint64_t))loadKernel("BringToSet");
    
    BringToSet2 = (sycl::event (*)(sycl::queue&, uint32_t,
                               sycl::buffer<ulong2>&, uint32_t,
                               uint32_t, uint, uint64_t))loadKernel("BringToSet2");
    
    TensorProduct = (sycl::event (*)(sycl::queue&, sycl::buffer<ulong4>&))loadKernel("TensorProduct");


    GetINTT1 = (L1::helib::bgv::intt1_t& (*)())loadKernel("GetINTT1");

    GetINTT2 = (L1::helib::bgv::intt2_t& (*)())loadKernel("GetINTT2");


}