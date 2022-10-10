// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __DL_KERNEL_INTERFACES_H__
#define __DL_KERNEL_INTERFACES_H__

#include <CL/sycl.hpp>
#include <dlfcn.h>
#include <string>

#include "../device/multlowlvl/include/L1/multLowLvl.h"

/// @brief
/// class Dynamic Interface
/// Parent Interface class for loading fpga bitstreams
/// contained in dynamic libraries
/// @param[in] libName name or path to library that is to be loaded

class DynamicIF {
public:
    explicit DynamicIF(const std::string& libName);
    void* loadKernel(const char* kernelName) const;
    std::string getLibName() const;
    ~DynamicIF();

private:
    void* m_lib_handle_;
    std::string m_lib_name_;
};


class MultLowLvlDynaimcIF : public DyanmicIF {
public:
    explicit MultLowLvlDynaimcIF(const std::string & lib);

    /**
     * see device/multlowlvl.cpp for kernel library interface.
     */


    /**
     * multlowlvl "load functions" interface.
     */

    sycl::event (*BringToSetLoad)(sycl::queue&, sycl::event&, 
                                 sycl::buffer<uint64_t>&,
                                 sycl::buffer<uint8_t>&);
    
    sycl::event (*BringToSetLoad2)(sycl::queue&, sycl::event&, 
                                 sycl::buffer<uint64_t>&,
                                 sycl::buffer<uint8_t>&);
    

    /**
     * multlowlvl "store functions" interface.
     */

    sycl::event (*TensorProductStore0)(sycl::queue&,
                                       sycl::buffer<ulong>&);
    
    sycl::event (*TensorProductStore12)(sycl::queue&,
                                        sycl::buffer<ulong>&,
                                        sycl::buffer<ulong>&);
    

    /**
     * BringToSet kernel interfaces
     */

    sycl::event (*BringToSet)(sycl::queue&, uint32_t,
                             sycl::buffer<ulong2>&, uint32_t,
                             uint32_t, uint, uint64_t);
    
    sycl::event (*BringToSet2)(sycl::queue&, uint32_t,
                               sycl::buffer<ulong2>&, uint32_t,
                               uint32_t, uint, uint64_t);


    /**
     * "TensorProduct" kernel interface.
     */

    sycl::event (*TensorProduct)(sycl::queue&, sycl::buffer<ulong4>&);


    /**
     * intt1_t and intt2_t are class/struct, not functions.
     * Let's try to use dynamic loading machanism load an object.
     * from shared library
     */

    L1::helib::bgv::intt1_t& (*GetINTT1)();

    L1::helib::bgv::intt2_t& (*GetINTT2)();


};

#endif