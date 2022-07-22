// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __DL_KERNEL_INTERFACES_H__
#define __DL_KERNEL_INTERFACES_H__

#include <CL/sycl.hpp>
#include <dlfcn.h>
#include <string>
#include "../../common/types.hpp"
namespace intel {
namespace hexl {
namespace fpga {

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

class NTTDynamicIF : public DynamicIF {
public:
    explicit NTTDynamicIF(const std::string& lib);
    /**
     * @brief function pointer to the forward ntt kernel.
     * counterpart of autotun kernel in OpenCL version.
     */
    void (*fwd_ntt)(sycl::queue& q);

    /**
     * @brief ntt input kernel, help stream input data to the ntt kernel.
     */
    sycl::event (*ntt_input)(sycl::queue& q, unsigned int,
                             uint64_t* __restrict__, uint64_t* __restrict__,
                             uint64_t* __restrict__, uint64_t* __restrict__,
                             uint64_t* __restrict__);

    /**
     * @brief ntt output kernel, help get the results of ntt kernel and write
     * to the device memory.
     */
    sycl::event (*ntt_output)(sycl::queue& q, int, uint64_t* __restrict__);
};

class INTTDynamicIF : public DynamicIF {
public:
    INTTDynamicIF();
    explicit INTTDynamicIF(std::string& lib);

    /**
     * @brief function pointer to the inverse ntt kernel.
     * conuterpart of autorun kernel in OpenCL version.
     */
    // info: integrated
    void (*inv_ntt)(sycl::queue& q);

    /**
     * @brief inverse intt output kernel, help write the results of intt kernel
     * to device memory
     */
    // info: integrated
    sycl::event (*intt_output)(sycl::queue&, unsigned int,
                               unsigned long* __restrict__);

    /**
     * @brief inverse intt input kernel, help stream input data to the intt
     * kernel.
     *
     */
    // info: integrated
    sycl::event (*intt_input)(sycl::queue&, unsigned int,
                              uint64_t* __restrict__, uint64_t* __restrict__,
                              uint64_t* __restrict__, uint64_t* __restrict__,
                              uint64_t* __restrict__, uint64_t* __restrict__);
};

class DyadicMultDynamicIF : public DynamicIF {
public:
    explicit DyadicMultDynamicIF(std::string& lib);
    sycl::event (*input_fifo_usm)(sycl::queue&, uint64_t* __restrict__,
                                  uint64_t* __restrict__, uint64_t,
                                  moduli_info_t* __restrict__, uint64_t, int,
                                  uint64_t*, uint64_t*, uint64_t);

    sycl::event (*output_nb_fifo_usm)(sycl::queue&, uint64_t*, int*, int*);
    void (*submit_autorun_kernels)(sycl::queue& q);
};

class KeySwitchDynamicIF : public DynamicIF {
public:
    explicit KeySwitchDynamicIF(std::string& lib);
    sycl::event (*load)(sycl::queue&, sycl::event*, sycl::buffer<uint64_t>&,
                        moduli_t, uint64_t, uint64_t, uint64_t, invn_t,
                        unsigned);
    sycl::event (*store)(sycl::queue&, sycl::event*,
                         sycl::buffer<sycl::ulong2>&, uint64_t, uint64_t,
                         uint64_t, moduli_t, unsigned, unsigned);
    void (*launchConfigurableKernels)(sycl::queue&, sycl::buffer<uint64_t>*,
                                      unsigned, bool);
    void (*launchStoreSwitchKeys)(sycl::queue&, sycl::buffer<uint256_t>&,
                                  sycl::buffer<uint256_t>&,
                                  sycl::buffer<uint256_t>&, int batch_size);
    void (*launchAllAutoRunKernels)(sycl::queue&);
};

class DyadicMultKeySwitchDynamicIF : public DynamicIF {
public:
    explicit DyadicMultKeySwitchDynamicIF(std::string& lib);

    /**
     * @brief dyadic multiply function pointers.
     */

    sycl::event (*input_fifo_usm)(sycl::queue&, uint64_t* __restrict__,
                                  uint64_t* __restrict__, uint64_t,
                                  moduli_info_t* __restrict__, uint64_t, int,
                                  uint64_t*, uint64_t*, uint64_t);

    sycl::event (*output_nb_fifo_usm)(sycl::queue&, uint64_t*, int*, int*);

    void (*submit_autorun_kernels)(sycl::queue& q);
    sycl::event (*load)(sycl::queue&, sycl::event*, sycl::buffer<uint64_t>&,
                        moduli_t, uint64_t, uint64_t, uint64_t, invn_t,
                        unsigned);

    sycl::event (*store)(sycl::queue&, sycl::event*,
                         sycl::buffer<sycl::ulong2>&, uint64_t, uint64_t,
                         uint64_t, moduli_t, unsigned, unsigned);

    void (*launchConfigurableKernels)(sycl::queue&, sycl::buffer<uint64_t>*,
                                      unsigned, bool);
    void (*launchStoreSwitchKeys)(sycl::queue&, sycl::buffer<uint256_t>&,
                                  sycl::buffer<uint256_t>&,
                                  sycl::buffer<uint256_t>&, int batch_size);

    void (*launchAllAutoRunKernels)(sycl::queue&);
};

}  // namespace fpga
}  // namespace hexl
}  // namespace intel

#endif /* __DL_KERNEL_INTERFACES_H__ */
