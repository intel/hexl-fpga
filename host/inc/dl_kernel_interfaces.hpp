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

/// @brief
/// class NTT Dynamic Interface
/// Interface class for loading NTT fpga bitstreams
/// contained in dynamic libraries
/// @param[in] lib name or path to library that is to be loaded
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

/// @brief
/// class INTT Dynamic Interface
/// Interface class for loading INTT fpga bitstreams
/// contained in dynamic libraries
/// @param[in] lib name or path to library that is to be loaded

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

/// @brief
/// class Dyadic Multiplier Dynamic Interface
/// Interface class for loading Dyadic Multiplier fpga bitstreams
/// contained in dynamic libraries
/// @param[in] lib name or path to library that is to be loaded
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

/// @brief
/// class KeySwitchDynamic Dynamic Interface
/// Interface class for loading KeySwitchDynamic fpga bitstreams
/// contained in dynamic libraries
/// @param[in] lib name or path to library that is to be loaded
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

/// @brief
/// class Dyadic Multiplier and KeySwitchDynamic Dynamic Interface
/// Interface class for loading Dyadic Multiplier and KeySwitchDynamic
/// fpga bitstreams contained in dynamic libraries
/// @param[in] lib name or path to library that is to be loaded
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

class MultiplyByDynamicIF : public DynamicIF {
public:
    explicit MultiplyByDynamicIF(std::string& lib);

    // BringToSet
    typedef void (*LaunchBringToSetINTT_func)(const std::vector<uint64_t>&,
                                              uint64_t, int flag);
    typedef void (*LaunchBringToSetNTT_func)(const std::vector<uint64_t>&,
                                             uint64_t, int flag);
    typedef sycl::event (*LoadBringToSet_func)(sycl::queue&,
                                               sycl::buffer<uint64_t>&,
                                               sycl::buffer<uint8_t>&, unsigned,
                                               int);
    typedef sycl::event (*BringToSet_func)(sycl::queue&, uint32_t,
                                           sycl::buffer<sycl::ulong2>&,
                                           uint32_t, uint32_t, uint, uint64_t);
    typedef sycl::event (*StoreBringToSet_func)(sycl::queue&,
                                                sycl::buffer<uint64_t>&,
                                                unsigned, int);
    LaunchBringToSetINTT_func LaunchBringToSetINTT;
    LaunchBringToSetNTT_func LaunchBringToSetNTT;
    LoadBringToSet_func LoadBringToSet;
    BringToSet_func BringToSet;
    StoreBringToSet_func StoreBringToSet;

    // TensorProduct
    typedef sycl::event (*TensorProduct_func)(
        sycl::queue&, sycl::buffer<ulong>&, sycl::buffer<sycl::ulong>&,
        sycl::buffer<sycl::ulong>&, sycl::buffer<sycl::ulong>&,
        sycl::buffer<sycl::ulong>&, sycl::buffer<sycl::ulong>&,
        sycl::buffer<sycl::ulong4>&, unsigned, int, int, int, int, sycl::event&,
        int);
    typedef sycl::event (*StoreTensorProduct_func)(sycl::queue&,
                                                   sycl::buffer<uint64_t>&,
                                                   sycl::buffer<uint64_t>&,
                                                   sycl::buffer<uint64_t>&,
                                                   unsigned, int);
    TensorProduct_func TensorProduct;
    StoreTensorProduct_func StoreTensorProduct;

    // BreakIntoDigits
    typedef sycl::event (*LoadBreakIntoDigits_func)(sycl::queue&,
                                                     sycl::buffer<uint64_t>&,
                                                     unsigned, int,
                                                     sycl::event);
    typedef sycl::event (*BreakIntoDigits_func)(sycl::queue&,
                                                sycl::buffer<sycl::ulong2>&,
                                                uint, uint, uint, uint, uint,
                                                uint, uint, sycl::event, int);
    typedef sycl::event (*StoreBreakIntoDigits_func)(sycl::queue&,
                                                     sycl::buffer<uint64_t>&,
                                                     unsigned, int);
    typedef void (*LaunchBreakIntoDigitsINTT_func)(const std::vector<uint64_t>&,
                                                   uint64_t, int);
    typedef void (*LaunchBreakIntoDigitsNTT_func)(const std::vector<uint64_t>&,
                                                  uint64_t, int);
    LoadBreakIntoDigits_func LoadBreakIntoDigits;
    BreakIntoDigits_func BreakIntoDigits;
    StoreBreakIntoDigits_func StoreBreakIntoDigits;
    LaunchBreakIntoDigitsINTT_func LaunchBreakIntoDigitsINTT;
    LaunchBreakIntoDigitsNTT_func LaunchBreakIntoDigitsNTT;

    // KeySwitchDigits
    typedef sycl::event (*KeySwitchDigits_func)(
        sycl::queue&, sycl::event, sycl::buffer<sycl::ulong4>&,
        sycl::buffer<sycl::ulong2>&, sycl::buffer<uint64_t>&,
        sycl::buffer<uint64_t>&, sycl::buffer<uint64_t>&,
        sycl::buffer<uint64_t>&, unsigned, unsigned, unsigned, unsigned,
        unsigned);
    typedef sycl::event (*StoreKeySwitchDigits_func)(sycl::queue&,
                                                     sycl::buffer<uint64_t>&,
                                                     sycl::buffer<uint64_t>&,
                                                     unsigned, unsigned);
    KeySwitchDigits_func KeySwitchDigits;
    StoreKeySwitchDigits_func StoreKeySwitchDigits;
};
}  // namespace fpga
}  // namespace hexl
}  // namespace intel

#endif /* __DL_KERNEL_INTERFACES_H__ */
