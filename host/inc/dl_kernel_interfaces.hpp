// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __DL_KERNEL_INTERFACES_H__
#define __DL_KERNEL_INTERFACES_H__

#include <CL/sycl.hpp>
#include <dlfcn.h>
#include <string>
#include "../../common/types.hpp"

#include "../../device/multlowlvl/include/L1/multLowLvl.h"
#include "../../device/multlowlvl/include/L1/tensorProduct.h"

using namespace L1::helib::bgv;

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


class MultLowLvlDynamicIF : public DynamicIF {
public:
    explicit MultLowLvlDynamicIF(const std::string& lib);

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

    INTT_Method& (*intt1_method_ops)();
    INTT_Method& (*intt2_method_ops)();

    NTT_Method& (*ntt1_method_ops)();
    NTT_Method& (*ntt2_method_ops)();

    std::vector<INTT_Method*> intt_ops_obj;
    std::vector<NTT_Method*> ntt_ops_obj;

};




}  // namespace fpga
}  // namespace hexl
}  // namespace intel

#endif /* __DL_KERNEL_INTERFACES_H__ */
