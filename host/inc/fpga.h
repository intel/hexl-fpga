// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#ifndef __FPGA_H__
#define __FPGA_H__

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "../../common/types.hpp"
#include "dl_kernel_interfaces.hpp"
#include <CL/sycl/INTEL/ac_types/ac_int.hpp>

#define HOST_MEM_ALIGNMENT 64
#define MEM_CHANNEL_K1 1
#define MEM_CHANNEL_K2 2
#define MEM_CHANNEL_K3 3
#define MEM_CHANNEL_K4 4
#define MEM_CHANNEL_TWIDDLES 4

namespace intel {
namespace hexl {
namespace fpga {

__extension__ typedef unsigned __int128 fpga_uint128_t;

/// @brief
/// Struct DyadmultKeys1_t
/// @param[in] key1-5 stores the bits of compressed switch key data
typedef struct {
    uint64_t key1 : 52;
    uint64_t key2 : 52;
    uint64_t key3 : 52;
    uint64_t key4 : 52;
    uint64_t key5 : 48;
} __attribute__((packed)) DyadmultKeys1_t;

/// @brief
/// Struct DyadmultKeys2_t
/// @param[in] key1-6 stores the bits of compressed switch key data
typedef struct {
    uint64_t key1 : 4;
    uint64_t key2 : 52;
    uint64_t key3 : 52;
    uint64_t key4 : 52;
    uint64_t key5 : 52;
    uint64_t key6 : 44;
} __attribute__((packed)) DyadmultKeys2_t;

/// @brief
/// Struct DyadmultKeys3_t
/// @param[in] key1-5 stores the bits of compressed switch key data
typedef struct {
    uint64_t key1 : 8;
    uint64_t key2 : 52;
    uint64_t key3 : 52;
    uint64_t key4 : 52;
    uint64_t key5 : 52;
    uint64_t NOT_USED : 40;
} __attribute__((packed)) DyadmultKeys3_t;

#define BIT_MASK(BITS) ((1UL << BITS) - 1)
#define MAX_RNS_MODULUS_SIZE 7
#define RWMEM_FLAG 1

enum KeySwitch_Kernels {
    KEYSWITCH_LOAD = 0,
    KEYSWITCH_STORE,
    KEYSWITCH_NUM_KERNELS
};

enum class kernel_t {
    NONE,
    DYADIC_MULTIPLY,
    NTT,
    INTT,
    KEYSWITCH,
    DYADIC_MULTIPLY_KEYSWITCH,
    MULTLOWLVL
};

/// @brief
/// Struct Object
/// @param[in] modulus stores the polynomial modulus
/// @param[in] ready_ flag indicating that the Object is ready for processing
/// @param[in] id_ Object local identifier
/// @param[in] g_wid_ Object global identifier
///
class Object {
public:
    explicit Object(kernel_t type = kernel_t::NONE, bool fence = false);
    virtual ~Object() = default;

    bool ready_;
    int id_;

    kernel_t type_;
    bool fence_;
    static unsigned int g_wid_;
};

/// @brief
/// class Object NTT
/// Stores the Number Theoretic Transform parameters
/// @param[in] coeff_poly polynomial coefficients
/// @param[out] coeff_poly polynomial coefficients
/// @param[in] root_of_unity_powers twiddle factors
/// @param[in] precon_root_of_unity_powers inverse twiddle factors
/// @param[in] coeff_modulus polynomial coefficients modulus
/// @param[in] n polynomial size in powers of two
///
class Object_NTT : public Object {
public:
    explicit Object_NTT(uint64_t* coeff_poly,
                        const uint64_t* root_of_unity_powers,
                        const uint64_t* precon_root_of_unity_powers,
                        uint64_t coeff_modulus, uint64_t n, bool fence = false);

    uint64_t* coeff_poly_;
    const uint64_t* root_of_unity_powers_;
    const uint64_t* precon_root_of_unity_powers_;
    uint64_t coeff_modulus_;
    uint64_t n_;
};

/// @brief
/// class Object INTT
/// Stores the Inverse Number Theoretic Transform parameters
/// @param[in] coeff_poly polynomial coefficients
/// @param[out] coeff_poly polynomial coefficients
/// @param[in] inv_root_of_unity_powers twiddle factors for the inverse
/// transform
/// @param[in] precon_inv_root_of_unity_powers inverse twiddle factors for the
/// inverse transform
/// @param[in] coeff_modulus polynomial coefficients modulus
/// @param[in] inv_n  normalization factor for the coefficients
/// @param[in] inv_n_w normalization factor for the constant factor
/// @param[in] n polynomial size
///
class Object_INTT : public Object {
public:
    explicit Object_INTT(uint64_t* coeff_poly,
                         const uint64_t* inv_root_of_unity_powers,
                         const uint64_t* precon_inv_root_of_unity_powers,
                         uint64_t coeff_modulus, uint64_t inv_n,
                         uint64_t inv_n_w, uint64_t n, bool fence = false);

    uint64_t* coeff_poly_;
    const uint64_t* inv_root_of_unity_powers_;
    const uint64_t* precon_inv_root_of_unity_powers_;
    uint64_t coeff_modulus_;
    uint64_t inv_n_;
    uint64_t inv_n_w_;
    uint64_t n_;
};
/// @brief
/// class Object_DyadicMultiply
/// Stores the parameters for the multiplication
/// @param[out] result stores the multiplication result
/// @param[in] operand1 stores the first operand for the multiplication
/// @param[in] operand2 stores the second operand for the multiplication
/// @param[in] n polynomial size
/// @param[in] moduli vector of moduli
/// @param[in] n_moduli size of the vector of moduli
///
class Object_DyadicMultiply : public Object {
public:
    explicit Object_DyadicMultiply(uint64_t* results, const uint64_t* operand1,
                                   const uint64_t* operand2, uint64_t n,
                                   const uint64_t* moduli, uint64_t n_moduli,
                                   bool fence = false);

    uint64_t* results_;
    const uint64_t* operand1_;
    const uint64_t* operand2_;
    uint64_t n_;
    const uint64_t* moduli_;
    uint64_t n_moduli_;
};

/// @brief
/// class Object_KeySwitch
/// Stores the parameters for the keyswitch
/// @param[out] results stores the keyswitch results
/// @param[in]  t_target_iter_ptr stores the input ciphertext data
/// @param[in]  n stores polynomial size
/// @param[in]  decomp_modulus_size stores modulus size
/// @param[in]  key_modulus_size stores key modulus size
/// @param[in]  rns_modulus_size stores the rns modulus size
/// @param[in]  key_component_size stores the key component size
/// @param[in]  moduli stores the moduli
/// @param[in]  k_switch_keys stores the keys for keyswitch operation
/// @param[in]  modswitch_factors stores the factors for modular switch
/// @param[in]  twiddle_factors stores the twiddle factors
/// @param[in]  fence indicates whether the object is a fenced object or not
///
class Object_KeySwitch : public Object {
public:
    explicit Object_KeySwitch(
        uint64_t* result, const uint64_t* t_target_iter_ptr, uint64_t n,
        uint64_t decomp_modulus_size, uint64_t key_modulus_size,
        uint64_t rns_modulus_size, uint64_t key_component_count,
        const uint64_t* moduli, const uint64_t** k_switch_keys,
        const uint64_t* modswitch_factors, const uint64_t* twiddle_factors,
        bool fence = false);

    uint64_t* result_;
    const uint64_t* t_target_iter_ptr_;
    uint64_t n_;
    uint64_t decomp_modulus_size_;
    uint64_t key_modulus_size_;
    uint64_t rns_modulus_size_;
    uint64_t key_component_count_;
    const uint64_t* moduli_;
    const uint64_t** k_switch_keys_;
    const uint64_t* modswitch_factors_;
    const uint64_t* twiddle_factors_;
};


class Object_MultLowLvl : public Object {
public:
    explicit Object_MultLowLvl(uint64_t* a0, uint64_t* a1, uint64_t a_primes_size, uint8_t* a_primes_index, 
                               uint64_t* b0, uint64_t* b1, uint64_t b_primes_size, uint8_t* b_primes_index,
                               uint64_t plainText, uint64_t coeff_count, 
                               uint64_t* c0, uint64_t* c1, uint64_t* c2, uint64_t c_primes_size,
                               uint8_t* output_primes_index, bool fence = false);
    uint64_t* a0_;
    uint64_t* a1_;
    uint64_t a_primes_size_;
    uint8_t* a_primes_index_;
    uint64_t* b0_;
    uint64_t* b1_;
    uint64_t b_primes_size_;
    uint8_t*  b_primes_index_;
    uint64_t plainText_;
    uint64_t coeff_count_;
    uint64_t* c0_;
    uint64_t* c1_;
    uint64_t* c2_;
    uint64_t c_primes_size_;
    uint8_t* output_primes_index_;
};



/// @brief
/// class Buffer
/// Structure containing information for the polynomial operations
/// @param[in] capacity of the buffer
/// @param[in] n_batch_dyadic_multiply batch size for the multiplication
/// @param[in] n_batch_ntt batch size for the Number Theoretical Transform
/// @param[in] n_batch_intt batch size for the inverse Number Theoretical
/// Transform
/// @param[in] n_batch_KeySwitch batch size for the keyswitch
/// @param[in] total_worksize_DyadicMultiply stores the worksize for the
/// multiplication
/// @param[in] num_DyadicMultiply stores the number of multiplications to be
/// performed
/// @param[in] total_worksize_NTT stores the worksize for the NTT
/// @param[in] num_NTT stores the number of NTT to be performed
/// @param[in] total_worksize_INTT stores the worksize for the INTT
/// @param[in] num_INTT stores the number of INTT to be performed
/// @param[in] total_worksize_KeySwitch stores the worksize for the keyswitch
/// @param[in] num_KeySwitch stores the number of keyswitch to be performed
/// @function push pushes an Object in the structure
/// @function front returns the front Object of the structure
/// @function back returns the last Object of the structure
/// @function pop pops the front Object out of the structure
/// @function size returns the size of the structure
/// @function get_worksize_DyadicMultiply returns the worksize of DyadicMultiply
/// @function get_worksize_NTT returns the worksize of NTT
/// @function get_worksize_INTT returns the worksize of INTT
/// @function get_worksize_KeySwitch returns the worksize of KeySwitch
/// @function set_worksize_DyadicMultiply sets the worksize of DyadicMultiply
/// @function set_worksize_NTT sets the worksize of NTT
/// @function set_worksize_INTT sets the worksize of INTT
/// @function set_worksize_KeySwitch sets the worksize of KeySwitch
///
class Buffer {
public:
    Buffer(uint64_t capacity, uint64_t n_batch_dyadic_multiply,
           uint64_t n_batch_ntt, uint64_t n_batch_intt,
           uint64_t n_batch_KeySwitch, uint64_t n_batch_MultLowLvl)
        : capacity_(capacity),
          n_batch_dyadic_multiply_(n_batch_dyadic_multiply),
          n_batch_ntt_(n_batch_ntt),
          n_batch_intt_(n_batch_intt),
          n_batch_KeySwitch_(n_batch_KeySwitch),
          n_batch_MultLowLvl_(n_batch_MultLowLvl),
          total_worksize_DyadicMultiply_(1),
          num_DyadicMultiply_(0),
          total_worksize_NTT_(1),
          num_NTT_(0),
          total_worksize_INTT_(1),
          num_INTT_(0),
          total_worksize_KeySwitch_(1),
          num_KeySwitch_(0),
          total_worksize_MultLowLvl_(1),
          num_MultLowLvl_(0) {}

    void push(Object* obj);
    Object* front() const;
    Object* back() const;
    std::vector<Object*> pop();

    uint64_t size();

    uint64_t get_worksize_DyadicMultiply() const {
        return total_worksize_DyadicMultiply_;
    }
    uint64_t get_worksize_NTT() const { return total_worksize_NTT_; }
    uint64_t get_worksize_INTT() const { return total_worksize_INTT_; }
    uint64_t get_worksize_KeySwitch() const {
        return total_worksize_KeySwitch_;
    }
    uint64_t get_worksize_MultLowLvl() const {
        return total_worksize_MultLowLvl_;
    }

    void set_worksize_DyadicMultiply(uint64_t ws) {
        total_worksize_DyadicMultiply_ = ws;
        num_DyadicMultiply_ = total_worksize_DyadicMultiply_;
    }
    void set_worksize_NTT(uint64_t ws) {
        total_worksize_NTT_ = ws;
        num_NTT_ = total_worksize_NTT_;
    }
    void set_worksize_INTT(uint64_t ws) {
        total_worksize_INTT_ = ws;
        num_INTT_ = total_worksize_INTT_;
    }
    void set_worksize_KeySwitch(uint64_t ws) {
        total_worksize_KeySwitch_ = ws;
        num_KeySwitch_ = total_worksize_KeySwitch_;
    }

    void set_worksize_MultLowLvl(uint64_t ws) {
        total_worksize_MultLowLvl_ = ws;
        num_MultLowLvl_ = total_worksize_MultLowLvl;
    }

private:
    uint64_t get_worksize_int_DyadicMultiply() const {
        return ((num_DyadicMultiply_ > n_batch_dyadic_multiply_)
                    ? n_batch_dyadic_multiply_
                    : num_DyadicMultiply_);
    }

    uint64_t get_worksize_int_NTT() const {
        return ((num_NTT_ > n_batch_ntt_) ? n_batch_ntt_ : num_NTT_);
    }

    uint64_t get_worksize_int_INTT() const {
        return ((num_INTT_ > n_batch_intt_) ? n_batch_intt_ : num_INTT_);
    }

    uint64_t get_worksize_int_KeySwitch() const {
        return ((num_KeySwitch_ > n_batch_KeySwitch_) ? n_batch_KeySwitch_
                                                      : num_KeySwitch_);
    }

    uint64_t get_worksize_int_MultLowLvl() const {
        return ((num_MultLowLvl_ > n_batch_MultLowLvl_) ? n_batch_MultLowLvl_
                                                        : num_MultLowLvl_);
    }

    void update_DyadicMultiply_work_size(uint64_t ws) {
        num_DyadicMultiply_ -= ws;
    }
    void update_NTT_work_size(uint64_t ws) { num_NTT_ -= ws; }
    void update_INTT_work_size(uint64_t ws) { num_INTT_ -= ws; }
    void update_KeySwitch_work_size(uint64_t ws) { num_KeySwitch_ -= ws; }
    void update_MultLowLvl_work_size(uint64_t ws) { num_MultLowLvl_ -= ws;}

    std::mutex mu_;
    std::mutex mu_size_;
    std::condition_variable cond_;
    std::deque<Object*> buffer_;
    const uint64_t capacity_;
    const uint64_t n_batch_dyadic_multiply_;
    const uint64_t n_batch_ntt_;
    const uint64_t n_batch_intt_;
    const uint64_t n_batch_KeySwitch_;
    const uint64_t n_batch_MultLowLvl_;

    uint64_t total_worksize_DyadicMultiply_;
    uint64_t num_DyadicMultiply_;

    uint64_t total_worksize_NTT_;
    uint64_t num_NTT_;

    uint64_t total_worksize_INTT_;
    uint64_t num_INTT_;

    uint64_t total_worksize_KeySwitch_;
    uint64_t num_KeySwitch_;


    uint64_t total_worksize_MultLowLvl_;
    uint64_t num_MultLowLvl_;

};
/// @brief
/// Parent class FPGAObject stores the blob of objects to be transfered to the
/// FPGA
///
/// @function fill_in_data
/// @param[in] vector of objects
/// @function fill_out_data
/// @param[out] vector of results
/// @function recycle releases the content
/// context_ stores the openCL context
/// tag_ stores the blob tag
/// n_batch_ stores the number of batches
/// in_objs_ vector of stored objects
/// g_tag_ stores the global tag identifier
///
class FPGAObject {
public:
    FPGAObject(sycl::queue& p_q, uint64_t n_batch,
               kernel_t type = kernel_t::NONE, bool fence = false);
    virtual ~FPGAObject() = default;
    virtual void fill_in_data(const std::vector<Object*>& objs) = 0;
    virtual void fill_out_data(uint64_t* results) = 0;

    void recycle();

    sycl::queue& m_q;
    int tag_;
    uint64_t n_batch_;
    uint64_t batch_size_;
    kernel_t type_;
    bool fence_;
    std::vector<Object*> in_objs_;

    static std::atomic<int> g_tag_;
};

/// @brief
/// class FPGAObject_NTT stores the NTT blob of objects to be transfered to the
/// FPGA
///
/// @function fill_in_data
/// @param[in] vector of objects
/// @function fill_out_data
/// @param[out] vector of polynomial coefficients
///
/// coeff_poly_in_svm vector of polynomial coefficients
/// root_of_unity_powers_in_svm twiddle factors
/// precon_root_of_unity_powers_in_svm inverse twiddle factors
/// coeff_modulus_in_svm_ polynomial coefficients modulus
/// n polynomial size
///
class FPGAObject_NTT : public FPGAObject {
public:
    explicit FPGAObject_NTT(sycl::queue& p_q, uint64_t coeff_count,
                            uint64_t batch_size);
    ~FPGAObject_NTT();

    FPGAObject_NTT(const FPGAObject_NTT&) = delete;
    FPGAObject_NTT& operator=(const FPGAObject_NTT&) = delete;

    void fill_in_data(const std::vector<Object*>& objs) override;
    void fill_out_data(uint64_t* coeff_poly) override;

    uint64_t* coeff_poly_in_svm_;
    uint64_t* root_of_unity_powers_in_svm_;
    uint64_t* precon_root_of_unity_powers_in_svm_;
    uint64_t* coeff_modulus_in_svm_;
    uint64_t n_;
};

/// @brief
/// class FPGAObject_INTT stores the INTT blob of objects to be transfered to
/// the FPGA
///
/// @function fill_in_data
/// @param[in] vector of objects
/// @function fill_out_data
/// @param[out] vector of polynomial coefficients
///
/// coeff_poly_in_svm vector of polynomial coefficients
/// inv_root_of_unity_powers_in_svm twiddle factors
/// precon_inv_root_of_unity_powers_in_svm inverse twiddle factors
/// coeff_modulus_in_svm_ polynomial coefficients modulus
/// inv_n_in_svm_  normalization factor 1/n for the polynomial coefficients
/// inv_n_w_in_svm_  normalization factor 1/n for the constant coefficient
/// n polynomial size
///
class FPGAObject_INTT : public FPGAObject {
public:
    explicit FPGAObject_INTT(sycl::queue& p_q, uint64_t coeff_count,
                             uint64_t batch_size);
    ~FPGAObject_INTT();
    FPGAObject_INTT(const FPGAObject_INTT&) = delete;
    FPGAObject_INTT& operator=(const FPGAObject_INTT&) = delete;

    void fill_in_data(const std::vector<Object*>& objs) override;
    void fill_out_data(uint64_t* coeff_poly) override;

    uint64_t* coeff_poly_in_svm_;
    uint64_t* inv_root_of_unity_powers_in_svm_;
    uint64_t* precon_inv_root_of_unity_powers_in_svm_;
    uint64_t* coeff_modulus_in_svm_;
    uint64_t* inv_n_in_svm_;
    uint64_t* inv_n_w_in_svm_;
    uint64_t n_;
};

/// @brief
/// class FPGAObject_DyadicMultiply
/// Stores the multiplication blob of objects to be transfered to the FPGA
///
/// @function fill_in_data
/// @param[in] vector of objects
/// @function fill_out_data
/// @param results vector containing the result of the multiplication
///
/// operand1_in_svm_ First operand for the multiplication
/// operand2_in_svm_ Second operand for the multiplication
/// moduli_info_ structure containing information about the moduli
/// n_ polynomial size
/// n_moduli number of moduli
/// operands_in_ddr_ pointer to operands in DDR memory
/// results_out_ddr_ pointer to multiplication results in DDR
///
class FPGAObject_DyadicMultiply : public FPGAObject {
public:
    explicit FPGAObject_DyadicMultiply(sycl::queue& p_q, uint64_t coeff_size,
                                       uint32_t modulus_size,
                                       uint64_t batch_size);
    ~FPGAObject_DyadicMultiply();
    FPGAObject_DyadicMultiply(const FPGAObject_DyadicMultiply&) = delete;
    FPGAObject_DyadicMultiply& operator=(const FPGAObject_DyadicMultiply&) =
        delete;

    void fill_in_data(const std::vector<Object*>& objs) override;
    void fill_out_data(uint64_t* results) override;

    uint64_t* operand1_in_svm_;
    uint64_t* operand2_in_svm_;
    moduli_info_t* moduli_info_;
    uint64_t n_;
    uint64_t n_moduli_;
    uint64_t* operands_in_ddr_;
    uint64_t* results_out_ddr_;
};

/// @brief
/// class FPGAObject_KeySwitch
/// Stores the keyswitch blob of objects to be transfered to the FPGA
///
/// @function fill_in_data
/// @param[in] vector of objects
/// @function fill_out_data
/// @param[out] results stores the keyswitch results
/// results stores the output ciphertext data
/// t_target_iter_ptr stores the input ciphertext data
/// n stores polynomial size
/// decomp_modulus_size stores modulus size
/// key_modulus_size stores key modulus size
/// rns_modulus_size stores the rns modulus size
/// key_component_size stores the key component size
/// moduli stores the moduli
/// k_switch_keys stores the keys for keyswitch operation
/// modswitch_factors stores the factors for modular switch
/// twiddle_factors stores the twiddle factors
///
class FPGAObject_KeySwitch : public FPGAObject {
public:
    explicit FPGAObject_KeySwitch(sycl::queue& p_q, uint64_t batch_size);

    ~FPGAObject_KeySwitch();

    // delete copy and assignment operators ////////////////////////////////
    FPGAObject_KeySwitch(const FPGAObject_KeySwitch&) = delete;
    FPGAObject_KeySwitch& operator=(const FPGAObject_KeySwitch&) = delete;
    ///////////////////////////////////////////////////////////////////////

    void fill_in_data(const std::vector<Object*>& objs) override;
    void fill_out_data(uint64_t* results) override;

    uint64_t n_;
    uint64_t decomp_modulus_size_;
    uint64_t key_modulus_size_;
    uint64_t rns_modulus_size_;
    uint64_t key_component_count_;
    uint64_t* moduli_;
    uint64_t** k_switch_keys_;
    uint64_t* modswitch_factors_;
    uint64_t* twiddle_factors_;
    uint64_t* ms_output_;

    sycl::buffer<uint64_t>* mem_t_target_iter_ptr_;
    sycl::buffer<sycl::ulong2>* mem_KeySwitch_results_;

private:
    enum {
        H_MAX_KEY_MODULUS_SIZE = 7,
        H_MAX_KEY_COMPONENT_SIZE = 2,
        H_MAX_COEFF_COUNT = 16384
    };
};


class FPGAObject_MultLowLvl : public FPGAObject {
public:
    explicit FPGAObject_MultLowLvl(sycl::queue& p_q, uint64_t batch_size);
    ~FPGAObject_MultLowLvl();

    // delete copy and assignment operators ////////////////////////////////
    FPGAObject_MultLowLvl(const FPGAObject_MultLowLvl&) = delete;
    FPGAObject_MultLowLvl& operator=(const FPGAObject_MultLowLvl&) = delete;
    ///////////////////////////////////////////////////////////////////////
    
    void fill_in_data(const std::vector<Object*>& objs) override;
    void fill_out_data(uint64_t* results) override;

    // use buffer to store input data.
    sycl::buffer<uint64_t>* a0_buf_;
    sycl::buffer<uint64_t>* a1_buf_;
    std::vector<uint8_t> *a_primes_index_;
    uint64_t a_primes_size_;

    sycl::buffer<uint64_t>* b0_buf_;
    sycl::buffer<uint64_t>* b1_buf_;
    std::vector<uint8_t> *b_primes_index_;
    uint64_t b_primes_size_;

    uint64_t plainText_;
    uint64_t coeff_count_;

    // store kernel.
    uint64_t c_primes_size_;
    sycl::buffer<uint64_t>* mem_output1_buf_;
    sycl::buffer<uint64_t>* mem_output2_buf_[2];
    sycl::buffer<uint64_t>* mem_output3_buf_[2];

};


template <class t_type = uint256_t>
struct KeySwitchMemKeys {
    //
    explicit KeySwitchMemKeys(sycl::buffer<t_type>* k1 = nullptr,
                              sycl::buffer<t_type>* k2 = nullptr,
                              sycl::buffer<t_type>* k3 = nullptr,
                              t_type* host_k1 = nullptr,
                              t_type* host_k2 = nullptr,
                              t_type* host_k3 = nullptr);
    ~KeySwitchMemKeys();
    sycl::buffer<t_type>* k_switch_keys_1_;
    sycl::buffer<t_type>* k_switch_keys_2_;
    sycl::buffer<t_type>* k_switch_keys_3_;
    t_type* host_k_switch_keys_1_;
    t_type* host_k_switch_keys_2_;
    t_type* host_k_switch_keys_3_;
};

/// @brief
/// enum DEV_TYPE
/// Lists the available device mode: CPU, EMU, FPGA
///
typedef enum { NONE = -1, CPU = 0, EMU, FPGA } DEV_TYPE;
/// @brief
/// Class Device
///
/// @function Device Constructor
/// @param[in] device choice between emulation and FPGA
/// @param[in] buffer memory blob where objects are stored
/// @param[in] exit_signal flag signaling data available
/// @param[in] coeff_size polynomial coefficient size
/// @param[in] modulus_size modulus size
/// @param[in] batch_size_dyadic_multiply batch size for the multiplication
/// operation
/// @param[in] batch_size_ntt batch size for the NTT operation
/// @param[in] batch_size_intt batch size for the INTT operation
/// @param[in] batch_size_KeySwitch batch size for the KeySwitch operation
/// @param[in] debug flag indicating debug mode
///
/// @function run function to launch the operation on the FPGA
///
class Device {
public:
    //
    Device(sycl::device& p_device, Buffer& buffer,
           std::shared_future<bool> exit_signal, uint64_t coeff_size,
           uint32_t modulus_size, uint64_t batch_size_dyadic_multiply,
           uint64_t batch_size_ntt, uint64_t batch_size_intt,
           uint64_t batch_size_KeySwitch, uint64_t batch_size_MultLowLvl, uint32_t debug);
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void run();

private:
    enum { CREDIT = 2 };

    void process_blocking_api();
    bool process_input(int index);
    bool process_output();

    bool process_output_dyadic_multiply();
    bool process_output_NTT();
    bool process_output_INTT();
    bool process_output_KeySwitch();
    bool process_output_MultLowLvl();

    void enqueue_input_data(FPGAObject* fpga_obj);
    void enqueue_input_data_dyadic_multiply(
        FPGAObject_DyadicMultiply* fpga_obj);
    void enqueue_input_data_NTT(FPGAObject_NTT* fpga_obj);
    void enqueue_input_data_INTT(FPGAObject_INTT* fpga_obj);
    void enqueue_input_data_KeySwitch(FPGAObject_KeySwitch* fpga_obj);
    void enqueue_input_data_MultLowLvl(FPGAObject_MultLowLvl* fpga_obj);

    int device_id() { return id_; }

    void KeySwitch_load_twiddles(FPGAObject_KeySwitch* fpga_obj);
    KeySwitchMemKeys<uint256_t>* KeySwitch_check_keys(uint64_t** keys);
    KeySwitchMemKeys<uint256_t>* KeySwitch_load_keys(
        FPGAObject_KeySwitch* fpga_obj);
    void build_modulus_meta(FPGAObject_KeySwitch* fpga_obj);
    void build_invn_meta(FPGAObject_KeySwitch* fpga_obj);
    void KeySwitch_read_output();
    uint64_t precompute_modulus_k(uint64_t modulus);
    void copyKeySwitchBatch(FPGAObject_KeySwitch* fpga_obj, int obj_id);
    
    // MultLowlvl heler functions, added by need.

    template <int id>
    void launch_ntt_config_tf(sycl::queue& q, uint64_t degree, const std::vector<uint64_t> &primes);

    template <int id>
    void launch_compute_forward(sycl::queue &q, uint64_t degree, const std::vector<uint64_t> &primes);

    template <int id>
    void launch_ntt(sycl::queue &q, uint64_t degree, const std::vector<uint64_t> &primes);


    template <int id>
    void launch_intt_config_tf(sycl::queue &q, uint64_t degree, const std::vector<uint64_t> &primes);

    template <int id>
    void launch_compute_inverse(sycl::queue &q, uint64_t degree, const std::vector<uint64_t> &primes);

    template <int id>
    void launch_intt(sycl::queue &q, uint64_t degree, const std::vector<uint64_t> &primes);



    uint64_t precompute_modulus_r(uint64_t modulus);
    void MultLowLvl_Init(uint64_t* primes, uint64_t primes_size);
    void copyMultLowlvlBatch(FPGAObject_MultLowLvl* fpga_obj, int obj_id);
    void MultLowLvl_read_output();
    
    // dynamic loading functions.
    kernel_t get_kernel_type();
    std::string get_bitstream_name();
    void load_kernel_symbols();

    sycl::device device_;
    Buffer& buffer_;
    unsigned int credit_;
    std::shared_future<bool> future_exit_;
    uint64_t* dyadic_multiply_results_out_svm_;
    int* dyadic_multiply_tag_out_svm_;
    int* dyadic_multiply_results_out_valid_svm_;
    uint64_t* NTT_coeff_poly_svm_;
    uint64_t* INTT_coeff_poly_svm_;
    sycl::buffer<uint64_t>* KeySwitch_mem_root_of_unity_powers_;
    bool KeySwitch_load_once_;
    uint64_t* root_of_unity_powers_ptr_;
    moduli_t modulus_meta_;
    invn_t invn_;
    uint64_t KeySwitch_id_;
    std::unordered_map<uint64_t**, KeySwitchMemKeys<uint256_t>*>::iterator
        keys_map_iter_;
    uint32_t debug_;

    // dynamic loading objects
    NTTDynamicIF* ntt_kernel_container_;
    INTTDynamicIF* intt_kernel_container_;
    DyadicMultDynamicIF* dyadicmult_kernel_container_;
    KeySwitchDynamicIF* KeySwitch_kernel_container_;
    MultLowLvlDynamicIF* MultLowLvl_kernel_container_;

    sycl::context context_;
    sycl::queue dyadic_multiply_input_queue_;
    sycl::queue dyadic_multiply_output_queue_;

    // NTT section
    sycl::queue ntt_load_queue_;
    sycl::queue ntt_store_queue_;

    // INTT section
    sycl::queue intt_load_queue_;
    sycl::queue intt_store_queue_;

    // KeySwitch section
    sycl::queue keyswitch_queues_[KEYSWITCH_NUM_KERNELS];
    sycl::event KeySwitch_events_write_[2][1024];
    sycl::event KeySwitch_events_enqueue_[2][2];
    std::unordered_map<uint64_t**, KeySwitchMemKeys<uint256_t>*> keys_map_;

    // MultLowLvl section
    sycl::queue multlowlvl_queues_[MULTLOWLVL_NUM_KERNELS];
    sycl::queue multlowlvl_init_ntt_queues_[2];
    sycl::queue multlowlvl_init_intt_queues_[2];
    std::vector<std::vector<uint8_t>*> *pi_reorder_primes_index;

    // LaunchBringToSet dynamically allocated sycl::buffer.
    sycl::buffer<sycl::ulong2>* scale_param_set_buf_[2];

    // TensorProduct dynamiclly allocated sycl::buffer.
    sycl::buffer<sycl::ulong4>* primes_mulmod_buf_;

    // 

    static int device_id_;
    int id_;
    kernel_t kernel_type_;
    std::vector<FPGAObject*> fpga_objects_;
    static const std::unordered_map<std::string, kernel_t> kernels_;
    
};

/// @brief
/// Class DevicePool
///
/// @function DevicePool constructor
/// @param[in] choice selector for the mode of operation: [CPU,emulation,FPGA]
/// @param[in] buffer to store the blob transferred to the device
/// @param[in] exit_signal flag to indicated data ready
/// @param[in] coeff_size size of the polynomial coefficients
/// @param[in] modulus_size size of the coefficient modulus
/// @param[in] batch_size_dyadic_multiply batch size for the multiplication
/// operation
/// @param[in] batch_size_ntt batch size for the NTT operation
/// @param[in] batch_size_intt batch size for the INTT operation
/// @param[in] batch_size_KeySwitch batch size for the KeySwitch operation
/// @param[in] debug flag indicating debug mode
///
class DevicePool {
public:
    DevicePool(int choice, Buffer& buffer, std::future<bool>& exit_signal,
               uint64_t coeff_size, uint32_t modulus_size,
               uint64_t batch_size_dyadic_multiply, uint64_t batch_size_ntt,
               uint64_t batch_size_intt, uint64_t batch_size_KeySwitch,
               uint32_t debug);
    ~DevicePool();

private:
    DevicePool(const DevicePool& d) = delete;
    DevicePool& operator=(const DevicePool& d) = delete;
    void getDevices(int numDevicesToUse, int choice);
    sycl::cl_uint device_count_;
    std::vector<sycl::device> device_list_;
    Device** devices_;
    std::shared_future<bool> future_exit_;
    std::vector<std::thread> runners_;
};
/// @brief
/// @function attach_fpga_pooling
/// Attach a device to this thread
///
void attach_fpga_pooling();
/// @brief
/// @function detach_fpga_pooling
/// Detach a device from this thread
///
void detach_fpga_pooling();

}  // namespace fpga
}  // namespace hexl
}  // namespace intel

#endif
