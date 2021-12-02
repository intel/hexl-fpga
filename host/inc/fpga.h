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
#include <thread>
#include <unordered_map>
#include <vector>

#include "CL/opencl.h"

namespace intel {
namespace hexl {
namespace fpga {

__extension__ typedef unsigned __int128 fpga_uint128_t;
/// @brief
/// Struct moduli_info_t
/// @param[in] modulus stores the polynomial modulus
/// @param[in] len stores the the modulus size in bits
/// @param[in] barr_lo stores n / modulus where n is the polynomial size
///
typedef struct {
    uint64_t modulus;
    uint64_t len;
    uint64_t barr_lo;
} moduli_info_t;

/// @brief
/// Struct Object
/// @param[in] modulus stores the polynomial modulus
/// @param[in] ready_ flag indicating that the Object is ready for processing
/// @param[in] id_ Object local identifier
/// @param[in] g_wid_ Object global identifier
///

struct Object {
public:
    Object();
    virtual ~Object() = default;

    bool ready_;
    int id_;
    static unsigned int g_wid_;
};

/// @brief
/// Struct Object NTT
/// Stores the Number Theoretic Transform parameters
/// @param[in] coeff_poly polynomial coefficients
/// @param[out] coeff_poly polynomial coefficients
/// @param[in] root_of_unity_powers twiddle factors
/// @param[in] precon_root_of_unity_powers inverse twiddle factors
/// @param[in] coeff_modulus polynomial coefficients modulus
/// @param[in] n polynomial size in powers of two
///
struct Object_NTT : public Object {
    explicit Object_NTT(uint64_t* coeff_poly,
                        const uint64_t* root_of_unity_powers,
                        const uint64_t* precon_root_of_unity_powers,
                        uint64_t coeff_modulus, uint64_t n);

    uint64_t* coeff_poly_;
    const uint64_t* root_of_unity_powers_;
    const uint64_t* precon_root_of_unity_powers_;
    uint64_t coeff_modulus_;
    uint64_t n_;
};

/// @brief
/// Struct Object INTT
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
struct Object_INTT : public Object {
    explicit Object_INTT(uint64_t* coeff_poly,
                         const uint64_t* inv_root_of_unity_powers,
                         const uint64_t* precon_inv_root_of_unity_powers,
                         uint64_t coeff_modulus, uint64_t inv_n,
                         uint64_t inv_n_w, uint64_t n);

    uint64_t* coeff_poly_;
    const uint64_t* inv_root_of_unity_powers_;
    const uint64_t* precon_inv_root_of_unity_powers_;
    uint64_t coeff_modulus_;
    uint64_t inv_n_;
    uint64_t inv_n_w_;
    uint64_t n_;
};
/// @brief
/// struct Object_DyadicMultiply
/// Stores the parameters for the multiplication
/// @param[out] result stores the multiplication result
/// @param[in] operand1 stores the first operand for the multiplication
/// @param[in] operand2 stores the second operand for the multiplication
/// @param[in] n polynomial size
/// @param[in] moduli vector of moduli
/// @param[in] n_moduli size of the vector of moduli
///
struct Object_DyadicMultiply : public Object {
    explicit Object_DyadicMultiply(uint64_t* results, const uint64_t* operand1,
                                   const uint64_t* operand2, uint64_t n,
                                   const uint64_t* moduli, uint64_t n_moduli);

    uint64_t* results_;
    const uint64_t* operand1_;
    const uint64_t* operand2_;
    uint64_t n_;
    const uint64_t* moduli_;
    uint64_t n_moduli_;
};
/// @brief
/// Struct Buffer
/// Structure containing information for the polynomial operations
/// @param[in] capacity of the buffer
/// @param[in] n_batch_dyadic_multiply batch size for the multiplication
/// @param[in] n_batch_ntt batch size for the Number Theoretical Transform
/// @param[in] n_batch_intt batch size for the inverse Number Theoretical
/// Transform
/// @param[in] total_worksize_DyadicMultiply stores the worksize for the
/// multiplication
/// @param[in] num_DyadicMultiply stores the number of multiplications to be
/// performed
/// @param[in] total_worksize_NTT stores the worksize for the NTT
/// @param[in] num_NTT stores the number of NTT to be performed
/// @param[in] total_worksize_INTT stores the worksize for the INTT
/// @param[in] num_INTT stores the number of INTT to be performed
/// @function push pushes an Object in the structure
/// @function front returns the front Object of the structure
/// @function pop pops the front Object out of the structure
/// @function size returns the size of the structure
/// @function get_worksize_DyadicMultiply returns the worksize of DyadicMultiply
/// @function get_worksize_NTT returns the worksize of NTT
/// @function get_worksize_INTT returns the worksize of INTT
/// @function set_worksize_DyadicMultiply sets the worksize of DyadicMultiply
/// @function set_worksize_NTT sets the worksize of NTT
/// @function set_worksize_INTT setsthe worksize of INTT
///
class Buffer {
public:
    Buffer(uint64_t capacity, uint64_t n_batch_dyadic_multiply,
           uint64_t n_batch_ntt, uint64_t n_batch_intt)
        : capacity_(capacity),
          n_batch_dyadic_multiply_(n_batch_dyadic_multiply),
          n_batch_ntt_(n_batch_ntt),
          n_batch_intt_(n_batch_intt),
          total_worksize_DyadicMultiply_(1),
          num_DyadicMultiply_(0),
          total_worksize_NTT_(1),
          num_NTT_(0),
          total_worksize_INTT_(1),
          num_INTT_(0) {}

    void push(Object* obj);
    Object* front();
    std::vector<Object*> pop();

    uint64_t size();

    uint64_t get_worksize_DyadicMultiply() const {
        return total_worksize_DyadicMultiply_;
    }
    uint64_t get_worksize_NTT() const { return total_worksize_NTT_; }
    uint64_t get_worksize_INTT() const { return total_worksize_INTT_; }

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

    void update_work_size(uint64_t ws) { num_DyadicMultiply_ -= ws; }
    void update_DyadicMultiply_work_size(uint64_t ws) {
        num_DyadicMultiply_ -= ws;
    }
    void update_NTT_work_size(uint64_t ws) { num_NTT_ -= ws; }
    void update_INTT_work_size(uint64_t ws) { num_INTT_ -= ws; }

    std::mutex mu_;
    std::mutex mu_size_;
    std::condition_variable cond_;
    std::deque<Object*> buffer_;
    const uint64_t capacity_;
    const uint64_t n_batch_dyadic_multiply_;
    const uint64_t n_batch_ntt_;
    const uint64_t n_batch_intt_;

    uint64_t total_worksize_DyadicMultiply_;
    uint64_t num_DyadicMultiply_;

    uint64_t total_worksize_NTT_;
    uint64_t num_NTT_;

    uint64_t total_worksize_INTT_;
    uint64_t num_INTT_;
};
/// @brief
/// Parent Struct FPGAObject stores the blob of objects to be transfered to the
/// FPGA
///
/// @function fill_in_data
/// @param[in] vector of objects
/// @function fill_out_data
/// @param[out] vector of results
/// @function recycle releases the content
/// context stores the openCL context
/// tag stores the blob tag
/// n_batch stores the number of batches
/// in_objs_ vector of stored objects
/// g_tag_ stores the global tag identifier
///
struct FPGAObject {
    FPGAObject(const cl_context& context, uint64_t n_batch);
    virtual ~FPGAObject() = default;
    virtual void fill_in_data(const std::vector<Object*>& objs) = 0;
    virtual void fill_out_data(uint64_t* results) = 0;

    void recycle();

    const cl_context& context_;
    int tag_;
    uint64_t n_batch_;

    std::vector<Object*> in_objs_;

    static std::atomic<int> g_tag_;
};

/// @brief
/// Struct FPGAObject_NTT stores the NTT blob of objects to be transfered to the
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
struct FPGAObject_NTT : public FPGAObject {
    explicit FPGAObject_NTT(const cl_context& context, uint64_t coeff_count,
                            uint64_t batch_size);
    ~FPGAObject_NTT();
    void fill_in_data(const std::vector<Object*>& objs) override;
    void fill_out_data(uint64_t* coeff_poly) override;

    uint64_t* coeff_poly_in_svm_;
    uint64_t* root_of_unity_powers_in_svm_;
    uint64_t* precon_root_of_unity_powers_in_svm_;
    uint64_t* coeff_modulus_in_svm_;
    uint64_t n_;
};

/// @brief
/// Struct FPGAObject_INTT stores the INTT blob of objects to be transfered to
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
struct FPGAObject_INTT : public FPGAObject {
    explicit FPGAObject_INTT(const cl_context& context, uint64_t coeff_count,
                             uint64_t batch_size);
    ~FPGAObject_INTT();
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
/// Struct FPGAObject_DyadicMultiply
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
struct FPGAObject_DyadicMultiply : public FPGAObject {
    explicit FPGAObject_DyadicMultiply(const cl_context& context,
                                       uint64_t coeff_size,
                                       uint32_t modulus_size,
                                       uint64_t batch_size);
    ~FPGAObject_DyadicMultiply();
    void fill_in_data(const std::vector<Object*>& objs) override;
    void fill_out_data(uint64_t* results) override;

    uint64_t* operand1_in_svm_;
    uint64_t* operand2_in_svm_;
    moduli_info_t* moduli_info_;
    uint64_t n_;
    uint64_t n_moduli_;
    cl_mem operands_in_ddr_;
    cl_mem results_out_ddr_;
};
/// @brief
/// enum DEV_TYPE
/// Lists the available device mode: CPU, emulation mode, FPGA
///
typedef enum { NONE = 0, EMU, FPGA } DEV_TYPE;
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
/// @param[in] debug flag indicating debug mode
///
/// @function run function to launch the operation on the FPGA
///
class Device {
public:
    Device(const cl_device_id& device, Buffer& buffer,
           std::shared_future<bool> exit_signal, uint64_t coeff_size,
           uint32_t modulus_size, uint64_t batch_size_dyadic_multiply,
           uint64_t batch_size_ntt, uint64_t batch_size_intt, uint32_t debug);
    ~Device();

    void run();

private:
    enum { CREDIT = 8 };

    enum class kernel_t { INTEGRATED, DYADIC_MULTIPLY, NTT, INTT };
    void process_blocking_api();
    bool process_input(int index);
    bool process_output();

    bool process_output_dyadic_multiply();
    bool process_output_NTT();
    bool process_output_INTT();

    void enqueue_input_data(FPGAObject* fpga_obj);
    void enqueue_input_data_dyadic_multiply(
        FPGAObject_DyadicMultiply* fpga_obj);
    void enqueue_input_data_NTT(FPGAObject_NTT* fpga_obj);
    void enqueue_input_data_INTT(FPGAObject_INTT* fpga_obj);

    int device_id() { return id_; }

    kernel_t get_kernel_type();
    std::string get_bitstream_name();

    const cl_device_id& device_;
    Buffer& buffer_;
    unsigned int credit_;
    std::shared_future<bool> future_exit_;
    int id_;
    static int device_id_;
    kernel_t kernel_type_;

    std::vector<FPGAObject*> fpgaObjects_;

    cl_context context_;
    cl_program program_;

    // DYADIC_MULTIPLY section
    cl_command_queue dyadic_multiply_input_queue_;
    cl_command_queue dyadic_multiply_output_queue_;
    cl_kernel dyadic_multiply_input_fifo_kernel_;
    cl_kernel dyadic_multiply_output_fifo_nb_kernel_;

    uint64_t* dyadic_multiply_results_out_svm_;
    int* dyadic_multiply_tag_out_svm_;
    int* dyadic_multiply_results_out_valid_svm_;
    //

    // NTT section
    cl_command_queue ntt_load_queue_;
    cl_command_queue ntt_store_queue_;
    cl_kernel ntt_load_kernel_;
    cl_kernel ntt_store_kernel_;

    uint64_t* NTT_coeff_poly_svm_;

    // INTT section
    cl_command_queue intt_INTT_queue_;
    cl_command_queue intt_load_queue_;
    cl_command_queue intt_store_queue_;
    cl_kernel intt_INTT_kernel_;
    cl_kernel intt_load_kernel_;
    cl_kernel intt_store_kernel_;

    uint64_t* INTT_coeff_poly_svm_;
    //

    uint32_t debug_;

    static const std::unordered_map<std::string, kernel_t> kernels;
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
/// @param[in] debug flag indicating debug mode
///
class DevicePool {
public:
    DevicePool(int choice, Buffer& buffer, std::future<bool>& exit_signal,
               uint64_t coeff_size, uint32_t modulus_size,
               uint64_t batch_size_dyadic_multiply, uint64_t batch_size_ntt,
               uint64_t batch_size_intt, uint32_t debug);
    ~DevicePool();

private:
    DevicePool(const DevicePool& d) = delete;
    DevicePool& operator=(const DevicePool& d) = delete;

    cl_platform_id platform_;
    cl_uint device_count_;
    cl_device_id* cl_devices_;
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
