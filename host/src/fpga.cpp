// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <string.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#define CL_VERSION_2_0
#include "AOCLUtils/aocl_utils.h"
#include "CL/opencl.h"
#include "fpga.h"
#include "fpga_assert.h"

namespace intel {
namespace hexl {
namespace fpga {

unsigned int Object::g_wid_ = 0;

Object::Object() : ready_(false) { id_ = Object::g_wid_++; }

Object_DyadicMultiply::Object_DyadicMultiply(uint64_t* results,
                                             const uint64_t* operand1,
                                             const uint64_t* operand2,
                                             uint64_t n, const uint64_t* moduli,
                                             uint64_t n_moduli)
    : Object(),
      results_(results),
      operand1_(operand1),
      operand2_(operand2),
      n_(n),
      moduli_(moduli),
      n_moduli_(n_moduli) {}

Object_NTT::Object_NTT(uint64_t* coeff_poly,
                       const uint64_t* root_of_unity_powers,
                       const uint64_t* precon_root_of_unity_powers,
                       uint64_t coeff_modulus, uint64_t n)
    : Object(),
      coeff_poly_(coeff_poly),
      root_of_unity_powers_(root_of_unity_powers),
      precon_root_of_unity_powers_(precon_root_of_unity_powers),
      coeff_modulus_(coeff_modulus),
      n_(n) {}

Object_INTT::Object_INTT(uint64_t* coeff_poly,
                         const uint64_t* inv_root_of_unity_powers,
                         const uint64_t* precon_inv_root_of_unity_powers,
                         uint64_t coeff_modulus, uint64_t inv_n,
                         uint64_t inv_n_w, uint64_t n)
    : Object(),
      coeff_poly_(coeff_poly),
      inv_root_of_unity_powers_(inv_root_of_unity_powers),
      precon_inv_root_of_unity_powers_(precon_inv_root_of_unity_powers),
      coeff_modulus_(coeff_modulus),
      inv_n_(inv_n),
      inv_n_w_(inv_n_w),
      n_(n) {}

Object* Buffer::front() {
    Object* obj = buffer_.front();
    return obj;
}

void Buffer::push(Object* obj) {
    std::unique_lock<std::mutex> locker(mu_);
    cond_.wait(locker, [this]() { return buffer_.size() < capacity_; });
    buffer_.push_back(obj);
    locker.unlock();
    cond_.notify_all();
}

std::vector<Object*> Buffer::pop() {
    std::unique_lock<std::mutex> locker(mu_);

    uint64_t work_size = 1;

    Object_DyadicMultiply* obj_dyadic_multiply = nullptr;
    Object_NTT* obj_ntt = nullptr;
    Object_INTT* obj_intt = nullptr;

    if (buffer_.size() > 0) {
        Object* object = buffer_.front();
        if (object) {
            {  // DyadicMultiply section
                obj_dyadic_multiply =
                    dynamic_cast<Object_DyadicMultiply*>(object);
                if (obj_dyadic_multiply) {
                    work_size = get_worksize_int_DyadicMultiply();
                }
            }
            {  // NTT section
                obj_ntt = dynamic_cast<Object_NTT*>(object);
                if (obj_ntt) {
                    work_size = get_worksize_int_NTT();
                }
            }
            {  // INTT section
                obj_intt = dynamic_cast<Object_INTT*>(object);
                if (obj_intt) {
                    work_size = get_worksize_int_INTT();
                }
            }
        }
    }
    cond_.wait(locker,
               [this, &work_size]() { return buffer_.size() >= work_size; });
    std::vector<Object*> objs;
    uint64_t batch = 0;
    while (batch++ < work_size) {
        Object* obj = buffer_.front();
        objs.emplace_back(obj);
        buffer_.pop_front();
    }
    if (obj_dyadic_multiply) {
        update_DyadicMultiply_work_size(work_size);
    } else if (obj_ntt) {
        update_NTT_work_size(work_size);
    } else if (obj_intt) {
        update_INTT_work_size(work_size);
    }

    locker.unlock();
    cond_.notify_all();

    return objs;
}

uint64_t Buffer::size() {
    std::unique_lock<std::mutex> locker(mu_size_);

    uint64_t buf_size = buffer_.size();

    locker.unlock();

    return buf_size;
}

std::atomic<int> FPGAObject::g_tag_(0);

FPGAObject::FPGAObject(const cl_context& context, uint64_t n_batch)
    : context_(context), tag_(-1), n_batch_(n_batch) {}

void FPGAObject::recycle() {
    tag_ = -1;
    in_objs_.resize(0);
}

FPGAObject_NTT::FPGAObject_NTT(const cl_context& context, uint64_t coeff_count,
                               uint64_t batch_size)
    : FPGAObject(context, batch_size),

      n_(coeff_count) {
    uint64_t data_size = batch_size * coeff_count;
    coeff_poly_in_svm_ =
        (uint64_t*)clSVMAlloc(context_, 0, data_size * sizeof(uint64_t), 0);
    root_of_unity_powers_in_svm_ =
        (uint64_t*)clSVMAlloc(context_, 0, coeff_count * sizeof(uint64_t), 0);
    precon_root_of_unity_powers_in_svm_ =
        (uint64_t*)clSVMAlloc(context_, 0, coeff_count * sizeof(uint64_t), 0);
    coeff_modulus_in_svm_ =
        (uint64_t*)clSVMAlloc(context_, 0, sizeof(uint64_t), 0);
}

FPGAObject_INTT::FPGAObject_INTT(const cl_context& context,
                                 uint64_t coeff_count, uint64_t batch_size)
    : FPGAObject(context, batch_size),

      n_(coeff_count) {
    uint64_t data_size = batch_size * coeff_count;
    coeff_poly_in_svm_ =
        (uint64_t*)clSVMAlloc(context_, 0, data_size * sizeof(uint64_t), 0);
    inv_n_in_svm_ =
        (unsigned long*)clSVMAlloc(context_, 0, sizeof(unsigned long), 0);
    inv_n_w_in_svm_ =
        (unsigned long*)clSVMAlloc(context_, 0, sizeof(unsigned long), 0);
    inv_root_of_unity_powers_in_svm_ =
        (uint64_t*)clSVMAlloc(context_, 0, coeff_count * sizeof(uint64_t), 0);
    precon_inv_root_of_unity_powers_in_svm_ =
        (uint64_t*)clSVMAlloc(context_, 0, coeff_count * sizeof(uint64_t), 0);
    coeff_modulus_in_svm_ =
        (uint64_t*)clSVMAlloc(context_, 0, sizeof(uint64_t), 0);
}

FPGAObject_DyadicMultiply::FPGAObject_DyadicMultiply(const cl_context& context,
                                                     uint64_t coeff_size,
                                                     uint32_t modulus_size,
                                                     uint64_t batch_size)
    : FPGAObject(context, batch_size), n_(coeff_size), n_moduli_(0) {
    uint64_t n = batch_size * modulus_size * coeff_size;
    operand1_in_svm_ =
        (uint64_t*)clSVMAlloc(context_, 0, n * 2 * sizeof(uint64_t), 0);
    operand2_in_svm_ =
        (uint64_t*)clSVMAlloc(context_, 0, n * 2 * sizeof(uint64_t), 0);
    moduli_info_ = (moduli_info_t*)clSVMAlloc(
        context_, 0, batch_size * modulus_size * sizeof(moduli_info_t), 0);

    cl_int status;
    operands_in_ddr_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                      n * 4 * sizeof(uint64_t), NULL, &status);
    results_out_ddr_ = clCreateBuffer(context_, CL_MEM_READ_WRITE,
                                      n * 3 * sizeof(uint64_t), NULL, &status);
}

FPGAObject_DyadicMultiply::~FPGAObject_DyadicMultiply() {
    clSVMFree(context_, operand1_in_svm_);
    operand1_in_svm_ = nullptr;
    clSVMFree(context_, operand2_in_svm_);
    operand2_in_svm_ = nullptr;
    clSVMFree(context_, moduli_info_);
    moduli_info_ = nullptr;

    if (operands_in_ddr_) {
        clReleaseMemObject(operands_in_ddr_);
    }
    if (results_out_ddr_) {
        clReleaseMemObject(results_out_ddr_);
    }
}

FPGAObject_NTT::~FPGAObject_NTT() {
    clSVMFree(context_, coeff_poly_in_svm_);
    coeff_poly_in_svm_ = nullptr;
    clSVMFree(context_, root_of_unity_powers_in_svm_);
    root_of_unity_powers_in_svm_ = nullptr;
    clSVMFree(context_, precon_root_of_unity_powers_in_svm_);
    precon_root_of_unity_powers_in_svm_ = nullptr;
    clSVMFree(context_, coeff_modulus_in_svm_);
    coeff_modulus_in_svm_ = nullptr;
}

FPGAObject_INTT::~FPGAObject_INTT() {
    clSVMFree(context_, coeff_poly_in_svm_);
    coeff_poly_in_svm_ = nullptr;
    clSVMFree(context_, inv_root_of_unity_powers_in_svm_);
    inv_root_of_unity_powers_in_svm_ = nullptr;
    clSVMFree(context_, precon_inv_root_of_unity_powers_in_svm_);
    precon_inv_root_of_unity_powers_in_svm_ = nullptr;
    clSVMFree(context_, coeff_modulus_in_svm_);
    coeff_modulus_in_svm_ = nullptr;
    clSVMFree(context_, inv_n_in_svm_);
    inv_n_in_svm_ = nullptr;
    clSVMFree(context_, inv_n_w_in_svm_);
    inv_n_w_in_svm_ = nullptr;
}

void FPGAObject_DyadicMultiply::fill_in_data(const std::vector<Object*>& objs) {
    uint64_t batch = 0;
    for (const auto& obj_in : objs) {
        Object_DyadicMultiply* obj =
            dynamic_cast<Object_DyadicMultiply*>(obj_in);
        FPGA_ASSERT(obj);
        in_objs_.emplace_back(obj);

        n_moduli_ = obj->n_moduli_;
        n_ = obj->n_;

        for (uint64_t i = 0; i < n_moduli_; i++) {
            uint64_t modulus = obj->moduli_[i];
            uint64_t len = uint64_t(floorl(std::log2l(modulus)) - 1);
            fpga_uint128_t n = fpga_uint128_t(1) << (len + 64);
            uint64_t barr_lo = uint64_t(n / modulus);
            moduli_info_[batch * n_moduli_ + i] =
                (moduli_info_t){modulus, len, barr_lo};
        }
        batch++;
    }

    n_batch_ = batch;

    // assuming the batch of operand1s and operand2s are in contiguous space
    // respectively
    uint64_t n_data = n_moduli_ * n_ * 2;
    Object_DyadicMultiply* obj =
        dynamic_cast<Object_DyadicMultiply*>(in_objs_.front());
    FPGA_ASSERT(obj);
    memcpy(operand1_in_svm_, obj->operand1_,
           n_batch_ * n_data * sizeof(uint64_t));
    memcpy(operand2_in_svm_, obj->operand2_,
           n_batch_ * n_data * sizeof(uint64_t));

    tag_ = g_tag_++;
}

void FPGAObject_NTT::fill_in_data(const std::vector<Object*>& objs) {
    uint64_t batch = 0;
    for (const auto& obj_in : objs) {
        Object_NTT* obj = dynamic_cast<Object_NTT*>(obj_in);
        FPGA_ASSERT(obj);
        in_objs_.emplace_back(obj);

        n_ = obj->n_;

        batch++;
    }

    n_batch_ = batch;

    uint64_t coeff_count = n_;
    Object_NTT* obj = dynamic_cast<Object_NTT*>(in_objs_.front());
    FPGA_ASSERT(obj);

    memcpy(coeff_poly_in_svm_, obj->coeff_poly_,
           n_batch_ * coeff_count * sizeof(uint64_t));
    coeff_modulus_in_svm_[0] = obj->coeff_modulus_;
    memcpy(root_of_unity_powers_in_svm_, obj->root_of_unity_powers_,
           coeff_count * sizeof(uint64_t));
    memcpy(precon_root_of_unity_powers_in_svm_,
           obj->precon_root_of_unity_powers_, coeff_count * sizeof(uint64_t));

    tag_ = g_tag_++;
}

void FPGAObject_INTT::fill_in_data(const std::vector<Object*>& objs) {
    uint64_t batch = 0;
    for (const auto& obj_in : objs) {
        Object_INTT* obj = dynamic_cast<Object_INTT*>(obj_in);
        FPGA_ASSERT(obj);
        in_objs_.emplace_back(obj);

        n_ = obj->n_;

        batch++;
    }

    n_batch_ = batch;

    uint64_t coeff_count = n_;
    Object_INTT* obj = dynamic_cast<Object_INTT*>(in_objs_.front());
    FPGA_ASSERT(obj);

    memcpy(coeff_poly_in_svm_, obj->coeff_poly_,
           n_batch_ * coeff_count * sizeof(uint64_t));

    coeff_modulus_in_svm_[0] = obj->coeff_modulus_;
    memcpy(inv_root_of_unity_powers_in_svm_, obj->inv_root_of_unity_powers_,
           coeff_count * sizeof(uint64_t));
    memcpy(precon_inv_root_of_unity_powers_in_svm_,
           obj->precon_inv_root_of_unity_powers_,
           coeff_count * sizeof(uint64_t));

    *inv_n_in_svm_ = obj->inv_n_;
    *inv_n_w_in_svm_ = obj->inv_n_w_;

    tag_ = g_tag_++;
}

void FPGAObject_DyadicMultiply::fill_out_data(uint64_t* results_in_svm) {
    uint64_t n_data = n_moduli_ * n_ * 3;
    // assuming the batch of results are in contiguous space
    Object_DyadicMultiply* obj_dyadic_multiply =
        dynamic_cast<Object_DyadicMultiply*>(in_objs_.front());
    FPGA_ASSERT(obj_dyadic_multiply);
    memcpy(obj_dyadic_multiply->results_, results_in_svm,
           n_batch_ * n_data * sizeof(uint64_t));

    uint64_t batch = 0;
    for (auto& obj : in_objs_) {
        obj->ready_ = true;
        batch++;
    }
    FPGA_ASSERT(batch == n_batch_);
}

void FPGAObject_NTT::fill_out_data(uint64_t* results_in_svm_) {
    uint64_t coeff_count = n_;
    Object_NTT* obj_NTT = dynamic_cast<Object_NTT*>(in_objs_.front());
    FPGA_ASSERT(obj_NTT);
    memcpy(obj_NTT->coeff_poly_, results_in_svm_,
           n_batch_ * coeff_count * sizeof(uint64_t));

    uint64_t batch = 0;
    for (auto& obj : in_objs_) {
        obj->ready_ = true;
        batch++;
    }
    FPGA_ASSERT(batch == n_batch_);
}

void FPGAObject_INTT::fill_out_data(uint64_t* results_in_svm_) {
    uint64_t coeff_count = n_;
    Object_INTT* obj_INTT = dynamic_cast<Object_INTT*>(in_objs_.front());
    FPGA_ASSERT(obj_INTT);
    memcpy(obj_INTT->coeff_poly_, results_in_svm_,
           n_batch_ * coeff_count * sizeof(uint64_t));

    uint64_t batch = 0;
    for (auto& obj : in_objs_) {
        obj->ready_ = true;
        batch++;
    }
    FPGA_ASSERT(batch == n_batch_);
}

int Device::device_id_ = 0;

const std::unordered_map<std::string, Device::kernel_t> Device::kernels =
    std::unordered_map<std::string, kernel_t>{
        {"INTEGRATED", kernel_t::INTEGRATED},
        {"DYADIC_MULTIPLY", kernel_t::DYADIC_MULTIPLY},
        {"NTT", kernel_t::NTT},
        {"INTT", kernel_t::INTT}};

Device::kernel_t Device::get_kernel_type() {
    kernel_t kernel = kernel_t::INTEGRATED;
    const char* env_kernel = getenv("FPGA_KERNEL");
    if (env_kernel) {
        auto found = kernels.find(std::string(env_kernel));

        if (found != kernels.end()) {
            kernel = found->second;
        }
    }
    return kernel;
}

std::string Device::get_bitstream_name() {
    const char* bitstream = getenv("FPGA_BITSTREAM");
    if (bitstream) {
        std::string s(bitstream);
        // remove postfix .aocx
        s.erase(s.end() - 5, s.end());
        return s;
    }

    switch (kernel_type_) {
    case kernel_t::INTEGRATED:
        return std::string("hexl_fpga");
    case kernel_t::DYADIC_MULTIPLY:
        return std::string("dyadic_multiply");
    case kernel_t::NTT:
        return std::string("fwd_ntt");
    case kernel_t::INTT:
        return std::string("inv_ntt");
    default:
        FPGA_ASSERT(0);
        return std::string("bad");
    }
}

Device::Device(const cl_device_id& device, Buffer& buffer,
               std::shared_future<bool> exit_signal, uint64_t coeff_size,
               uint32_t modulus_size, uint64_t batch_size_dyadic_multiply,
               uint64_t batch_size_ntt, uint64_t batch_size_intt,
               uint32_t debug)
    : device_(device),
      buffer_(buffer),
      credit_(CREDIT),
      future_exit_(exit_signal),
      dyadic_multiply_input_queue_(nullptr),
      dyadic_multiply_output_queue_(nullptr),
      dyadic_multiply_input_fifo_kernel_(nullptr),
      dyadic_multiply_output_fifo_nb_kernel_(nullptr),
      dyadic_multiply_results_out_svm_(nullptr),
      dyadic_multiply_tag_out_svm_(nullptr),
      dyadic_multiply_results_out_valid_svm_(nullptr),
      ntt_load_queue_(nullptr),
      ntt_store_queue_(nullptr),
      ntt_load_kernel_(nullptr),
      ntt_store_kernel_(nullptr),
      NTT_coeff_poly_svm_(nullptr),
      intt_INTT_queue_(nullptr),
      intt_load_queue_(nullptr),
      intt_store_queue_(nullptr),
      intt_INTT_kernel_(nullptr),
      intt_load_kernel_(nullptr),
      intt_store_kernel_(nullptr),
      INTT_coeff_poly_svm_(nullptr),
      debug_(debug) {
    id_ = device_id_++;
    std::cout << "Acquiring Device ... " << id_ << std::endl;

    kernel_type_ = get_kernel_type();

    cl_int status;
    context_ = clCreateContext(NULL, 1, &device_, NULL, NULL, &status);
    aocl_utils::checkError(status, "Failed to create context");

    std::string bitstream = get_bitstream_name();
    std::string binary_file =
        aocl_utils::getBoardBinaryFile(bitstream.c_str(), device_);
    std::cout << "Running with FPGA bitstream: " << binary_file << std::endl;

    program_ = aocl_utils::createProgramFromBinary(
        context_, binary_file.c_str(), &device_, 1);
    status = clBuildProgram(program_, 0, NULL, "", NULL, NULL);
    aocl_utils::checkError(status, "Failed to build program");

    // DYADIC_MULTIPLY section
    if ((kernel_type_ == kernel_t::INTEGRATED) ||
        (kernel_type_ == kernel_t::DYADIC_MULTIPLY)) {
        cl_queue_properties props[] = {0};
        dyadic_multiply_input_queue_ = clCreateCommandQueueWithProperties(
            context_, device_, props, &status);
        aocl_utils::checkError(status, "Failed to create command input_queue");
        dyadic_multiply_output_queue_ = clCreateCommandQueueWithProperties(
            context_, device_, props, &status);
        aocl_utils::checkError(status, "Failed to create command output_queue");

        dyadic_multiply_input_fifo_kernel_ =
            clCreateKernel(program_, "input_fifo", &status);
        aocl_utils::checkError(status, "Failed to create input_fifo_kernel");
        dyadic_multiply_output_fifo_nb_kernel_ =
            clCreateKernel(program_, "output_nb_fifo", &status);
        aocl_utils::checkError(status,
                               "Failed to create output_nb_fifo_kernel");
        uint64_t size = batch_size_dyadic_multiply * 3 * modulus_size *
                        coeff_size * sizeof(uint64_t);
        dyadic_multiply_results_out_svm_ =
            (uint64_t*)clSVMAlloc(context_, 0, size, 0);
        dyadic_multiply_tag_out_svm_ =
            (int*)clSVMAlloc(context_, 0, sizeof(int), 0);
        dyadic_multiply_results_out_valid_svm_ =
            (int*)clSVMAlloc(context_, 0, sizeof(int), 0);
    }

    // INTT section
    if ((kernel_type_ == kernel_t::INTEGRATED) ||
        (kernel_type_ == kernel_t::INTT)) {
        cl_queue_properties props[] = {0};
        intt_load_queue_ = clCreateCommandQueueWithProperties(context_, device_,
                                                              props, &status);
        aocl_utils::checkError(status, "Failed to create command load_queue");
        intt_store_queue_ = clCreateCommandQueueWithProperties(
            context_, device_, props, &status);
        aocl_utils::checkError(status, "Failed to create command store_queue");

        intt_load_kernel_ =
            clCreateKernel(program_, "intt_input_kernel", &status);
        aocl_utils::checkError(status, "Failed to create intt_input_kernel");
        intt_store_kernel_ =
            clCreateKernel(program_, "intt_output_kernel", &status);
        aocl_utils::checkError(status, "Failed to create intt_output_kernel");
        uint64_t size = batch_size_intt * 16834 * sizeof(uint64_t);
        INTT_coeff_poly_svm_ = (uint64_t*)clSVMAlloc(context_, 0, size, 0);
    }

    // NTT section
    if ((kernel_type_ == kernel_t::INTEGRATED) ||
        (kernel_type_ == kernel_t::NTT)) {
        cl_queue_properties props[] = {0};
        ntt_load_queue_ = clCreateCommandQueueWithProperties(context_, device_,
                                                             props, &status);
        aocl_utils::checkError(status, "Failed to create command load_queue");
        ntt_store_queue_ = clCreateCommandQueueWithProperties(context_, device_,
                                                              props, &status);
        aocl_utils::checkError(status, "Failed to create command store_queue");

        ntt_load_kernel_ =
            clCreateKernel(program_, "ntt_input_kernel", &status);
        aocl_utils::checkError(status, "Failed to create ntt_input_kernel");
        ntt_store_kernel_ =
            clCreateKernel(program_, "ntt_output_kernel", &status);
        aocl_utils::checkError(status, "Failed to create ntt_output_kernel");
        uint64_t size = batch_size_ntt * 16834 * sizeof(uint64_t);
        NTT_coeff_poly_svm_ = (uint64_t*)clSVMAlloc(context_, 0, size, 0);
    }

    // DYADIC_MULTIPLY: [0, CREDIT)
    for (int i = 0; i < CREDIT; i++) {
        fpgaObjects_.emplace_back(new FPGAObject_DyadicMultiply(
            context_, coeff_size, modulus_size, batch_size_dyadic_multiply));
    }
    // INTT: CREDIT
    fpgaObjects_.emplace_back(
        new FPGAObject_INTT(context_, 16384, batch_size_intt));
    // NTT:  CREDIT + 1
    fpgaObjects_.emplace_back(
        new FPGAObject_NTT(context_, 16384, batch_size_ntt));
}

Device::~Device() {
    device_id_ = 0;
    for (auto& fpga_obj : fpgaObjects_) {
        if (fpga_obj) {
            delete fpga_obj;
            fpga_obj = nullptr;
        }
    }
    fpgaObjects_.clear();

    // DYADIC_MULTIPLY section
    if ((kernel_type_ == kernel_t::INTEGRATED) ||
        (kernel_type_ == kernel_t::DYADIC_MULTIPLY)) {
        if (dyadic_multiply_input_fifo_kernel_) {
            clReleaseKernel(dyadic_multiply_input_fifo_kernel_);
        }
        if (dyadic_multiply_input_queue_) {
            clReleaseCommandQueue(dyadic_multiply_input_queue_);
        }

        if (dyadic_multiply_output_queue_) {
            clReleaseCommandQueue(dyadic_multiply_output_queue_);
        }
        if (dyadic_multiply_output_fifo_nb_kernel_) {
            clReleaseKernel(dyadic_multiply_output_fifo_nb_kernel_);
        }
        clSVMFree(context_, dyadic_multiply_results_out_valid_svm_);
        dyadic_multiply_results_out_valid_svm_ = nullptr;
        clSVMFree(context_, dyadic_multiply_tag_out_svm_);
        dyadic_multiply_tag_out_svm_ = nullptr;
        clSVMFree(context_, dyadic_multiply_results_out_svm_);
        dyadic_multiply_results_out_svm_ = nullptr;
    }

    // NTT section
    if ((kernel_type_ == kernel_t::INTEGRATED) ||
        (kernel_type_ == kernel_t::NTT)) {
        if (ntt_load_kernel_) {
            clReleaseKernel(ntt_load_kernel_);
        }
        if (ntt_store_kernel_) {
            clReleaseKernel(ntt_store_kernel_);
        }

        if (ntt_load_queue_) {
            clReleaseCommandQueue(ntt_load_queue_);
        }

        if (ntt_store_queue_) {
            clReleaseCommandQueue(ntt_store_queue_);
        }
        clSVMFree(context_, NTT_coeff_poly_svm_);
        NTT_coeff_poly_svm_ = nullptr;
    }

    // INTT section
    if ((kernel_type_ == kernel_t::INTEGRATED) ||
        (kernel_type_ == kernel_t::INTT)) {
        if (intt_load_kernel_) {
            clReleaseKernel(intt_load_kernel_);
        }
        if (intt_store_kernel_) {
            clReleaseKernel(intt_store_kernel_);
        }

        if (intt_load_queue_) {
            clReleaseCommandQueue(intt_load_queue_);
        }

        if (intt_store_queue_) {
            clReleaseCommandQueue(intt_store_queue_);
        }
        clSVMFree(context_, INTT_coeff_poly_svm_);
        INTT_coeff_poly_svm_ = nullptr;
    }

    if (context_) {
        clReleaseContext(context_);
    }
    if (program_) {
        clReleaseProgram(program_);
    }
}

void Device::process_blocking_api() {
    process_input(CREDIT);
    process_output_INTT();
}

void Device::run() {
    while (future_exit_.wait_for(std::chrono::milliseconds(0)) ==
           std::future_status::timeout) {
        if (buffer_.size()) {
            // Look ahead
            Object* front = buffer_.front();

            Object_DyadicMultiply* obj_dyadic_multiply =
                dynamic_cast<Object_DyadicMultiply*>(front);
            Object_INTT* obj_intt = dynamic_cast<Object_INTT*>(front);
            Object_NTT* obj_ntt = dynamic_cast<Object_NTT*>(front);

            // Run blocking for NTT and INTT or non-blocking for Dyadic Multiply
            if (obj_intt) {
                process_input(CREDIT);
                process_output_INTT();
            } else if (obj_ntt) {
                process_input(CREDIT + 1);
                process_output_NTT();
            } else if (obj_dyadic_multiply) {
                if ((credit_ > 0) && process_input(CREDIT - credit_)) {
                    credit_ -= 1;
                }
            }
        }

        if ((credit_ < CREDIT) && process_output()) {
            credit_ += 1;
        }
    }
    std::cout << "Releasing Device ... " << device_id() << std::endl;
}

bool Device::process_input(int credit_id) {
    std::vector<Object*> objs = buffer_.pop();

    if (objs.empty()) {
        return false;
    }

    FPGAObject* fpga_obj = fpgaObjects_[credit_id];

    const auto& start_io = std::chrono::high_resolution_clock::now();
    fpga_obj->fill_in_data(objs);
    const auto& end_io = std::chrono::high_resolution_clock::now();

    enqueue_input_data(fpga_obj);

    if (debug_ == 2) {
        const auto& end_api = std::chrono::high_resolution_clock::now();
        const auto& duration_io =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_io -
                                                                      start_io);
        const auto& duration_api =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_api -
                                                                      start_io);

        std::string kernel;
        FPGAObject_DyadicMultiply* fpga_obj_dyadic_multiply =
            dynamic_cast<FPGAObject_DyadicMultiply*>(fpga_obj);
        if (fpga_obj_dyadic_multiply) {
            kernel = "DYADIC_MULTIPLY";
        }
        FPGAObject_NTT* fpga_obj_NTT = dynamic_cast<FPGAObject_NTT*>(fpga_obj);
        if (fpga_obj_NTT) {
            kernel = "NTT";
        }
        FPGAObject_INTT* fpga_obj_INTT =
            dynamic_cast<FPGAObject_INTT*>(fpga_obj);
        if (fpga_obj_INTT) {
            kernel = "INTT";
        }

        double unit = 1.0e+6;  // microseconds
        std::cout << kernel << " input I/O   time taken: " << std::fixed
                  << std::setprecision(8) << duration_io.count() * unit << " us"
                  << std::endl;
        std::cout << kernel << " input API   time taken: " << std::fixed
                  << std::setprecision(8) << duration_api.count() * unit
                  << " us" << std::endl
                  << std::endl;
    }
    return true;
}

void Device::enqueue_input_data(FPGAObject* fpga_obj) {
    FPGAObject_DyadicMultiply* fpga_obj_dyadic_multiply =
        dynamic_cast<FPGAObject_DyadicMultiply*>(fpga_obj);
    if (fpga_obj_dyadic_multiply) {
        enqueue_input_data_dyadic_multiply(fpga_obj_dyadic_multiply);
        return;
    }

    FPGAObject_INTT* fpga_obj_INTT = dynamic_cast<FPGAObject_INTT*>(fpga_obj);
    if (fpga_obj_INTT) {
        enqueue_input_data_INTT(fpga_obj_INTT);
        return;
    }

    FPGAObject_NTT* fpga_obj_NTT = dynamic_cast<FPGAObject_NTT*>(fpga_obj);
    if (fpga_obj_NTT) {
        enqueue_input_data_NTT(fpga_obj_NTT);
        return;
    }

    FPGA_ASSERT(0);
}

void Device::enqueue_input_data_dyadic_multiply(
    FPGAObject_DyadicMultiply* fpga_obj) {
    int i = 0;
    const size_t gws = 1;
    const size_t lws = 1;

    const auto& start_ocl = std::chrono::high_resolution_clock::now();

    clSetKernelArgSVMPointer(dyadic_multiply_input_fifo_kernel_, i++,
                             (void*)fpga_obj->operand1_in_svm_);
    clSetKernelArgSVMPointer(dyadic_multiply_input_fifo_kernel_, i++,
                             (void*)fpga_obj->operand2_in_svm_);
    clSetKernelArg(dyadic_multiply_input_fifo_kernel_, i++, sizeof(uint64_t),
                   &fpga_obj->n_);
    clSetKernelArgSVMPointer(dyadic_multiply_input_fifo_kernel_, i++,
                             (void*)fpga_obj->moduli_info_);
    clSetKernelArg(dyadic_multiply_input_fifo_kernel_, i++, sizeof(uint64_t),
                   &fpga_obj->n_moduli_);
    clSetKernelArg(dyadic_multiply_input_fifo_kernel_, i++, sizeof(int),
                   &fpga_obj->tag_);
    clSetKernelArg(dyadic_multiply_input_fifo_kernel_, i++, sizeof(cl_mem),
                   &fpga_obj->operands_in_ddr_);
    clSetKernelArg(dyadic_multiply_input_fifo_kernel_, i++, sizeof(cl_mem),
                   &fpga_obj->results_out_ddr_);
    clSetKernelArg(dyadic_multiply_input_fifo_kernel_, i++, sizeof(uint64_t),
                   &fpga_obj->n_batch_);

    cl_int status = clEnqueueNDRangeKernel(dyadic_multiply_input_queue_,
                                           dyadic_multiply_input_fifo_kernel_,
                                           1, NULL, &gws, &lws, 0, NULL, NULL);
    aocl_utils::checkError(status, "Failed to launch input_fifo_kernel");
    status = clFlush(dyadic_multiply_input_queue_);
    aocl_utils::checkError(status, "Failed to flush input_queue");

    if (debug_ == 1) {
        const auto& end_ocl = std::chrono::high_resolution_clock::now();
        const auto& duration_ocl =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_ocl - start_ocl);
        double unit = 1.0e+6;  // microseconds
        std::cout << "DYADIC_MULTIPLY"
                  << " OCL-in      time taken: " << std::fixed
                  << std::setprecision(8) << duration_ocl.count() * unit
                  << " us" << std::endl;
        std::cout << "DYADIC_MULTIPLY"
                  << " OCL-in avg  time taken: " << std::fixed
                  << std::setprecision(8)
                  << duration_ocl.count() / fpga_obj->n_batch_ * unit << " us"
                  << std::endl;
    }
}

void Device::enqueue_input_data_INTT(FPGAObject_INTT* fpga_obj) {
    int i = 0;

    const size_t gws = 1;
    const size_t lws = 1;
    unsigned int batch = fpga_obj->n_batch_;

    const auto& start_ocl = std::chrono::high_resolution_clock::now();

    clSetKernelArg(intt_load_kernel_, i++, sizeof(unsigned int), &batch);
    clSetKernelArgSVMPointer(intt_load_kernel_, i++,
                             (void*)fpga_obj->coeff_poly_in_svm_);
    clSetKernelArgSVMPointer(intt_load_kernel_, i++,
                             (void*)&fpga_obj->coeff_modulus_in_svm_[0]);
    clSetKernelArgSVMPointer(intt_load_kernel_, i++,
                             (void*)&fpga_obj->inv_n_in_svm_[0]);
    clSetKernelArgSVMPointer(intt_load_kernel_, i++,
                             (void*)&fpga_obj->inv_n_w_in_svm_[0]);
    clSetKernelArgSVMPointer(intt_load_kernel_, i++,
                             (void*)fpga_obj->inv_root_of_unity_powers_in_svm_);
    clSetKernelArgSVMPointer(
        intt_load_kernel_, i++,
        (void*)fpga_obj->precon_inv_root_of_unity_powers_in_svm_);

    cl_int status = clEnqueueNDRangeKernel(intt_load_queue_, intt_load_kernel_,
                                           1, NULL, &gws, &lws, 0, NULL, NULL);
    aocl_utils::checkError(status, "Failed to launch intt_load_kernel");

    if (debug_ == 1) {
        const auto& end_ocl = std::chrono::high_resolution_clock::now();
        const auto& duration_ocl =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_ocl - start_ocl);
        double unit = 1.0e+6;  // microseconds
        std::cout << "INTT"
                  << " OCL-in      time taken: " << std::fixed
                  << std::setprecision(8) << duration_ocl.count() * unit
                  << " us" << std::endl;
        std::cout << "INTT"
                  << " OCL-in avg  time taken: " << std::fixed
                  << std::setprecision(8)
                  << duration_ocl.count() / fpga_obj->n_batch_ * unit << " us"
                  << std::endl;
    }
}

void Device::enqueue_input_data_NTT(FPGAObject_NTT* fpga_obj) {
    int i = 0;

    const size_t gws = 1;
    const size_t lws = 1;
    unsigned int batch = fpga_obj->n_batch_;

    const auto& start_ocl = std::chrono::high_resolution_clock::now();

    clSetKernelArg(ntt_load_kernel_, i++, sizeof(unsigned int), &batch);
    clSetKernelArgSVMPointer(ntt_load_kernel_, i++,
                             (void*)fpga_obj->coeff_poly_in_svm_);
    clSetKernelArgSVMPointer(ntt_load_kernel_, i++,
                             (void*)fpga_obj->coeff_poly_in_svm_);
    clSetKernelArgSVMPointer(ntt_load_kernel_, i++,
                             (void*)&fpga_obj->coeff_modulus_in_svm_[0]);
    clSetKernelArgSVMPointer(ntt_load_kernel_, i++,
                             (void*)fpga_obj->root_of_unity_powers_in_svm_);
    clSetKernelArgSVMPointer(
        ntt_load_kernel_, i++,
        (void*)fpga_obj->precon_root_of_unity_powers_in_svm_);

    cl_int status = clEnqueueNDRangeKernel(ntt_load_queue_, ntt_load_kernel_, 1,
                                           NULL, &gws, &lws, 0, NULL, NULL);
    aocl_utils::checkError(status, "Failed to launch ntt_load_kernel");

    if (debug_ == 1) {
        const auto& end_ocl = std::chrono::high_resolution_clock::now();
        const auto& duration_ocl =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_ocl - start_ocl);
        double unit = 1.0e+6;  // microseconds
        std::cout << "NTT"
                  << " OCL-in      time taken: " << std::fixed
                  << std::setprecision(8) << duration_ocl.count() * unit
                  << " us" << std::endl;
        std::cout << "NTT"
                  << " OCL-in avg  time taken: " << std::fixed
                  << std::setprecision(8)
                  << duration_ocl.count() / fpga_obj->n_batch_ * unit << " us"
                  << std::endl;
    }
}

bool Device::process_output() {
    bool rsl = false;
    rsl |= process_output_dyadic_multiply();
    return rsl;
}

bool Device::process_output_dyadic_multiply() {
    bool rsl = false;

    dyadic_multiply_tag_out_svm_[0] = -1;
    dyadic_multiply_results_out_valid_svm_[0] = 0;

    int idx = 0;
    const size_t gws = 1;
    const size_t lws = 1;

    const auto& start_ocl = std::chrono::high_resolution_clock::now();
    clSetKernelArgSVMPointer(dyadic_multiply_output_fifo_nb_kernel_, idx++,
                             (void*)dyadic_multiply_results_out_svm_);
    clSetKernelArgSVMPointer(dyadic_multiply_output_fifo_nb_kernel_, idx++,
                             (void*)dyadic_multiply_tag_out_svm_);
    clSetKernelArgSVMPointer(dyadic_multiply_output_fifo_nb_kernel_, idx++,
                             (void*)dyadic_multiply_results_out_valid_svm_);
    cl_int status = clEnqueueNDRangeKernel(
        dyadic_multiply_output_queue_, dyadic_multiply_output_fifo_nb_kernel_,
        1, NULL, &gws, &lws, 0, NULL, NULL);
    aocl_utils::checkError(status, "Failed to launch output_fifo_nb_kernel");
    status = clFinish(dyadic_multiply_output_queue_);
    const auto& end_ocl = std::chrono::high_resolution_clock::now();

    if (*dyadic_multiply_results_out_valid_svm_ == 1) {
        FPGA_ASSERT(dyadic_multiply_tag_out_svm_[0] >= 0);

        const auto& start_io = std::chrono::high_resolution_clock::now();
        FPGAObject* completed = nullptr;
        for (int credit = 0; credit < CREDIT; credit++) {
            if (fpgaObjects_[credit]->tag_ == dyadic_multiply_tag_out_svm_[0]) {
                completed = fpgaObjects_[credit];
                break;
            }
        }

        if (completed) {
            completed->fill_out_data(dyadic_multiply_results_out_svm_);

            completed->recycle();
            rsl = true;

            for (int credit = 0; credit < CREDIT - 1; credit++) {
                fpgaObjects_[credit] = fpgaObjects_[credit + 1];
            }
            fpgaObjects_[CREDIT - 1] = completed;

            if (debug_) {
                const auto& end_io = std::chrono::high_resolution_clock::now();
                const auto& duration_io =
                    std::chrono::duration_cast<std::chrono::duration<double>>(
                        end_io - start_io);
                const auto& duration_ocl =
                    std::chrono::duration_cast<std::chrono::duration<double>>(
                        end_ocl - start_ocl);
                const auto& duration_api =
                    std::chrono::duration_cast<std::chrono::duration<double>>(
                        end_io - start_ocl);

                double unit = 1.0e+6;  // microseconds
                if (debug_ == 1) {
                    std::cout << "DYADIC_MULTIPLY OCL-out     time taken: "
                              << std::fixed << std::setprecision(8)
                              << duration_ocl.count() * unit << " us"
                              << std::endl;
                    std::cout
                        << "DYADIC_MULTIPLY OCL-out avg time taken: "
                        << std::fixed << std::setprecision(8)
                        << duration_ocl.count() / completed->n_batch_ * unit
                        << " us" << std::endl;
                }
                if (debug_ == 2) {
                    std::cout << "DYADIC_MULTIPLY out I/O     time taken: "
                              << std::fixed << std::setprecision(8)
                              << duration_io.count() * unit << " us"
                              << std::endl;
                    std::cout << "DYADIC_MULTIPLY out API     time taken: "
                              << std::fixed << std::setprecision(8)
                              << duration_api.count() * unit << " us"
                              << std::endl
                              << std::endl;
                }
            }
        } else {
            FPGA_ASSERT(0);
        }
    }

    return rsl;
}

bool Device::process_output_NTT() {
    int i = 0;
    unsigned int batch = 1;
    int ntt_instance_index = CREDIT + 1;

    FPGAObject* completed = fpgaObjects_[ntt_instance_index];
    FPGAObject_NTT* kernel_inf = dynamic_cast<FPGAObject_NTT*>(completed);

    FPGA_ASSERT(kernel_inf);

    batch = kernel_inf->n_batch_;

    const size_t gws = 1;
    const size_t lws = 1;

    const auto& start_ocl = std::chrono::high_resolution_clock::now();
    clSetKernelArg(ntt_store_kernel_, i++, sizeof(unsigned int), &batch);
    clSetKernelArgSVMPointer(ntt_store_kernel_, i++,
                             (void*)NTT_coeff_poly_svm_);

    cl_int status = clEnqueueNDRangeKernel(ntt_store_queue_, ntt_store_kernel_,
                                           1, NULL, &gws, &lws, 0, NULL, NULL);
    aocl_utils::checkError(status, "Failed to launch ntt_store_kernel");

    status = clFinish(ntt_store_queue_);
    aocl_utils::checkError(status, "Failed to finish ntt_store_queue");
    const auto& end_ocl = std::chrono::high_resolution_clock::now();

    const auto& start_io = std::chrono::high_resolution_clock::now();
    completed->fill_out_data(NTT_coeff_poly_svm_);
    completed->recycle();

    if (debug_) {
        const auto& end_io = std::chrono::high_resolution_clock::now();
        const auto& duration_io =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_io -
                                                                      start_io);
        const auto& duration_ocl =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_ocl - start_ocl);
        const auto& duration_api =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_io - start_ocl);

        double unit = 1.0e+6;  // microseconds
        if (debug_ == 1) {
            std::cout << "NTT OCL-out     time taken: " << std::fixed
                      << std::setprecision(8) << duration_ocl.count() * unit
                      << " us" << std::endl;
            std::cout << "NTT OCL-out avg time taken: " << std::fixed
                      << std::setprecision(8)
                      << duration_ocl.count() / completed->n_batch_ * unit
                      << " us" << std::endl;
        }
        if (debug_ == 2) {
            std::cout << "NTT out I/O     time taken: " << std::fixed
                      << std::setprecision(8) << duration_io.count() * unit
                      << " us" << std::endl;
            std::cout << "NTT out API     time taken: " << std::fixed
                      << std::setprecision(8) << duration_api.count() * unit
                      << " us" << std::endl
                      << std::endl;
        }
    }

    return 0;
}

bool Device::process_output_INTT() {
    int i = 0;
    unsigned int batch = 1;
    int intt_instance_index = CREDIT;

    FPGAObject* completed = fpgaObjects_[intt_instance_index];
    FPGAObject_INTT* kernel_inf = dynamic_cast<FPGAObject_INTT*>(completed);

    FPGA_ASSERT(kernel_inf);

    batch = kernel_inf->n_batch_;
    const size_t gws = 1;
    const size_t lws = 1;

    const auto& start_ocl = std::chrono::high_resolution_clock::now();
    clSetKernelArg(intt_store_kernel_, i++, sizeof(unsigned int), &batch);
    clSetKernelArgSVMPointer(intt_store_kernel_, i++,
                             (void*)INTT_coeff_poly_svm_);

    cl_int status =
        clEnqueueNDRangeKernel(intt_store_queue_, intt_store_kernel_, 1, NULL,
                               &gws, &lws, 0, NULL, NULL);
    aocl_utils::checkError(status, "Failed to launch intt_store_kernel");
    status = clFinish(intt_store_queue_);
    const auto& end_ocl = std::chrono::high_resolution_clock::now();
    aocl_utils::checkError(status, "Failed to finish intt_store_queue");

    const auto& start_io = std::chrono::high_resolution_clock::now();
    completed->fill_out_data(INTT_coeff_poly_svm_);
    completed->recycle();

    if (debug_) {
        const auto& end_io = std::chrono::high_resolution_clock::now();
        const auto& duration_io =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_io -
                                                                      start_io);
        const auto& duration_ocl =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_ocl - start_ocl);
        const auto& duration_api =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_io - start_ocl);

        double unit = 1.0e+6;  // microseconds
        if (debug_ == 1) {
            std::cout << "INTT OCL-out     time taken: " << std::fixed
                      << std::setprecision(8) << duration_ocl.count() * unit
                      << " us" << std::endl;
            std::cout << "INTT OCL-out avg time taken: " << std::fixed
                      << std::setprecision(8)
                      << duration_ocl.count() / completed->n_batch_ * unit
                      << " us" << std::endl;
        }
        if (debug_ == 2) {
            std::cout << "INTT out I/O     time taken: " << std::fixed
                      << std::setprecision(8) << duration_io.count() * unit
                      << " us" << std::endl;
            std::cout << "INTT out API     time taken: " << std::fixed
                      << std::setprecision(8) << duration_api.count() * unit
                      << " us" << std::endl
                      << std::endl;
        }
    }

    return 0;
}

DevicePool::DevicePool(int choice, Buffer& buffer,
                       std::future<bool>& exit_signal, uint64_t coeff_size,
                       uint32_t modulus_size,
                       uint64_t batch_size_dyadic_multiply,
                       uint64_t batch_size_ntt, uint64_t batch_size_intt,
                       uint32_t debug) {
    if (choice == EMU) {
        platform_ = aocl_utils::findPlatform("Intel(R) FPGA Emulation");
    } else {
        platform_ =
            aocl_utils::findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
    }

    if (platform_ == NULL) {
        printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
        FPGA_ASSERT(0);
    }

    cl_int status =
        clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, 0, NULL, &device_count_);
    if (status != CL_SUCCESS) {
        printf("No Success with clGetDeviceIDs!\n");
    }

    const char* n_dev = getenv("NUM_DEV");
    cl_uint dev_count = 1;
    if (n_dev) {
        dev_count = atoi(n_dev);
        if ((dev_count > device_count_) || (dev_count < 0)) {
            std::cout << "   [WARN] Maximal NUM_DEV is " << device_count_
                      << " on this platform." << std::endl;
            dev_count = device_count_;
        }
    }
    device_count_ = dev_count;
    std::cout << "   [INFO] Using " << device_count_ << " FPGA device(s)."
              << std::endl;
    cl_devices_ = new cl_device_id[device_count_];
    status = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, device_count_,
                            cl_devices_, NULL);
    aocl_utils::checkError(status, "Failed to create devices");

    future_exit_ = exit_signal.share();
    devices_ = new Device*[device_count_];
    for (unsigned int i = 0; i < device_count_; i++) {
        devices_[i] = new Device(
            cl_devices_[i], buffer, future_exit_, coeff_size, modulus_size,
            batch_size_dyadic_multiply, batch_size_ntt, batch_size_intt, debug);
        std::thread runner(&Device::run, devices_[i]);
        runners_.emplace_back(std::move(runner));
    }
}

DevicePool::~DevicePool() {
    for (auto& runner : runners_) {
        runner.join();
    }
    for (unsigned int i = 0; i < device_count_; i++) {
        delete devices_[i];
        devices_[i] = nullptr;
    }
    delete[] devices_;
    devices_ = nullptr;
    delete[] cl_devices_;
    cl_devices_ = nullptr;
}

}  // namespace fpga
}  // namespace hexl
}  // namespace intel
