// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
namespace L0 {
// According to the OpenCL C spec, the format string must be in the constant
// address space. To simplify code when invoking printf, the following macros
// are defined.

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

#ifdef FPGA_EMULATOR
#define PRINTF(format, ...)                                          \
  {                                                                  \
    static const CL_CONSTANT char _format[] = format;                \
    sycl::ext::oneapi::experimental::printf(_format, ##__VA_ARGS__); \
  }
#else
#define PRINTF(format, ...)
#endif

#define ASSERT(cond, message, ...)         \
  if (!(cond)) {                           \
    PRINTF("%s#%d: ", __FILE__, __LINE__); \
    PRINTF(message, ##__VA_ARGS__);        \
  }

#define ASSERT2(cond)                      \
  if (!(cond)) {                           \
    PRINTF("%s#%d\n", __FILE__, __LINE__); \
  }

#define MAX_PRIME_BITS 60
#define BIT_MASK(BITS) ((1UL << BITS) - 1)

constexpr uint64_t READY = 1;

typedef ac_int<MAX_PRIME_BITS, false> ntt_data_t;

template <unsigned int VEC>
struct WideVector_t {
  ntt_data_t data[VEC * 2];
};
template <unsigned int VEC>
struct TwiddleFactor_t {
  uint64_t data[VEC];
};

static unsigned get_ntt_log(unsigned size) {
  unsigned log;
  if (size == 65536) {
    log = 16;
  } else if (size == 32768) {
    size = 15;
  } else if (size == 16384) {
    log = 14;
  } else if (size == 8192) {
    log = 13;
  } else if (size == 4096) {
    log = 12;
  } else if (size == 2048) {
    log = 11;
  } else if (size == 1024) {
    log = 10;
  } else {
    ASSERT(false, "ntt/intt size (%u) is not supported\n", size);
  }
  return log;
}

static unsigned get_vec_log(unsigned VEC) {
  unsigned log;
  if (VEC == 1) {
    log = 0;
  } else if (VEC == 2) {
    log = 1;
  } else if (VEC == 4) {
    log = 2;
  } else if (VEC == 8) {
    log = 3;
  } else if (VEC == 16) {
    log = 4;
  } else if (VEC == 32) {
    log = 5;
  } else {
    ASSERT(false, "VEC (%u) is not supported\n", VEC);
  }
  ASSERT((1 << log) == VEC, "Failed to get_vec_log\n");
  return log;
}
}  // namespace L0
