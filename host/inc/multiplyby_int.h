// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef __KEYSWITCH_INT_H__
#define __KEYSWITCH_INT_H__

#include <cstdint>

namespace intel {
namespace hexl {
namespace fpga {
/// @brief
/// Function set_worksize_MultiplyBy_int
/// Reserves software resources for the MultiplyBy
/// @param ws integer storing the worksize
///
void set_worksize_MultiplyBy_int(uint64_t ws);

/// @brief Perform multiplication operand1 * operand2
/// @param context
/// @param operand1
/// @param operand1_primes_index
/// @param operand2
/// @param operand2_primes_index
/// @param result
/// @param result_primes_index
void MultiplyBy_int(const MultiplyByContext& context,
                    const std::vector<uint64_t>& operand1,
                    const std::vector<uint8_t>& operand1_primes_index,
                    const std::vector<uint64_t>& operand2,
                    const std::vector<uint8_t>& operand2_primes_index,
                    std::vector<uint64_t>& result,
                    const std::vector<uint8_t>& result_primes_index);

/// @brief
///
/// Function MultiplyByCompleted_int
/// Executed after MultiplyBy to sync up the outstanding MultiplyBy tasks
bool MultiplyByCompleted_int();

}  // namespace fpga
}  // namespace hexl
}  // namespace intel

#endif
