// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "channels.h"

NTT2_INS(0, 7, MAX_RNS_MODULUS_SIZE)
NTT2_INS(0, 8, MAX_RNS_MODULUS_SIZE + 1)

#if CORES > 1
NTT2_INS(1, 7, MAX_RNS_MODULUS_SIZE)
NTT2_INS(1, 8, MAX_RNS_MODULUS_SIZE + 1)
#endif
