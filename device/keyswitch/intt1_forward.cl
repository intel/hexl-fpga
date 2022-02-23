// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "channels.h"

void _intt_broadcast(int COREID) {
    while (true) {
        uint64_t data = read_channel_intel(ch_intt_elements_out[COREID][0]);

// broadcast to rns_modulus_size NTTs
#pragma unroll
        for (int ins = 0; ins < MAX_RNS_MODULUS_SIZE; ins++) {
            write_channel_intel(ch_intt_elements_out_rep[COREID][ins], data);
        }
    }
}

__single_task __autorun void intt_broadcast0() { _intt_broadcast(0); }

#if CORES > 1
__single_task __autorun void intt_broadcast1() { _intt_broadcast(1); }
#endif
