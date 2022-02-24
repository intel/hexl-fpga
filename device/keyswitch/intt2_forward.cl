// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "channels.h"

void _intt2_forward(int COREID, int key_component) {
    while (true) {
        uint64_t data =
            read_channel_intel(ch_t_poly_prod_iter_last[COREID][key_component]);
        write_channel_intel(ch_intt_elements_in[COREID][1 + key_component],
                            data);
    }
}

__single_task __autorun void intt_forward01() { _intt2_forward(0, 0); }
__single_task __autorun void intt_forward02() { _intt2_forward(0, 1); }

#if CORES > 1
__single_task __autorun void intt_forward11() { _intt2_forward(1, 0); }
__single_task __autorun void intt_forward12() { _intt2_forward(1, 1); }
#endif
