// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "keyswitch/channels.h"

// load
#include "keyswitch/intt_core.cl"
#include "keyswitch/load.cl"
#include "keyswitch/ntt_core.cl"

// intt1
#include "keyswitch/intt1.cl"
#include "keyswitch/intt1_forward.cl"
#include "keyswitch/intt1_redu.cl"

// ntt1
#include "keyswitch/ntt1.cl"

// dyadmult
#include "keyswitch/dyadmult.cl"

// intt2
#include "keyswitch/intt2.cl"
#include "keyswitch/intt2_forward.cl"
#include "keyswitch/intt2_redu.cl"

// ntt2
#include "keyswitch/ntt2.cl"

// ms
#include "keyswitch/ms.cl"

// store
#include "keyswitch/store.cl"
