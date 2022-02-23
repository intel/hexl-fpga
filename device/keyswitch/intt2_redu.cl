// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

void _intt2_redu(int COREID, uint64_t key_modulus_size, int key_component) {
    uint64_t elements[MAX_COFF_COUNT];

#pragma disable_loop_pipelining
    while (true) {
        ulong4 moduli =
            read_channel_intel(ch_intt2_redu_params[COREID][key_component]);
        uint64_t decomp_modulus_index = moduli.s2 & 0xf;
        uint64_t qk = moduli.s2 >> 4;
        uint64_t qk_half = qk >> 1;
        uint64_t qi = moduli.s0 & MODULUS_BIT_MASK;

        write_channel_intel(
            ch_ntt_modulus[COREID][MAX_RNS_MODULUS_SIZE + key_component],
            moduli);

        uint64_t barrett_factor = moduli.s1;
        uint64_t fix = qi - BarrettReduce64(qk_half, qi, barrett_factor);
        unsigned coeff_count = GET_COEFF_COUNT(moduli.s0);

        for (int j = 0; j < coeff_count; j++) {
            uint64_t val;
            if (decomp_modulus_index == 0) {
                val = read_channel_intel(
                    ch_intt_elements_out[COREID][1 + key_component]);
                ASSERT(val < qk, "x >= modulus\n");
                val = AddUIntMod(val, qk_half, qk);
                elements[j] = val;
            } else {
                val = elements[j];
            }

            // TO BE CONFIRMED: add the fix before the barrett reduce
            val += fix;
            uint64_t val_redu = BarrettReduce64(val, qi, barrett_factor);
            // val_redu += fix;

            write_channel_intel(
                ch_ntt_elements_in[COREID]
                                  [MAX_RNS_MODULUS_SIZE + key_component],
                val_redu);
        }
    }
}

__single_task __autorun void intt_redu01() {
    _intt2_redu(0, MAX_KEY_MODULUS_SIZE, 0);
}

__single_task __autorun void intt_redu02() {
    _intt2_redu(0, MAX_KEY_MODULUS_SIZE, 1);
}

#if CORES > 1
__single_task __autorun void intt_redu10() {
    _intt2_redu(1, MAX_KEY_MODULUS_SIZE, 0);
}

__single_task __autorun void intt_redu11() {
    _intt2_redu(1, MAX_KEY_MODULUS_SIZE, 1);
}
#endif
