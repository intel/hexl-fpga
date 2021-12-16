#include "channels.h"

void _intt1_redu(moduli_t moduli, uint64_t key_modulus_size,
                 uint64_t coeff_count, uint64_t decomp_modulus_size,
                 int ntt_ins) {
  size_t key_index =
      (ntt_ins == decomp_modulus_size ? key_modulus_size - 1 : ntt_ins);
  ulong4 cur_moduli = moduli.data[key_index];

  unsigned n = 0;
  unsigned j = 0;
  uint64_t modulus = cur_moduli.s0;
  uint64_t barrett_factor = cur_moduli.s1;
  cur_moduli.s2 = key_index;

  while (true) {
    if (n == 0) {
      write_channel_intel(ch_ntt_modulus[ntt_ins], cur_moduli);
      STEP(j, decomp_modulus_size);
    }
    uint64_t val = read_channel_intel(ch_intt_elements_out_rep[ntt_ins]);
    uint64_t val_redu = BarrettReduce64(val, modulus, barrett_factor);
    write_channel_intel(ch_ntt_elements_in[ntt_ins], val_redu);
    STEP(n, coeff_count);
  }
}

#define INTT1_REDU(INS_ID)                                                  \
  __single_task __kernel void intt1_redu##INS_ID(                           \
      moduli_t moduli, uint64_t key_modulus_size, uint64_t coeff_count,     \
      uint64_t decomp_modulus_size) {                                       \
    _intt1_redu(moduli, key_modulus_size, coeff_count, decomp_modulus_size, \
                INS_ID);                                                    \
  }

#if MAX_RNS_MODULUS_SIZE > 0
INTT1_REDU(0)
#endif

#if MAX_RNS_MODULUS_SIZE > 1
INTT1_REDU(1)
#endif

#if MAX_RNS_MODULUS_SIZE > 2
INTT1_REDU(2)
#endif

#if MAX_RNS_MODULUS_SIZE > 3
INTT1_REDU(3)
#endif

#if MAX_RNS_MODULUS_SIZE > 4
INTT1_REDU(4)
#endif

#if MAX_RNS_MODULUS_SIZE > 5
INTT1_REDU(5)
#endif

#if MAX_RNS_MODULUS_SIZE > 6
INTT1_REDU(6)
#endif