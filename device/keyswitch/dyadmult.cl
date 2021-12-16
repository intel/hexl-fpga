#include "channels.h"

#define BIT_MASK(BITS) ((1UL << BITS) - 1)
#define BIT_MASK_52 BIT_MASK(52)
#define BIT_MASK_4 BIT_MASK(4)
#define BIT_MASK_8 BIT_MASK(8)

__single_task __kernel void broadcast_keys(
    __global DEVICE_MEM uint256_t* restrict k_switch_keys1,
    __global DEVICE_MEM uint256_t* restrict k_switch_keys2,
    __global DEVICE_MEM uint256_t* restrict k_switch_keys3) {
  unsigned i = 0;
  unsigned num_keys = MAX_DECOMP_MODULUS_SIZE * MAX_COFF_COUNT;
  while (true) {
    uint256_t keys1 = k_switch_keys1[i];
    uint256_t keys2 = k_switch_keys2[i];
    uint256_t keys3 = k_switch_keys3[i];
    STEP(i, num_keys);

    ulong keys[MAX_RNS_MODULUS_SIZE * 2];
    int j = 0;
    keys[j++] = keys1 & BIT_MASK_52;
    keys[j++] = (keys1 >> 52) & BIT_MASK_52;
    keys[j++] = (keys1 >> (52 * 2)) & BIT_MASK_52;
    keys[j++] = (keys1 >> (52 * 3)) & BIT_MASK_52;
    keys[j++] =
        ((keys1 >> (52 * 4)) & BIT_MASK_52) | ((keys2 & BIT_MASK_4) << 48);

    keys[j++] = (keys2 >> 4) & BIT_MASK_52;
    keys[j++] = (keys2 >> (4 + 52)) & BIT_MASK_52;
    keys[j++] = (keys2 >> (4 + 52 * 2)) & BIT_MASK_52;
    keys[j++] = (keys2 >> (4 + 52 * 3)) & BIT_MASK_52;
    keys[j++] =
        ((keys2 >> (4 + 52 * 4)) & BIT_MASK_52) | ((keys3 & BIT_MASK_8) << 44);

    keys[j++] = (keys3 >> 8) & BIT_MASK_52;
    keys[j++] = (keys3 >> (8 + 52)) & BIT_MASK_52;
    keys[j++] = (keys3 >> (8 + 52 * 2)) & BIT_MASK_52;
    keys[j++] = (keys3 >> (8 + 52 * 3)) & BIT_MASK_52;

    #pragma unroll
    for (int ins = 0; ins < MAX_RNS_MODULUS_SIZE; ins++) {
      ulong2 key;
      key.s0 = keys[ins * 2];
      key.s1 = keys[ins * 2 + 1];
      ASSERT(key.s0 < MAX_KEY, "key > MAX_KEY\n");
      ASSERT(key.s1 < MAX_KEY, "key > MAX_KEY\n");
      write_channel_intel(ch_dyadmult_keys[ins], key);
    }
  }
}

void _dyadmult(moduli_t moduli, uint64_t _key_modulus_size,
               uint64_t _coeff_count, uint64_t decomp_modulus_size,
               uint64_t rns_modulus_size, uint64_t key_component_count,
               int ntt_ins, unsigned rmem) {
  uint64_t t_poly_lazy[MAX_COFF_COUNT][MAX_KEY_COMPONENT_SIZE];

  unsigned coeff_count = _coeff_count;
  unsigned key_modulus_size = _key_modulus_size;

  unsigned key_index =
      (ntt_ins == decomp_modulus_size ? key_modulus_size - 1 : ntt_ins);

  ulong4 curr_moduli = moduli.data[key_index];
  ulong4 last_moduli = moduli.data[key_modulus_size - 1];
  last_moduli.s2 = key_modulus_size - 1;

  unsigned l = 0;
  unsigned j = 0;
  unsigned k_switch_keys_off;

  #pragma ivdep
  while (true) {
    if (l == 0) {
      if (j == 0) {
        k_switch_keys_off = ntt_ins * decomp_modulus_size * coeff_count;
      }
      if (ntt_ins == 0 && j == 0) {
        #pragma unroll
        for (int key_component = 0; key_component < MAX_KEY_COMPONENT_SIZE;
             key_component++) {
          write_channel_intel(ch_intt_modulus[1 + key_component], last_moduli);
        }
      }
    }
    uint64_t val = read_channel_intel(ch_ntt_elements_out[ntt_ins]);
    // ulong2 keys = rmem ? k_switch_keys[k_switch_keys_off++] : 999;
    ulong2 keys = read_channel_intel(ch_dyadmult_keys[ntt_ins]);

    #pragma unroll
    for (unsigned k = 0; k < key_component_count; ++k) {
      uint64_t t_poly_idx = l;
      uint64_t prod = MultiplyUIntMod(val, ((ulong*)&keys)[k], curr_moduli.s0,
                                      curr_moduli.s3);

      uint64_t prev = t_poly_lazy[t_poly_idx][k];
      prev = j == 0 ? 0 : prev;

      uint64_t sum = AddUIntMod(prod, prev, curr_moduli.s0);
      t_poly_lazy[t_poly_idx][k] = sum;

      // save in the last iteration
      if (j == (decomp_modulus_size - 1)) {
        // 0 - n, 7n - 8n
        // n - 2n, 8n - 9n
        // 2n - 3n, 9n - 10n
        // ...
        // 6n - 7n, 13n - 14n
        // overall 14n
        write_channel_intel(ch_t_poly_prod_iter[ntt_ins][k], sum);
      }
    }
    STEP(l, coeff_count);
    if (l == 0) {
      STEP(j, decomp_modulus_size);
    }
  }
}

#define DYADMULT_INS(INS_ID)                                              \
  __single_task __kernel void dyadmult##INS_ID(                           \
      moduli_t moduli, uint64_t key_modulus_size, uint64_t coeff_count,   \
      uint64_t decomp_modulus_size, uint64_t rns_modulus_size,            \
      unsigned rmem) {                                                    \
    _dyadmult(moduli, key_modulus_size, coeff_count, decomp_modulus_size, \
              rns_modulus_size, MAX_KEY_COMPONENT_SIZE, INS_ID, rmem);    \
  }

#if MAX_RNS_MODULUS_SIZE > 0
DYADMULT_INS(0)
#endif

#if MAX_RNS_MODULUS_SIZE > 1
DYADMULT_INS(1)
#endif

#if MAX_RNS_MODULUS_SIZE > 2
DYADMULT_INS(2)
#endif

#if MAX_RNS_MODULUS_SIZE > 3
DYADMULT_INS(3)
#endif

#if MAX_RNS_MODULUS_SIZE > 4
DYADMULT_INS(4)
#endif

#if MAX_RNS_MODULUS_SIZE > 5
DYADMULT_INS(5)
#endif

#if MAX_RNS_MODULUS_SIZE > 6
DYADMULT_INS(6)
#endif