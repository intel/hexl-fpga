void _intt2_redu(moduli_t moduli, uint64_t key_modulus_size,
                 uint64_t coeff_count, uint64_t decomp_modulus_size,
                 int key_component) {
  uint64_t elements[MAX_COFF_COUNT];

  unsigned j = 0;
  unsigned i = 0;
  ulong4 modulus = moduli.data[key_modulus_size - 1];
  uint64_t qk = modulus.s0;
  uint64_t qk_half = qk >> 1;

  uint64_t qi;
  uint64_t barrett_factor;
  uint64_t fix;

  while (true) {
    if (j == 0) {
      int ntt_ins = i;
      // (ct mod 4qk) mod qi
      ulong4 modulus_cur = moduli.data[i];
      qi = modulus_cur.s0;

      modulus_cur.s2 = i;
      write_channel_intel(ch_ntt_modulus[MAX_RNS_MODULUS_SIZE + key_component],
                          modulus_cur);

      barrett_factor = modulus_cur.s1;
      fix = qi - BarrettReduce64(qk_half, qi, barrett_factor);
    }
    uint64_t val;
    if (i == 0) {
      val = read_channel_intel(ch_intt_elements_out[1 + key_component]);
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
        ch_ntt_elements_in[MAX_RNS_MODULUS_SIZE + key_component], val_redu);
    STEP(j, coeff_count);
    if (j == 0) {
      STEP(i, decomp_modulus_size);
    }
  }
}

__single_task void intt21_redu(moduli_t moduli, uint64_t key_modulus_size,
                               uint64_t coeff_count,
                               uint64_t decomp_modulus_size) {
  _intt2_redu(moduli, key_modulus_size, coeff_count, decomp_modulus_size, 0);
}

__single_task void intt22_redu(moduli_t moduli, uint64_t key_modulus_size,
                               uint64_t coeff_count,
                               uint64_t decomp_modulus_size) {
  _intt2_redu(moduli, key_modulus_size, coeff_count, decomp_modulus_size, 1);
}