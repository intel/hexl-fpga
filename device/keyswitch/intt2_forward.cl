#include "channels.h"

void _intt2_forward(int key_component) {
  while (true) {
    uint64_t data = read_channel_intel(
        ch_t_poly_prod_iter[MAX_RNS_MODULUS_SIZE - 1][key_component]);
    write_channel_intel(ch_intt_elements_in[1 + key_component], data);
  }
}

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0))) __attribute__((autorun))
__kernel void
intt21_forward() {
  _intt2_forward(0);
}

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0))) __attribute__((autorun))
__kernel void
intt22_forward() {
  _intt2_forward(1);
}