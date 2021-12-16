#include "channels.h"

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0))) __kernel void
intt1_forward(uint64_t coeff_count, uint64_t rns_modulus_size) {
  while (true) {
    uint64_t data = read_channel_intel(ch_intt_elements_out[0]);

    // broadcast to rns_modulus_size NTTs
    #pragma unroll
    for (int ins = 0; ins < MAX_RNS_MODULUS_SIZE; ins++) {
      write_channel_intel(ch_intt_elements_out_rep[ins], data);
    }
  }
}
