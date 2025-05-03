#include "mat_mult_trgt.h"
#include <cstdint>
#include "ap_int.h"

uint32_t mac_accumulate(
    const uint32_t *A_buf,
    uint32_t B_buf[K][N0],
    const uint8_t *k_list,
    uint8_t count,
    int n0
) {
    #pragma HLS INLINE
    uint32_t total_sum = 0;

    for (int base = 0; base < count; base += 8) {
        #pragma HLS UNROLL
        uint32_t partial[8] = {0};
        #pragma HLS ARRAY_PARTITION variable=partial complete

        for (int i = 0; i < 8; i++) {
            #pragma HLS UNROLL
            int idx = base + i;
            if (idx < count) {
                uint8_t k = k_list[idx];
                partial[i] = A_buf[k] * B_buf[k][n0];
            }
        }

        for (int i = 0; i < 8; i++) {
            #pragma HLS UNROLL
            total_sum += partial[i];
        }
    }

    return total_sum;
}

void load_B_buf(const uint32_t *B, int n1, uint32_t B_buf[K][N0]) {
    #pragma HLS INLINE off
    for (int k = 0; k < K; k++) {
        for (int n0 = 0; n0 < N0; n0++) {
            #pragma HLS PIPELINE II=1
            B_buf[k][n0] = B[n1 * K * N0 + k * N0 + n0];
        }
    }
}

void compute_tile(
    const uint32_t *A,
    uint32_t B_buf[K][N0],
    uint32_t C_tile[M1][M0][N0],
    int m2, int n1
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=B_buf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=C_tile complete dim=3

    ap_uint<K> B_bitvec[N0];
    #pragma HLS ARRAY_PARTITION variable=B_bitvec complete dim=1

    for (int n0 = 0; n0 < N0; n0++) {
        ap_uint<K> mask = 0;
        for (int k = 0; k < K; k++) {
            if (B_buf[k][n0] != 0) mask.set_bit(k, 1);
        }
        B_bitvec[n0] = mask;
    }

    for (int m1 = 0; m1 < M1; m1++) {
        for (int m0 = 0; m0 < M0; m0++) {
            for (int n0 = 0; n0 < N0; n0++) {
                #pragma HLS PIPELINE II=1 rewind

                uint32_t A_buf[K];
                #pragma HLS ARRAY_PARTITION variable=A_buf complete dim=1

                ap_uint<K> A_bitvec = 0;
                for (int k = 0; k < K; k++) {
                    uint32_t val = A[m2 * M1 * M0 * K + m1 * M0 * K + m0 * K + k];
                    A_buf[k] = val;
                    if (val != 0) A_bitvec.set_bit(k, 1);
                }

                ap_uint<K> active_k = A_bitvec & B_bitvec[n0];
                if (active_k == 0) {
                    C_tile[m1][m0][n0] = 0;
                    continue;
                }

                uint8_t k_list[K];
                uint8_t count = 0;
                #pragma HLS ARRAY_PARTITION variable=k_list complete dim=1

                for (int k = 0; k < K; k++) {
                    if (active_k[k]) {
                        k_list[count++] = k;
                    }
                }

                uint32_t sum = mac_accumulate(A_buf, B_buf, k_list, count, n0);
                C_tile[m1][m0][n0] = sum;
            }
        }
    }
}

void store_C_tile(uint32_t *C, uint32_t C_tile[M1][M0][N0], int m2, int n1) {
    #pragma HLS INLINE off
    for (int m1 = 0; m1 < M1; m1++) {
        for (int m0 = 0; m0 < M0; m0++) {
            for (int n0 = 0; n0 < N0; n0++) {
                #pragma HLS PIPELINE II=1
                int idx = n1 * M2 * M1 * M0 * N0 +
                          m2 * M1 * M0 * N0 +
                          m1 * M0 * N0 +
                          m0 * N0 + n0;
                C[idx] = C_tile[m1][m0][n0];
            }
        }
    }
}

void mat_mult_trgt(
    const uint32_t *A,
    const uint32_t *B,
    uint32_t *C
) {
    #pragma HLS ALLOCATION instances=mul limit=8 operation
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=MAT_A_SIZE
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=MAT_B_SIZE
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=MAT_C_SIZE
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    uint32_t B_buf[K][N0];
    uint32_t C_tile[M1][M0][N0];

    #pragma HLS ARRAY_PARTITION variable=B_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=B_buf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=C_tile complete dim=3

    for (int tile_id = 0; tile_id < N1 * M2; tile_id++) {
        #pragma HLS DATAFLOW
        int n1 = tile_id / M2;
        int m2 = tile_id % M2;

        load_B_buf(B, n1, B_buf);
        compute_tile(A, B_buf, C_tile, m2, n1);
        store_C_tile(C, C_tile, m2, n1);
    }
}
