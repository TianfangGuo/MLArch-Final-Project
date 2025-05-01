#include "mat_mult_trgt.h"
#include <cstdint>
#include "ap_int.h"

void mat_mult_trgt(
    const uint32_t *A,
    const uint32_t *B,
    uint32_t *C
) {
    #pragma HLS ALLOCATION instances=mul limit=8 operation

    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=MAT_A_SIZE
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=MAT_B_SIZE
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0 depth=MAT_C_SIZE
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    uint32_t A_buf[M1][M0][K];
    uint32_t B_tmp[K][N0];
    uint32_t B_tmp_t[N0][K];
    uint32_t C_tile[M1][M0][N0];

    #pragma HLS ARRAY_PARTITION variable=A_buf complete dim=3
    #pragma HLS ARRAY_PARTITION variable=B_tmp_t complete dim=2
    #pragma HLS ARRAY_PARTITION variable=C_tile complete

    #pragma HLS RESOURCE variable=A_buf core=RAM_2P_BRAM
    #pragma HLS RESOURCE variable=B_tmp_t core=RAM_2P_BRAM
    #pragma HLS RESOURCE variable=C_tile core=RAM_2P_BRAM

    ap_uint<K> A_bitvec[M2][M1][M0];
    ap_uint<K> B_bitvec[N1][N0];

    loop_bitvec_a_m2:
    for (int m2 = 0; m2 < M2; m2++) {
        loop_bitvec_a_m1:
        for (int m1 = 0; m1 < M1; m1++) {
            loop_bitvec_a_m0:
            for (int m0 = 0; m0 < M0; m0++) {
                uint32_t A_tile[K];
                ap_uint<K> mask = 0;

                loop_load_A_tile_k:
                for (int k = 0; k < K; k++) {
                    #pragma HLS PIPELINE
                    uint32_t val = A[m2 * M1 * M0 * K + m1 * M0 * K + m0 * K + k];
                    A_tile[k] = val;
                }

                loop_set_A_bitvec_k:
                for (int k = 0; k < K; k++) {
                    if (A_tile[k] != 0) mask.set_bit(k, 1);
                }

                A_bitvec[m2][m1][m0] = mask;
            }
        }
    }

    loop_bitvec_b_n1:
    for (int n1 = 0; n1 < N1; n1++) {
        loop_bitvec_b_n0:
        for (int n0 = 0; n0 < N0; n0++) {
            ap_uint<K> mask = 0;

            loop_set_B_bitvec_k:
            for (int k = 0; k < K; k++) {
                #pragma HLS PIPELINE
                uint32_t val = B[n1 * K * N0 + k * N0 + n0];
                if (val != 0) mask.set_bit(k, 1);
            }

            B_bitvec[n1][n0] = mask;
        }
    }

    loop_n1:
    for (int n1 = 0; n1 < N1; n1++) {
        loop_m2:
        for (int m2 = 0; m2 < M2; m2++) {

            // Load B[n1,:,:] into B_tmp
            loop_b_tile_linear:
            for (int i = 0; i < K * N0; i++) {
                #pragma HLS PIPELINE
                B_tmp[i / N0][i % N0] = B[n1 * K * N0 + i];
            }

            // Transpose B_tmp → B_tmp_t
            loop_transpose_k:
            for (int k = 0; k < K; k++) {
                #pragma HLS PIPELINE
                loop_transpose_n0:
                for (int n0 = 0; n0 < N0; n0++) {
                    B_tmp_t[n0][k] = B_tmp[k][n0];
                }
            }

            loop_m1:
            for (int m1 = 0; m1 < M1; m1++) {
                loop_m0:
                for (int m0 = 0; m0 < M0; m0++) {

                    loop_clear_c_tile_n0:
                    for (int n0 = 0; n0 < N0; n0++) {
                        #pragma HLS UNROLL
                        C_tile[m1][m0][n0] = 0;
                    }

                    loop_load_A_buf_k:
                    for (int k = 0; k < K; k++) {
                        A_buf[m1][m0][k] = A[m2 * M1 * M0 * K + m1 * M0 * K + m0 * K + k];
                    }

                    // === Compute kernel ===
                    loop_compute_n0:
                    for (int n0 = 0; n0 < N0; n0++) {
                        ap_uint<K> active_k = A_bitvec[m2][m1][m0] & B_bitvec[n1][n0];
                        if (active_k == 0) continue;

                        uint32_t sum = 0;

                        loop_compute_k:
                        for (int k = 0; k < K; k++) {
                            if (active_k[k]) {
                                sum += A_buf[m1][m0][k] * B_tmp_t[n0][k];
                            }
                        }

                        C_tile[m1][m0][n0] = sum;
                    }

                    loop_store_c_tile_n0:
                    for (int n0 = 0; n0 < N0; n0++) {
                        int idx = n1 * M2 * M1 * M0 * N0 +
                                  m2 * M1 * M0 * N0 +
                                  m1 * M0 * N0 +
                                  m0 * N0 + n0;
                        C[idx] = C_tile[m1][m0][n0];
                    }
                }
            }
        }
    }
}
