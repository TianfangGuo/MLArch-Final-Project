#include "mat_mult_trgt.h"
#include <cstdint>
#include "ap_int.h"

void mat_mult_trgt(
    const uint32_t *A,
    const uint32_t *B,
    uint32_t *C
) {
	// HLS pragma tuning: tile registers and loop pipelining
	#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=MAT_A_SIZE
	#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=MAT_B_SIZE
	#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem0 depth=MAT_C_SIZE
	#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local buffers
    uint32_t A_buf[M1][M0][K];
    uint32_t B_tmp[K][N0];
    uint32_t B_tmp_t[N0][K]; // Transposed tile of B
    uint32_t C_tile[M1][M0][N0];

    #pragma HLS ARRAY_PARTITION variable=A_buf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=B_tmp complete dim=1
    #pragma HLS ARRAY_PARTITION variable=B_tmp_t complete dim=1
    #pragma HLS ARRAY_PARTITION variable=C_tile complete dim=2

    loop_n1: for (int n1 = 0; n1 < N1; n1++) {
        loop_m2: for (int m2 = 0; m2 < M2; m2++) {

            // Load & tile B[n1,:,:] into B_tmp
            loop_k_b: for (int k = 0; k < K; k++) {
                #pragma HLS PIPELINE
                for (int n0 = 0; n0 < N0; n0++) {
                    B_tmp[k][n0] = B[n1 * K * N0 + k * N0 + n0];
                }
            }

            // Transpose B_tmp[K][N0] -> B_tmp_t[N0][K]
            loop_transpose: for (int k = 0; k < K; k++) {
                #pragma HLS PIPELINE
                for (int n0 = 0; n0 < N0; n0++) {
                    B_tmp_t[n0][k] = B_tmp[k][n0];
                }
            }

            loop_m1: for (int m1 = 0; m1 < M1; m1++) {
                loop_m0: for (int m0 = 0; m0 < M0; m0++) {

                    // Clear output tile
                    for (int n0 = 0; n0 < N0; n0++) {
                        #pragma HLS UNROLL
                        C_tile[m1][m0][n0] = 0;
                    }

                    // Load A tile A[m2][m1][m0][k] into register
                    for (int k = 0; k < K; k++) {
                        A_buf[m1][m0][k] = A[m2 * M1 * M0 * K + m1 * M0 * K + m0 * K + k];
                    }

                    // Outer-product style accumulation
                    loop_n0: for (int n0 = 0; n0 < N0; n0++) {
                        #pragma HLS PIPELINE II=1
                        for (int k = 0; k < K; k++) {
                            #pragma HLS unroll factor=8
                            C_tile[m1][m0][n0] += A_buf[m1][m0][k] * B_tmp_t[n0][k];
                        }
                    }

                    // Store results back to global C buffer
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
