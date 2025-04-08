#include "mat_mult.h"
#include <cstdint>

void mat_mult(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const uint32_t *A,
    const uint32_t *B,
    uint32_t *C
) {
    #pragma HLS INTERFACE m_axi port=A offset=slave depth=1024 bundle=wrapper0
    #pragma HLS INTERFACE m_axi port=B offset=slave depth=1024 bundle=wrapper0
    #pragma HLS INTERFACE m_axi port=C offset=slave depth=1024 bundle=wrapper0
    #pragma HLS INTERFACE s_axilite port=M bundle=control
    #pragma HLS INTERFACE s_axilite port=N bundle=control
    #pragma HLS INTERFACE s_axilite port=K bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Create buffers for input and output matrices
    uint32_t A_buf[SIZE_M][SIZE_K];
    uint32_t B_buf[SIZE_K][SIZE_N];
    uint32_t C_buf[SIZE_M][SIZE_N];

    #pragma HLS array_partition variable=A_buf complete dim=2
    #pragma HLS array_partition variable=B_buf complete dim=1

    // Perform matrix multiplication (inner product)
    for (uint32_t block_m = 0; block_m < M; block_m += SIZE_M) {
        for (uint32_t block_n = 0; block_n < N; block_n += SIZE_N) {
            for (uint32_t block_k = 0; block_k < K; block_k += SIZE_K) {

                // Load A and B blocks into buffers
                // Reset C_buf to zero
                for (uint32_t i = 0; i < SIZE_M; i++) {
                    for (uint32_t j = 0; j < SIZE_N; j++) {
                        C_buf[i][j] = 0;
                    }
                }
                for (uint32_t i = 0; i < SIZE_M; i++) {
                    for (uint32_t k = 0; k < SIZE_K; k++) {
                        if (block_m + i < M && block_k + k < K) {
                            A_buf[i][k] = A[(block_m + i) * K + (block_k + k)];
                        } else {
                            A_buf[i][k] = 0;
                        }
                    }
                }
                for (uint32_t k = 0; k < SIZE_K; k++) {
                    for (uint32_t j = 0; j < SIZE_N; j++) {
                        if (block_k + k < K && block_n + j < N) {
                            B_buf[k][j] = B[(block_k + k) * N + (block_n + j)];
                        } else {
                            B_buf[k][j] = 0;
                        }
                    }
                }

                // Compute the product of A and B blocks
                for (uint32_t m = 0; m < SIZE_M; m++) {
                    for (uint32_t n = 0; n < SIZE_N; n++) {
                        for (uint32_t k = 0; k < SIZE_K; k++) {
                            if (block_m + m < M && block_n + n < N) {
                                C_buf[m][n] += A_buf[m][k] * B_buf[k][n];
                            }
                        }
                    }
                }

                // Store the result in the output matrix C
                for (uint32_t m = 0; m < SIZE_M; m++) {
                    for (uint32_t n = 0; n < SIZE_N; n++) {
                        if (block_m + m < M && block_n + n < N) {
                            C[(block_m + m) * N + (block_n + n)] = C_buf[m][n];
                        }
                    }
                }

            }
        }
    }


}