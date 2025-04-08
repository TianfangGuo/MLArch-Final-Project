#include "mat_mult.h"
#include <cstdint>

// Assumption: Dimensions are a multiple of the block sizes

void mat_mult(
    const uint32_t M,
    const uint32_t N,
    const uint32_t K,
    const uint32_t *A,
    const uint32_t *B,
    uint32_t *C
) {
    #pragma HLS INTERFACE m_axi port=A offset=slave depth=1024 bundle=gmem0
    #pragma HLS INTERFACE m_axi port=B offset=slave depth=1024 bundle=gmem1
    #pragma HLS INTERFACE m_axi port=C offset=slave depth=1024 bundle=gmem0
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
    loop_block_m: for (uint32_t block_m = 0; block_m < M; block_m += SIZE_M) {
    #pragma HLS LOOP_TRIPCOUNT min=1 max=1
        loop_block_n: for (uint32_t block_n = 0; block_n < N; block_n += SIZE_N) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=1
        	// Reset C_buf to zero
        	rst_c_m: for (uint32_t m = 0; m < SIZE_M; m++) {
        	    rst_c_n: for (uint32_t n = 0; n < SIZE_N; n++) {
        	        C_buf[m][n] = 0;
        	    }
        	}

            loop_block_k: for (uint32_t block_k = 0; block_k < K; block_k += SIZE_K) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1
                // Load A and B blocks into buffers
                load_A_m: for (uint32_t m = 0; m < SIZE_M; m++) {
                #pragma HLS PIPELINE off
                    load_A_k: for (uint32_t k = 0; k < SIZE_K; k++) {
                    #pragma HLS PIPELINE
                        A_buf[m][k] = A[(block_m + m) * K + (block_k + k)];
                    }
                }
                load_B_k: for (uint32_t k = 0; k < SIZE_K; k++) {
                #pragma HLS PIPELINE off
                    load_B_n: for (uint32_t n = 0; n < SIZE_N; n++) {
                    #pragma HLS PIPELINE
                        B_buf[k][n] = B[(block_k + k) * N + (block_n + n)];
                    }
                }

                // Compute the product of A and B blocks
                compute_m: for (uint32_t m = 0; m < SIZE_M; m++) {
                #pragma HLS PIPELINE off
                    compute_n: for (uint32_t n = 0; n < SIZE_N; n++) {
                    #pragma HLS PIPELINE
                        compute_k: for (uint32_t k = 0; k < SIZE_K; k++) {
                        #pragma HLS UNROLL
                            C_buf[m][n] += A_buf[m][k] * B_buf[k][n];
                        }
                    }
                }

                // Store the result in the output matrix C
                store_C_m: for (uint32_t m = 0; m < SIZE_M; m++) {
                #pragma HLS PIPELINE off
                    store_C_n: for (uint32_t n = 0; n < SIZE_N; n++) {
                    #pragma HLS PIPELINE
                        C[(block_m + m) * N + (block_n + n)] = C_buf[m][n];
                    }
                }

            }
        }
    }


}
