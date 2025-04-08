#include "mat_mult_sparse.h"
#include <cstdint>

void mat_mult_sparse(
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

    // Sparse Matrix Multiply Game Plan
    // 1. Capture a complete row of A
    // 2. In this row of A, capture all the non-zero elements and their corresponding column indices
    // 3. For each non-zero element of A, capture the corresponding row of B

    // Create buffers for input and output matrices
    uint32_t SRAM_A[BLOCK_SIZE][BLOCK_SIZE];        // 32 x 32
    uint32_t SRAM_B[BLOCK_SIZE][BLOCK_SIZE];        // 32 x 32
    uint32_t SRAM_C[BLOCK_SIZE][BLOCK_SIZE];        // 32 x 32

    uint32_t REG_A[PE_SIZE][BLOCK_SIZE];            // 16 x 32
    uint32_t REG_B[BLOCK_SIZE][PE_SIZE];            // 32 x 16
    uint32_t REG_C[PE_SIZE][PE_SIZE];               // 16 x 16

    loop_block_m: for (uint32_t block_m = 0; block_m < M; block_m += BLOCK_SIZE) {
        loop_block_n: for (uint32_t block_n = 0; block_n < N; block_n += BLOCK_SIZE) {
            // Init SRAM_C to zero
            init_C_m: for (uint32_t m = 0; m < BLOCK_SIZE; m++) {
                init_C_n: for (uint32_t n = 0; n < BLOCK_SIZE; n++) {
                    SRAM_C[m][n] = 0;
                }
            }

            loop_block_k: for (uint32_t block_k = 0; block_k < K; block_k += BLOCK_SIZE) {
                // Load A and B blocks into SRAM
                load_A_m: for (uint32_t m = 0; m < BLOCK_SIZE; m++) {
                    load_A_k: for (uint32_t k = 0; k < BLOCK_SIZE; k++) {
                        SRAM_A[m][k] = A[(block_m + m) * K + (block_k + k)];
                    }
                }
                load_B_k: for (uint32_t k = 0; k < BLOCK_SIZE; k++) {
                    load_B_n: for (uint32_t n = 0; n < BLOCK_SIZE; n++) {
                        SRAM_B[k][n] = B[(block_k + k) * N + (block_n + n)];
                    }
                }

                // INSERT SPARSE LOGIC AND COMPUTE HERE
                compute_block_m: for (uint32_t cblock_m = 0; cblock_m < BLOCK_SIZE; cblock_m += PE_SIZE) {
                    compute_block_n: for (uint32_t cblock_n = 0; cblock_n < BLOCK_SIZE; cblock_n += PE_SIZE) {
                        // Load REG_A and REG_B from SRAM
                        // Clear REG_C
                        clear_REG_C_m: for (uint32_t m = 0; m < PE_SIZE; m++) {
                            clear_REG_C_n: for (uint32_t n = 0; n < PE_SIZE; n++) {
                                REG_C[m][n] = 0;
                            }
                        }
                        load_REG_A_m: for (uint32_t m = 0; m < PE_SIZE; m++) {
                            load_REG_A_k: for (uint32_t k = 0; k < BLOCK_SIZE; k++) {
                                REG_A[m][k] = SRAM_A[cblock_m + m][k];
                            }
                        }
                        load_REG_B_k: for (uint32_t k = 0; k < BLOCK_SIZE; k++) {
                            load_REG_B_n: for (uint32_t n = 0; n < PE_SIZE; n++) {
                                REG_B[k][n] = SRAM_B[k][cblock_n + n];
                            }
                        }

                        // Compute the product of A and B blocks
                        compute_m: for (uint32_t m = 0; m < PE_SIZE; m++) {
                            compute_n: for (uint32_t n = 0; n < PE_SIZE; n++) {
                                compute_k: for (uint32_t k = 0; k < BLOCK_SIZE; k++) {
                                    REG_C[m][n] += REG_A[m][k] * REG_B[k][n];
                                }
                            }
                        }

                        // Store the result in SRAM_C
                        store_SRAM_C_m: for (uint32_t m = 0; m < PE_SIZE; m++) {
                            store_SRAM_C_n: for (uint32_t n = 0; n < PE_SIZE; n++) {
                                SRAM_C[cblock_m + m][cblock_n + n] += REG_C[m][n];
                            }
                        }
                    }
                }
                
            }

            // Store the result in the output matrix C
            store_C_m: for (uint32_t m = 0; m < BLOCK_SIZE; m++) {
                store_C_n: for (uint32_t n = 0; n < BLOCK_SIZE; n++) {
                    C[(block_m + m) * N + (block_n + n)] = SRAM_C[m][n];
                }
            }
        }
    }

    /*
    uint32_t A_buf[SIZE_K];
    uint32_t B_buf[SIZE_K][SIZE_N];
    uint32_t C_buf[SIZE_N];

    #pragma HLS array_partition variable=A_buf complete dim=0
    #pragma HLS array_partition variable=B_buf complete dim=0

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
    */

}
