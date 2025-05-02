#include "mat_mult.h"
#include <cstdint>

// Assumption: Dimensions are a multiple of the block sizes

void mat_mult(
    const uint32_t *A,
    const uint32_t *B,
    uint32_t *C
) {
    #pragma HLS INTERFACE m_axi port=A offset=slave depth=1024 bundle=gmem0
    #pragma HLS INTERFACE m_axi port=B offset=slave depth=1024 bundle=gmem1
    #pragma HLS INTERFACE m_axi port=C offset=slave depth=1024 bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=return bundle=control


	/*
	// Inner version
    // Create buffers for input and output matrices
    uint32_t A_buf[MAT_DIM][MAT_DIM];
    uint32_t B_buf[MAT_DIM][MAT_DIM];
    uint32_t C_buf[MAT_DIM][MAT_DIM];

    // Array partitioning for parallel optimization
    #pragma HLS array_partition variable=A_buf cyclic factor=8 dim=2
    #pragma HLS array_partition variable=B_buf cyclic factor=8 dim=1

    // Load the matrices
    load_x: for (uint8_t x = 0; x < MAT_DIM; x++) {
    	load_y: for (uint8_t y = 0; y < MAT_DIM; y++) {
    		A_buf[x][y] = A[x * MAT_DIM + y];
    		B_buf[x][y] = B[x * MAT_DIM + y];
    		C_buf[x][y] = 0;
    	}
    }

    // Perform the matrix multiplication (inner product)
    compute_m: for (uint8_t m = 0; m < MAT_DIM; m++) {
    	compute_n: for (uint8_t n = 0; n < MAT_DIM; n++) {
        #pragma HLS allocation operation instances=mul limit=8
        #pragma HLS pipeline
    		uint32_t sum = 0;

    		compute_k: for (uint8_t k = 0; k < MAT_DIM; k++) {
            #pragma HLS unroll factor=8 skip_exit_check
    			sum += A_buf[m][k] * B_buf[k][n];
    		}

    		C_buf[m][n] = sum;
    	}
    }


    // Store the matrix
    store_x: for (uint8_t x = 0; x < MAT_DIM; x++) {
    	store_y: for (uint8_t y = 0; y < MAT_DIM; y++) {
    		C[x * MAT_DIM + y] = C_buf[x][y];
    	}
    }
    */


    // Gustavson version
	// Create buffers for input and output matrices
	uint32_t A_buf[MAT_DIM][MAT_DIM];
	uint32_t B_buf[MAT_DIM][MAT_DIM];
	uint32_t C_buf[MAT_DIM][MAT_DIM];

	// Array partitioning for parallel optimization
	#pragma HLS array_partition variable=C_buf cyclic factor=8 dim=2
	#pragma HLS array_partition variable=B_buf cyclic factor=8 dim=2

	// Load the matrices
	load_x: for (uint8_t x = 0; x < MAT_DIM; x++) {
	 	load_y: for (uint8_t y = 0; y < MAT_DIM; y++) {
	   		A_buf[x][y] = A[x * MAT_DIM + y];
	   		B_buf[x][y] = B[x * MAT_DIM + y];
	   		C_buf[x][y] = 0;
	   	}
	}


	compute_m: for (uint8_t m = 0; m < MAT_DIM; m++) {
		compute_k: for (uint8_t k = 0; k < MAT_DIM; k++) {
        #pragma HLS allocation operation instances=mul limit=8
	    #pragma HLS pipeline
			uint32_t A_val = A_buf[m][k];
			compute_n: for (uint8_t n = 0; n < MAT_DIM; n++) {
            #pragma HLS unroll factor=8 skip_exit_check
				C_buf[m][n] += A_val * B_buf[k][n];
			}
		}
	}

	// Store the matrix
	store_x: for (uint8_t x = 0; x < MAT_DIM; x++) {
	    store_y: for (uint8_t y = 0; y < MAT_DIM; y++) {
	    	C[x * MAT_DIM + y] = C_buf[x][y];
	    }
	}

    // Outer version
	/*
	// Create buffers for input and output matrices
	uint32_t A_buf[MAT_DIM][MAT_DIM];
	uint32_t B_buf[MAT_DIM][MAT_DIM];
	uint32_t C_buf[8][MAT_DIM][MAT_DIM];

	// Array partitioning for parallel optimization
	#pragma HLS array_partition variable=A_buf block factor=8 dim=2
	#pragma HLS array_partition variable=B_buf block factor=8 dim=1
    #pragma HLS array_partition variable=C_buf complete dim=1

	// Load the matrices
	load_x: for (uint32_t x = 0; x < MAT_DIM; x++) {
	    load_y: for (uint32_t y = 0; y < MAT_DIM; y++) {
	    	A_buf[x][y] = A[x * MAT_DIM + y];
	        B_buf[x][y] = B[x * MAT_DIM + y];
	        clear_c: for (uint32_t k = 0; k < 8; k++) {
            #pragma HLS UNROLL
	        	C_buf[k][x][y] = 0;
	        }
	    }
	}

	// Perform the matrix multiplication (outer product)
	compute_k: for (uint32_t k = 0; k < 4; k++) {
		compute_m: for (uint32_t m = 0; m < MAT_DIM; m++) {
			compute_n: for (uint32_t n = 0; n < MAT_DIM; n++) {
            #pragma HLS pipeline
				compute_k_block: for (uint32_t k_block = 0; k_block < 8; k_block++) {
                #pragma HLS UNROLL
					C_buf[k_block][m][n] += A_buf[m][k_block*4 + k] * B_buf[k_block*4 + k][n];
				}
			}
		}
	}

	// Store the matrix
	store_m: for (uint32_t m = 0; m < MAT_DIM; m++) {
		store_n: for (uint32_t n = 0; n < MAT_DIM; n++) {
			uint32_t sum = 0;
			store_k: for (uint32_t k = 0; k < 8; k++) {
				sum += C_buf[k][m][n];
			}
			C[m * MAT_DIM + n] = sum;
		}
	}
	*/
}
