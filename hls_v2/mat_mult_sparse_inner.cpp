#include "mat_mult.h"
#include <cstdint>
#include "stdio.h"

void mat_mult(
    const uint32_t *A,
    const uint32_t *B,
    uint32_t *C
) {
    #pragma HLS INTERFACE m_axi port=A offset=slave depth=1024 bundle=gmem0
    #pragma HLS INTERFACE m_axi port=B offset=slave depth=1024 bundle=gmem1
    #pragma HLS INTERFACE m_axi port=C offset=slave depth=1024 bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Create buffers for input and output matrices
    uint32_t A_buf[MAT_DIM][MAT_DIM];
    uint32_t B_buf[MAT_DIM][MAT_DIM];
    uint32_t C_buf[MAT_DIM][MAT_DIM];

    //#pragma HLS array_partition variable=A_buf cyclic factor=8 dim=2
    //#pragma HLS array_partition variable=B_buf cyclic factor=8 dim=1

    // Create a CSR data structure for A_buf
    uint16_t A_csr_row_ptr[MAT_DIM + 1];
    uint8_t A_csr_col_indices[MAT_DIM * MAT_DIM];
    uint32_t A_csr_values[MAT_DIM * MAT_DIM];

    //#pragma HLS array_partition variable=A_csr_row_ptr cyclic factor=2 dim=1

    // Load the matrices
    load_x: for (uint8_t x = 0; x < MAT_DIM; x++) {
    	load_y: for (uint8_t y = 0; y < MAT_DIM; y++) {
    		A_buf[x][y] = A[x * MAT_DIM + y];
    		B_buf[x][y] = B[x * MAT_DIM + y];
    		C_buf[x][y] = 0;
    	}
    }

    // Convert A_buf to CSR representation
    uint16_t i = 0;
    A_csr_x: for (uint8_t x = 0; x < MAT_DIM; x++) {
    	A_csr_row_ptr[x] = i;
    	A_csr_y: for (uint8_t y = 0; y < MAT_DIM; y++) {
    		if (A_buf[x][y] != 0) {
    			A_csr_values[i] = A_buf[x][y];
    			A_csr_col_indices[i] = y;
    			i += 1;
    		}
    	}
    }
    A_csr_row_ptr[MAT_DIM] = i;

    // Perform the matrix multiplication (inner product)
    compute_m: for (uint8_t m = 0; m < MAT_DIM; m++) {
    	compute_n: for (uint8_t n = 0; n < MAT_DIM; n++) {
    		uint32_t sum = 0;
    		compute_k: for (uint16_t idx = A_csr_row_ptr[m]; idx < A_csr_row_ptr[m+1]; idx++) {
    			sum += A_csr_values[idx] * B_buf[A_csr_col_indices[idx]][n];
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


}

