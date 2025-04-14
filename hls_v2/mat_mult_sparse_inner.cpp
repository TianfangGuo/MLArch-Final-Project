#include "mat_mult.h"
#include <cstdint>
#include "ap_int.h"
#include "stdio.h"


void intersection_unit(
    const uint32_t *A_row,
	const uint32_t *B_col,
	const ap_uint<32> A_row_bitvec,
	const ap_uint<32> B_col_bitvec,
	uint32_t *A_row_shift,
	uint32_t *B_col_shift,
	uint8_t *result_len
) {
    //#pragma HLS DATAFLOW
	// Step 1: Find all the pairwise interactions
	ap_uint<32> AB_bitvec = A_row_bitvec & B_col_bitvec;
	ap_uint<32> AB_bitvec_inv = ~(A_row_bitvec & B_col_bitvec);

	// Set the total length
	uint8_t len = 0;
	for (uint8_t i = 0; i < 32; i++) {
    #pragma HLS unroll
		len += AB_bitvec[i];
	}
	*result_len = len;

	// Step 2: Produce a prefix sum from the computed bitvector
	ap_uint<2> AB_prefix_sum_1[32];
	ap_uint<3> AB_prefix_sum_2[32];
	ap_uint<4> AB_prefix_sum_3[32];
	ap_uint<5> AB_prefix_sum_4[32];
	ap_uint<6> AB_prefix_sum[32];

    #pragma HLS ARRAY_PARTITION variable=AB_prefix_sum_1 type=complete
    #pragma HLS ARRAY_PARTITION variable=AB_prefix_sum_2 type=complete
    #pragma HLS ARRAY_PARTITION variable=AB_prefix_sum_3 type=complete
    #pragma HLS ARRAY_PARTITION variable=AB_prefix_sum_4 type=complete
    #pragma HLS ARRAY_PARTITION variable=AB_prefix_sum type=complete

	prefix_sum_1: for (uint8_t i = 0; i < 32; i++) {
    #pragma HLS unroll
	    if (i == 0) {
	    	AB_prefix_sum_1[i] = 0;
	    } else if (i == 1) {
	    	AB_prefix_sum_1[i] = (ap_uint<2>)AB_bitvec_inv[i-1];
	    } else {
	    	AB_prefix_sum_1[i] = (ap_uint<2>)AB_bitvec_inv[i-1] + (ap_uint<2>)AB_bitvec_inv[i-2];
	    }
	}

	for (uint8_t i = 0; i < 32; i++) {
    #pragma HLS unroll
		if (i >= 2) {
			AB_prefix_sum_2[i] = (ap_uint<3>)AB_prefix_sum_1[i] + (ap_uint<3>)AB_prefix_sum_1[i-2];
		} else {
			AB_prefix_sum_2[i] = (ap_uint<3>)AB_prefix_sum_1[i];
		}
	}

	for (uint8_t i = 0; i < 32; i++) {
    #pragma HLS unroll
		if (i >= 4) {
			AB_prefix_sum_3[i] = (ap_uint<4>)AB_prefix_sum_2[i] + (ap_uint<4>)AB_prefix_sum_2[i-4];
		} else {
			AB_prefix_sum_3[i] = (ap_uint<4>)AB_prefix_sum_2[i];
		}
	}

	for (uint8_t i = 0; i < 32; i++) {
    #pragma HLS unroll
		if (i >= 8) {
			AB_prefix_sum_4[i] = (ap_uint<5>)AB_prefix_sum_3[i] + (ap_uint<5>)AB_prefix_sum_3[i-8];
		} else {
			AB_prefix_sum_4[i] = (ap_uint<5>)AB_prefix_sum_3[i];
		}
	}

	for (uint8_t i = 0; i < 32; i++) {
	#pragma HLS unroll
		if (i >= 16) {
			AB_prefix_sum[i] = (ap_uint<6>)AB_prefix_sum_4[i] + (ap_uint<6>)AB_prefix_sum_4[i-16];
		} else {
			AB_prefix_sum[i] = (ap_uint<6>)AB_prefix_sum_4[i];
		}
	}

	// Step 3: Shift A_row and B_col so that the matching nonzero indexes are aligned
	uint32_t A_row_shift_0[32];
	uint32_t A_row_shift_1[32];
	uint32_t A_row_shift_2[32];
	uint32_t A_row_shift_3[32];
	uint32_t A_row_shift_4[32];
	//uint32_t A_row_shift[32];

	uint32_t B_col_shift_0[32];
	uint32_t B_col_shift_1[32];
	uint32_t B_col_shift_2[32];
	uint32_t B_col_shift_3[32];
	uint32_t B_col_shift_4[32];
	//uint32_t B_col_shift[32];

	#pragma HLS ARRAY_PARTITION variable=A_row_shift_0 type=complete
	#pragma HLS ARRAY_PARTITION variable=A_row_shift_1 type=complete
	#pragma HLS ARRAY_PARTITION variable=A_row_shift_2 type=complete
	#pragma HLS ARRAY_PARTITION variable=A_row_shift_3 type=complete
	#pragma HLS ARRAY_PARTITION variable=A_row_shift_4 type=complete
	#pragma HLS ARRAY_PARTITION variable=B_col_shift_0 type=complete
	#pragma HLS ARRAY_PARTITION variable=B_col_shift_1 type=complete
	#pragma HLS ARRAY_PARTITION variable=B_col_shift_2 type=complete
	#pragma HLS ARRAY_PARTITION variable=B_col_shift_3 type=complete
	#pragma HLS ARRAY_PARTITION variable=B_col_shift_4 type=complete

	for (uint8_t i = 0; i < 32; i++) {
	#pragma HLS unroll
		if (i < 31) {
			A_row_shift_0[i] = (AB_prefix_sum[i+1][0]) ? A_row[i+1] : A_row[i];
			B_col_shift_0[i] = (AB_prefix_sum[i+1][0]) ? B_col[i+1] : B_col[i];
		} else {
			A_row_shift_0[i] = A_row[i];
			B_col_shift_0[i] = B_col[i];
		}
	}
	for (uint8_t i = 0; i < 32; i++) {
	#pragma HLS unroll
		if (i < 30) {
			A_row_shift_1[i] = (AB_prefix_sum[i+2][1]) ? A_row_shift_0[i+2] : A_row_shift_0[i];
			B_col_shift_1[i] = (AB_prefix_sum[i+2][1]) ? B_col_shift_0[i+2] : B_col_shift_0[i];
		} else {
			A_row_shift_1[i] = A_row_shift_0[i];
			B_col_shift_1[i] = B_col_shift_0[i];
		}
	}
	for (uint8_t i = 0; i < 32; i++) {
	#pragma HLS unroll
		if (i < 28) {
			A_row_shift_2[i] = (AB_prefix_sum[i+4][2]) ? A_row_shift_1[i+4] : A_row_shift_1[i];
			B_col_shift_2[i] = (AB_prefix_sum[i+4][2]) ? B_col_shift_1[i+4] : B_col_shift_1[i];
		} else {
			A_row_shift_2[i] = A_row_shift_1[i];
			B_col_shift_2[i] = B_col_shift_1[i];
		}
	}
	for (uint8_t i = 0; i < 32; i++) {
	#pragma HLS unroll
		if (i < 24) {
			A_row_shift_3[i] = (AB_prefix_sum[i+8][3]) ? A_row_shift_2[i+8] : A_row_shift_2[i];
			B_col_shift_3[i] = (AB_prefix_sum[i+8][3]) ? B_col_shift_2[i+8] : B_col_shift_2[i];
		} else {
			A_row_shift_3[i] = A_row_shift_2[i];
			B_col_shift_3[i] = B_col_shift_2[i];
		}
	}
	for (uint8_t i = 0; i < 32; i++) {
	#pragma HLS unroll
		if (i < 16) {
			A_row_shift[i] = (AB_prefix_sum[i+16][4]) ? A_row_shift_3[i+16] : A_row_shift_3[i];
			B_col_shift[i] = (AB_prefix_sum[i+16][4]) ? B_col_shift_3[i+16] : B_col_shift_3[i];
		} else {
			A_row_shift[i] = A_row_shift_3[i];
			B_col_shift[i] = B_col_shift_3[i];
		}
	}

}



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

    #pragma HLS ARRAY_PARTITION variable=A_buf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=B_buf complete dim=1

    // Create bitvectors for A_buf and B_buf
    ap_uint<32> A_buf_bitvec[MAT_DIM];
    ap_uint<32> B_buf_bitvec[MAT_DIM];

    // Load the matrices
    load_x: for (uint8_t x = 0; x < MAT_DIM; x++) {
    	load_y: for (uint8_t y = 0; y < MAT_DIM; y++) {
    		A_buf[x][y] = A[x * MAT_DIM + y];
    		B_buf[x][y] = B[x * MAT_DIM + y];
    		C_buf[x][y] = 0;
    		A_buf_bitvec[x].set_bit(y, A[x * MAT_DIM + y] != 0);
    		B_buf_bitvec[y].set_bit(x, B[x * MAT_DIM + y] != 0);
    	}
    }

    /*
    printf("MATRIX A:\n");
    for (uint8_t x = 0; x < MAT_DIM; x++) {
    	for (uint8_t y = 0; y < MAT_DIM; y++) {
    		printf("%d ", A_buf[x][y]);
    	}
    	printf("\n");
    }

    printf("BIT VECTOR A:\n");
    for (uint8_t x = 0; x < MAT_DIM; x++) {
       	for (uint8_t y = 0; y < MAT_DIM; y++) {
       		printf("%d ", A_buf_bitvec[x].range(y, y).to_uint());
      	}
     	printf("\n");
    }
    */

    /*
    uint32_t A_row[MAT_DIM];
    uint32_t B_col[MAT_DIM];
    uint32_t A_res[MAT_DIM];
    uint32_t B_res[MAT_DIM];
    uint8_t len_res[1];
    for (int i = 0; i < MAT_DIM; i++) {
    	A_row[i] = A_buf[0][i];
    	B_col[i] = B_buf[i][0];
    }

    intersection_unit(A_row, B_col, A_buf_bitvec[0], B_buf_bitvec[0], A_res, B_res, len_res);

    printf("Original A and B:\n");
    for (int i = 0; i < MAT_DIM; i++) {
    	printf("%d ", A_row[i]);
    }
    printf("\n");
    for (int i = 0; i < MAT_DIM; i++) {
        printf("%d ", B_col[i]);
    }
    printf("\n");

    printf("New A and B:\n");
    for (int i = 0; i < MAT_DIM; i++) {
    	printf("%d ", A_res[i]);
    }
    printf("\n");
    for (int i = 0; i < MAT_DIM; i++) {
        printf("%d ", B_res[i]);
    }
    printf("\n");
    printf("Length: %d\n", len_res[0]);
	*/

    // Perform the matrix multiplication (inner product)
    compute_m: for (uint8_t m = 0; m < MAT_DIM; m++) {
    	compute_n: for (uint8_t n = 0; n < MAT_DIM; n++) {

    		/*
    		uint32_t sum = 0;
    		compute_k: for (uint8_t k = 0; k < MAT_DIM; k += PE) {
    		#pragma HLS pipeline
    		    compute_pe: for (uint8_t pe = 0; pe < PE; pe++) {
    		    #pragma HLS unroll
    		        sum += A_buf[m][k + pe] * B_buf[k + pe][n];
    		    }
    		}
    		C_buf[m][n] = sum;
			*/

    		uint32_t A_row[MAT_DIM];
    		uint32_t B_col[MAT_DIM];
    		uint32_t A_row_shifted[MAT_DIM];
    		uint32_t B_col_shifted[MAT_DIM];
    		uint8_t intersect_len[1];
    		uint32_t sum = 0;

            #pragma HLS array_partition variable=A_row complete
            #pragma HLS array_partition variable=B_col complete
            #pragma HLS array_partition variable=A_row_shifted complete
            #pragma HLS array_partition variable=B_col_shifted complete

    		// Load a row of A and column of B
    		compute_k_load: for (uint8_t k = 0; k < MAT_DIM; k++) {
            #pragma HLS UNROLL
    			A_row[k] = A_buf[m][k];
    			B_col[k] = B_buf[k][n];
    		}

    		// Intersect the vectors, find as many nonzero intersections and align them
    		intersection_unit(A_row, B_col, A_buf_bitvec[m], B_buf_bitvec[n], A_row_shifted, B_col_shifted, intersect_len);

    		// Perform the dot product
    		compute_k: for (uint8_t k = 0; k < intersect_len[0]; k += PE) {
    			compute_pe: for (uint8_t pe = 0; pe < PE; pe++) {
                #pragma HLS unroll
    				sum += A_row_shifted[k + pe] * B_col_shifted[k + pe];
    			}
    		}

    		// Store the sum
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

