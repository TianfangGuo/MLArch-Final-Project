#include "mat_mult_sparse_inner.h"
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
    #pragma HLS INLINE
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

	uint32_t B_col_shift_0[32];
	uint32_t B_col_shift_1[32];
	uint32_t B_col_shift_2[32];
	uint32_t B_col_shift_3[32];
	uint32_t B_col_shift_4[32];


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
			A_row_shift_4[i] = (AB_prefix_sum[i+16][4]) ? A_row_shift_3[i+16] : A_row_shift_3[i];
			B_col_shift_4[i] = (AB_prefix_sum[i+16][4]) ? B_col_shift_3[i+16] : B_col_shift_3[i];
		} else {
			A_row_shift_4[i] = A_row_shift_3[i];
			B_col_shift_4[i] = B_col_shift_3[i];
		}
	}

	// Step 4: Mask the shifted values. Preserve only the values that matter after shifting
	ap_uint<32> final_mask;
	for (uint8_t i = 0; i < 32; i++) {
    #pragma HLS unroll
		final_mask.set_bit(i, len > i);
	}

	for (uint8_t i = 0; i < 32; i++) {
    #pragma HLS unroll
		if (final_mask[i]) {
			A_row_shift[i] = A_row_shift_4[i];
			B_col_shift[i] = B_col_shift_4[i];
		} else {
			A_row_shift[i] = 0;
			B_col_shift[i] = 0;
		}
	}
}



void mat_mult(
    const uint32_t *A,
    const uint32_t *B,
	uint32_t *C_row_ind,
	uint32_t *C_col_ind,
	uint32_t *C_val,
	uint32_t *C_len
) {
    #pragma HLS INTERFACE m_axi port=A offset=slave depth=1024 bundle=gmem0
    #pragma HLS INTERFACE m_axi port=B offset=slave depth=1024 bundle=gmem1
    #pragma HLS INTERFACE m_axi port=C_row_ind offset=slave depth=1024 bundle=gmem0
    #pragma HLS INTERFACE m_axi port=C_col_ind offset=slave depth=1024 bundle=gmem1
    #pragma HLS INTERFACE m_axi port=C_val offset=slave depth=1024 bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=C_len bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Create buffers for input and output matrices
    uint32_t A_buf[MAT_DIM][MAT_DIM];
    uint32_t B_buf[MAT_DIM][MAT_DIM];
    uint32_t C_row_ind_buf[MAT_DIM * MAT_DIM];
    uint32_t C_col_ind_buf[MAT_DIM * MAT_DIM];
    uint32_t C_val_buf[MAT_DIM * MAT_DIM];
    uint32_t C_len_buf = 0;

    #pragma HLS ARRAY_PARTITION variable=A_buf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=B_buf complete dim=1

    // Create bitvectors for A_buf and B_buf
    ap_uint<32> A_buf_bitvec[MAT_DIM];
    ap_uint<32> B_buf_bitvec[MAT_DIM];

    // Load the matrices
    load_x: for (uint8_t x = 0; x < MAT_DIM; x++) {
    	load_y: for (uint8_t y = 0; y < MAT_DIM; y++) {
        #pragma HLS LOOP_FLATTEN
        #pragma HLS PIPELINE
    		A_buf[x][y] = A[x * MAT_DIM + y];
    		B_buf[x][y] = B[x * MAT_DIM + y];
    		A_buf_bitvec[x].set_bit(y, A[x * MAT_DIM + y] != 0);
    		B_buf_bitvec[y].set_bit(x, B[x * MAT_DIM + y] != 0);
    	}
    }

    // With the bitvectors, track only the rows with nonzeros in A_buf and columns with nonzeros in B_buf
    uint8_t nonzero_rows_A[MAT_DIM];
    uint8_t nonzero_cols_B[MAT_DIM];
    uint8_t nonzero_len_A = 0;
    uint8_t nonzero_len_B = 0;

    get_nonzeros: for (uint8_t k = 0; k < MAT_DIM; k++) {
    	if (A_buf_bitvec[k]) {
    		nonzero_rows_A[nonzero_len_A++] = k;
    	}
    	if (B_buf_bitvec[k]) {
    		nonzero_cols_B[nonzero_len_B++] = k;
    	}
    }


    // Perform the matrix multiplication (inner product)
    compute_m: for (uint8_t i = 0; i < nonzero_len_A; i++) {
    	compute_n: for (uint8_t j = 0; j < nonzero_len_B; j++) {
        //#pragma HLS pipeline

    		uint8_t m = nonzero_rows_A[i];
    		uint8_t n = nonzero_cols_B[j];
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
    		C_row_ind_buf[C_len_buf] = m;
    		C_col_ind_buf[C_len_buf] = n;
    		C_val_buf[C_len_buf] = sum;
    		C_len_buf++;
    	}
    }

    // Store the matrix
    *C_len = C_len_buf;
    store_C_ind: for (uint32_t i = 0; i < C_len_buf; i++) {
    #pragma HLS pipeline
    	C_row_ind[i] = C_row_ind_buf[i];
    	C_col_ind[i] = C_col_ind_buf[i];
    }
    store_C_val: for (uint32_t i = 0; i < C_len_buf; i++) {
    #pragma HLS pipeline
    	C_val[i] = C_val_buf[i];
    }


}

