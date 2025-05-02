#include "mat_mult_sparse_outer.h"
#include <cstdint>

void partial_mat_mult(
	const uint32_t   k_block,
	const uint32_t*  A_row_ind,
	const uint32_t*  A_col_ptr,
	const uint32_t*  A_val,
	const uint32_t*  B_row_ptr,
	const uint32_t*  B_col_ind,
	const uint32_t*  B_val,
	uint32_t         C_temp[MAT_DIM][MAT_DIM]
) {
    #pragma HLS INLINE
    #pragma HLS allocation operation instances=mul limit=1
	// Load the matrices
	compute_k: for (uint32_t k = 0; k < 4; k++) {
		uint32_t A_start = (k_block == 0 && k == 0) ? 0 : A_col_ptr[k_block*4 + k - 1];
		uint32_t A_end = A_col_ptr[k_block*4 + k];
		uint32_t B_start = (k_block == 0 && k == 0) ? 0 : B_row_ptr[k_block*4 + k - 1];
		uint32_t B_end = B_row_ptr[k_block*4 + k];
		bool A_nonzero = A_end != A_start;
		bool B_nonzero = B_end != B_start;

		if (A_nonzero && B_nonzero) {
			compute_i: for (uint32_t i = A_start; i < A_end; i++) {
				uint32_t m = A_row_ind[i];
				uint32_t a_val = A_val[i];
				compute_j: for (uint32_t j = B_start; j < B_end; j++) {
					uint32_t n = B_col_ind[j];
					uint32_t b_val = B_val[j];
					C_temp[m][n] += a_val * b_val;
				}
			}
		}
	}
}

void mat_mult(
	const uint32_t*  A_row_ind,
    const uint32_t*  A_col_ptr,
    const uint32_t*  A_val,
    const uint32_t   A_nnz,
    const uint32_t*  B_row_ptr,
    const uint32_t*  B_col_ind,
    const uint32_t*  B_val,
    const uint32_t   B_nnz,
	uint32_t*        C
) {
	#pragma HLS INTERFACE m_axi port=A_row_ind offset=slave depth=1024 bundle=gmem0
	#pragma HLS INTERFACE m_axi port=A_col_ptr offset=slave depth=32 bundle=gmem0
	#pragma HLS INTERFACE m_axi port=A_val offset=slave depth=1024 bundle=gmem0

	#pragma HLS INTERFACE m_axi port=B_row_ptr offset=slave depth=32 bundle=gmem1
	#pragma HLS INTERFACE m_axi port=B_col_ind offset=slave depth=1024 bundle=gmem1
	#pragma HLS INTERFACE m_axi port=B_val offset=slave depth=1024 bundle=gmem1

    #pragma HLS INTERFACE m_axi port=C offset=slave depth=1024 bundle=gmem0

    #pragma HLS INTERFACE s_axilite port=A_nnz bundle=control
	#pragma HLS INTERFACE s_axilite port=B_nnz bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Create buffers for input and output matrices
    uint32_t A_row_ind_buf[MAT_DIM * MAT_DIM];
	uint32_t A_col_ptr_buf[MAT_DIM];
	uint32_t A_val_buf[MAT_DIM * MAT_DIM];
	#pragma HLS ARRAY_PARTITION variable=A_col_ptr_buf type=block factor=8
	#pragma HLS ARRAY_PARTITION variable=A_row_ind_buf type=cyclic factor=32
	#pragma HLS ARRAY_PARTITION variable=A_val_buf type=cyclic factor=32

	uint32_t B_row_ptr_buf[MAT_DIM];
	uint32_t B_col_ind_buf[MAT_DIM * MAT_DIM];
	uint32_t B_val_buf[MAT_DIM * MAT_DIM];
	#pragma HLS ARRAY_PARTITION variable=B_row_ptr_buf type=block factor=8
	#pragma HLS ARRAY_PARTITION variable=B_col_ind_buf type=cyclic factor=32
	#pragma HLS ARRAY_PARTITION variable=B_val_buf type=cyclic factor=32

	uint32_t C_temp[8][MAT_DIM][MAT_DIM];
	#pragma HLS ARRAY_PARTITION variable=C_temp dim=1 type=complete


	// Load the matrices
	load_ptr: for (uint8_t i = 0; i < MAT_DIM; i++) {
	#pragma HLS pipeline
		A_col_ptr_buf[i] = A_col_ptr[i];
		B_row_ptr_buf[i] = B_row_ptr[i];
	}

	load_A_row_ind: for (uint32_t i = 0; i < A_nnz; i++) {
	#pragma HLS pipeline
		A_row_ind_buf[i] = A_row_ind[i];
	}
	load_A_val: for (uint32_t i = 0; i < A_nnz; i++) {
	#pragma HLS pipeline
		A_val_buf[i] = A_val[i];
	}

	load_B_col_ind: for (uint32_t i = 0; i < B_nnz; i++) {
	#pragma HLS pipeline
		B_col_ind_buf[i] = B_col_ind[i];
	}
	load_B_val: for (uint32_t i = 0; i < B_nnz; i++) {
	#pragma HLS pipeline
		B_val_buf[i] = B_val[i];
	}


	clear_C_temp_i: for (uint32_t i = 0; i < MAT_DIM; i++) {
		clear_C_temp_j: for (uint32_t j = 0; j < MAT_DIM; j++) {
		#pragma HLS pipeline
			clear_C_temp_k: for (uint32_t k = 0; k < 8; k++) {
        	#pragma HLS UNROLL
				C_temp[k][i][j] = 0;
			}
		}
	}

	// Perform the matrix multiplication (outer product)
	compute_k_block: for (uint32_t k_block = 0; k_block < 8; k_block++) {
	#pragma HLS unroll
    //#pragma HLS pipeline
    //#pragma HLS dataflow
		partial_mat_mult(k_block, A_row_ind_buf, A_col_ptr_buf, A_val_buf, B_row_ptr_buf, B_col_ind_buf, B_val_buf, C_temp[k_block]);
	}


	// Store the matrix
	store_C_m: for (uint32_t m = 0; m < MAT_DIM; m++) {
		store_C_n: for (uint32_t n = 0; n < MAT_DIM; n++) {
        #pragma HLS pipeline
			uint32_t C_temp_sum = 0;
			store_C_k: for (uint32_t k = 0; k < 8; k++) {
			#pragma HLS UNROLL
				C_temp_sum += C_temp[k][m][n];
			}
		    C[m * MAT_DIM + n] = C_temp_sum;
		}
	}

}
