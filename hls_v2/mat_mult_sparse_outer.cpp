#include "mat_mult.h"
#include <cstdint>

// Assumption: Dimensions are a multiple of the block sizes

void mat_mult(
	const uint32_t*  A_row_ind,
    const uint32_t*  A_col_ptr,
    const uint32_t*  A_val,
    const uint32_t   A_nnz,
    const uint32_t*  B_row_ptr,
    const uint32_t*  B_col_ind,
    const uint32_t*  B_val,
    const uint32_t   B_nnz,
    uint32_t*        C_row_ind,
    uint32_t*        C_col_ind,
    uint32_t*        C_val,
    uint32_t*        C_nnz
) {
	#pragma HLS INTERFACE m_axi port=A_row_ind offset=slave depth=1024 bundle=gmem0
	#pragma HLS INTERFACE m_axi port=A_col_ptr offset=slave depth=32 bundle=gmem0
	#pragma HLS INTERFACE m_axi port=A_val offset=slave depth=1024 bundle=gmem0

	#pragma HLS INTERFACE m_axi port=B_row_ptr offset=slave depth=32 bundle=gmem1
	#pragma HLS INTERFACE m_axi port=B_col_ind offset=slave depth=1024 bundle=gmem1
	#pragma HLS INTERFACE m_axi port=B_val offset=slave depth=1024 bundle=gmem1

	#pragma HLS INTERFACE m_axi port=C_row_ind offset=slave depth=1024 bundle=gmem0
	#pragma HLS INTERFACE m_axi port=C_col_ind offset=slave depth=1024 bundle=gmem0
	#pragma HLS INTERFACE m_axi port=C_val offset=slave depth=1024 bundle=gmem0

    #pragma HLS INTERFACE s_axilite port=A_nnz bundle=control
	#pragma HLS INTERFACE s_axilite port=B_nnz bundle=control
	#pragma HLS INTERFACE s_axilite port=C_nnz bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Create buffers for input and output matrices
    uint32_t A_row_ind_buf[MAT_DIM * MAT_DIM];
	uint32_t A_col_ptr_buf[MAT_DIM + 1];
	uint32_t A_val_buf[MAT_DIM * MAT_DIM];
	uint32_t A_nnz_buf;

	uint32_t B_row_ptr_buf[MAT_DIM + 1];
	uint32_t B_col_ind_buf[MAT_DIM * MAT_DIM];
	uint32_t B_val_buf[MAT_DIM * MAT_DIM];
	uint32_t B_nnz_buf;

	uint32_t C_row_ind_buf[MAT_DIM * MAT_DIM];
	uint32_t C_col_ind_buf[MAT_DIM * MAT_DIM];
	uint32_t C_val_buf[MAT_DIM * MAT_DIM];
	uint32_t C_nnz_buf;

	A_col_ptr_buf[0] = 0;
	B_row_ptr_buf[0] = 0;
	C_nnz_buf = 0;
	A_nnz_buf = A_nnz;
	B_nnz_buf = B_nnz;

	// Load the matrices
	load_ptr: for (uint8_t i = 0; i < MAT_DIM; i++) {
	#pragma HLS pipeline
		A_col_ptr_buf[i+1] = A_col_ptr[i];
		B_row_ptr_buf[i+1] = B_row_ptr[i];
	}

	load_A_row_ind: for (uint32_t i = 0; i < A_nnz_buf; i++) {
	#pragma HLS pipeline
		A_row_ind_buf[i] = A_row_ind[i];
	}
	load_A_val: for (uint32_t i = 0; i < A_nnz_buf; i++) {
	#pragma HLS pipeline
		A_val_buf[i] = A_val[i];
	}

	load_B_col_ind: for (uint32_t i = 0; i < B_nnz_buf; i++) {
	#pragma HLS pipeline
		B_col_ind_buf[i] = B_col_ind[i];
	}
	load_B_val: for (uint32_t i = 0; i < B_nnz_buf; i++) {
	#pragma HLS pipeline
		B_val_buf[i] = B_val[i];
	}

	uint32_t C_temp[MAT_DIM][MAT_DIM];
	clear_C_temp_i: for (uint32_t i = 0; i < MAT_DIM; i++) {
		clear_C_temp_j: for (uint32_t j = 0; j < MAT_DIM; j++) {
        #pragma HLS LOOP_FLATTEN
        #pragma HLS PIPELINE
			C_temp[i][j] = 0;
		}
	}

	// Perform the matrix multiplication (outer product)
	compute_k: for (uint32_t k = 0; k < MAT_DIM; k++) {
    #pragma HLS pipeline off
		uint32_t A_start = A_col_ptr_buf[k];
		uint32_t A_end = A_col_ptr_buf[k + 1];
		uint32_t B_start = B_row_ptr_buf[k];
		uint32_t B_end = B_row_ptr_buf[k + 1];

		compute_i: for (uint32_t i = A_start; i < A_end; i++) {
        #pragma HLS pipeline
			uint32_t m = A_row_ind_buf[i];
			uint32_t a_val = A_val_buf[i];

			compute_j: for (uint32_t j = B_start; j < B_end; j++) {
				uint32_t n = B_col_ind_buf[j];
				uint32_t b_val = B_val_buf[j];

				// Temporary storage for the result
				C_temp[m][n] += a_val * b_val;
			}
		}
	}

	// Temporary conversion to COO format
	coo_i: for (uint32_t i = 0; i < MAT_DIM; i++) {
		coo_j: for (uint32_t j = 0; j < MAT_DIM; j++) {
		#pragma HLS pipeline
			if (C_temp[i][j] != 0) {
				C_row_ind_buf[C_nnz_buf] = i;
				C_col_ind_buf[C_nnz_buf] = j;
				C_val_buf[C_nnz_buf] = C_temp[i][j];
				C_nnz_buf++;
			}
		}
	}

	// Store the matrix
	*C_nnz = C_nnz_buf;
	store_C_row_ind: for (uint32_t i = 0; i < C_nnz_buf; i++) {
	#pragma HLS pipeline
		C_row_ind[i] = C_row_ind_buf[i];
	}
	store_C_col_ind: for (uint32_t i = 0; i < C_nnz_buf; i++) {
	#pragma HLS pipeline
		C_col_ind[i] = C_col_ind_buf[i];
	}
	store_C_val: for (uint32_t i = 0; i < C_nnz_buf; i++) {
	#pragma HLS pipeline
		C_val[i] = C_val_buf[i];
	}

}
