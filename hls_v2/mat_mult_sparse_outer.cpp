#include "mat_mult_sparse_outer.h"
#include <cstdint>

void mat_mult(
	const uint32_t*  A_m_ind,
    const uint32_t*  A_val,
    const uint32_t   A_m_dim,
    const uint32_t*  B_n_ind,
    const uint32_t*  B_val,
    const uint32_t   B_n_dim,
	uint32_t*        C
) {
	#pragma HLS INTERFACE m_axi port=A_m_ind offset=slave depth=1024 bundle=gmem0
	#pragma HLS INTERFACE m_axi port=A_val offset=slave depth=1024 bundle=gmem0

	#pragma HLS INTERFACE m_axi port=B_n_ind offset=slave depth=1024 bundle=gmem1
	#pragma HLS INTERFACE m_axi port=B_val offset=slave depth=1024 bundle=gmem1

    #pragma HLS INTERFACE m_axi port=C offset=slave depth=1024 bundle=gmem0

    #pragma HLS INTERFACE s_axilite port=A_m_dim bundle=control
	#pragma HLS INTERFACE s_axilite port=B_n_dim bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

	// Create buffers for input and output matrices
	uint32_t A_m_ind_buf[MAT_DIM][MAT_DIM];
	uint32_t A_val_buf[MAT_DIM][MAT_DIM];

	uint32_t B_n_ind_buf[MAT_DIM][MAT_DIM];
	uint32_t B_val_buf[MAT_DIM][MAT_DIM];

	uint32_t C_buf[8][MAT_DIM][MAT_DIM];

	#pragma HLS array_partition variable=C_buf complete dim=1
	#pragma HLS array_partition variable=A_m_ind_buf block factor=8 dim=2
	#pragma HLS array_partition variable=A_val_buf block factor=8 dim=2
	#pragma HLS array_partition variable=B_n_ind_buf block factor=8 dim=1
	#pragma HLS array_partition variable=B_val_buf block factor=8 dim=1

	// Initialize the matrices
	load_A_m: for (uint32_t m = 0; m < A_m_dim; m++) {
		load_A_k: for (uint32_t k = 0; k < MAT_DIM; k++) {
			#pragma HLS pipeline
			A_m_ind_buf[m][k] = A_m_ind[m * MAT_DIM + k];
			A_val_buf[m][k] = A_val[m * MAT_DIM + k];
		}
	}
	load_B_n: for (uint32_t n = 0; n < B_n_dim; n++) {
		load_B_k: for (uint32_t k = 0; k < MAT_DIM; k++) {
			#pragma HLS pipeline
			B_n_ind_buf[k][n] = B_n_ind[n * MAT_DIM + k];
			B_val_buf[k][n] = B_val[n * MAT_DIM + k];
		}
	}
	clear_C_m: for (uint32_t m = 0; m < MAT_DIM; m++) {
		clear_C_n: for (uint32_t n = 0; n < MAT_DIM; n++) {
			#pragma HLS pipeline
			clear_C_k_block: for (uint32_t k = 0; k < 8; k++) {
				#pragma HLS unroll
				C_buf[k][m][n] = 0;
			}
		}
	}

	// Perform the matrix multiplication (outer product)
	compute_k: for (uint32_t k = 0; k < 4; k++) {
		compute_i: for (uint32_t i = 0; i < A_m_dim; i++) {
			compute_j: for (uint32_t j = 0; j < B_n_dim; j++) {
				compute_k_block: for (uint32_t k_block = 0; k_block < 8; k_block++) {
				#pragma HLS unroll
					uint32_t true_k = k_block * 4 + k;
					uint32_t m = A_m_ind_buf[i][true_k];
					uint32_t n = B_n_ind_buf[true_k][j];
					uint32_t a_val = A_val_buf[i][true_k];
					uint32_t b_val = B_val_buf[true_k][j];
					C_buf[k_block][m][n] += a_val * b_val;
				}
			}
		}
	}

	// Store the matrix
	store_C_m: for (uint32_t m = 0; m < MAT_DIM; m++) {
		store_C_n: for (uint32_t n = 0; n < MAT_DIM; n++) {
        #pragma HLS pipeline
			uint32_t C_sum = 0;
			store_C_acc: for (uint32_t k = 0; k < 8; k++) {
			#pragma HLS unroll
				C_sum += C_buf[k][m][n];
			}
		    C[m * MAT_DIM + n] = C_sum;
		}
	}

}
