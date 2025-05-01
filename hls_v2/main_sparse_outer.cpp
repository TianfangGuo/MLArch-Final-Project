#include "mat_mult_sparse_outer.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib> 
#include <ctime> 

int main(int argc, char **argv) {
    // Allocate memory for input and output data
    std::vector<uint32_t> A(MAT_DIM * MAT_DIM);
    std::vector<uint32_t> B(MAT_DIM * MAT_DIM);
    std::vector<uint32_t> C_hw(MAT_DIM * MAT_DIM);
    std::vector<uint32_t> C_sw(MAT_DIM * MAT_DIM);

    // Clear the data
    std::fill(A.begin(), A.end(), 0);
    std::fill(B.begin(), B.end(), 0);
    std::fill(C_hw.begin(), C_hw.end(), 0);
    std::fill(C_sw.begin(), C_sw.end(), 0);

    // Initialize input data
    //for (int i = 0; i < MAT_DIM * MAT_DIM; i++) {
        //A[i] = (i % 5) == 0;
        //A[i] = i % 10;
    	//A[i] = (i == 0);
    //}
    //for (int i = 0; i < MAT_DIM * MAT_DIM; i++) {
    	//B[i] = (i % 5) == 0;
        //B[i] = (i + 1) % 10;
    	//B[i] = (i == 0);
    //}

    float target_density = 0.1f; // 10% non-zeros
    float prob_nonzero = target_density;

    srand(time(0)); // seed random number generator

    for (int i = 0; i < MAT_DIM * MAT_DIM; i++) {
        A[i] = (rand() / (float)RAND_MAX) < prob_nonzero ? (i % 10 + 1) : 0;
        B[i] = (rand() / (float)RAND_MAX) < prob_nonzero ? (i % 10 + 1) : 0;
    }

    /*
    for (int i = 0; i < MAT_DIM; i++) {
    	for (int j = 0; j < MAT_DIM; j++) {
    		if (j < MAT_DIM / 2) {
    		    A[i * MAT_DIM + j] = 1;
    		    B[j * MAT_DIM + i] = 1;
    		} else {
    			A[i * MAT_DIM + j] = 0;
    			B[j * MAT_DIM + i] = 0;
    		}
    	}
    }
    */

    // Produce golden data for matrix multiplication
    for (int i = 0; i < MAT_DIM; i++) {
        for (int j = 0; j < MAT_DIM; j++) {
            for (int k = 0; k < MAT_DIM; k++) {
                C_sw[i * MAT_DIM + j] += A[i * MAT_DIM + k] * B[k * MAT_DIM + j];
            }
        }
    }

    // Convert the input matrices to CSR/CSC format
    std::vector<uint8_t>  A_row_ind(MAT_DIM * MAT_DIM);
    std::vector<uint8_t>  A_col_ptr(MAT_DIM);
    std::vector<uint32_t> A_val(MAT_DIM * MAT_DIM);
    uint16_t              A_nnz = 0;

    std::vector<uint8_t>  B_row_ptr(MAT_DIM);
    std::vector<uint8_t>  B_col_ind(MAT_DIM * MAT_DIM);
    std::vector<uint32_t> B_val(MAT_DIM * MAT_DIM);
    uint16_t              B_nnz = 0;

    for (int k = 0; k < MAT_DIM; k++) {
        for (int m = 0; m < MAT_DIM; m++) {
            if (A[m * MAT_DIM + k] != 0) {
                A_row_ind[A_nnz] = m;
                A_val[A_nnz] = A[m * MAT_DIM + k];
                A_nnz++;
            }
        }
        A_col_ptr[k] = A_nnz;
    }

    for (int k = 0; k < MAT_DIM; k++) {
        for (int n = 0; n < MAT_DIM; n++) {
            if (B[k * MAT_DIM + n] != 0) {
                B_col_ind[B_nnz] = n;
                B_val[B_nnz] = B[k * MAT_DIM + n];
                B_nnz++;
            }
        }
        B_row_ptr[k] = B_nnz;
    }

    // Result matrix in COO format
    std::vector<uint8_t>  C_row_ind(MAT_DIM * MAT_DIM);
    std::vector<uint8_t>  C_col_ind(MAT_DIM * MAT_DIM);
    std::vector<uint32_t> C_val(MAT_DIM * MAT_DIM);
    uint16_t              C_nnz = 0;

    // Obtain the hardware results (C_hw) from the accelerator
    //mat_mult(A.data(), B.data(), C_hw.data());
    mat_mult(
        A_row_ind.data(),
        A_col_ptr.data(),
        A_val.data(), 
        A_nnz, 
        B_row_ptr.data(), 
        B_col_ind.data(), 
        B_val.data(), 
        B_nnz, 
        C_row_ind.data(),
        C_col_ind.data(),
        C_val.data(),
        &C_nnz
    );

    // Convert the result matrix back to dense format
    for (int i = 0; i < C_nnz; i++) {
        C_hw[(uint16_t)C_row_ind[i] * MAT_DIM + (uint16_t)C_col_ind[i]] = C_val[i];
    }

    // Compare the results
    bool match = true;
    for (int i = 0; i < MAT_DIM; i++) {
    	for (int j = 0; j < MAT_DIM; j++) {
			if (C_hw[i * MAT_DIM + j] != C_sw[i * MAT_DIM + j]) {
				std::cout << "Mismatch at index (" << i << ", " << j << "): HW = " << (uint32_t)C_hw[i * MAT_DIM + j] << ", SW = " << (uint32_t)C_sw[i * MAT_DIM + j] << std::endl;
				match = false;
			}
    	}
    }

    if (match) {
        std::cout << "All results match!" << std::endl;
        return 0;
    } else {
        std::cout << "Results do not match!" << std::endl;
        return 1;
    }
    return 0;
}
