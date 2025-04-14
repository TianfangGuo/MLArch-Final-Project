#include "mat_mult.h"
#include <iostream>
#include <vector>
#include <cstdint>

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
    for (int i = 0; i < MAT_DIM * MAT_DIM; i++) {
        A[i] = (i % 20) == 0;
        //A[i] = i % 10;
    	//A[i] = (i == 0);
    	//A[i] = 1;
    }
    for (int i = 0; i < MAT_DIM * MAT_DIM; i++) {
    	B[i] = (i % 20) == 0;
        //B[i] = (i + 1) % 10;
    	//B[i] = (i == 0);
    	//B[i] = 1;
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

    // Obtain the hardware results (C_hw) from the accelerator
    mat_mult(A.data(), B.data(), C_hw.data());

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
