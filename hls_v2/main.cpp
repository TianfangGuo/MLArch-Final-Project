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
        A[i] = i % 10;
    }
    for (int i = 0; i < MAT_DIM * MAT_DIM; i++) {
        B[i] = (i + 1) % 10;
    }

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
    for (int i = 0; i < MAT_DIM * MAT_DIM; i++) {
        if (C_hw[i] != C_sw[i]) {
            std::cout << "Mismatch at index " << i << ": HW = " << (uint32_t)C_hw[i] << ", SW = " << (uint32_t)C_sw[i] << std::endl;
            match = false;
        }
    }

    if (match) {
        std::cout << "All results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }
    return 0;
}
