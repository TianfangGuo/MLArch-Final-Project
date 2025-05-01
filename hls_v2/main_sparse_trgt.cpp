#include "mat_mult_trgt.h"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <ctime>

int main(int argc, char **argv) {
    std::vector<uint32_t> A(MAT_A_SIZE);
    std::vector<uint32_t> B(MAT_B_SIZE);
    std::vector<uint32_t> C_hw(MAT_C_SIZE);
    std::vector<uint32_t> C_sw(MAT_C_SIZE);

    std::fill(A.begin(), A.end(), 0);
    std::fill(B.begin(), B.end(), 0);
    std::fill(C_hw.begin(), C_hw.end(), 0);
    std::fill(C_sw.begin(), C_sw.end(), 0);

    // Random sparse initialization
    float A_density = 0.1f;
    float B_density = 0.1f;

    std::cout << "Initializing A with density " << A_density * 100 << "%\n";
    std::cout << "Initializing B with density " << B_density * 100 << "%\n";

    srand(time(0));
    for (int i = 0; i < MAT_A_SIZE; i++) {
        A[i] = (rand() / (float)RAND_MAX) < A_density ? (i % 10 + 1) : 0;
    }
    for (int i = 0; i < MAT_B_SIZE; i++) {
        B[i] = (rand() / (float)RAND_MAX) < B_density ? (i % 10 + 1) : 0;
    }

    // Actual sparsity
    int A_nnz = 0;
    int B_nnz = 0;
    for (int i = 0; i < MAT_A_SIZE; i++) {
        if (A[i] != 0) A_nnz++;
    }
    for (int i = 0; i < MAT_B_SIZE; i++) {
        if (B[i] != 0) B_nnz++;
    }

    float A_actual_density = A_nnz / (float)MAT_A_SIZE;
    float B_actual_density = B_nnz / (float)MAT_B_SIZE;

    std::cout << "A actual density: " << A_actual_density * 100 << "% ("
            << (100.0f - A_actual_density * 100) << "% sparse)" << std::endl;
    std::cout << "B actual density: " << B_actual_density * 100 << "% ("
            << (100.0f - B_actual_density * 100) << "% sparse)" << std::endl;


    // Golden reference: same loop structure as TrGT
    for (int n1 = 0; n1 < N1; n1++) {
        for (int m2 = 0; m2 < M2; m2++) {
            for (int m1 = 0; m1 < M1; m1++) {
                for (int m0 = 0; m0 < M0; m0++) {
                    for (int n0 = 0; n0 < N0; n0++) {
                        uint32_t sum = 0;
                        for (int k = 0; k < K; k++) {
                            uint32_t a_val = A[m2 * M1 * M0 * K + m1 * M0 * K + m0 * K + k];
                            uint32_t b_val = B[n1 * K * N0 + k * N0 + n0];
                            sum += a_val * b_val;
                        }
                        int c_idx = n1 * M2 * M1 * M0 * N0 +
                                    m2 * M1 * M0 * N0 +
                                    m1 * M0 * N0 +
                                    m0 * N0 + n0;
                        C_sw[c_idx] = sum;
                    }
                }
            }
        }
    }

    // Run HLS kernel
    mat_mult_trgt(A.data(), B.data(), C_hw.data());

    // Compare results
    bool match = true;
    for (int i = 0; i < MAT_C_SIZE; i++) {
        if (C_hw[i] != C_sw[i]) {
            std::cout << "Mismatch at index " << i
                      << ": HW = " << C_hw[i]
                      << ", SW = " << C_sw[i] << std::endl;
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
