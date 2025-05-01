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
    float target_density = 0.1f;
    srand(time(0));
    for (int i = 0; i < MAT_A_SIZE; i++) {
        A[i] = (rand() / (float)RAND_MAX) < target_density ? (i % 10 + 1) : 0;
    }
    for (int i = 0; i < MAT_B_SIZE; i++) {
        B[i] = (rand() / (float)RAND_MAX) < target_density ? (i % 10 + 1) : 0;
    }

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
