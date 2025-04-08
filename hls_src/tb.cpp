#include "accelerator.h"
#include <iostream>

int main() {
    // Parameters
    const int M = 4;
    const int N = 4;
    const int K = 4;

    // Example sparse matrix A in CSR
    const int nnzA = 6;
    float valuesA[MAX_NNZ] = {1, 2, 3, 4, 5, 6};
    int col_indicesA[MAX_NNZ] = {0, 3, 0, 2, 0, 3};
    int row_ptrA[SIZE_M + 1] = {0, 2, 3, 4, 6}; // 4 rows

    // Dense matrix B (flattened)
    float B[SIZE_K * SIZE_N] = {
        7, 0, 0, 0,
        0, 8, 0, 0,
        9, 0, 10, 0,
        0, 0, 0, 11
    };

    // Output matrix C
    float C[SIZE_M * SIZE_N] = {0};

    // Call HLS kernel
    gustavson_hls(M, N, nnzA, valuesA, col_indicesA, row_ptrA, B, C);

    // Print output
    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
