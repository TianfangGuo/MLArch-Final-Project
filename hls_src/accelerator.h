#ifndef ACCELERATOR_H
#define ACCELERATOR_H

// Parameterized sizes (you can tune these later)
#define SIZE_M 16
#define SIZE_N 16
#define SIZE_K 16
#define MAX_NNZ 256

// Gustavson HLS kernel function prototype
void gustavson_hls(
    int M, int N, int nnzA,
    const float valuesA[MAX_NNZ],
    const int col_indicesA[MAX_NNZ],
    const int row_ptrA[SIZE_M + 1],
    const float B[SIZE_K * SIZE_N],
    float C[SIZE_M * SIZE_N]
);

#endif // ACCELERATOR_H
