#include "accelerator.h"

void gustavson_hls(
    int M, int N, int nnzA,
    const float valuesA[MAX_NNZ],
    const int col_indicesA[MAX_NNZ],
    const int row_ptrA[SIZE_M + 1],
    const float B[SIZE_K * SIZE_N],
    float C[SIZE_M * SIZE_N]
) {
    #pragma HLS INTERFACE ap_memory port=valuesA depth=MAX_NNZ
    #pragma HLS INTERFACE ap_memory port=col_indicesA depth=MAX_NNZ
    #pragma HLS INTERFACE ap_memory port=row_ptrA depth=SIZE_M+1
    #pragma HLS INTERFACE ap_memory port=B depth=SIZE_K*SIZE_N
    #pragma HLS INTERFACE ap_memory port=C depth=SIZE_M*SIZE_N

    #pragma HLS INTERFACE s_axilite port=M bundle=control
    #pragma HLS INTERFACE s_axilite port=N bundle=control
    #pragma HLS INTERFACE s_axilite port=nnzA bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Pipeline kernel itself
    #pragma HLS PIPELINE II=1

    // Initialize output matrix C
    init_output:
    for (int i = 0; i < SIZE_M * SIZE_N; i++) {
        #pragma HLS PIPELINE II=1
        C[i] = 0;
    }

    compute_rows:
    for (int i = 0; i < M; i++) {
        float row_acc[SIZE_N] = {0};
        #pragma HLS ARRAY_PARTITION variable=row_acc complete dim=1

        compute_nnz:
        for (int idx = row_ptrA[i]; idx < row_ptrA[i + 1]; idx++) {
            #pragma HLS PIPELINE II=1
            int col = col_indicesA[idx];
            float val = valuesA[idx];

            compute_inner:
            for (int j = 0; j < N; j++) {
                #pragma HLS UNROLL factor=4
                row_acc[j] += val * B[col * N + j];
            }
        }

        write_back:
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            C[i * N + j] = row_acc[j];
        }
    }
}
