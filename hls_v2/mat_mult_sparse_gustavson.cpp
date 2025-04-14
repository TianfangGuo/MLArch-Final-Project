#include "mat_mult.h"
#include <cstdint>
#include "ap_int.h"
#include "hls_stream.h"
#include "stdio.h"

// Define matrix dimensions
#define M2 1
#define M1 1
#define M0 MAT_DIM
#define N1 1
#define N0 MAT_DIM
#define K MAT_DIM

// Stage 1: Load A into local buffer
void load_A(const uint32_t *A, uint32_t A_buf[M2][M1][M0][K]) {
    for (int m2 = 0; m2 < M2; ++m2) {
        for (int m1 = 0; m1 < M1; ++m1) {
            for (int m0 = 0; m0 < M0; ++m0) {
                for (int k = 0; k < K; ++k) {
                    #pragma HLS PIPELINE II=1
                    A_buf[m2][m1][m0][k] = A[(m2 * M1 * M0 * K) + (m1 * M0 * K) + (m0 * K) + k];
                }
            }
        }
    }
}

// Stage 2: Gather B to stream
void gather_B(const uint32_t *B, hls::stream<uint32_t> B_tmp_stream[K]) {
    for (int k = 0; k < K; ++k) {
        for (int n0 = 0; n0 < N0; ++n0) {
            #pragma HLS PIPELINE II=1
            uint32_t B_val = B[k * N0 + n0]; // Flattened access, assuming n1=0
            B_tmp_stream[k].write(B_val);
        }
    }
}

// Stage 3: Merge reduction (with local accumulation + partitioned buffers)
void merge_reduction(
    uint32_t A_buf[M2][M1][M0][K],
    hls::stream<uint32_t> B_tmp_stream[K],
    uint32_t C_buf[M2][M1][M0][N0]
) {
    // ✅ Partition buffers for full throughput
    #pragma HLS ARRAY_PARTITION variable=A_buf complete dim=3
    #pragma HLS ARRAY_PARTITION variable=C_buf complete dim=4

    // ✅ Local register accumulation buffer
    uint32_t local_C_buf[M0][N0];
    #pragma HLS ARRAY_PARTITION variable=local_C_buf complete dim=0

    // Step 1: Initialize local accumulator
    for (int m0 = 0; m0 < M0; ++m0) {
        for (int n0 = 0; n0 < N0; ++n0) {
            #pragma HLS PIPELINE II=1
            local_C_buf[m0][n0] = 0;
        }
    }

    // Step 2: Perform accumulation into local register buffer
    for (int k = 0; k < K; ++k) {
        for (int n0 = 0; n0 < N0; ++n0) {
            #pragma HLS PIPELINE II=1
            uint32_t B_val = B_tmp_stream[k].read();

            for (int m0 = 0; m0 < M0; ++m0) {
                uint32_t A_val = A_buf[0][0][m0][k];
                if (A_val != 0 && B_val != 0) {
                    local_C_buf[m0][n0] += A_val * B_val;
                }
            }
        }
    }

    // Step 3: Write back local accumulators to C_buf
    for (int m0 = 0; m0 < M0; ++m0) {
        for (int n0 = 0; n0 < N0; ++n0) {
            #pragma HLS PIPELINE II=1
            C_buf[0][0][m0][n0] = local_C_buf[m0][n0];
        }
    }
}

// Stage 4: Store C buffer to output
void store_C(uint32_t C_buf[M2][M1][M0][N0], uint32_t *C) {
    for (int m2 = 0; m2 < M2; ++m2) {
        for (int m1 = 0; m1 < M1; ++m1) {
            for (int m0 = 0; m0 < M0; ++m0) {
                for (int n0 = 0; n0 < N0; ++n0) {
                    #pragma HLS PIPELINE II=1
                    C[(m2 * M1 * M0 * N0) + (m1 * M0 * N0) + (m0 * N0) + n0] = C_buf[m2][m1][m0][n0];
                }
            }
        }
    }
}

// Top-level function
void mat_mult(
    const uint32_t *A,
    const uint32_t *B,
    uint32_t *C
) {
    #pragma HLS INTERFACE m_axi port=A offset=slave depth=1024 bundle=gmem0
    #pragma HLS INTERFACE m_axi port=B offset=slave depth=1024 bundle=gmem1
    #pragma HLS INTERFACE m_axi port=C offset=slave depth=1024 bundle=gmem0
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    uint32_t A_buf[M2][M1][M0][K];
    uint32_t C_buf[M2][M1][M0][N0];

    // Stream for B
    hls::stream<uint32_t> B_tmp_stream[K];
    #pragma HLS STREAM variable=B_tmp_stream depth=32
    #pragma HLS ARRAY_PARTITION variable=B_tmp_stream complete dim=1

    // ✅ Dataflow region
    #pragma HLS DATAFLOW

    load_A(A, A_buf);
    gather_B(B, B_tmp_stream);
    merge_reduction(A_buf, B_tmp_stream, C_buf);
    store_C(C_buf, C);
}
