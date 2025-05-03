#ifndef MAT_MULT_TRGT_H
#define MAT_MULT_TRGT_H

#include <cstdint>

// === Configurable tile and matrix dimensions ===
// These values must match the tiling scheme in your TrGT loop nest
#define M2 2
#define M1 2
#define M0 8
#define K 32
#define N1 4
#define N0 8

#define MAT_DIM 32
#define PE 8
#define K 32

// === Derived constants ===
#define MAT_A_SIZE (M2 * M1 * M0 * K)
#define MAT_B_SIZE (N1 * K * N0)
#define MAT_C_SIZE (N1 * M2 * M1 * M0 * N0)

void mat_mult_trgt(
    const uint32_t *A, // shape: [M2, M1, M0, K]
    const uint32_t *B, // shape: [N1, K, N0]
    uint32_t *C        // shape: [N1, M2, M1, M0, N0]
);

#endif // MAT_MULT_TRGT_H
