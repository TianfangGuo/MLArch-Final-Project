#include <cstdint>

#ifndef MAT_MULT_SPARSE_H
#define MAT_MULT_SPARSE_H

// Parameterized sizes (you can tune these later)
#define SIZE_M 64
#define SIZE_N 64
#define SIZE_K 64
#define BLOCK_SIZE 32
#define PE_SIZE 16

void mat_mult_sparse(
    const uint32_t M, 
    const uint32_t N, 
    const uint32_t K,
    const uint32_t *A,
    const uint32_t *B,
    uint32_t *C
);


#endif