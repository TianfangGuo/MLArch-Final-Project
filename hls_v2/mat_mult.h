#include <cstdint>

#ifndef MAT_MULT_H
#define MAT_MULT_H

// Parameterized sizes (you can tune these later)
#define SIZE_M 32
#define SIZE_N 32
#define SIZE_K 32

void mat_mult(
    const uint32_t M, 
    const uint32_t N, 
    const uint32_t K,
    const uint32_t *A,
    const uint32_t *B,
    uint32_t *C
);


#endif