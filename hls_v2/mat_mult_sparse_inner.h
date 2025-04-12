#include <cstdint>

#ifndef MAT_MULT_H
#define MAT_MULT_H

// Parameterized sizes (you can tune these later)
#define MAT_DIM 32
#define PE 8

void mat_mult(
    const uint32_t *A,
    const uint32_t *B,
    uint32_t *C
);


#endif
