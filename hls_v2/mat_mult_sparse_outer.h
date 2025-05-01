#include <cstdint>

#ifndef MAT_MULT_H
#define MAT_MULT_H

// Parameterized sizes (you can tune these later)
#define MAT_DIM 32
#define PE 8

void mat_mult(
    const uint32_t*  A_row_ind,
    const uint32_t*  A_col_ptr,
    const uint32_t*  A_val,
    const uint32_t   A_nnz,
    const uint32_t*  B_row_ptr,
    const uint32_t*  B_col_ind,
    const uint32_t*  B_val,
    const uint32_t   B_nnz,
    uint32_t*        C_row_ind,
    uint32_t*        C_col_ind,
    uint32_t*        C_val,
    uint32_t*        C_nnz
);

#endif
