#include <cstdint>

#ifndef MAT_MULT_H
#define MAT_MULT_H

// Parameterized sizes (you can tune these later)
#define MAT_DIM 32
#define PE 8

void mat_mult(
    const uint32_t*  A_m_ind,
    const uint32_t*  A_val,
    const uint32_t   A_m_dim,
    const uint32_t*  B_n_ind,
    const uint32_t*  B_val,
    const uint32_t   B_n_dim,
	uint32_t*        C
);

#endif
