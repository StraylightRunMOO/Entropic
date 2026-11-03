/**
 * @file sobol.h
 * @brief Fast N‑dimensional Sobol sequence generator (C99)
 * @author NotoriousLOB
 * @version 0.0.1
 * 
 * Quasi‑Monte‑Carlo low‑discrepancy sequences using Gray‑code acceleration.
 * Direction numbers from Joe & Kuo (2008) for dimensions 1‑100.
 */

#ifndef SK_SPATIAL_SOBOL_H
#define SK_SPATIAL_SOBOL_H

#include "gray_codes.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Maximum bits per coordinate (64 → full double precision) */
#define SOBOL_MAX_BITS 64
#define SOBOL_MAX_DIM 10  /* Only 10 dims provided in table; full table would be 100+ */
#define SOBOL_POLY_DEGREE_MAX 18

/* ============================================================================
 * Direction numbers for each dimension (Joe & Kuo, 2008)
 * ========================================================================= */

typedef struct {
    unsigned int degree;                 /* Degree of primitive polynomial */
    unsigned int a;                      /* Polynomial coefficients (binary) */
    unsigned int init[SOBOL_POLY_DEGREE_MAX]; /* m_i values, i=1..degree */
} sob31_dirnum_t;

/* First 10 dimensions of direction numbers (Joe & Kuo)
 * Note: Full 100-dim table would require ~3KB but is omitted for brevity.
 * For production use, include the complete table from the original paper.
 */
static const sob31_dirnum_t sob31_dirnum_table[SOBOL_MAX_DIM] = {
    {1, 0, {1}},  /* dim 1 */
    {2, 1, {1, 1}},
    {3, 1, {1, 3, 1}},
    {3, 2, {1, 1, 1}},
    {4, 1, {1, 1, 3, 3}},
    {4, 4, {1, 3, 5, 13}},
    {5, 2, {1, 1, 3, 5, 1}},
    {5, 4, {1, 3, 1, 11, 1}},
    {5, 7, {1, 1, 5, 11, 1}},
    {5, 11, {1, 5, 1, 9, 1}},  /* dim 10 */
    /* Dimensions 11-100: not populated (would need full table) */
};

/* ============================================================================
 * Per‑dimension generator state
 * ========================================================================= */

typedef struct {
    uint64_t v[SOBOL_MAX_BITS];          /* Direction numbers v_i = m_i / 2^i */
    unsigned int degree;                 /* Cached degree */
} sob31_dim_state_t;

/* ============================================================================
 * Main generator state
 * ========================================================================= */

typedef struct {
    size_t dim;                          /* Number of dimensions */
    sob31_dim_state_t *state;            /* Per‑dim direction numbers */
    uint64_t *x;                         /* Current integer vector (unscaled) */
    uint64_t count;                      /* Points generated so far */
} sob31_generator_t;

/* ============================================================================
 * Internal: Build v_i = m_i / 2^i for one dimension
 * ========================================================================= */

GC_INLINE void sob31_build_direction_numbers(sob31_dim_state_t *s, 
                                             const sob31_dirnum_t *dir) {
    assert(s && dir);
    
    s->degree = dir->degree;
    unsigned int *m = (unsigned int*)alloca((SOBOL_MAX_BITS + 1) * sizeof(unsigned int));
    assert(m);
    
    /* Seed initial m_i from table */
    for (unsigned int i = 1; i <= dir->degree; i++) {
        m[i] = dir->init[i - 1];
    }
    
    /* Compute remaining m_i using recurrence */
    const unsigned int a = dir->a;
    const unsigned int deg = dir->degree;
    for (unsigned int i = deg + 1; i <= SOBOL_MAX_BITS; i++) {
        m[i] = m[i - deg] ^ (m[i - deg] >> deg);
        for (unsigned int k = 1; k < deg; k++) {
            m[i] ^= (((a >> (deg - 1 - k)) & 1) * m[i - k]);
        }
    }
    
    /* Build v_i = m_i / 2^i (i.e., shift left by 64-i bits) */
    for (unsigned int i = 1; i <= SOBOL_MAX_BITS; i++) {
        s->v[i - 1] = (uint64_t)m[i] << (64 - i);
    }
}

/* ============================================================================
 * Public API: Initialize generator
 * ========================================================================= */

GC_INLINE int sob31_init(sob31_generator_t *gen, size_t dim) {
    if (!gen || dim == 0 || dim > SOBOL_MAX_DIM) return 0;
    
    gen->dim = dim;
    gen->state = NULL;
    gen->x = NULL;
    gen->count = 0;
    
    gen->state = (sob31_dim_state_t *)calloc(dim, sizeof(sob31_dim_state_t));
    if (!gen->state) {
        return 0;
    }
    
    gen->x = (uint64_t *)calloc(dim, sizeof(uint64_t));
    if (!gen->x) {
        free(gen->state);
        gen->state = NULL;
        return 0;
    }
    
    /* Load direction numbers for each dimension */
    for (size_t d = 0; d < dim; d++) {
        sob31_build_direction_numbers(&gen->state[d], &sob31_dirnum_table[d]);
    }
    
    return 1;
}

/* ============================================================================
 * Public API: Free generator
 * ========================================================================= */

GC_INLINE void sob31_free(sob31_generator_t *gen) {
    if (!gen) return;
    free(gen->state);
    free(gen->x);
    gen->state = NULL;
    gen->x = NULL;
}

/* ============================================================================
 * Public API: Generate next point (fast Gray‑code update)
 * ========================================================================= */

GC_INLINE uint64_t sob31_next(sob31_generator_t *gen, size_t dim_idx) {
    assert(gen && gen->state && gen->x);
    assert(dim_idx < gen->dim);
    
    /* Use your Gray‑code kernel: g = encode(count) */
    const uint64_t g = gc_encode(gen->count);
    
    /* Find least‑significant zero bit in Gray code (the bit that flipped) */
    const unsigned int flipped_bit = gc_ctz64(~g);
    
    /* Update x[dim] ^= v[flipped_bit] */
    gen->x[dim_idx] ^= gen->state[dim_idx].v[flipped_bit];
    
    return gen->x[dim_idx];
}

/* ============================================================================
 * Public API: Generate N points in one dimension (batch mode)
 * ========================================================================= */

GC_INLINE size_t sob31_generate(sob31_generator_t *gen, size_t dim_idx, 
                                size_t n, uint64_t *restrict out) {
    assert(gen && gen->state && gen->x);
    assert(dim_idx < gen->dim);
    assert(out);
    
    const sob31_dim_state_t *s = &gen->state[dim_idx];
    uint64_t x = gen->x[dim_idx];

    for (size_t i = 0; i < n; i++) {
        const uint64_t g = gc_encode(gen->count + i);
        const unsigned int flipped_bit = gc_ctz64(~g);
        x ^= s->v[flipped_bit];
        out[i] = x;
    }

    gen->x[dim_idx] = x;
    gen->count += n;
    return n;
}

/* ============================================================================
 * Utility: Convert uint64_t to double in [0, 1)
 * ========================================================================= */

GC_INLINE double sob31_to_double(uint64_t x) {
    /* Multiply by 2^-64 */
    return x * 0x1.0p-64;
}

/* ============================================================================
 * Utility: Skip ahead by 2^L points (using Gray‑code jumps)
 * ========================================================================= */

GC_INLINE void sob31_skip_ahead(sob31_generator_t *gen, size_t dim_idx, 
                                unsigned int power_of_two) {
    assert(gen && gen->state && gen->x);
    assert(dim_idx < gen->dim);
    assert(power_of_two <= SOBOL_MAX_BITS);
    
    /* Gray code at position 2^power_of_two has single bit set */
    const uint64_t g = (uint64_t)1 << power_of_two;
    const unsigned int flipped_bit = gc_ctz64(g);
    
    gen->x[dim_idx] ^= gen->state[dim_idx].v[flipped_bit];
    gen->count += (1ULL << power_of_two);
}

#endif /* SK_SPATIAL_SOBOL_H */

/* Example: generate 1000 points in 3D
#include <stdio.h>
#include "sobol.h"

int main() {
    sob31_generator_t gen;
    if (!sob31_init(&gen, 3)) {
        fprintf(stderr, "Failed to init generator\n");
        return 1;
    }
    
    for (size_t i = 0; i < 1000; i++) {
        double x[3];
        for (size_t d = 0; d < 3; d++) {
            x[d] = sob31_to_double(sob31_next(&gen, d));
        }
        printf("%lf %lf %lf\n", x[0], x[1], x[2]);
    }
    
    sob31_free(&gen);
    return 0;
}
*/


