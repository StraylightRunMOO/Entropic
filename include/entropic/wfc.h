/* wfc.h — Entropic: Wave Function Collapse HPC Engine
 * Single-header C11. Zero heap traffic on the hot path.
 * Version 1.0.0
 *
 * Compile-time configuration (define BEFORE including this header):
 *   WFC_MAX_PATTERNS          — must be power of 2, <= 256 (default 256)
 *   WFC_MAX_CLASSES           — tile classes for global constraints (default 64)
 *   WFC_MAX_LAYERS            — per-pattern layer slots (default 4)
 *   WFC_MAX_BACKTRACK_DEPTH   — snapshot stack depth (default 64)
 *   WFC_ENABLE_AVX2           — force AVX2 bitset paths
 *   WFC_ENABLE_NEON           — force NEON bitset paths
 *   WFC_USE_FFTW              — enable spectral Gumbel / Matérn / heat fields
 *   WFC_ENABLE_COPULA         — enable Gaussian copula entropy biasing
 *   WFC_ENABLE_BACKTRACK      — enable delta-trail backtracking
 *   WFC_MALLOC / WFC_FREE / WFC_REALLOC — allocator overrides
 *
 * Usage:
 *   #define WFC_IMPLEMENTATION
 *   #include "wfc.h"
 *
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025-2026 StraylightRunMOO
 */

#ifndef WFC_H
#define WFC_H

/* ============================================================
 * 1. Configuration & defaults
 * ============================================================ */
#ifndef WFC_MAX_PATTERNS
#define WFC_MAX_PATTERNS 256
#endif
#if ((WFC_MAX_PATTERNS) & ((WFC_MAX_PATTERNS) - 1)) != 0 || (WFC_MAX_PATTERNS) > 256
#error "WFC_MAX_PATTERNS must be a power of 2 and <= 256"
#endif

#ifndef WFC_MAX_CLASSES
#define WFC_MAX_CLASSES 64
#endif
#ifndef WFC_MAX_LAYERS
#define WFC_MAX_LAYERS 4
#endif
#ifndef WFC_MAX_BACKTRACK_DEPTH
#define WFC_MAX_BACKTRACK_DEPTH 64
#endif

#ifndef WFC_MALLOC
#include <stdlib.h>
#define WFC_MALLOC(sz)       malloc(sz)
#endif
#ifndef WFC_FREE
#include <stdlib.h>
#define WFC_FREE(p)          free(p)
#endif
#ifndef WFC_REALLOC
#include <stdlib.h>
#define WFC_REALLOC(p, sz)   realloc(p, sz)
#endif

/* ============================================================
 * 2. SIMD / ISA detection
 * ============================================================ */
#if defined(WFC_ENABLE_AVX2) && defined(__AVX2__)
#include <immintrin.h>
#define WFC_HAS_AVX2 1
#endif
#if defined(WFC_ENABLE_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define WFC_HAS_NEON 1
#endif

/* ============================================================
 * 3. Optional FFTW spectral backend
 * ============================================================ */
#ifdef WFC_USE_FFTW
#include <fftw3.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#endif

#include <stdint.h>
#include <string.h>
#include <float.h>
#include <math.h>

/* C/C++ compatibility */
#ifdef __cplusplus
#define WFC_RESTRICT __restrict__
#else
#define WFC_RESTRICT restrict
#endif

/* ============================================================
 * 4. Constants
 * ============================================================ */
#define WFC_PATTERN_INVALID 0xFFFFu
#define WFC_CLASS_INVALID   0xFFFFu
#define WFC_HEAP_INVALID    0xFFFFFFFFu
#define WFC_MASK_U64S       ((WFC_MAX_PATTERNS) / 64)

/* ============================================================
 * 5. Types
 * ============================================================ */

typedef enum {
    WFC_OK = 0,
    WFC_EINVAL = -1,
    WFC_ENOMEM = -2,
    WFC_ECONTRADICTION = -3,
    WFC_EUNINIT = -4,
    WFC_EUNSUPPORTED = -5
} wfc_err_t;

/* Directions: 2D uses N,S,E,W. 3D adds U,D. */
typedef enum {
    WFC_DIR_N = 0,
    WFC_DIR_S = 1,
    WFC_DIR_E = 2,
    WFC_DIR_W = 3,
    WFC_DIR_U = 4,
    WFC_DIR_D = 5,
    WFC_DIR_COUNT = 6
} wfc_dir_t;

/* A pattern is a complete tile configuration (all layers fused). */
typedef struct {
    uint16_t id;                        /* < WFC_MAX_PATTERNS */
    uint16_t class_id;                  /* < WFC_MAX_CLASSES */
    float    weight;                    /* empirical frequency (>= 0) */
    uint32_t freq;                      /* raw count from sample */
    uint8_t  num_layers;                /* <= WFC_MAX_LAYERS */
    uint16_t layer_types[WFC_MAX_LAYERS]; /* e.g. 1=grass, 2=bush */
} wfc_pattern_def_t;

/* Grid cell. _Alignas(64) keeps masks on cache lines and prevents false sharing. */
#ifdef __cplusplus
struct alignas(64) wfc_cell {
#else
_Alignas(64) struct wfc_cell {
#endif
    uint64_t mask[WFC_MASK_U64S];       /* valid patterns bitset */
    uint16_t collapsed;                 /* WFC_PATTERN_INVALID if uncollapsed */
    uint16_t reserved;
    float    entropy;                   /* cached Shannon entropy */
    float    prior;                     /* copula/spectral prior offset */
    uint8_t  dirty;                     /* entropy needs recomputation */
};
typedef struct wfc_cell wfc_cell_t;

/* World configuration */
typedef struct {
    uint32_t nx, ny, nz;                /* requested output dimensions (nz=1 for 2D) */
    uint32_t seed;
    int      enable_backtracking;       /* requires WFC_ENABLE_BACKTRACK */
    int      enable_spectral;           /* requires WFC_USE_FFTW */
    int      enable_copula;             /* requires WFC_ENABLE_COPULA */
    int      seamless;                  /* toroidal boundary conditions */
    float    spectral_weight;           /* scale of Gumbel/Matérn prior */
    float    copula_weight;             /* scale of copula prior */
} wfc_config_t;

/* Backtracking trail entry (delta snapshot) */
typedef struct {
    uint32_t cell_idx;
    uint64_t mask[WFC_MASK_U64S];
    uint16_t collapsed;
} wfc_trail_entry_t;

typedef struct {
    uint32_t trail_top;
    uint32_t class_placed[WFC_MAX_CLASSES];
} wfc_snapshot_t;

/* Contradiction callback type */
typedef void (*wfc_contradiction_cb_t)(uint32_t x, uint32_t y, uint32_t z, void* user_data);

/* The world holds everything. All variable-size arrays are allocated once in wfc_world_create. */
typedef struct wfc_world {
    /* Geometry */
    uint32_t nx, ny, nz;                /* requested */
    uint32_t nx_p2, ny_p2, nz_p2;       /* power-of-two allocated */
    uint32_t log2_nx, log2_ny, log2_nz;
    uint32_t num_cells;
    uint32_t num_patterns;
    uint32_t num_classes;

    /* Grid */
    wfc_cell_t* cells;

    /* Pattern catalogue */
    wfc_pattern_def_t patterns[WFC_MAX_PATTERNS];
    float pattern_plogp[WFC_MAX_PATTERNS]; /* -p*log2(p) for fast entropy */

    /* Precomputed class bitmasks */
    uint64_t class_mask[WFC_MAX_CLASSES][WFC_MASK_U64S];

    /* Compatibility: compat[dir][pat][block] = bitmask of patterns compatible with pat in dir. */
    uint64_t compat[WFC_DIR_COUNT][WFC_MAX_PATTERNS][WFC_MASK_U64S];

    /* Class hard constraints */
    uint32_t class_min[WFC_MAX_CLASSES];
    uint32_t class_max[WFC_MAX_CLASSES];
    uint32_t class_placed[WFC_MAX_CLASSES];
    uint32_t class_constrained;         /* bitmask of classes with min>0 || max<UINT32_MAX */

    /* Min-entropy heap (binary min-heap of cell indices) */
    uint32_t* heap;                     /* size num_cells */
    uint32_t* heap_pos;                 /* cell -> index in heap, WFC_HEAP_INVALID if collapsed */
    uint32_t heap_size;

    /* Propagation queue (ring buffer) */
    uint32_t* queue;
    uint32_t  queue_head, queue_tail, queue_cap;
    uint8_t*  in_queue;                 /* BUG FIX: track queued cells to prevent overflow */

    /* Backtracking */
#if defined(WFC_ENABLE_BACKTRACK)
    wfc_trail_entry_t* trail;
    uint32_t          trail_cap;
    uint32_t          trail_top;
    uint32_t*         trail_gen;        /* generation tag per cell */
    uint32_t          trail_current_gen;
    wfc_snapshot_t    snapshots[WFC_MAX_BACKTRACK_DEPTH];
    uint32_t          snapshot_top;
    int               contradiction;
#endif

    /* Spectral fields (FFTW) */
#if defined(WFC_USE_FFTW)
    float*    spectral_field;           /* additive entropy bias per cell */
    fftw_plan fftw_plan_fwd;
    fftw_plan fftw_plan_inv;
    double*   fftw_workspace;
    int       fftw_nx, fftw_ny;         /* power-of-two dims for FFTW */
#endif

    /* Copula */
#if defined(WFC_ENABLE_COPULA)
    /* Factorized Gaussian copula: rho[dir][c1][c2] for class pairs. */
    float copula_rho[WFC_DIR_COUNT][WFC_MAX_CLASSES][WFC_MAX_CLASSES];
    float copula_marginal[WFC_MAX_CLASSES]; /* empirical CDF value per class */
    float copula_z[WFC_MAX_CLASSES];        /* PRECOMPUTED: Phi^{-1}(marginal[c]) */
    uint8_t copula_class_pop[WFC_MAX_CLASSES]; /* number of patterns in class */
#endif

    /* RNG */
    uint64_t rng_state;

    /* Flags */
    int is_3d;
    int seamless;
    int backtrack_enabled;
    int spectral_enabled;
    int copula_enabled;
    float spectral_weight;
    float copula_weight;

    /* Step mode state */
    int step_mode;
    uint32_t step_target_collapses;
    uint32_t step_collapsed_count;

    /* Contradiction callback */
    wfc_contradiction_cb_t on_contradiction;
    void* contradiction_user_data;
} wfc_world_t;

/* ============================================================
 * 6. Public API
 * ============================================================ */

#ifdef __cplusplus
extern "C" {
#endif

/* Lifecycle */
wfc_world_t* wfc_world_create(const wfc_config_t* cfg);
void         wfc_world_destroy(wfc_world_t* w);

/* Pattern / rule definition */
wfc_err_t wfc_add_pattern(wfc_world_t* w, const wfc_pattern_def_t* pat);
wfc_err_t wfc_add_rule(wfc_world_t* w, uint16_t pat_a, uint16_t pat_b, wfc_dir_t dir);

/* Global class constraints (hard min / max counts). max=UINT32_MAX means unlimited. */
wfc_err_t wfc_set_class_limit(wfc_world_t* w, uint16_t class_id, uint32_t min_count, uint32_t max_count);

/* Learn rules, frequencies, and copula correlations from a sample grid. */
wfc_err_t wfc_learn_from_sample(wfc_world_t* w, const uint16_t* sample_grid,
                                uint32_t sx, uint32_t sy, uint32_t sz);

/* Spectral priors (FFTW required). Call before wfc_collapse. */
#if defined(WFC_USE_FFTW)
wfc_err_t wfc_set_gumbel_field(wfc_world_t* w, double H, uint64_t noise_seed);
wfc_err_t wfc_set_matern_field(wfc_world_t* w, double kappa, double nu, uint64_t noise_seed);
#endif

/* Solve: full collapse or step-by-step */
wfc_err_t wfc_collapse(wfc_world_t* w);

/* Step-by-step execution (new in 2.0) */
wfc_err_t wfc_step_init(wfc_world_t* w);
wfc_err_t wfc_step(wfc_world_t* w);
int       wfc_step_done(const wfc_world_t* w);

/* Contradiction callback (new in 2.0) */
void wfc_set_contradiction_callback(wfc_world_t* w, wfc_contradiction_cb_t cb, void* user_data);

/* Query result */
uint16_t wfc_get_cell_pattern(const wfc_world_t* w, uint32_t x, uint32_t y, uint32_t z);
const wfc_pattern_def_t* wfc_get_pattern(const wfc_world_t* w, uint16_t pat_id);

/* Utility: export grid to buffer */
wfc_err_t wfc_export_grid(const wfc_world_t* w, uint16_t* out_grid, size_t out_count);

/* Utility: reset world for re-solving */
void wfc_world_reset(wfc_world_t* w);

#ifdef __cplusplus
}
#endif

#endif /* WFC_H */

/* ============================================================
 * 7. IMPLEMENTATION
 * ============================================================ */
#ifdef WFC_IMPLEMENTATION

/* --- Utilities --- */
static inline uint32_t wfc_next_pow2_u32(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

/* SplitMix64 PRNG. One 64-bit state, excellent quality, no malloc. */
static inline uint64_t wfc_splitmix64(uint64_t* state)
{
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline float wfc_randf(uint64_t* state)
{
    /* 53-bit precision float in [0,1) */
    return (float)((wfc_splitmix64(state) >> 11) * (1.0 / (1ULL << 53)));
}

static inline uint32_t wfc_rand_range(uint64_t* state, uint32_t n)
{
    return (uint32_t)(wfc_randf(state) * (float)n);
}

/* --- Bitset primitives (the hot path) --- */
static inline void wfc_mask_copy(uint64_t* WFC_RESTRICT dst, const uint64_t* WFC_RESTRICT src)
{
    for (int i = 0; i < WFC_MASK_U64S; ++i) dst[i] = src[i];
}

static inline void wfc_mask_and(uint64_t* WFC_RESTRICT dst, const uint64_t* WFC_RESTRICT src)
{
#if defined(WFC_HAS_AVX2) && WFC_MASK_U64S == 4
    __m256i d = _mm256_loadu_si256((const __m256i*)dst);
    __m256i s = _mm256_loadu_si256((const __m256i*)src);
    _mm256_storeu_si256((__m256i*)dst, _mm256_and_si256(d, s));
#elif defined(WFC_HAS_NEON) && WFC_MASK_U64S == 4
    uint64x2_t d0 = vld1q_u64(dst + 0);
    uint64x2_t d1 = vld1q_u64(dst + 2);
    uint64x2_t s0 = vld1q_u64(src + 0);
    uint64x2_t s1 = vld1q_u64(src + 2);
    vst1q_u64(dst + 0, vandq_u64(d0, s0));
    vst1q_u64(dst + 2, vandq_u64(d1, s1));
#else
    for (int i = 0; i < WFC_MASK_U64S; ++i) dst[i] &= src[i];
#endif
}

static inline void wfc_mask_andnot(uint64_t* WFC_RESTRICT dst, const uint64_t* WFC_RESTRICT src)
{
    /* dst &= ~src */
#if defined(WFC_HAS_AVX2) && WFC_MASK_U64S == 4
    __m256i d = _mm256_loadu_si256((const __m256i*)dst);
    __m256i s = _mm256_loadu_si256((const __m256i*)src);
    _mm256_storeu_si256((__m256i*)dst, _mm256_andnot_si256(s, d));
#elif defined(WFC_HAS_NEON) && WFC_MASK_U64S == 4
    uint64x2_t d0 = vld1q_u64(dst + 0);
    uint64x2_t d1 = vld1q_u64(dst + 2);
    uint64x2_t s0 = vld1q_u64(src + 0);
    uint64x2_t s1 = vld1q_u64(src + 2);
    vst1q_u64(dst + 0, vbicq_u64(d0, s0));
    vst1q_u64(dst + 2, vbicq_u64(d1, s1));
#else
    for (int i = 0; i < WFC_MASK_U64S; ++i) dst[i] &= ~src[i];
#endif
}

static inline int wfc_mask_empty(const uint64_t* WFC_RESTRICT m)
{
#if defined(WFC_HAS_AVX2) && WFC_MASK_U64S == 4
    __m256i v = _mm256_loadu_si256((const __m256i*)m);
    return _mm256_testz_si256(v, v);
#elif defined(WFC_HAS_NEON) && WFC_MASK_U64S == 4
    /* BUG FIX (v2.0): Check ALL 256 bits, not just lower 128 */
    uint64x2_t v0 = vld1q_u64(m + 0);
    uint64x2_t v1 = vld1q_u64(m + 2);
    uint64x2_t z  = vdupq_n_u64(0);
    /* OR all lanes together, then check if zero */
    uint64_t lane0 = vgetq_lane_u64(vceqq_u64(v0, z), 0) & vgetq_lane_u64(vceqq_u64(v0, z), 1);
    uint64_t lane1 = vgetq_lane_u64(vceqq_u64(v1, z), 0) & vgetq_lane_u64(vceqq_u64(v1, z), 1);
    return (lane0 & lane1) & 1;
#else
    for (int i = 0; i < WFC_MASK_U64S; ++i) if (m[i]) return 0;
    return 1;
#endif
}

static inline int wfc_mask_eq(const uint64_t* WFC_RESTRICT a, const uint64_t* WFC_RESTRICT b)
{
#if defined(WFC_HAS_AVX2) && WFC_MASK_U64S == 4
    __m256i x = _mm256_loadu_si256((const __m256i*)a);
    __m256i y = _mm256_loadu_si256((const __m256i*)b);
    __m256i c = _mm256_cmpeq_epi64(x, y);
    return _mm256_testc_si256(c, _mm256_set1_epi64x(-1));
#else
    for (int i = 0; i < WFC_MASK_U64S; ++i) if (a[i] != b[i]) return 0;
    return 1;
#endif
}

static inline uint32_t wfc_mask_popcnt(const uint64_t* WFC_RESTRICT m)
{
    uint32_t c = 0;
#if defined(WFC_HAS_AVX2) && WFC_MASK_U64S == 4
    __m256i v = _mm256_loadu_si256((const __m256i*)m);
    const __m256i lookup = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4
    );
    __m256i low  = _mm256_and_si256(v, _mm256_set1_epi8(0x0F));
    __m256i high = _mm256_srli_epi16(v, 4);
    high = _mm256_and_si256(high, _mm256_set1_epi8(0x0F));
    __m256i pc = _mm256_add_epi8(_mm256_shuffle_epi8(lookup, low),
                                 _mm256_shuffle_epi8(lookup, high));
    __m256i sum = _mm256_sad_epu8(pc, _mm256_setzero_si256());
    c = (uint32_t)(_mm256_extract_epi64(sum, 0) + _mm256_extract_epi64(sum, 1)
                 + _mm256_extract_epi64(sum, 2) + _mm256_extract_epi64(sum, 3));
#elif defined(WFC_HAS_NEON) && WFC_MASK_U64S == 4
    uint8x16_t v0 = vreinterpretq_u8_u64(vld1q_u64(m + 0));
    uint8x16_t v1 = vreinterpretq_u8_u64(vld1q_u64(m + 2));
    uint8x16_t c0 = vcntq_u8(v0);
    uint8x16_t c1 = vcntq_u8(v1);
    uint8x16_t s  = vaddq_u8(c0, c1);
    c = vaddvq_u8(s);
#else
    for (int i = 0; i < WFC_MASK_U64S; ++i)
        c += (uint32_t)__builtin_popcountll(m[i]);
#endif
    return c;
}

/* --- Indexing (power-of-two) --- */
static inline uint32_t wfc_idx_2d(const wfc_world_t* w, uint32_t x, uint32_t y)
{
    return (y << w->log2_nx) | x;
}

static inline uint32_t wfc_idx_3d(const wfc_world_t* w, uint32_t x, uint32_t y, uint32_t z)
{
    return (z << (w->log2_nx + w->log2_ny)) | (y << w->log2_nx) | x;
}

static inline uint32_t wfc_idx(const wfc_world_t* w, uint32_t x, uint32_t y, uint32_t z)
{
    return w->is_3d ? wfc_idx_3d(w, x, y, z) : wfc_idx_2d(w, x, y);
}

/* --- Entropy --- */
static inline float wfc_cell_entropy(const wfc_world_t* w, const wfc_cell_t* cell)
{
    if (cell->collapsed != WFC_PATTERN_INVALID) return -1.0f;
    float e = 0.0f;
    for (int b = 0; b < WFC_MASK_U64S; ++b) {
        uint64_t m = cell->mask[b];
        while (m) {
            uint64_t lsb = m & -m;
            int bit = __builtin_ctzll(m) + (b << 6);
            e += w->pattern_plogp[bit];
            m ^= lsb;
        }
    }
    return e;
}

/* --- Heap --- */
static inline void wfc_heap_swap(wfc_world_t* w, uint32_t i, uint32_t j)
{
    uint32_t a = w->heap[i];
    uint32_t b = w->heap[j];
    w->heap[i] = b;
    w->heap[j] = a;
    w->heap_pos[a] = j;
    w->heap_pos[b] = i;
}

static inline void wfc_heap_sift_up(wfc_world_t* w, uint32_t idx)
{
    while (idx > 0) {
        uint32_t parent = (idx - 1) >> 1;
        uint32_t a = w->heap[idx];
        uint32_t b = w->heap[parent];
        if (w->cells[a].entropy >= w->cells[b].entropy) break;
        wfc_heap_swap(w, idx, parent);
        idx = parent;
    }
}

static inline void wfc_heap_sift_down(wfc_world_t* w, uint32_t idx)
{
    uint32_t n = w->heap_size;
    while (1) {
        uint32_t l = (idx << 1) + 1;
        uint32_t r = l + 1;
        uint32_t smallest = idx;
        if (l < n && w->cells[w->heap[l]].entropy < w->cells[w->heap[smallest]].entropy)
            smallest = l;
        if (r < n && w->cells[w->heap[r]].entropy < w->cells[w->heap[smallest]].entropy)
            smallest = r;
        if (smallest == idx) break;
        wfc_heap_swap(w, idx, smallest);
        idx = smallest;
    }
}

static inline void wfc_heap_insert(wfc_world_t* w, uint32_t cell_idx)
{
    uint32_t pos = w->heap_size++;
    w->heap[pos] = cell_idx;
    w->heap_pos[cell_idx] = pos;
    wfc_heap_sift_up(w, pos);
}

static inline uint32_t wfc_heap_pop(wfc_world_t* w)
{
    if (!w->heap_size) return WFC_HEAP_INVALID;
    uint32_t root = w->heap[0];
    w->heap_pos[root] = WFC_HEAP_INVALID;
    uint32_t last = w->heap[--w->heap_size];
    if (w->heap_size) {
        w->heap[0] = last;
        w->heap_pos[last] = 0;
        wfc_heap_sift_down(w, 0);
    }
    return root;
}

static inline void wfc_heap_update(wfc_world_t* w, uint32_t cell_idx)
{
    uint32_t pos = w->heap_pos[cell_idx];
    if (pos == WFC_HEAP_INVALID) return;
    wfc_heap_sift_up(w, pos);
    wfc_heap_sift_down(w, pos);
}

/* --- Backtracking (delta trail) --- */
#if defined(WFC_ENABLE_BACKTRACK)

static inline void wfc_trail_push(wfc_world_t* w, uint32_t cell_idx)
{
    if (!w->backtrack_enabled || !w->trail_gen) return;
    if (w->trail_gen[cell_idx] == w->trail_current_gen) return;
    w->trail_gen[cell_idx] = w->trail_current_gen;

    if (w->trail_top >= w->trail_cap) {
        w->contradiction = 1;
        return;
    }
    wfc_trail_entry_t* e = &w->trail[w->trail_top++];
    e->cell_idx = cell_idx;
    wfc_mask_copy(e->mask, w->cells[cell_idx].mask);
    e->collapsed = w->cells[cell_idx].collapsed;
}

static inline void wfc_snapshot_push(wfc_world_t* w)
{
    if (!w->backtrack_enabled) return;
    if (w->snapshot_top >= WFC_MAX_BACKTRACK_DEPTH) {
        w->contradiction = 1;
        return;
    }
    wfc_snapshot_t* s = &w->snapshots[w->snapshot_top++];
    s->trail_top = w->trail_top;
    memcpy(s->class_placed, w->class_placed, sizeof(w->class_placed));
    w->trail_current_gen++;
}

static inline int wfc_snapshot_pop(wfc_world_t* w)
{
    if (!w->backtrack_enabled || !w->snapshot_top) return 0;
    wfc_snapshot_t* s = &w->snapshots[--w->snapshot_top];
    /* Restore trail entries */
    while (w->trail_top > s->trail_top) {
        wfc_trail_entry_t* e = &w->trail[--w->trail_top];
        wfc_cell_t* cell = &w->cells[e->cell_idx];
        wfc_mask_copy(cell->mask, e->mask);
        cell->collapsed = e->collapsed;
        cell->dirty = 1;  /* Mark as dirty since entropy needs recomputation */
        if (cell->collapsed == WFC_PATTERN_INVALID) {
            if (w->heap_pos[e->cell_idx] == WFC_HEAP_INVALID)
                wfc_heap_insert(w, e->cell_idx);
        }
    }
    memcpy(w->class_placed, s->class_placed, sizeof(w->class_placed));

    /* BUG FIX (v2.0): Clear the propagation queue to prevent ghost state */
    w->queue_head = w->queue_tail = 0;
    /* Clear in_queue flags */
    if (w->in_queue) memset(w->in_queue, 0, w->num_cells * sizeof(uint8_t));

    w->contradiction = 0;
    return 1;
}

#endif /* WFC_ENABLE_BACKTRACK */

/* --- Propagation (AC-3) --- */
static inline void wfc_queue_push(wfc_world_t* w, uint32_t cell_idx)
{
    /* BUG FIX (v2.0): Prevent queue overflow by checking in_queue flag */
    if (w->in_queue[cell_idx]) return;
    w->in_queue[cell_idx] = 1;
    w->queue[w->queue_tail] = cell_idx;
    w->queue_tail = (w->queue_tail + 1) & (w->queue_cap - 1);
}

static inline uint32_t wfc_queue_pop(wfc_world_t* w)
{
    if (w->queue_head == w->queue_tail) return WFC_HEAP_INVALID;
    uint32_t v = w->queue[w->queue_head];
    w->queue_head = (w->queue_head + 1) & (w->queue_cap - 1);
    w->in_queue[v] = 0;  /* Clear flag on pop */
    return v;
}


/* Coordinates from cell index */
static inline void wfc_cell_coords(const wfc_world_t* w, uint32_t cell_idx,
                                    uint32_t* x, uint32_t* y, uint32_t* z)
{
    if (w->is_3d) {
        uint32_t p2 = w->nx_p2 * w->ny_p2;
        *z = cell_idx / p2;
        uint32_t rem = cell_idx % p2;
        *y = rem >> w->log2_nx;
        *x = rem & (w->nx_p2 - 1);
    } else {
        *z = 0;
        *y = cell_idx >> w->log2_nx;
        *x = cell_idx & (w->nx_p2 - 1);
    }
}

static inline void wfc_propagate_cell(wfc_world_t* w, uint32_t cell_idx, wfc_dir_t dir)
{
    const wfc_cell_t* cell = &w->cells[cell_idx];
    if (cell->collapsed == WFC_PATTERN_INVALID) return;

    uint16_t pat = cell->collapsed;
    const uint64_t* allow = w->compat[dir][pat];

    /* Neighbor coordinate from cell_idx */
    uint32_t x, y, z;
    wfc_cell_coords(w, cell_idx, &x, &y, &z);

    uint32_t nx2 = w->nx_p2, ny2 = w->ny_p2, nz2 = w->nz_p2;
    uint32_t nidx = WFC_HEAP_INVALID;

    switch (dir) {
        case WFC_DIR_N: if (y > 0 || w->seamless) nidx = cell_idx - nx2; break;
        case WFC_DIR_S: if (y + 1 < ny2 || w->seamless) nidx = cell_idx + nx2; break;
        case WFC_DIR_E: if (x + 1 < nx2 || w->seamless) nidx = cell_idx + 1; break;
        case WFC_DIR_W: if (x > 0 || w->seamless) nidx = cell_idx - 1; break;
        case WFC_DIR_U: if (z > 0 || w->seamless) nidx = cell_idx - nx2*ny2; break;
        case WFC_DIR_D: if (z + 1 < nz2 || w->seamless) nidx = cell_idx + nx2*ny2; break;
        default: break;
    }

    if (w->seamless && nidx == WFC_HEAP_INVALID) {
        switch (dir) {
            case WFC_DIR_N: nidx = cell_idx + (ny2 - 1) * nx2; break;
            case WFC_DIR_S: nidx = cell_idx - (ny2 - 1) * nx2; break;
            case WFC_DIR_E: nidx = cell_idx - (nx2 - 1); break;
            case WFC_DIR_W: nidx = cell_idx + (nx2 - 1); break;
            case WFC_DIR_U: nidx = cell_idx + (nz2 - 1) * nx2 * ny2; break;
            case WFC_DIR_D: nidx = cell_idx - (nz2 - 1) * nx2 * ny2; break;
            default: break;
        }
    }

    if (nidx >= w->num_cells) return;
    wfc_cell_t* nbr = &w->cells[nidx];
    if (nbr->collapsed != WFC_PATTERN_INVALID) return;

#if defined(WFC_ENABLE_BACKTRACK)
    wfc_trail_push(w, nidx);
#endif

    uint64_t old_mask[WFC_MASK_U64S];
    wfc_mask_copy(old_mask, nbr->mask);
    wfc_mask_and(nbr->mask, allow);

    if (wfc_mask_eq(old_mask, nbr->mask)) return;

    if (wfc_mask_empty(nbr->mask)) {
#if defined(WFC_ENABLE_BACKTRACK)
        w->contradiction = 1;
#endif
        return;
    }

    /* Mark dirty for lazy entropy update */
    nbr->dirty = 1;
    nbr->entropy = 0.0f;  /* Will be recomputed when popped */
    wfc_heap_update(w, nidx);

    /* Enqueue neighbor's neighbors */
    for (int d = 0; d < (w->is_3d ? 6 : 4); ++d)
        wfc_queue_push(w, nidx);
}

static inline void wfc_propagate_from(wfc_world_t* w, uint32_t cell_idx)
{
    for (int d = 0; d < (w->is_3d ? 6 : 4); ++d) {
        wfc_propagate_cell(w, cell_idx, (wfc_dir_t)d);
#if defined(WFC_ENABLE_BACKTRACK)
        if (w->contradiction) return;
#endif
    }
}

/* --- Class constraint enforcement --- */
static inline void wfc_enforce_class_limits(wfc_world_t* w)
{
    for (int c = 0; c < WFC_MAX_CLASSES; ++c) {
        if (!(w->class_constrained & (1u << c))) continue;
        if (w->class_placed[c] >= w->class_max[c]) {
            /* Remove all patterns of class c from every uncollapsed cell. */
            for (uint32_t i = 0; i < w->num_cells; ++i) {
                wfc_cell_t* cell = &w->cells[i];
                if (cell->collapsed != WFC_PATTERN_INVALID) continue;
#if defined(WFC_ENABLE_BACKTRACK)
                wfc_trail_push(w, i);
#endif
                wfc_mask_andnot(cell->mask, w->class_mask[c]);
                if (wfc_mask_empty(cell->mask)) {
#if defined(WFC_ENABLE_BACKTRACK)
                    w->contradiction = 1;
                    return;
#endif
                }
                /* Lazy entropy: mark dirty instead of immediate recompute */
                cell->dirty = 1;
                cell->entropy = 0.0f;
                wfc_heap_update(w, i);
            }
        }
    }
}

/* --- Copula (factorized Gaussian C-vine) --- */
#if defined(WFC_ENABLE_COPULA)

static inline float wfc_inv_cdf_normal(float u)
{
    /* Acklam approximation for Phi^{-1}(u), accurate enough for copula bias. */
    if (u <= 0.0f) u = 1e-7f;
    if (u >= 1.0f) u = 1.0f - 1e-7f;
    const float a1 = -3.969683028665376e+01f, a2 =  2.209460984245205e+02f;
    const float a3 = -2.759285104469687e+02f, a4 =  1.383577518672690e+02f;
    const float a5 = -3.066479806614716e+01f, a6 =  2.506628277459239e+00f;
    const float b1 = -5.447609879822406e+01f, b2 =  1.615858368580409e+02f;
    const float b3 = -1.556989798598866e+02f, b4 =  6.680131188771972e+01f;
    const float b5 = -1.328068155288572e+01f;
    const float c1 = -7.784894002430293e-03f, c2 = -3.223964580411365e-01f;
    const float c3 = -2.400758277161838e+00f, c4 = -2.549732539343734e+00f;
    const float c5 =  4.374664141464968e+00f, c6 =  2.938163982698783e+00f;
    const float d1 =  7.784695709041462e-03f, d2 =  3.224671290700398e-01f;
    const float d3 =  2.445134137142996e+00f, d4 =  3.754408661907416e+00f;
    const float p_low  = 0.02425f;
    const float p_high = 1.0f - p_low;
    float q, r, x;
    if (u < p_low) {
        q = sqrtf(-2.0f * logf(u));
        x = (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1.0f);
    } else if (u <= p_high) {
        q = u - 0.5f;
        r = q*q;
        x = (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q /
            (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0f);
    } else {
        q = sqrtf(-2.0f * logf(1.0f - u));
        x = -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1.0f);
    }
    return x;
}

static inline float wfc_cdf_normal(float x)
{
    return 0.5f * erfcf(-x * 0.7071067811865475f);
}

static inline float wfc_copula_class_prior(wfc_world_t* w, uint32_t cell_idx, uint16_t center_class)
{
    /* C-vine: center is root. Sum log-conditional from each collapsed neighbor.
     * PERFORMANCE FIX (v2.0): Use precomputed Z_c values instead of inv_cdf per call. */
    float log_prior = 0.0f;
    uint32_t x, y, z;
    wfc_cell_coords(w, cell_idx, &x, &y, &z);

    const int ndirs = w->is_3d ? 6 : 4;
    for (int d = 0; d < ndirs; ++d) {
        uint32_t nx = x, ny = y, nz = z;
        switch (d) {
            case WFC_DIR_N: ny = (ny + w->ny_p2 - 1) & (w->ny_p2 - 1); break;
            case WFC_DIR_S: ny = (ny + 1) & (w->ny_p2 - 1); break;
            case WFC_DIR_E: nx = (nx + 1) & (w->nx_p2 - 1); break;
            case WFC_DIR_W: nx = (nx + w->nx_p2 - 1) & (w->nx_p2 - 1); break;
            case WFC_DIR_U: nz = (nz + w->nz_p2 - 1) & (w->nz_p2 - 1); break;
            case WFC_DIR_D: nz = (nz + 1) & (w->nz_p2 - 1); break;
        }
        uint32_t nidx = wfc_idx(w, nx, ny, nz);
        const wfc_cell_t* nbr = &w->cells[nidx];
        if (nbr->collapsed == WFC_PATTERN_INVALID) continue;

        uint16_t nc = w->patterns[nbr->collapsed].class_id;
        float rho = w->copula_rho[d][nc][center_class];
        if (rho == 0.0f) continue;

        /* Use precomputed Z values instead of computing inv_cdf every time */
        float z_n = w->copula_z[nc];
        float z_c = w->copula_z[center_class];

        float cond_mean = rho * z_n;
        float cond_var  = 1.0f - rho * rho;
        float cond_std  = sqrtf(cond_var);
        if (cond_std < 1e-6f) continue;

        float z_cond = (z_c - cond_mean) / cond_std;
        float p = wfc_cdf_normal(z_cond);
        if (p > 0.0f) log_prior += logf(p);
    }
    return log_prior;
}

static inline void wfc_copula_update_cell(wfc_world_t* w, uint32_t cell_idx)
{
    wfc_cell_t* cell = &w->cells[cell_idx];
    if (cell->collapsed != WFC_PATTERN_INVALID) return;

    float class_log_prior[WFC_MAX_CLASSES];
    for (int c = 0; c < WFC_MAX_CLASSES; ++c) class_log_prior[c] = 0.0f;

    for (int c = 0; c < WFC_MAX_CLASSES; ++c) {
        if (w->copula_class_pop[c] == 0) continue;
        class_log_prior[c] = wfc_copula_class_prior(w, cell_idx, (uint16_t)c);
    }

    float exp_log_prior = 0.0f;
    float total_w = 0.0f;
    for (int b = 0; b < WFC_MASK_U64S; ++b) {
        uint64_t m = cell->mask[b];
        while (m) {
            uint64_t lsb = m & -m;
            int bit = __builtin_ctzll(m) + (b << 6);
            uint16_t c = w->patterns[bit].class_id;
            float wgt = w->patterns[bit].weight;
            exp_log_prior += wgt * class_log_prior[c];
            total_w += wgt;
            m ^= lsb;
        }
    }
    if (total_w > 0.0f) {
        cell->prior = -exp_log_prior / total_w;
        cell->entropy += cell->prior * w->copula_weight;
    }
}

#endif /* WFC_ENABLE_COPULA */

/* --- Spectral field generation (FFTW) --- */
#if defined(WFC_USE_FFTW)

static inline void wfc_flandrin_spectrum(double* lambda, int n, double H)
{
    double pow_n_2H = pow((double)n, 2.0 * H);
    double twoH_plus_1 = 2.0 * H + 1.0;
    for (int k = 1; k < n; ++k) {
        double s = sin(M_PI * (double)k / (2.0 * (double)n));
        if (s < 1e-12) s = 1e-12;
        lambda[k] = pow_n_2H * pow(s, -twoH_plus_1);
    }
    if (n >= 3) lambda[0] = lambda[1] * (lambda[1] / lambda[2]);
    else if (n >= 2) lambda[0] = 2.0 * lambda[1];
}

static inline void wfc_fill_normal(double* buf, int n, uint64_t* rng)
{
    int i = 0;
    for (; i + 2 <= n; i += 2) {
        double u1 = wfc_randf(rng); if (u1 < 1e-7) u1 = 1e-7;
        double u2 = wfc_randf(rng);
        double r = sqrt(-2.0 * log(u1));
        double t = 2.0 * M_PI * u2;
        buf[i+0] = r * cos(t);
        buf[i+1] = r * sin(t);
    }
    if (i < n) {
        double u1 = wfc_randf(rng); if (u1 < 1e-7) u1 = 1e-7;
        double u2 = wfc_randf(rng);
        buf[i] = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    }
}

static inline void wfc_gumbel_link(double* buf, int n)
{
    for (int i = 0; i < n; ++i) {
        double u = 0.5 * erfc(-buf[i] * 0.7071067811865475);
        if (u < 1e-15) u = 1e-15;
        if (u > 1.0 - 1e-15) u = 1.0 - 1e-15;
        buf[i] = -log(-log(u));
    }
}

wfc_err_t wfc_set_gumbel_field(wfc_world_t* w, double H, uint64_t noise_seed)
{
    if (!w || !w->spectral_enabled) return WFC_EUNINIT;
    int nx = (int)w->nx_p2;
    int ny = (int)w->ny_p2;
    int n = nx * ny;

    /* BUG FIX (v2.0): Check for stale plans and cleanup properly */
    if (w->fftw_workspace && (w->fftw_nx != nx || w->fftw_ny != ny)) {
        if (w->fftw_plan_fwd) fftw_destroy_plan(w->fftw_plan_fwd);
        if (w->fftw_plan_inv) fftw_destroy_plan(w->fftw_plan_inv);
        fftw_free(w->fftw_workspace);
        w->fftw_workspace = NULL;
        w->fftw_plan_fwd = NULL;
        w->fftw_plan_inv = NULL;
    }

    if (!w->fftw_workspace) {
        w->fftw_workspace = (double*)fftw_malloc((size_t)n * sizeof(double));
        if (!w->fftw_workspace) return WFC_ENOMEM;
        w->fftw_plan_fwd = fftw_plan_r2r_2d(ny, nx, w->fftw_workspace, w->fftw_workspace,
                                             FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);
        w->fftw_plan_inv = fftw_plan_r2r_2d(ny, nx, w->fftw_workspace, w->fftw_workspace,
                                             FFTW_REDFT01, FFTW_REDFT01, FFTW_ESTIMATE);
        w->fftw_nx = nx;
        w->fftw_ny = ny;
    }

    double* lambda_x = (double*)WFC_MALLOC((size_t)nx * sizeof(double));
    double* lambda_y = (double*)WFC_MALLOC((size_t)ny * sizeof(double));
    if (!lambda_x || !lambda_y) { WFC_FREE(lambda_x); WFC_FREE(lambda_y); return WFC_ENOMEM; }

    wfc_flandrin_spectrum(lambda_x, nx, H);
    wfc_flandrin_spectrum(lambda_y, ny, H);

    uint64_t rng = noise_seed;
    wfc_fill_normal(w->fftw_workspace, n, &rng);

    fftw_execute(w->fftw_plan_fwd);

    for (int j = 0; j < ny; ++j) {
        double sy = sqrt(lambda_y[j]);
        for (int i = 0; i < nx; ++i) {
            double sx = sqrt(lambda_x[i]);
            w->fftw_workspace[j * nx + i] *= (sx * sy);
        }
    }

    fftw_execute(w->fftw_plan_inv);
    double norm = 1.0 / (4.0 * (double)nx * (double)ny);
    for (int i = 0; i < n; ++i) w->fftw_workspace[i] *= norm;

    wfc_gumbel_link(w->fftw_workspace, n);

    for (uint32_t i = 0; i < w->num_cells; ++i)
        w->spectral_field[i] = (float)w->fftw_workspace[i];

    WFC_FREE(lambda_x);
    WFC_FREE(lambda_y);
    return WFC_OK;
}

wfc_err_t wfc_set_matern_field(wfc_world_t* w, double kappa, double nu, uint64_t noise_seed)
{
    if (!w || !w->spectral_enabled) return WFC_EUNINIT;
    int nx = (int)w->nx_p2;
    int ny = (int)w->ny_p2;
    int n = nx * ny;

    if (w->fftw_workspace && (w->fftw_nx != nx || w->fftw_ny != ny)) {
        if (w->fftw_plan_fwd) fftw_destroy_plan(w->fftw_plan_fwd);
        if (w->fftw_plan_inv) fftw_destroy_plan(w->fftw_plan_inv);
        fftw_free(w->fftw_workspace);
        w->fftw_workspace = NULL;
        w->fftw_plan_fwd = NULL;
        w->fftw_plan_inv = NULL;
    }
    if (!w->fftw_workspace) {
        w->fftw_workspace = (double*)fftw_malloc((size_t)n * sizeof(double));
        if (!w->fftw_workspace) return WFC_ENOMEM;
        w->fftw_plan_fwd = fftw_plan_r2r_2d(ny, nx, w->fftw_workspace, w->fftw_workspace,
                                             FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);
        w->fftw_plan_inv = fftw_plan_r2r_2d(ny, nx, w->fftw_workspace, w->fftw_workspace,
                                             FFTW_REDFT01, FFTW_REDFT01, FFTW_ESTIMATE);
        w->fftw_nx = nx; w->fftw_ny = ny;
    }

    uint64_t rng = noise_seed;
    wfc_fill_normal(w->fftw_workspace, n, &rng);
    fftw_execute(w->fftw_plan_fwd);

    double kk = kappa * kappa;
    for (int j = 0; j < ny; ++j) {
        double sy = sin(M_PI * (double)j / (2.0 * (double)ny));
        double ly = 4.0 * sy * sy;
        for (int i = 0; i < nx; ++i) {
            double sx = sin(M_PI * (double)i / (2.0 * (double)nx));
            double lx = 4.0 * sx * sx;
            double base = kk + lx + ly;
            if (base < 1e-12) base = 1e-12;
            w->fftw_workspace[j * nx + i] *= pow(base, -nu);
        }
    }

    fftw_execute(w->fftw_plan_inv);
    double norm = 1.0 / (4.0 * (double)nx * (double)ny);
    for (int i = 0; i < n; ++i) w->fftw_workspace[i] *= norm;

    for (uint32_t i = 0; i < w->num_cells; ++i)
        w->spectral_field[i] = (float)w->fftw_workspace[i];

    return WFC_OK;
}

#endif /* WFC_USE_FFTW */

/* --- Pattern choice --- */
static inline uint16_t wfc_choose_pattern(wfc_world_t* w, const wfc_cell_t* cell)
{
    /* Weighted reservoir sampling from the valid mask. */
    float total = 0.0f;
    uint16_t choice = WFC_PATTERN_INVALID;
    for (int b = 0; b < WFC_MASK_U64S; ++b) {
        uint64_t m = cell->mask[b];
        while (m) {
            uint64_t lsb = m & -m;
            int bit = __builtin_ctzll(m) + (b << 6);
            float wt = w->patterns[bit].weight;
            total += wt;
            if (wfc_randf(&w->rng_state) * total < wt)
                choice = (uint16_t)bit;
            m ^= lsb;
        }
    }
    return choice;
}


/* --- Lazy entropy recomputation --- */
static inline void wfc_ensure_entropy_fresh(wfc_world_t* w, uint32_t cell_idx)
{
    wfc_cell_t* cell = &w->cells[cell_idx];
    if (!cell->dirty || cell->collapsed != WFC_PATTERN_INVALID) return;

    cell->entropy = wfc_cell_entropy(w, cell);
    cell->prior = 0.0f;

#if defined(WFC_USE_FFTW)
    if (w->spectral_enabled && cell_idx < w->num_cells)
        cell->entropy += w->spectral_field[cell_idx] * w->spectral_weight;
#endif

#if defined(WFC_ENABLE_COPULA)
    if (w->copula_enabled) wfc_copula_update_cell(w, cell_idx);
#endif

    cell->dirty = 0;
}

/* --- Tie-breaking: deterministic micro-noise based on coordinates --- */
static inline float wfc_tiebreak_noise(const wfc_world_t* w, uint32_t cell_idx)
{
    /* Use a simple hash of cell_idx and seed to create micro-perturbations
     * that break symmetry deterministically. */
    uint64_t h = w->rng_state ^ ((uint64_t)cell_idx * 0x9e3779b97f4a7c15ULL);
    h = (h ^ (h >> 33)) * 0xff51afd7ed558ccdULL;
    h = (h ^ (h >> 33)) * 0xc4ceb9fe1a85ec53ULL;
    h = h ^ (h >> 33);
    /* Return a tiny value in [-1e-6, 1e-6] */
    return ((float)(h & 0xFFFFF) / 1048576.0f - 0.5f) * 2e-6f;
}

/* --- Single collapse iteration (shared by wfc_collapse and wfc_step) --- */
static inline wfc_err_t wfc_collapse_one(wfc_world_t* w, uint32_t* out_cell_idx)
{
    uint32_t cell_idx = WFC_HEAP_INVALID;

    /* Find uncollapsed cell with minimum entropy */
    while (w->heap_size > 0) {
        cell_idx = wfc_heap_pop(w);
        if (cell_idx >= w->num_cells) {
            cell_idx = WFC_HEAP_INVALID;
            break;
        }
        wfc_cell_t* c = &w->cells[cell_idx];
        if (c->collapsed != WFC_PATTERN_INVALID) {
            cell_idx = WFC_HEAP_INVALID;
            continue;
        }
        /* Lazy entropy: recompute if dirty */
        wfc_ensure_entropy_fresh(w, cell_idx);
        break;
    }

    if (cell_idx == WFC_HEAP_INVALID) return WFC_OK; /* All collapsed */

    wfc_cell_t* cell = &w->cells[cell_idx];

#if defined(WFC_ENABLE_BACKTRACK)
    if (w->backtrack_enabled) wfc_snapshot_push(w);
#endif

#if defined(WFC_ENABLE_COPULA)
    if (w->copula_enabled) wfc_copula_update_cell(w, cell_idx);
#endif

    uint16_t pat = wfc_choose_pattern(w, cell);
    if (pat == WFC_PATTERN_INVALID) {
        /* Contradiction callback */
        if (w->on_contradiction) {
            uint32_t cx, cy, cz;
            wfc_cell_coords(w, cell_idx, &cx, &cy, &cz);
            w->on_contradiction(cx, cy, cz, w->contradiction_user_data);
        }
#if defined(WFC_ENABLE_BACKTRACK)
        if (w->backtrack_enabled && wfc_snapshot_pop(w)) {
            return WFC_OK; /* Will retry on next step */
        }
#endif
        return WFC_ECONTRADICTION;
    }

    cell->collapsed = pat;
    cell->entropy = -1.0f;
    cell->dirty = 0;
    w->class_placed[w->patterns[pat].class_id]++;

    /* Propagate */
    w->queue_head = w->queue_tail = 0;
    memset(w->in_queue, 0, w->num_cells * sizeof(uint8_t));
    for (int d = 0; d < (w->is_3d ? 6 : 4); ++d)
        wfc_queue_push(w, cell_idx);

    while (w->queue_head != w->queue_tail) {
        uint32_t cur = wfc_queue_pop(w);
        if (cur == WFC_HEAP_INVALID) break;
        wfc_propagate_from(w, cur);
#if defined(WFC_ENABLE_BACKTRACK)
        if (w->contradiction) break;
#endif
    }

#if defined(WFC_ENABLE_BACKTRACK)
    if (w->contradiction) {
        if (w->backtrack_enabled && wfc_snapshot_pop(w)) {
            w->class_placed[w->patterns[pat].class_id]--;
            return WFC_OK;
        }
        if (w->on_contradiction) {
            uint32_t cx, cy, cz;
            wfc_cell_coords(w, cell_idx, &cx, &cy, &cz);
            w->on_contradiction(cx, cy, cz, w->contradiction_user_data);
        }
        return WFC_ECONTRADICTION;
    }
#endif

    /* Hard class limits */
    wfc_enforce_class_limits(w);
#if defined(WFC_ENABLE_BACKTRACK)
    if (w->contradiction) {
        if (w->backtrack_enabled && wfc_snapshot_pop(w)) {
            w->class_placed[w->patterns[pat].class_id]--;
            return WFC_OK;
        }
        if (w->on_contradiction) {
            uint32_t cx, cy, cz;
            wfc_cell_coords(w, cell_idx, &cx, &cy, &cz);
            w->on_contradiction(cx, cy, cz, w->contradiction_user_data);
        }
        return WFC_ECONTRADICTION;
    }
#endif

    if (out_cell_idx) *out_cell_idx = cell_idx;
    return WFC_OK;
}

/* --- Core collapse loop --- */
wfc_err_t wfc_collapse(wfc_world_t* w)
{
    if (!w || !w->num_patterns) return WFC_EINVAL;

    w->step_mode = 0;
    w->step_collapsed_count = 0;
    w->step_target_collapses = w->nx * w->ny * w->nz;

    /* Initialize heap with all uncollapsed cells */
    w->heap_size = 0;
    for (uint32_t i = 0; i < w->num_cells; ++i) {
        wfc_cell_t* cell = &w->cells[i];
        cell->entropy = wfc_cell_entropy(w, cell);
        cell->prior = 0.0f;
        cell->dirty = 0;
        /* Tie-breaking: add micro-noise to prevent directional bias */
        cell->entropy += wfc_tiebreak_noise(w, i);
#if defined(WFC_USE_FFTW)
        if (w->spectral_enabled) cell->entropy += w->spectral_field[i] * w->spectral_weight;
#endif
        wfc_heap_insert(w, i);
    }

    while (w->step_collapsed_count < w->step_target_collapses) {
        uint32_t collapsed_cell;
        wfc_err_t err = wfc_collapse_one(w, &collapsed_cell);
        if (err != WFC_OK) return err;
        if (collapsed_cell == WFC_HEAP_INVALID) break; /* nothing left to collapse */
        w->step_collapsed_count++;
    }

    return WFC_OK;
}

/* --- Step-by-step execution API (new in 2.0) --- */
wfc_err_t wfc_step_init(wfc_world_t* w)
{
    if (!w || !w->num_patterns) return WFC_EINVAL;

    w->step_mode = 1;
    w->step_collapsed_count = 0;
    w->step_target_collapses = w->nx * w->ny * w->nz;

    w->heap_size = 0;
    for (uint32_t i = 0; i < w->num_cells; ++i) {
        wfc_cell_t* cell = &w->cells[i];
        cell->entropy = wfc_cell_entropy(w, cell);
        cell->prior = 0.0f;
        cell->dirty = 0;
        cell->entropy += wfc_tiebreak_noise(w, i);
#if defined(WFC_USE_FFTW)
        if (w->spectral_enabled) cell->entropy += w->spectral_field[i] * w->spectral_weight;
#endif
        wfc_heap_insert(w, i);
    }

    w->queue_head = w->queue_tail = 0;
    if (w->in_queue) memset(w->in_queue, 0, w->num_cells * sizeof(uint8_t));

    return WFC_OK;
}

wfc_err_t wfc_step(wfc_world_t* w)
{
    if (!w || !w->step_mode) return WFC_EINVAL;
    if (w->step_collapsed_count >= w->step_target_collapses) return WFC_OK;

    uint32_t collapsed_cell;
    wfc_err_t err = wfc_collapse_one(w, &collapsed_cell);
    if (err != WFC_OK) return err;
    if (collapsed_cell != WFC_HEAP_INVALID) {
        w->step_collapsed_count++;
    } else if (w->heap_size == 0) {
        /* No more cells in heap - mark as complete to prevent infinite loop */
        w->step_collapsed_count = w->step_target_collapses;
    }
    return WFC_OK;
}

int wfc_step_done(const wfc_world_t* w)
{
    if (!w) return 1;
    return w->step_collapsed_count >= w->step_target_collapses;
}

/* --- Contradiction callback --- */
void wfc_set_contradiction_callback(wfc_world_t* w, wfc_contradiction_cb_t cb, void* user_data)
{
    if (!w) return;
    w->on_contradiction = cb;
    w->contradiction_user_data = user_data;
}

/* --- API implementation --- */
wfc_world_t* wfc_world_create(const wfc_config_t* cfg)
{
    if (!cfg || !cfg->nx || !cfg->ny) return NULL;
    wfc_world_t* w = (wfc_world_t*)WFC_MALLOC(sizeof(wfc_world_t));
    if (!w) return NULL;
    memset(w, 0, sizeof(*w));

    w->nx = cfg->nx;
    w->ny = cfg->ny;
    w->nz = cfg->nz ? cfg->nz : 1;
    w->is_3d = (cfg->nz > 1);
    w->seamless = cfg->seamless;
    w->rng_state = cfg->seed ? cfg->seed : 0x123456789ABCDEF0ULL;

    w->nx_p2 = wfc_next_pow2_u32(w->nx);
    w->ny_p2 = wfc_next_pow2_u32(w->ny);
    w->nz_p2 = wfc_next_pow2_u32(w->nz);
    w->log2_nx = (uint32_t)__builtin_ctz(w->nx_p2);
    w->log2_ny = (uint32_t)__builtin_ctz(w->ny_p2);
    w->log2_nz = (uint32_t)__builtin_ctz(w->nz_p2);

    w->num_cells = w->nx_p2 * w->ny_p2 * w->nz_p2;

    w->cells = (wfc_cell_t*)WFC_MALLOC((size_t)w->num_cells * sizeof(wfc_cell_t));
    w->heap = (uint32_t*)WFC_MALLOC((size_t)w->num_cells * sizeof(uint32_t));
    w->heap_pos = (uint32_t*)WFC_MALLOC((size_t)w->num_cells * sizeof(uint32_t));
    w->queue = (uint32_t*)WFC_MALLOC((size_t)w->num_cells * sizeof(uint32_t));
    w->in_queue = (uint8_t*)WFC_MALLOC((size_t)w->num_cells * sizeof(uint8_t));
    if (!w->cells || !w->heap || !w->heap_pos || !w->queue || !w->in_queue) goto fail;

    w->queue_cap = w->num_cells;

    /* Initialize cells: all patterns valid */
    for (uint32_t i = 0; i < w->num_cells; ++i) {
        wfc_cell_t* c = &w->cells[i];
        c->collapsed = WFC_PATTERN_INVALID;
        c->entropy = 0.0f;
        c->prior = 0.0f;
        c->dirty = 0;
        for (int b = 0; b < WFC_MASK_U64S; ++b) c->mask[b] = ~0ULL;
        w->heap_pos[i] = WFC_HEAP_INVALID;
    }
    memset(w->in_queue, 0, w->num_cells * sizeof(uint8_t));

    /* Backtracking */
#if defined(WFC_ENABLE_BACKTRACK)
    w->backtrack_enabled = cfg->enable_backtracking;
    if (w->backtrack_enabled) {
        w->trail = (wfc_trail_entry_t*)WFC_MALLOC((size_t)w->num_cells * 8 * sizeof(wfc_trail_entry_t));
        w->trail_gen = (uint32_t*)WFC_MALLOC((size_t)w->num_cells * sizeof(uint32_t));
        if (!w->trail || !w->trail_gen) goto fail;
        w->trail_cap = w->num_cells * 8;
        memset(w->trail_gen, 0, (size_t)w->num_cells * sizeof(uint32_t));
    }
#endif

    /* Spectral */
#if defined(WFC_USE_FFTW)
    w->spectral_enabled = cfg->enable_spectral;
    w->spectral_weight = cfg->spectral_weight;
    if (w->spectral_enabled) {
        w->spectral_field = (float*)WFC_MALLOC((size_t)w->num_cells * sizeof(float));
        if (!w->spectral_field) goto fail;
        memset(w->spectral_field, 0, (size_t)w->num_cells * sizeof(float));
    }
    /* BUG FIX (v2.0): Zero-initialize FFTW plan pointers */
    w->fftw_plan_fwd = NULL;
    w->fftw_plan_inv = NULL;
    w->fftw_workspace = NULL;
    w->fftw_nx = 0;
    w->fftw_ny = 0;
#endif

    /* Copula */
#if defined(WFC_ENABLE_COPULA)
    w->copula_enabled = cfg->enable_copula;
    w->copula_weight = cfg->copula_weight;
    memset(w->copula_rho, 0, sizeof(w->copula_rho));
    memset(w->copula_marginal, 0, sizeof(w->copula_marginal));
    memset(w->copula_z, 0, sizeof(w->copula_z));
    memset(w->copula_class_pop, 0, sizeof(w->copula_class_pop));
#endif

    /* Class limits default to unlimited */
    for (int i = 0; i < WFC_MAX_CLASSES; ++i) {
        w->class_min[i] = 0;
        w->class_max[i] = 0xFFFFFFFFu;
    }

    return w;

fail:
    wfc_world_destroy(w);
    return NULL;
}

void wfc_world_destroy(wfc_world_t* w)
{
    if (!w) return;
#if defined(WFC_USE_FFTW)
    /* BUG FIX (v2.0): Properly cleanup FFTW resources */
    if (w->fftw_plan_fwd) fftw_destroy_plan(w->fftw_plan_fwd);
    if (w->fftw_plan_inv) fftw_destroy_plan(w->fftw_plan_inv);
    if (w->fftw_workspace) fftw_free(w->fftw_workspace);
    WFC_FREE(w->spectral_field);
#endif
#if defined(WFC_ENABLE_BACKTRACK)
    WFC_FREE(w->trail);
    WFC_FREE(w->trail_gen);
#endif
    WFC_FREE(w->cells);
    WFC_FREE(w->heap);
    WFC_FREE(w->heap_pos);
    WFC_FREE(w->queue);
    WFC_FREE(w->in_queue);
    WFC_FREE(w);
}

wfc_err_t wfc_add_pattern(wfc_world_t* w, const wfc_pattern_def_t* pat)
{
    if (!w || !pat || pat->id >= WFC_MAX_PATTERNS) return WFC_EINVAL;
    if (w->num_patterns >= WFC_MAX_PATTERNS) return WFC_EINVAL;
    w->patterns[pat->id] = *pat;
    if (pat->class_id < WFC_MAX_CLASSES) {
        w->class_mask[pat->class_id][pat->id >> 6] |= (1ULL << (pat->id & 63));
#if defined(WFC_ENABLE_COPULA)
        w->copula_class_pop[pat->class_id]++;
#endif
    }
    w->num_patterns++;
    if (pat->class_id >= w->num_classes) w->num_classes = pat->class_id + 1;
    return WFC_OK;
}

wfc_err_t wfc_add_rule(wfc_world_t* w, uint16_t pat_a, uint16_t pat_b, wfc_dir_t dir)
{
    if (!w || pat_a >= WFC_MAX_PATTERNS || pat_b >= WFC_MAX_PATTERNS) return WFC_EINVAL;
    if ((int)dir >= WFC_DIR_COUNT) return WFC_EINVAL;
    w->compat[dir][pat_a][pat_b >> 6] |= (1ULL << (pat_b & 63));
    /* Add symmetric rule */
    wfc_dir_t opp;
    switch (dir) {
        case WFC_DIR_N: opp = WFC_DIR_S; break;
        case WFC_DIR_S: opp = WFC_DIR_N; break;
        case WFC_DIR_E: opp = WFC_DIR_W; break;
        case WFC_DIR_W: opp = WFC_DIR_E; break;
        case WFC_DIR_U: opp = WFC_DIR_D; break;
        case WFC_DIR_D: opp = WFC_DIR_U; break;
        default: return WFC_EINVAL;
    }
    w->compat[opp][pat_b][pat_a >> 6] |= (1ULL << (pat_a & 63));
    return WFC_OK;
}

wfc_err_t wfc_set_class_limit(wfc_world_t* w, uint16_t class_id, uint32_t min_count, uint32_t max_count)
{
    if (!w || class_id >= WFC_MAX_CLASSES) return WFC_EINVAL;
    w->class_min[class_id] = min_count;
    w->class_max[class_id] = max_count;
    w->class_constrained |= (1u << class_id);
    return WFC_OK;
}

wfc_err_t wfc_learn_from_sample(wfc_world_t* w, const uint16_t* sample_grid,
                                uint32_t sx, uint32_t sy, uint32_t sz)
{
    if (!w || !sample_grid || !sx || !sy) return WFC_EINVAL;

    /* Count frequencies and adjacencies */
    uint32_t freq[WFC_MAX_PATTERNS] = {0};
    uint32_t class_freq[WFC_MAX_CLASSES] = {0};
    uint32_t total = 0;

    for (uint32_t z = 0; z < sz; ++z) {
        for (uint32_t y = 0; y < sy; ++y) {
            for (uint32_t x = 0; x < sx; ++x) {
                uint16_t p = sample_grid[(z * sy + y) * sx + x];
                if (p >= WFC_MAX_PATTERNS) continue;
                freq[p]++;
                total++;
                uint16_t c = w->patterns[p].class_id;
                if (c < WFC_MAX_CLASSES) class_freq[c]++;
            }
        }
    }

    /* Set weights from frequencies */
    for (int i = 0; i < WFC_MAX_PATTERNS; ++i) {
        if (freq[i] && w->patterns[i].freq == 0) {
            w->patterns[i].freq = freq[i];
            w->patterns[i].weight = (float)freq[i];
        }
    }

    /* Normalize weights */
    float wsum = 0.0f;
    for (int i = 0; i < WFC_MAX_PATTERNS; ++i) wsum += w->patterns[i].weight;
    if (wsum > 0.0f) {
        for (int i = 0; i < WFC_MAX_PATTERNS; ++i)
            w->patterns[i].weight /= wsum;
    }

    /* Precompute pattern_plogp */
    for (int i = 0; i < WFC_MAX_PATTERNS; ++i) {
        float p = w->patterns[i].weight;
        w->pattern_plogp[i] = (p > 0.0f) ? -p * log2f(p) : 0.0f;
    }

    /* Learn adjacencies */
    for (uint32_t z = 0; z < sz; ++z) {
        for (uint32_t y = 0; y < sy; ++y) {
            for (uint32_t x = 0; x < sx; ++x) {
                uint16_t p = sample_grid[(z * sy + y) * sx + x];
                if (p >= WFC_MAX_PATTERNS) continue;
                const int ndirs = (sz > 1) ? 6 : 4;
                for (int d = 0; d < ndirs; ++d) {
                    uint32_t nx = x, ny = y, nz = z;
                    switch (d) {
                        case WFC_DIR_N: if (y > 0) ny--; else continue; break;
                        case WFC_DIR_S: if (y + 1 < sy) ny++; else continue; break;
                        case WFC_DIR_E: if (x + 1 < sx) nx++; else continue; break;
                        case WFC_DIR_W: if (x > 0) nx--; else continue; break;
                        case WFC_DIR_U: if (z > 0) nz--; else continue; break;
                        case WFC_DIR_D: if (z + 1 < sz) nz++; else continue; break;
                    }
                    uint16_t q = sample_grid[(nz * sy + ny) * sx + nx];
                    if (q < WFC_MAX_PATTERNS) wfc_add_rule(w, p, q, (wfc_dir_t)d);
                }
            }
        }
    }

#if defined(WFC_ENABLE_COPULA)
    /* Learn copula marginals and precompute Z values */
    for (int c = 0; c < WFC_MAX_CLASSES; ++c) {
        w->copula_marginal[c] = (total > 0) ? (float)class_freq[c] / (float)total : 0.0f;
        /* PERFORMANCE FIX (v2.0): Precompute inv_cdf for each class */
        w->copula_z[c] = wfc_inv_cdf_normal(w->copula_marginal[c]);
    }
#endif

    return WFC_OK;
}

uint16_t wfc_get_cell_pattern(const wfc_world_t* w, uint32_t x, uint32_t y, uint32_t z)
{
    if (!w) return WFC_PATTERN_INVALID;
    if (x >= w->nx || y >= w->ny || z >= w->nz) return WFC_PATTERN_INVALID;
    uint32_t idx = wfc_idx(w, x, y, z);
    return w->cells[idx].collapsed;
}

const wfc_pattern_def_t* wfc_get_pattern(const wfc_world_t* w, uint16_t pat_id)
{
    if (!w || pat_id >= WFC_MAX_PATTERNS) return NULL;
    return &w->patterns[pat_id];
}

wfc_err_t wfc_export_grid(const wfc_world_t* w, uint16_t* out_grid, size_t out_count)
{
    if (!w || !out_grid) return WFC_EINVAL;
    uint32_t total = w->nx * w->ny * w->nz;
    if (out_count < total) return WFC_EINVAL;

    for (uint32_t z = 0; z < w->nz; ++z) {
        for (uint32_t y = 0; y < w->ny; ++y) {
            for (uint32_t x = 0; x < w->nx; ++x) {
                uint32_t idx = (z * w->ny + y) * w->nx + x;
                out_grid[idx] = wfc_get_cell_pattern(w, x, y, z);
            }
        }
    }
    return WFC_OK;
}

void wfc_world_reset(wfc_world_t* w)
{
    if (!w) return;

    /* Reset cells to uncollapsed with all patterns valid */
    for (uint32_t i = 0; i < w->num_cells; ++i) {
        wfc_cell_t* c = &w->cells[i];
        c->collapsed = WFC_PATTERN_INVALID;
        c->entropy = 0.0f;
        c->prior = 0.0f;
        c->dirty = 0;
        for (int b = 0; b < WFC_MASK_U64S; ++b) c->mask[b] = ~0ULL;
        w->heap_pos[i] = WFC_HEAP_INVALID;
    }

    memset(w->class_placed, 0, sizeof(w->class_placed));
    w->heap_size = 0;
    w->queue_head = w->queue_tail = 0;
    if (w->in_queue) memset(w->in_queue, 0, w->num_cells * sizeof(uint8_t));
    w->step_mode = 0;
    w->step_collapsed_count = 0;

#if defined(WFC_ENABLE_BACKTRACK)
    w->trail_top = 0;
    w->snapshot_top = 0;
    w->contradiction = 0;
    w->trail_current_gen = 0;
    if (w->trail_gen) memset(w->trail_gen, 0, w->num_cells * sizeof(uint32_t));
#endif
}

#endif /* WFC_IMPLEMENTATION */
