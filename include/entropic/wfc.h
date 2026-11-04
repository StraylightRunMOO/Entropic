/* wfc.h — Entropic: Wave Function Collapse HPC Engine
 * Single-header C11. Zero heap traffic on the hot path.
 * Version 1.1.0
 *
 * Compile-time configuration (define BEFORE including this header):
 *   WFC_MAX_PATTERNS          — must be power of 2, <= 256 (default 256)
 *   WFC_MAX_CLASSES           — tile classes for global constraints (default 64)
 *   WFC_MAX_LAYERS            — per-pattern layer slots (default 4)
 *   WFC_MAX_BACKTRACK_DEPTH   — snapshot stack depth (default 64)
 *   WFC_ENABLE_AVX2           — force AVX2 bitset paths
 *   WFC_ENABLE_NEON           — force NEON bitset paths
 *   WFC_USE_FAF               — enable spectral Gumbel / Matérn fields via FastAndFourier DWT
 *   WFC_ENABLE_HIERARCHY      — enable residual multiscale hierarchy (requires WFC_USE_FAF)
 *   WFC_ENABLE_COPULA         — enable Gaussian copula entropy biasing
 *   WFC_ENABLE_BACKTRACK      — enable delta-trail backtracking
 *   WFC_MALLOC / WFC_FREE / WFC_REALLOC — allocator overrides
 *   WFC_ALIGNED_ALLOC / WFC_ALIGNED_FREE — 64-byte cell-array allocator
 *   WFC_CELL_ALIGN                        — cell alignment (default 64)
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

/* posix_memalign needs _POSIX_C_SOURCE >= 200112L. Defining that alone
 * turns off glibc's default feature set, which hides usleep and the other
 * misc declarations suspenders.h uses. _DEFAULT_SOURCE puts them back.
 * Both must be set before any libc header. */
#if !defined(_WIN32)
#  if !defined(_POSIX_C_SOURCE) && !defined(_XOPEN_SOURCE)
#    define _POSIX_C_SOURCE 200112L
#  endif
#  ifndef _DEFAULT_SOURCE
#    define _DEFAULT_SOURCE 1
#  endif
#endif

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
#if (WFC_MAX_CLASSES) > 64
#error "WFC_MAX_CLASSES must be <= 64 (class_constrained is a uint64_t bitset)"
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

#ifndef WFC_CELL_ALIGN
#define WFC_CELL_ALIGN 64
#endif

#ifdef __cplusplus
#define WFC_ALIGNAS(n) alignas(n)
#else
#define WFC_ALIGNAS(n) _Alignas(n)
#endif

/* ============================================================
 * 2. SIMD / ISA detection
 * ============================================================
 * A pattern mask is at most 256 bits (WFC_MAX_PATTERNS <= 256): one AVX2
 * register, or two NEON registers. AVX-512 does not widen that op, so it
 * is not used here. The paths follow the ISA the TU was actually compiled
 * for. Define WFC_DISABLE_AVX2 or WFC_DISABLE_NEON to force the scalar loop.
 */
#if !defined(WFC_DISABLE_AVX2) && defined(__AVX2__)
#include <immintrin.h>
#define WFC_HAS_AVX2 1
#endif
#if !defined(WFC_DISABLE_NEON) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define WFC_HAS_NEON 1
#endif

/* ============================================================
 * 3. Optional wavelet spectral backend (FastAndFourier)
 * ============================================================ */
#ifdef WFC_USE_FAF
#include <fastandfourier.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#endif

#if defined(WFC_ENABLE_HIERARCHY) && !defined(WFC_USE_FAF)
#error "WFC_ENABLE_HIERARCHY requires WFC_USE_FAF"
#endif

/* ============================================================
 * 3b. Optional coroutine parallelism (suspenders.h)
 * ============================================================ */
#if defined(WFC_ENABLE_SUSPENDERS) && !defined(WFC_USE_FAF)
#error "WFC_ENABLE_SUSPENDERS requires WFC_USE_FAF"
#endif

#include <stdint.h>
#include <string.h>
#include <float.h>
#include <math.h>

#ifdef WFC_ENABLE_SUSPENDERS
#ifdef WFC_IMPLEMENTATION
#ifndef MEMENTO_IMPLEMENTATION
#define MEMENTO_IMPLEMENTATION
#endif
#ifndef SUSPENDERS_IMPLEMENTATION
#define SUSPENDERS_IMPLEMENTATION
#endif
#endif /* WFC_IMPLEMENTATION */
#include <memento.h>
#include <suspenders.h>
#include <sched.h>

/* C11 <stdatomic.h> names do not exist in C++. Include <stdint.h> first so
 * uint_fast32_t is visible, then route every op through WFC_ATOMIC_*. */
#ifndef __cplusplus
#include <stdatomic.h>
typedef atomic_uint_fast32_t wfc_atomic_u32_t;
typedef atomic_int           wfc_atomic_int_t;
#define WFC_ATOMIC_STORE(p, v, mo)     atomic_store_explicit((p), (v), memory_order_##mo)
#define WFC_ATOMIC_LOAD(p, mo)         atomic_load_explicit((p), memory_order_##mo)
#define WFC_ATOMIC_FETCH_SUB(p, v, mo) atomic_fetch_sub_explicit((p), (v), memory_order_##mo)
#define WFC_ATOMIC_EXCHANGE(p, v, mo)  atomic_exchange_explicit((p), (v), memory_order_##mo)
#else
#include <cstdint>
#include <atomic>
typedef std::atomic<uint_fast32_t> wfc_atomic_u32_t;
typedef std::atomic<int>           wfc_atomic_int_t;
#define WFC_ATOMIC_STORE(p, v, mo)     (p)->store((v), std::memory_order_##mo)
#define WFC_ATOMIC_LOAD(p, mo)         (p)->load(std::memory_order_##mo)
#define WFC_ATOMIC_FETCH_SUB(p, v, mo) (p)->fetch_sub((v), std::memory_order_##mo)
#define WFC_ATOMIC_EXCHANGE(p, v, mo)  (p)->exchange((v), std::memory_order_##mo)
#endif
#ifndef WFC_PAR_MIN_DIM
#define WFC_PAR_MIN_DIM 32
#endif
#endif /* WFC_ENABLE_SUSPENDERS */

/* C/C++ compatibility */
#ifdef __cplusplus
#define WFC_RESTRICT __restrict__
#else
#define WFC_RESTRICT restrict
#endif

#ifdef __cplusplus
#define WFC_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define WFC_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
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

/* Grid cell. Alignment is pinned three ways because `_Alignas` on a struct
 * tag is ignored by clang, and GCC has discarded a lone member `_Alignas`
 * (sizeof 48 in C vs 64 in C++). Type-level alignas/attribute, first-member
 * alignas, and explicit padding to a 64-byte stride must all agree.
 * Allocate arrays with the aligned allocator in wfc_world_create. */
#define WFC_CELL_PAYLOAD_BYTES ((WFC_MASK_U64S) * 8 + 13)
#define WFC_CELL_PAD_BYTES \
    (((WFC_CELL_PAYLOAD_BYTES) % (WFC_CELL_ALIGN)) \
        ? ((WFC_CELL_ALIGN) - ((WFC_CELL_PAYLOAD_BYTES) % (WFC_CELL_ALIGN))) \
        : 0)

#if defined(__cplusplus)
struct alignas(WFC_CELL_ALIGN) wfc_cell
#elif defined(_MSC_VER)
__declspec(align(64)) struct wfc_cell
#elif defined(__GNUC__) || defined(__clang__)
struct __attribute__((aligned(WFC_CELL_ALIGN))) wfc_cell
#else
struct wfc_cell
#endif
{
    WFC_ALIGNAS(WFC_CELL_ALIGN) uint64_t mask[WFC_MASK_U64S];
    uint16_t collapsed;                 /* WFC_PATTERN_INVALID if uncollapsed */
    uint16_t reserved;
    float    entropy;                   /* cached Shannon entropy */
    float    prior;                     /* copula/spectral prior offset */
    uint8_t  dirty;                     /* entropy needs recomputation */
#if WFC_CELL_PAD_BYTES > 0
    uint8_t  _pad[WFC_CELL_PAD_BYTES];
#endif
};
typedef struct wfc_cell wfc_cell_t;

WFC_STATIC_ASSERT(sizeof(wfc_cell_t) % WFC_CELL_ALIGN == 0,
    "wfc_cell_t size must be a multiple of WFC_CELL_ALIGN for aligned_alloc");
WFC_STATIC_ASSERT(sizeof(wfc_cell_t) >= WFC_CELL_ALIGN,
    "wfc_cell_t must occupy at least one aligned slot");
#ifdef __cplusplus
WFC_STATIC_ASSERT(alignof(wfc_cell_t) == WFC_CELL_ALIGN,
    "wfc_cell_t alignment must match WFC_CELL_ALIGN");
#else
WFC_STATIC_ASSERT(_Alignof(wfc_cell_t) == WFC_CELL_ALIGN,
    "wfc_cell_t alignment must match WFC_CELL_ALIGN");
#endif

/* World configuration */
typedef struct {
    uint32_t nx, ny, nz;                /* requested output dimensions (nz=1 for 2D) */
    uint32_t seed;
    int      enable_backtracking;       /* requires WFC_ENABLE_BACKTRACK */
    int      enable_spectral;           /* requires WFC_USE_FAF */
    int      enable_copula;             /* requires WFC_ENABLE_COPULA */
    int      seamless;                  /* toroidal boundary conditions */
    float    spectral_weight;           /* scale of Gumbel/Matérn prior */
    float    copula_weight;             /* scale of copula prior */
} wfc_config_t;

/* Multiscale hierarchy types */
#if defined(WFC_ENABLE_HIERARCHY)
typedef struct wfc_scale {
    uint32_t nx, ny;
    uint32_t nx_p2, ny_p2;
    uint32_t ncells;
    float   *residual_field;            /* signature observation (Gaussian target) */
    float   *recon_field;               /* 2x upsample of coarser coarse_codes */
    /* Mallat details owned by this level. detail_lh/hl/hh alias the primary
     * octave inside detail_pack (band size nx/2 by ny/2) when that octave
     * belongs here. detail_pack holds every owned LH/HL/HH sample. */
    float   *detail_lh;
    float   *detail_hl;
    float   *detail_hh;
    float   *detail_pack;
    uint32_t detail_count;              /* live samples in detail_pack */
    uint32_t detail_cap;
    uint32_t octave_lo;                 /* inclusive, 0 = finest Mallat octave */
    uint32_t octave_hi;                 /* exclusive */
    uint16_t *coarse_codes;             /* collapsed pattern per cell at this scale */
    float    weight;                    /* kept; the cascade no longer blends by it */
} wfc_scale_t;

typedef struct wfc_hierarchy {
    uint32_t       nlevels;
    wfc_scale_t   *levels;
    uint32_t       finest_idx;
    int            wavelet_family;
    uint32_t       wavelet_levels;      /* 0 = full residual depth */
    int            mode;                /* 0 coarse-to-fine, 1 fine-to-coarse */
    int            decomposed;          /* 1: adopt_spectral already split the field */
    float         *target_field;
    float         *running;             /* z^(k): target minus synthesized contributions */
} wfc_hierarchy_t;
#endif

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
    float pattern_plogp[WFC_MAX_PATTERNS]; /* -p*log2(p); kept for readers */
    float pattern_wlogw[WFC_MAX_PATTERNS]; /* p*log2(p), 0 when p <= 0 */
    uint64_t support_mask[WFC_MASK_U64S];  /* patterns with weight > 0 */
    /* OR of compat[dir][p] over the support. Used when a cell still holds it. */
    uint64_t compat_support[WFC_DIR_COUNT][WFC_MASK_U64S];

    /* Precomputed class bitmasks */
    uint64_t class_mask[WFC_MAX_CLASSES][WFC_MASK_U64S];

    /* Compatibility: compat[dir][pat][block] = bitmask of patterns compatible with pat in dir. */
    uint64_t compat[WFC_DIR_COUNT][WFC_MAX_PATTERNS][WFC_MASK_U64S];

    /* Class hard constraints */
    uint32_t class_min[WFC_MAX_CLASSES];
    uint32_t class_max[WFC_MAX_CLASSES];
    uint32_t class_placed[WFC_MAX_CLASSES];
    uint64_t class_constrained;         /* bit i set iff class i has a non-default limit */

    /* Min-entropy heap (binary min-heap of cell indices) */
    uint32_t* heap;                     /* size num_cells */
    uint32_t* heap_pos;                 /* cell -> index in heap, WFC_HEAP_INVALID if collapsed */
    uint32_t heap_size;

    /* Propagation queue (ring buffer). head == tail means empty, so the
     * capacity is 2*num_cells: one spare slot, and still a power of two.
     * in_queue keeps occupancy at or below num_cells. */
    uint32_t* queue;
    uint32_t  queue_head, queue_tail, queue_cap;
    uint8_t*  in_queue;

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

    /* Spectral fields (FastAndFourier DWT) */
#if defined(WFC_USE_FAF)
    float*          spectral_field;     /* additive entropy bias per cell */
    faf_transform*  faf_row_fwd;
    faf_transform*  faf_row_inv;
    faf_transform*  faf_col_fwd;
    faf_transform*  faf_col_inv;
    float*          faf_workspace;      /* 2 * nx_p2 * ny_p2 interleaved */
    float*          faf_col_buf;        /* 2 * max(nx_p2, ny_p2) temp */
    int             faf_initialized;
    int             faf_wavelet_family; /* FAF_WAVELET_*; default CDF 9/7 */
    uint32_t        faf_wavelet_levels; /* 0 = full decomposition */
    int             faf_live_family;    /* family the live transforms were built with */
    int             faf_row_shared;     /* row plans live in the process cache */
    int             faf_col_shared;     /* col plans live in the process cache */
    int             spectral_guides_choice; /* 1: spectral_field is a signature target */
#endif

    /* Copula */
#if defined(WFC_ENABLE_COPULA)
    /* Factorized Gaussian copula: rho[dir][c1][c2] for class pairs. */
    float copula_rho[WFC_DIR_COUNT][WFC_MAX_CLASSES][WFC_MAX_CLASSES];
    float copula_marginal[WFC_MAX_CLASSES]; /* empirical CDF value per class */
    float copula_z[WFC_MAX_CLASSES];        /* PRECOMPUTED: Phi^{-1}(marginal[c]) */
    uint8_t copula_class_pop[WFC_MAX_CLASSES]; /* number of patterns in class */
#endif

    /* Multiscale hierarchy */
#if defined(WFC_ENABLE_HIERARCHY)
    wfc_hierarchy_t *hierarchy;
    int              owns_compat;
#endif

    /* Coroutine parallelism */
#if defined(WFC_ENABLE_SUSPENDERS)
    int              susp_initialized;
    unsigned         susp_num_workers;
    float           *faf_par_scratch;    /* 4 * nx_p2 * ny_p2, column DWT pool */
#endif

    /* RNG */
    uint64_t rng_seed;                  /* original seed; restored by wfc_world_reset */
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

/* Pin one logical cell to `pat` before the solve. A zero-weight pattern is
 * legal here and stays out of the support, so the solver will not choose it
 * anywhere else. The opening AC-3 propagates from pinned cells. A second pin
 * on the same cell returns WFC_EINVAL; reset the world first. */
wfc_err_t wfc_fix_cell(wfc_world_t* w, uint32_t x, uint32_t y, uint32_t z, uint16_t pat);

/* Learn rules, frequencies, and copula correlations from a sample grid. */
wfc_err_t wfc_learn_from_sample(wfc_world_t* w, const uint16_t* sample_grid,
                                uint32_t sx, uint32_t sy, uint32_t sz);

/* Spectral priors (FAF DWT). Call before wfc_collapse. */
#if defined(WFC_USE_FAF)
wfc_err_t wfc_set_gumbel_field(wfc_world_t* w, double H, uint64_t noise_seed);
wfc_err_t wfc_set_matern_field(wfc_world_t* w, double kappa, double nu, uint64_t noise_seed);
#endif

/* Multiscale hierarchy. Call wfc_hierarchy_init, then optionally set_wavelet/set_target/set_mode, then collapse.
 * set_wavelet selects the DWT family and the residual decomposition depth. dwt_levels == 0
 * decomposes every octave the padded grid can hold; any other value is that many octaves.
 * The 2× step between adjacent hierarchy scales stays one Mallat octave (signature upsample).
 * Detail subbands are retained on each scale and steer the next finer observation.
 * hierarchy_collapse requires spectral fields: the residual is a signature target.
 * Observe order is the Shannon entropy of the Gaussian-reweighted posterior, and pattern
 * choice samples that posterior. Without spectral the call returns WFC_EUNINIT.
 * set_mode: 0 = coarse-to-fine generation (default), 1 = fine-to-coarse analysis.
 * residual_energy is the L2 of the detail samples stored at that level (0 if level is out of range).
 * adopt_spectral copies the live fBm/Matérn field and decomposes it once into the
 * detail buffers. That split is z^(0). A spectral field is not a target until
 * this call or set_target. set_wavelet and set_target drop a previous split. */
#if defined(WFC_ENABLE_HIERARCHY)
wfc_err_t wfc_hierarchy_init(wfc_world_t* w, uint32_t nlevels);
wfc_err_t wfc_hierarchy_set_wavelet(wfc_world_t* w, int family, uint32_t dwt_levels);
wfc_err_t wfc_hierarchy_set_target(wfc_world_t* w, const float* target, uint32_t count);
wfc_err_t wfc_hierarchy_adopt_spectral(wfc_world_t* w);
wfc_err_t wfc_hierarchy_set_mode(wfc_world_t* w, int mode);
float     wfc_hierarchy_residual_energy(const wfc_world_t* w, uint32_t level);
wfc_err_t wfc_hierarchy_collapse(wfc_world_t* w);
void      wfc_hierarchy_destroy(wfc_world_t* w);
#endif

/* Coroutine parallelism. Call before spectral or hierarchy work that should
 * use the parallel DWT. Worlds share one runtime; the worker count is fixed
 * by the first init (0 means 4). The last shutdown, including the one from
 * wfc_world_destroy, stops that runtime. */
#if defined(WFC_ENABLE_SUSPENDERS)
wfc_err_t wfc_parallel_init(wfc_world_t* w, unsigned num_workers);
void      wfc_parallel_shutdown(wfc_world_t* w);
#endif

/* Solve: full collapse or step-by-step */
wfc_err_t wfc_collapse(wfc_world_t* w);

/* Step-by-step execution */
wfc_err_t wfc_step_init(wfc_world_t* w);
wfc_err_t wfc_step(wfc_world_t* w);
int       wfc_step_done(const wfc_world_t* w);

/* Contradiction callback */
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

#if defined(_WIN32)
#include <malloc.h>
#endif

/* --- Utilities --- */
static inline uint32_t wfc_next_pow2_u32(uint32_t v)
{
    if (v <= 1) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

/* 64-byte (or WFC_CELL_ALIGN) allocation for the cell array.
 * Override with WFC_ALIGNED_ALLOC / WFC_ALIGNED_FREE if you ship a custom heap. */
#ifndef WFC_ALIGNED_ALLOC
static inline void* wfc_alloc_aligned(size_t align, size_t bytes)
{
    if (bytes == 0) return NULL;
    if (align < sizeof(void*)) align = sizeof(void*);
    size_t n = (bytes + align - 1) / align * align;
#if defined(_WIN32)
    return _aligned_malloc(n, align);
#else
    void* p = NULL;
    if (posix_memalign(&p, align, n) != 0) return NULL;
    return p;
#endif
}
#ifndef WFC_ALIGNED_FREE
#if defined(_WIN32)
#define WFC_ALIGNED_FREE(p) _aligned_free(p)
#else
#define WFC_ALIGNED_FREE(p) free(p)
#endif
#endif
#else
#define wfc_alloc_aligned(align, bytes) WFC_ALIGNED_ALLOC((align), (bytes))
#ifndef WFC_ALIGNED_FREE
#define WFC_ALIGNED_FREE(p) WFC_FREE(p)
#endif
#endif

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

/* Class bitset: 1ULL so class_id 32..63 is defined (1u << c is UB at 32). */
static inline uint64_t wfc_class_bit(unsigned c)
{
    return (c < 64u) ? (UINT64_C(1) << c) : UINT64_C(0);
}

static inline void wfc_note_class_placed(wfc_world_t* w, uint16_t pat, int delta)
{
    uint16_t c = w->patterns[pat].class_id;
    if (c >= (uint16_t)WFC_MAX_CLASSES) return;
    if (delta > 0) {
        w->class_placed[c]++;
    } else if (w->class_placed[c] > 0) {
        w->class_placed[c]--;
    }
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

static inline void wfc_mask_or(uint64_t* WFC_RESTRICT dst, const uint64_t* WFC_RESTRICT src)
{
#if defined(WFC_HAS_AVX2) && WFC_MASK_U64S == 4
    __m256i d = _mm256_loadu_si256((const __m256i*)dst);
    __m256i s = _mm256_loadu_si256((const __m256i*)src);
    _mm256_storeu_si256((__m256i*)dst, _mm256_or_si256(d, s));
#elif defined(WFC_HAS_NEON) && WFC_MASK_U64S == 4
    uint64x2_t d0 = vld1q_u64(dst + 0);
    uint64x2_t d1 = vld1q_u64(dst + 2);
    uint64x2_t s0 = vld1q_u64(src + 0);
    uint64x2_t s1 = vld1q_u64(src + 2);
    vst1q_u64(dst + 0, vorrq_u64(d0, s0));
    vst1q_u64(dst + 2, vorrq_u64(d1, s1));
#else
    for (int i = 0; i < WFC_MASK_U64S; ++i) dst[i] |= src[i];
#endif
}

static inline void wfc_mask_clear(uint64_t* WFC_RESTRICT m)
{
#if defined(WFC_HAS_AVX2) && WFC_MASK_U64S == 4
    _mm256_storeu_si256((__m256i*)m, _mm256_setzero_si256());
#elif defined(WFC_HAS_NEON) && WFC_MASK_U64S == 4
    uint64x2_t z = vdupq_n_u64(0);
    vst1q_u64(m + 0, z);
    vst1q_u64(m + 2, z);
#else
    for (int i = 0; i < WFC_MASK_U64S; ++i) m[i] = 0;
#endif
}

static inline int wfc_mask_empty(const uint64_t* WFC_RESTRICT m)
{
#if defined(WFC_HAS_AVX2) && WFC_MASK_U64S == 4
    __m256i v = _mm256_loadu_si256((const __m256i*)m);
    return _mm256_testz_si256(v, v);
#elif defined(WFC_HAS_NEON) && WFC_MASK_U64S == 4
    uint64x2_t v0 = vld1q_u64(m + 0);
    uint64x2_t v1 = vld1q_u64(m + 2);
    uint64x2_t o = vorrq_u64(v0, v1);
    return (vgetq_lane_u64(o, 0) | vgetq_lane_u64(o, 1)) == 0;
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
#elif defined(WFC_HAS_NEON) && WFC_MASK_U64S == 4
    uint64x2_t d0 = veorq_u64(vld1q_u64(a + 0), vld1q_u64(b + 0));
    uint64x2_t d1 = veorq_u64(vld1q_u64(a + 2), vld1q_u64(b + 2));
    uint64x2_t o = vorrq_u64(d0, d1);
    return (vgetq_lane_u64(o, 0) | vgetq_lane_u64(o, 1)) == 0;
#else
    for (int i = 0; i < WFC_MASK_U64S; ++i) if (a[i] != b[i]) return 0;
    return 1;
#endif
}

static inline uint32_t wfc_mask_popcnt(const uint64_t* WFC_RESTRICT m)
{
    /* Four hardware popcnts beat a nibble-LUT over one 256-bit register.
     * On NEON, vcnt is the hardware popcnt. */
#if defined(WFC_HAS_NEON) && WFC_MASK_U64S == 4
    uint8x16_t v0 = vreinterpretq_u8_u64(vld1q_u64(m + 0));
    uint8x16_t v1 = vreinterpretq_u8_u64(vld1q_u64(m + 2));
    return vaddvq_u8(vaddq_u8(vcntq_u8(v0), vcntq_u8(v1)));
#else
    uint32_t c = 0;
    for (int i = 0; i < WFC_MASK_U64S; ++i)
        c += (uint32_t)__builtin_popcountll(m[i]);
    return c;
#endif
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

/* --- Weights, support, Shannon entropy ---
 * H(mask) = log2(Z) - (Σ w_i log2 w_i) / Z, Z = Σ w_i over the mask.
 * A singleton is exactly 0. Summing -p log p over a subset is not this.
 */
static inline void wfc_prepare_weights(wfc_world_t* w)
{
    wfc_mask_clear(w->support_mask);
    for (int d = 0; d < WFC_DIR_COUNT; ++d)
        wfc_mask_clear(w->compat_support[d]);

    for (int i = 0; i < WFC_MAX_PATTERNS; ++i) {
        /* Ids are not required to be a dense prefix. Unadded slots stay weight 0. */
        float p = w->patterns[i].weight;
        if (p > 0.0f) {
            w->support_mask[i >> 6] |= (UINT64_C(1) << (i & 63));
            w->pattern_wlogw[i] = p * log2f(p);
            w->pattern_plogp[i] = -w->pattern_wlogw[i];
            for (int d = 0; d < WFC_DIR_COUNT; ++d)
                wfc_mask_or(w->compat_support[d], w->compat[d][i]);
        } else {
            w->pattern_wlogw[i] = 0.0f;
            w->pattern_plogp[i] = 0.0f;
        }
    }
}

static inline float wfc_cell_entropy(const wfc_world_t* w, const wfc_cell_t* cell)
{
    if (cell->collapsed != WFC_PATTERN_INVALID) return -1.0f;
    float sum = 0.0f;
    float wlog = 0.0f;
    for (int b = 0; b < WFC_MASK_U64S; ++b) {
        uint64_t m = cell->mask[b];
        while (m) {
            uint64_t lsb = m & -m;
            int bit = __builtin_ctzll(m) + (b << 6);
            float wt = (bit < WFC_MAX_PATTERNS) ? w->patterns[bit].weight : 0.0f;
            if (wt > 0.0f) {
                sum += wt;
                wlog += w->pattern_wlogw[bit];
            }
            m ^= lsb;
        }
    }
    if (sum <= 0.0f) return 0.0f;
    return log2f(sum) - wlog / sum;
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
static inline void wfc_refresh_entropy(wfc_world_t* w, uint32_t cell_idx);

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

/* Logical cell: padding is pre-collapsed and must not enter AC-3. */
static inline int wfc_cell_logical(const wfc_world_t* w, uint32_t cell_idx)
{
    uint32_t x, y, z;
    wfc_cell_coords(w, cell_idx, &x, &y, &z);
    return x < w->nx && y < w->ny && z < w->nz;
}

/* Neighbor in `dir`, wrapping on a torus. Returns 0 at a hard edge.
 * Index arithmetic stays on the logical rectangle, so a non-power-of-two
 * seamless grid does not step into the padding. */
static inline int wfc_neighbor_index(const wfc_world_t* w, uint32_t cell_idx,
                                     wfc_dir_t dir, uint32_t* out)
{
    uint32_t x, y, z;
    wfc_cell_coords(w, cell_idx, &x, &y, &z);
    switch (dir) {
        case WFC_DIR_N: if (y > 0) y--; else if (w->seamless) y = w->ny - 1; else return 0; break;
        case WFC_DIR_S: if (y + 1 < w->ny) y++; else if (w->seamless) y = 0; else return 0; break;
        case WFC_DIR_E: if (x + 1 < w->nx) x++; else if (w->seamless) x = 0; else return 0; break;
        case WFC_DIR_W: if (x > 0) x--; else if (w->seamless) x = w->nx - 1; else return 0; break;
        case WFC_DIR_U: if (z > 0) z--; else if (w->seamless) z = w->nz - 1; else return 0; break;
        case WFC_DIR_D: if (z + 1 < w->nz) z++; else if (w->seamless) z = 0; else return 0; break;
        default: return 0;
    }
    *out = wfc_idx(w, x, y, z);
    return 1;
}

/* Patterns the neighbor may keep: the union of compat rows of every pattern
 * still in this cell. A collapsed cell is exactly one row. A cell that still
 * holds the full support uses the pre-OR'd row. */
static inline void wfc_build_allow(const wfc_world_t* w, const wfc_cell_t* cell,
                                   wfc_dir_t dir, uint64_t* allow)
{
    if (cell->collapsed != WFC_PATTERN_INVALID && cell->collapsed < WFC_MAX_PATTERNS) {
        wfc_mask_copy(allow, w->compat[dir][cell->collapsed]);
        return;
    }
    if (wfc_mask_eq(cell->mask, w->support_mask)) {
        wfc_mask_copy(allow, w->compat_support[dir]);
        return;
    }
    wfc_mask_clear(allow);
    for (int b = 0; b < WFC_MASK_U64S; ++b) {
        uint64_t m = cell->mask[b];
        while (m) {
            uint64_t lsb = m & -m;
            int bit = __builtin_ctzll(m) + (b << 6);
            if (bit < WFC_MAX_PATTERNS && w->patterns[bit].weight > 0.0f)
                wfc_mask_or(allow, w->compat[dir][bit]);
            m ^= lsb;
        }
    }
}

static inline void wfc_propagate_cell(wfc_world_t* w, uint32_t cell_idx, wfc_dir_t dir)
{
    if (cell_idx >= w->num_cells || !wfc_cell_logical(w, cell_idx)) return;
    const wfc_cell_t* cell = &w->cells[cell_idx];

    uint32_t nidx;
    if (!wfc_neighbor_index(w, cell_idx, dir, &nidx)) return;
    if (nidx >= w->num_cells) return;
    wfc_cell_t* nbr = &w->cells[nidx];
    if (nbr->collapsed != WFC_PATTERN_INVALID) return;

    uint64_t allow[WFC_MASK_U64S];
    wfc_build_allow(w, cell, dir, allow);

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

    /* True Shannon entropy, then sift. Do not park the cell at 0: that
     * makes every touched cell look like the minimum and the heap lies. */
    wfc_refresh_entropy(w, nidx);
    wfc_heap_update(w, nidx);
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
        if (!(w->class_constrained & wfc_class_bit((unsigned)c))) continue;
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
                wfc_refresh_entropy(w, i);
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
        uint32_t lx = x, ly = y, lz = z;
        int valid = 1;
        switch (d) {
            case WFC_DIR_N: if (ly > 0) ly--; else if (w->seamless) ly = w->ny - 1; else valid = 0; break;
            case WFC_DIR_S: if (ly + 1 < w->ny) ly++; else if (w->seamless) ly = 0; else valid = 0; break;
            case WFC_DIR_E: if (lx + 1 < w->nx) lx++; else if (w->seamless) lx = 0; else valid = 0; break;
            case WFC_DIR_W: if (lx > 0) lx--; else if (w->seamless) lx = w->nx - 1; else valid = 0; break;
            case WFC_DIR_U: if (lz > 0) lz--; else if (w->seamless) lz = w->nz - 1; else valid = 0; break;
            case WFC_DIR_D: if (lz + 1 < w->nz) lz++; else if (w->seamless) lz = 0; else valid = 0; break;
        }
        if (!valid) continue;
        uint32_t nidx = wfc_idx(w, lx, ly, lz);
        const wfc_cell_t* nbr = &w->cells[nidx];
        if (nbr->collapsed == WFC_PATTERN_INVALID) continue;

        uint16_t nc = w->patterns[nbr->collapsed].class_id;
        float rho = w->copula_rho[d][nc][center_class];
        if (rho == 0.0f) continue;

        /* Bivariate Gaussian copula log-density. The conditional CDF is not
         * the density: log Φ peaks in the upper tail, log c peaks when the
         * class quantile matches the neighbor. */
        float z_n = w->copula_z[nc];
        float z_c = w->copula_z[center_class];
        if (rho > 0.999f) rho = 0.999f;
        else if (rho < -0.999f) rho = -0.999f;
        float one = 1.0f - rho * rho;
        float quad = rho * rho * (z_n * z_n + z_c * z_c) - 2.0f * rho * z_n * z_c;
        log_prior += -0.5f * logf(one) - quad / (2.0f * one);
    }
    return log_prior;
}

static inline void wfc_copula_class_lp(wfc_world_t* w, uint32_t cell_idx, float* class_lp)
{
    for (int c = 0; c < WFC_MAX_CLASSES; ++c) class_lp[c] = 0.0f;
    if (!w->copula_enabled) return;
    for (int c = 0; c < WFC_MAX_CLASSES; ++c) {
        if (w->copula_class_pop[c] == 0) continue;
        class_lp[c] = wfc_copula_class_prior(w, cell_idx, (uint16_t)c);
    }
}

#endif /* WFC_ENABLE_COPULA */

/* --- Spectral field generation (FastAndFourier DWT) --- */
#if defined(WFC_USE_FAF)

static inline void wfc_fill_normal_f(float* buf, int n, uint64_t* rng)
{
    int i = 0;
    for (; i + 2 <= n; i += 2) {
        double u1 = wfc_randf(rng); if (u1 < 1e-7) u1 = 1e-7;
        double u2 = wfc_randf(rng);
        double r = sqrt(-2.0 * log(u1));
        double t = 2.0 * M_PI * u2;
        buf[i+0] = (float)(r * cos(t));
        buf[i+1] = (float)(r * sin(t));
    }
    if (i < n) {
        double u1 = wfc_randf(rng); if (u1 < 1e-7) u1 = 1e-7;
        double u2 = wfc_randf(rng);
        buf[i] = (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
    }
}

static inline void wfc_gumbel_link_f(float* buf, int n)
{
    for (int i = 0; i < n; ++i) {
        double v = (double)buf[i];
        double u = 0.5 * erfc(-v * 0.7071067811865475);
        if (u < 1e-15) u = 1e-15;
        if (u > 1.0 - 1e-15) u = 1.0 - 1e-15;
        buf[i] = (float)(-log(-log(u)));
    }
}

/* n >= 128 makes FAF spawn gcc for a JIT kernel on first execute.
 * Worlds used to build that kernel and throw it away. The cache keeps the
 * pair for the process. A full cache returns a private plan the caller frees. */
typedef struct wfc_faf_plan_slot {
    int used;
    int family;
    uint32_t n;
    uint32_t levels;
    faf_transform* fwd;
    faf_transform* inv;
} wfc_faf_plan_slot;

static wfc_faf_plan_slot wfc_faf_plans[16];

static int wfc_faf_acquire_plan(int family, uint32_t n, uint32_t levels,
                                faf_transform** fwd, faf_transform** inv, int* shared)
{
    int i;
    *fwd = NULL;
    *inv = NULL;
    *shared = 0;
    if (n == 0) return -1;
    for (i = 0; i < 16; ++i) {
        wfc_faf_plan_slot* s = &wfc_faf_plans[i];
        if (s->used && s->family == family && s->n == n && s->levels == levels) {
            *fwd = s->fwd;
            *inv = s->inv;
            *shared = 1;
            return 0;
        }
    }
    faf_init();
    faf_config rc = faf_config_init(n);
    rc.family = (faf_wavelet_family)family;
    rc.levels = levels;
    rc.layout = FAF_LAYOUT_INTERLEAVED;
    {
        faf_transform* created = faf_create_dwt(&rc);
        faf_transform* inverse;
        if (!created) return -1;
        inverse = faf_create_inverse(created);
        if (!inverse) {
            faf_destroy_transform(created);
            return -1;
        }
        for (i = 0; i < 16; ++i) {
            if (!wfc_faf_plans[i].used) {
                wfc_faf_plans[i].used = 1;
                wfc_faf_plans[i].family = family;
                wfc_faf_plans[i].n = n;
                wfc_faf_plans[i].levels = levels;
                wfc_faf_plans[i].fwd = created;
                wfc_faf_plans[i].inv = inverse;
                *fwd = created;
                *inv = inverse;
                *shared = 1;
                return 0;
            }
        }
        *fwd = created;
        *inv = inverse;
        *shared = 0;
    }
    return 0;
}

static inline void wfc_faf_teardown_transforms(wfc_world_t* w)
{
    if (!w) return;
    if (!w->faf_row_shared) {
        if (w->faf_row_inv) faf_destroy_transform(w->faf_row_inv);
        if (w->faf_row_fwd) faf_destroy_transform(w->faf_row_fwd);
    }
    w->faf_row_fwd = NULL;
    w->faf_row_inv = NULL;
    if (!w->faf_col_shared) {
        if (w->faf_col_inv) faf_destroy_transform(w->faf_col_inv);
        if (w->faf_col_fwd) faf_destroy_transform(w->faf_col_fwd);
    }
    w->faf_col_fwd = NULL;
    w->faf_col_inv = NULL;
    w->faf_row_shared = 0;
    w->faf_col_shared = 0;
    if (w->faf_workspace) { faf_aligned_free(w->faf_workspace); w->faf_workspace = NULL; }
    if (w->faf_col_buf) { faf_aligned_free(w->faf_col_buf); w->faf_col_buf = NULL; }
    w->faf_initialized = 0;
    w->faf_live_family = 0;
}

static inline wfc_err_t wfc_faf_ensure_transforms(wfc_world_t* w)
{
    if (w->faf_initialized) return WFC_OK;

    faf_init();

    uint32_t nx = w->nx_p2, ny = w->ny_p2;
    uint32_t max_dim = nx > ny ? nx : ny;
    size_t levels_row = 0, levels_col = 0;
    { uint32_t t = nx; while (t > 1) { t >>= 1; levels_row++; } }
    { uint32_t t = ny; while (t > 1) { t >>= 1; levels_col++; } }
    if (w->faf_wavelet_levels > 0) {
        if ((size_t)w->faf_wavelet_levels < levels_row) levels_row = w->faf_wavelet_levels;
        if ((size_t)w->faf_wavelet_levels < levels_col) levels_col = w->faf_wavelet_levels;
    }

    int family = w->faf_wavelet_family;
    int row_shared = 0, col_shared = 0;
    if (family < 0 || family >= (int)FAF_WAVELET_COUNT)
        family = (int)FAF_WAVELET_CDF97;

    w->faf_row_shared = 0;
    w->faf_col_shared = 0;
    if (wfc_faf_acquire_plan(family, nx, (uint32_t)levels_row,
                             &w->faf_row_fwd, &w->faf_row_inv, &row_shared) != 0)
        return WFC_ENOMEM;
    w->faf_row_shared = row_shared;
    if (nx == ny && levels_row == levels_col) {
        w->faf_col_fwd = w->faf_row_fwd;
        w->faf_col_inv = w->faf_row_inv;
        w->faf_col_shared = 1;
    } else if (wfc_faf_acquire_plan(family, ny, (uint32_t)levels_col,
                                    &w->faf_col_fwd, &w->faf_col_inv, &col_shared) != 0) {
        wfc_faf_teardown_transforms(w);
        return WFC_ENOMEM;
    } else {
        w->faf_col_shared = col_shared;
    }

    w->faf_workspace = (float*)faf_aligned_alloc(2 * (size_t)nx * ny * sizeof(float));
    w->faf_col_buf   = (float*)faf_aligned_alloc(2 * (size_t)max_dim * sizeof(float));
    if (!w->faf_workspace || !w->faf_col_buf) {
        wfc_faf_teardown_transforms(w);
        return WFC_ENOMEM;
    }

    w->faf_initialized = 1;
    w->faf_live_family = family;
    return WFC_OK;
}

/* Forward declarations for parallel DWT paths */
#if defined(WFC_ENABLE_SUSPENDERS)
static inline void wfc_faf_dwt_2d_forward_par(wfc_world_t* w, float* data);
static inline void wfc_faf_dwt_2d_inverse_par(wfc_world_t* w, float* data);
static inline void wfc_fill_noise_par(float* buf, uint32_t n, uint64_t noise_seed, unsigned nw);
#endif

/* Blocked transpose of interleaved complex samples.
 * dst[(x * ny + y) * 2 + {0,1}] = src[(y * nx + x) * 2 + {0,1}].
 * Column DWTs then read contiguous pairs instead of a 2*nx stride. */
static inline void wfc_interleaved_transpose(float* WFC_RESTRICT dst,
                                             const float* WFC_RESTRICT src,
                                             uint32_t nx, uint32_t ny)
{
    const uint32_t TB = 8;
    for (uint32_t y0 = 0; y0 < ny; y0 += TB) {
        uint32_t y1 = y0 + TB;
        if (y1 > ny) y1 = ny;
        for (uint32_t x0 = 0; x0 < nx; x0 += TB) {
            uint32_t x1 = x0 + TB;
            if (x1 > nx) x1 = nx;
            for (uint32_t y = y0; y < y1; ++y) {
                const float* s = src + ((size_t)y * nx + x0) * 2;
                for (uint32_t x = x0; x < x1; ++x) {
                    float* d = dst + ((size_t)x * ny + y) * 2;
                    d[0] = s[0];
                    d[1] = s[1];
                    s += 2;
                }
            }
        }
    }
}

/* Separable 2D forward DWT: rows then columns.
 * Data is in interleaved layout: [re0,im0,re1,im1,...].
 * Real data stored at even indices, im=0. FAF's faf_execute_f32 consumes
 * that layout; the native REAL layout is a different entry point. */
static inline void wfc_faf_dwt_2d_forward(wfc_world_t* w, float* data)
{
#if defined(WFC_ENABLE_SUSPENDERS)
    if (w->susp_initialized && w->nx_p2 >= WFC_PAR_MIN_DIM && w->ny_p2 >= WFC_PAR_MIN_DIM) {
        wfc_faf_dwt_2d_forward_par(w, data);
        return;
    }
#endif
    uint32_t nx = w->nx_p2, ny = w->ny_p2;

    /* Transform each row */
    float* row_out = w->faf_workspace;
    for (uint32_t j = 0; j < ny; ++j) {
        float* row = data + j * 2 * nx;
        faf_execute_f32(w->faf_row_fwd, row_out + j * 2 * nx, row);
    }
    memcpy(data, row_out, 2 * (size_t)nx * ny * sizeof(float));

    /* Columns. Large grids are transposed so each DWT is sequential.
     * col_buf is 2*max(nx,ny) and is not the workspace. */
    if (nx >= 32 && ny >= 32) {
        wfc_interleaved_transpose(w->faf_workspace, data, nx, ny);
        for (uint32_t i = 0; i < nx; ++i) {
            float* col = w->faf_workspace + (size_t)i * 2 * ny;
            faf_execute_f32(w->faf_col_fwd, w->faf_col_buf, col);
            memcpy(col, w->faf_col_buf, 2 * (size_t)ny * sizeof(float));
        }
        wfc_interleaved_transpose(data, w->faf_workspace, ny, nx);
    } else {
        for (uint32_t i = 0; i < nx; ++i) {
            float* col_buf = w->faf_col_buf;
            for (uint32_t j = 0; j < ny; ++j) {
                col_buf[2*j]   = data[j * 2 * nx + 2*i];
                col_buf[2*j+1] = data[j * 2 * nx + 2*i+1];
            }
            float* col_out = w->faf_workspace;
            faf_execute_f32(w->faf_col_fwd, col_out, col_buf);
            for (uint32_t j = 0; j < ny; ++j) {
                data[j * 2 * nx + 2*i]   = col_out[2*j];
                data[j * 2 * nx + 2*i+1] = col_out[2*j+1];
            }
        }
    }
}

static inline void wfc_faf_dwt_2d_inverse(wfc_world_t* w, float* data)
{
#if defined(WFC_ENABLE_SUSPENDERS)
    if (w->susp_initialized && w->nx_p2 >= WFC_PAR_MIN_DIM && w->ny_p2 >= WFC_PAR_MIN_DIM) {
        wfc_faf_dwt_2d_inverse_par(w, data);
        return;
    }
#endif
    uint32_t nx = w->nx_p2, ny = w->ny_p2;

    /* Inverse columns first */
    if (nx >= 32 && ny >= 32) {
        wfc_interleaved_transpose(w->faf_workspace, data, nx, ny);
        for (uint32_t i = 0; i < nx; ++i) {
            float* col = w->faf_workspace + (size_t)i * 2 * ny;
            faf_execute_f32(w->faf_col_inv, w->faf_col_buf, col);
            memcpy(col, w->faf_col_buf, 2 * (size_t)ny * sizeof(float));
        }
        wfc_interleaved_transpose(data, w->faf_workspace, ny, nx);
    } else {
        for (uint32_t i = 0; i < nx; ++i) {
            float* col_buf = w->faf_col_buf;
            for (uint32_t j = 0; j < ny; ++j) {
                col_buf[2*j]   = data[j * 2 * nx + 2*i];
                col_buf[2*j+1] = data[j * 2 * nx + 2*i+1];
            }
            float* col_out = w->faf_workspace;
            faf_execute_f32(w->faf_col_inv, col_out, col_buf);
            for (uint32_t j = 0; j < ny; ++j) {
                data[j * 2 * nx + 2*i]   = col_out[2*j];
                data[j * 2 * nx + 2*i+1] = col_out[2*j+1];
            }
        }
    }

    /* Inverse rows */
    float* row_out = w->faf_workspace;
    for (uint32_t j = 0; j < ny; ++j) {
        float* row = data + j * 2 * nx;
        faf_execute_f32(w->faf_row_inv, row_out + j * 2 * nx, row);
    }
    memcpy(data, row_out, 2 * (size_t)nx * ny * sizeof(float));
}

/* Scale DWT detail coefficients by 2^(j*H) per level (Flandrin). */
static inline void wfc_faf_scale_flandrin(float* data, uint32_t nx, uint32_t ny,
                                          uint32_t levels_x, uint32_t levels_y, double H)
{
    /* After separable 2D DWT the coefficient layout is a standard Mallat
     * decomposition: at each level j the LL band is in the upper-left
     * quadrant and detail bands (LH, HL, HH) fill the rest.  Level 0 is
     * the finest (full-resolution details) and level max-1 is coarsest.
     *
     * We scale each coefficient at level j by 2^(j*H) so coarser detail
     * bands get more energy — this is the wavelet-domain analogue of
     * Flandrin's 1/f^(2H+1) spectral weighting. */
    uint32_t bx = nx, by = ny;
    for (uint32_t j = 0; j < levels_x && j < levels_y; ++j) {
        float scale = (float)pow(2.0, (double)(j + 1) * H);
        uint32_t hx = bx >> 1;
        uint32_t hy = by >> 1;

        /* LH band: rows [0, hy), cols [hx, bx) */
        for (uint32_t r = 0; r < hy; ++r)
            for (uint32_t c = hx; c < bx; ++c)
                data[r * 2 * nx + 2 * c] *= scale;

        /* HL band: rows [hy, by), cols [0, hx) */
        for (uint32_t r = hy; r < by; ++r)
            for (uint32_t c = 0; c < hx; ++c)
                data[r * 2 * nx + 2 * c] *= scale;

        /* HH band: rows [hy, by), cols [hx, bx) */
        for (uint32_t r = hy; r < by; ++r)
            for (uint32_t c = hx; c < bx; ++c)
                data[r * 2 * nx + 2 * c] *= scale;

        bx = hx;
        by = hy;
    }
}

/* Scale DWT coefficients by Matérn spectral density. */
static inline void wfc_faf_scale_matern(float* data, uint32_t nx, uint32_t ny,
                                        uint32_t levels_x, uint32_t levels_y,
                                        double kappa, double nu)
{
    double kk = kappa * kappa;
    uint32_t bx = nx, by = ny;
    for (uint32_t j = 0; j < levels_x && j < levels_y; ++j) {
        double freq = M_PI * ldexp(1.0, -(int)(j + 1));
        double base = kk + freq * freq;
        if (base < 1e-12) base = 1e-12;
        float scale = (float)pow(base, -nu);
        uint32_t hx = bx >> 1;
        uint32_t hy = by >> 1;

        for (uint32_t r = 0; r < hy; ++r)
            for (uint32_t c = hx; c < bx; ++c)
                data[r * 2 * nx + 2 * c] *= scale;
        for (uint32_t r = hy; r < by; ++r)
            for (uint32_t c = 0; c < hx; ++c)
                data[r * 2 * nx + 2 * c] *= scale;
        for (uint32_t r = hy; r < by; ++r)
            for (uint32_t c = hx; c < bx; ++c)
                data[r * 2 * nx + 2 * c] *= scale;

        bx = hx;
        by = hy;
    }
}

/* Unit-variance, zero-mean. The Gumbel link assumes N(0,1) marginals;
 * an unnormalized IDWT is not that. Weight 1 then means one standard deviation. */
static inline void wfc_standardize_plane(float* v, uint32_t n)
{
    if (!n) return;
    double mean = 0.0;
    for (uint32_t i = 0; i < n; ++i) mean += (double)v[i];
    mean /= (double)n;
    double var = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        double d = (double)v[i] - mean;
        var += d * d;
    }
    var /= (double)n;
    if (var < 1e-20) {
        for (uint32_t i = 0; i < n; ++i) v[i] = 0.0f;
        return;
    }
    float inv = (float)(1.0 / sqrt(var));
    float m = (float)mean;
    for (uint32_t i = 0; i < n; ++i) v[i] = (v[i] - m) * inv;
}

/* Copy one XY plane onto every Z slice. The DWT is 2D; indexing
 * buf[2*i] for i up to num_cells walks off a 3D grid. */
static inline void wfc_store_spectral_plane(wfc_world_t* w, const float* plane, uint32_t n)
{
    uint32_t plane_n = w->nx_p2 * w->ny_p2;
    if (n < plane_n) plane_n = n;
    for (uint32_t z = 0; z < w->nz_p2; ++z) {
        uint32_t base = z * w->nx_p2 * w->ny_p2;
        for (uint32_t i = 0; i < plane_n; ++i) {
            uint32_t dst = base + i;
            if (dst < w->num_cells) w->spectral_field[dst] = plane[i];
        }
    }
}

wfc_err_t wfc_set_gumbel_field(wfc_world_t* w, double H, uint64_t noise_seed)
{
    if (!w || !w->spectral_enabled) return WFC_EUNINIT;
    wfc_err_t err = wfc_faf_ensure_transforms(w);
    if (err != WFC_OK) return err;

    uint32_t nx = w->nx_p2, ny = w->ny_p2;
    uint32_t n = nx * ny;
    uint32_t levels_x = 0, levels_y = 0;
    { uint32_t t = nx; while (t > 1) { t >>= 1; levels_x++; } }
    { uint32_t t = ny; while (t > 1) { t >>= 1; levels_y++; } }
#if defined(WFC_ENABLE_HIERARCHY)
    /* wfc_hierarchy_set_wavelet may cap the decomposition depth used by
     * the per-octave scaling below; 0 means use every level. */
    if (w->hierarchy && w->hierarchy->wavelet_levels > 0) {
        if (levels_x > w->hierarchy->wavelet_levels) levels_x = w->hierarchy->wavelet_levels;
        if (levels_y > w->hierarchy->wavelet_levels) levels_y = w->hierarchy->wavelet_levels;
    }
#endif

    /* Interleaved buffer: 2*n floats (re,im pairs) */
    float* buf = (float*)faf_aligned_alloc(2 * (size_t)n * sizeof(float));
    if (!buf) return WFC_ENOMEM;

#if defined(WFC_ENABLE_SUSPENDERS)
    if (w->susp_initialized && n >= WFC_PAR_MIN_DIM * WFC_PAR_MIN_DIM) {
        wfc_fill_noise_par(buf, n, noise_seed, w->susp_num_workers);
    } else
#endif
    {
        uint64_t rng = noise_seed;
        for (uint32_t i = 0; i < n; ++i) {
            double u1 = wfc_randf(&rng); if (u1 < 1e-7) u1 = 1e-7;
            double u2 = wfc_randf(&rng);
            double r = sqrt(-2.0 * log(u1));
            double t = 2.0 * M_PI * u2;
            buf[2*i]   = (float)(r * cos(t));
            buf[2*i+1] = 0.0f;
            if (i + 1 < n) {
                buf[2*(i+1)]   = (float)(r * sin(t));
                buf[2*(i+1)+1] = 0.0f;
                ++i;
            }
        }
    }

    wfc_faf_dwt_2d_forward(w, buf);
    wfc_faf_scale_flandrin(buf, nx, ny, levels_x, levels_y, H);
    wfc_faf_dwt_2d_inverse(w, buf);

    /* Compact reals in place (2*i >= i, so forward is safe), standardize, link. */
    for (uint32_t i = 0; i < n; ++i) buf[i] = buf[2 * i];
    wfc_standardize_plane(buf, n);
    wfc_gumbel_link_f(buf, (int)n);
    wfc_store_spectral_plane(w, buf, n);

    faf_aligned_free(buf);
    return WFC_OK;
}

wfc_err_t wfc_set_matern_field(wfc_world_t* w, double kappa, double nu, uint64_t noise_seed)
{
    if (!w || !w->spectral_enabled) return WFC_EUNINIT;
    wfc_err_t err = wfc_faf_ensure_transforms(w);
    if (err != WFC_OK) return err;

    uint32_t nx = w->nx_p2, ny = w->ny_p2;
    uint32_t n = nx * ny;
    uint32_t levels_x = 0, levels_y = 0;
    { uint32_t t = nx; while (t > 1) { t >>= 1; levels_x++; } }
    { uint32_t t = ny; while (t > 1) { t >>= 1; levels_y++; } }
#if defined(WFC_ENABLE_HIERARCHY)
    /* wfc_hierarchy_set_wavelet may cap the decomposition depth used by
     * the per-octave scaling below; 0 means use every level. */
    if (w->hierarchy && w->hierarchy->wavelet_levels > 0) {
        if (levels_x > w->hierarchy->wavelet_levels) levels_x = w->hierarchy->wavelet_levels;
        if (levels_y > w->hierarchy->wavelet_levels) levels_y = w->hierarchy->wavelet_levels;
    }
#endif

    float* buf = (float*)faf_aligned_alloc(2 * (size_t)n * sizeof(float));
    if (!buf) return WFC_ENOMEM;

#if defined(WFC_ENABLE_SUSPENDERS)
    if (w->susp_initialized && n >= WFC_PAR_MIN_DIM * WFC_PAR_MIN_DIM) {
        wfc_fill_noise_par(buf, n, noise_seed, w->susp_num_workers);
    } else
#endif
    {
        uint64_t rng = noise_seed;
        for (uint32_t i = 0; i < n; ++i) {
            double u1 = wfc_randf(&rng); if (u1 < 1e-7) u1 = 1e-7;
            double u2 = wfc_randf(&rng);
            double r = sqrt(-2.0 * log(u1));
            double t = 2.0 * M_PI * u2;
            buf[2*i]   = (float)(r * cos(t));
            buf[2*i+1] = 0.0f;
            if (i + 1 < n) {
                buf[2*(i+1)]   = (float)(r * sin(t));
                buf[2*(i+1)+1] = 0.0f;
                ++i;
            }
        }
    }

    wfc_faf_dwt_2d_forward(w, buf);
    wfc_faf_scale_matern(buf, nx, ny, levels_x, levels_y, kappa, nu);
    wfc_faf_dwt_2d_inverse(w, buf);

    for (uint32_t i = 0; i < n; ++i) buf[i] = buf[2 * i];
    wfc_standardize_plane(buf, n);
    wfc_store_spectral_plane(w, buf, n);

    faf_aligned_free(buf);
    return WFC_OK;
}

#endif /* WFC_USE_FAF */

#if defined(WFC_ENABLE_HIERARCHY)
/* Identity embedding of a collapsed pattern for residual coupling.
 * Class occupies the unit interval in equal bins; pattern id is a unique
 * sub-quantum so two tiles with equal weight never alias. Invalid codes
 * map to 0. Weights do not enter. */
static inline float wfc_pattern_signature(const wfc_world_t* w, uint16_t code)
{
    if (!w || code == WFC_PATTERN_INVALID || code >= w->num_patterns)
        return 0.0f;
    const wfc_pattern_def_t* p = &w->patterns[code];
    float nclass = (w->num_classes > 1) ? (float)w->num_classes : 1.0f;
    float frac = (float)(p->id + 1) * (1.0f / (float)(WFC_MAX_PATTERNS + 1));
    return ((float)p->class_id + frac) / nclass;
}
#endif

/* Posterior weight of one pattern.
 * Hierarchy: Gaussian likelihood of the signature against the residual
 * target, wt * exp(-|λ| (sig - target)^2). This is not a Gumbel perturbation.
 * Copula: wt * exp(λ * log c) = wt * c^λ, the Gaussian copula density.
 * A 1e-12 floor keeps a soft prior from banning a still-legal pattern.
 */
static inline float wfc_posterior_weight(const wfc_world_t* w, int bit,
                                         float target, int guide,
                                         const float* class_lp)
{
    float wt = w->patterns[bit].weight;
    if (wt <= 0.0f) return 0.0f;
#if defined(WFC_ENABLE_HIERARCHY) && defined(WFC_USE_FAF)
    if (guide) {
        float sig = wfc_pattern_signature(w, (uint16_t)bit);
        float d = sig - target;
        float lambda = w->spectral_weight;
        if (lambda < 0.0f) lambda = -lambda;
        wt *= expf(-lambda * d * d);
        if (wt < 1e-12f) wt = 1e-12f;
    }
#else
    (void)target;
    (void)guide;
#endif
#if defined(WFC_ENABLE_COPULA)
    if (class_lp) {
        uint16_t c = w->patterns[bit].class_id;
        if (c < WFC_MAX_CLASSES) {
            wt *= expf(w->copula_weight * class_lp[c]);
            if (wt < 1e-12f) wt = 1e-12f;
        }
    }
#else
    (void)class_lp;
#endif
    return wt;
}

/* --- Pattern choice --- */
static inline uint16_t wfc_choose_pattern(wfc_world_t* w, const wfc_cell_t* cell)
{
    /* Efraimidis-Spirakis one-pass weighted sample of the posterior. */
    float total = 0.0f;
    uint16_t choice = WFC_PATTERN_INVALID;
    float target = 0.0f;
    int guide = 0;
#if defined(WFC_ENABLE_HIERARCHY) && defined(WFC_USE_FAF)
    guide = w->spectral_guides_choice && w->spectral_enabled && w->spectral_field;
    if (guide) {
        uint32_t cidx = (uint32_t)(cell - w->cells);
        if (cidx < w->num_cells) target = w->spectral_field[cidx];
        if (target < 0.0f) target = 0.0f;
        else if (target > 1.0f) target = 1.0f;
    }
#endif
#if defined(WFC_ENABLE_COPULA)
    float class_lp[WFC_MAX_CLASSES];
    const float* lp = NULL;
    if (w->copula_enabled) {
        wfc_copula_class_lp(w, (uint32_t)(cell - w->cells), class_lp);
        lp = class_lp;
    }
#else
    const float* lp = NULL;
#endif
    int adjust = guide || lp != NULL;
    for (int b = 0; b < WFC_MASK_U64S; ++b) {
        uint64_t m = cell->mask[b];
        while (m) {
            uint64_t lsb = m & -m;
            int bit = __builtin_ctzll(m) + (b << 6);
            float wt = adjust ? wfc_posterior_weight(w, bit, target, guide, lp)
                              : w->patterns[bit].weight;
            if (wt <= 0.0f) { m ^= lsb; continue; }
            total += wt;
            if (wfc_randf(&w->rng_state) * total < wt)
                choice = (uint16_t)bit;
            m ^= lsb;
        }
    }
    return choice;
}

/* Shannon entropy of the same posterior wfc_choose_pattern samples. */
static inline float wfc_posterior_entropy(wfc_world_t* w, const wfc_cell_t* cell)
{
    float target = 0.0f;
    int guide = 0;
#if defined(WFC_ENABLE_HIERARCHY) && defined(WFC_USE_FAF)
    guide = w->spectral_guides_choice && w->spectral_enabled && w->spectral_field;
    if (guide) {
        uint32_t cidx = (uint32_t)(cell - w->cells);
        if (cidx < w->num_cells) target = w->spectral_field[cidx];
        if (target < 0.0f) target = 0.0f;
        else if (target > 1.0f) target = 1.0f;
    }
#endif
#if defined(WFC_ENABLE_COPULA)
    float class_lp[WFC_MAX_CLASSES];
    const float* lp = NULL;
    if (w->copula_enabled) {
        wfc_copula_class_lp(w, (uint32_t)(cell - w->cells), class_lp);
        lp = class_lp;
    }
#else
    const float* lp = NULL;
#endif
    if (!guide && !lp) return wfc_cell_entropy(w, cell);

    float sum = 0.0f, wlog = 0.0f;
    for (int b = 0; b < WFC_MASK_U64S; ++b) {
        uint64_t m = cell->mask[b];
        while (m) {
            uint64_t lsb = m & -m;
            int bit = __builtin_ctzll(m) + (b << 6);
            float wt = wfc_posterior_weight(w, bit, target, guide, lp);
            if (wt > 0.0f) {
                sum += wt;
                wlog += wt * log2f(wt);
            }
            m ^= lsb;
        }
    }
    if (sum <= 0.0f) return 0.0f;
    return log2f(sum) - wlog / sum;
}


/* --- Tie-breaking: deterministic micro-noise based on coordinates --- */
static inline float wfc_tiebreak_noise(const wfc_world_t* w, uint32_t cell_idx)
{
    /* Hash the original seed, not the live RNG, so a refresh mid-solve
     * does not reshuffle ties as SplitMix advances. */
    uint64_t h = w->rng_seed ^ ((uint64_t)cell_idx * 0x9e3779b97f4a7c15ULL);
    h = (h ^ (h >> 33)) * 0xff51afd7ed558ccdULL;
    h = (h ^ (h >> 33)) * 0xc4ceb9fe1a85ec53ULL;
    h = h ^ (h >> 33);
    /* Return a tiny value in [-1e-6, 1e-6] */
    return ((float)(h & 0xFFFFF) / 1048576.0f - 0.5f) * 2e-6f;
}

/* --- Entropy used by the heap --- */
static inline void wfc_refresh_entropy(wfc_world_t* w, uint32_t cell_idx)
{
    wfc_cell_t* cell = &w->cells[cell_idx];
    if (cell->collapsed != WFC_PATTERN_INVALID) {
        cell->entropy = -1.0f;
        cell->dirty = 0;
        return;
    }

    /* Guided hierarchy: the field is a signature target, already inside the
     * posterior. Unguided spectral: the field is a standardized spatial bias
     * added in bits (weight 1 moves the cell by one standard deviation). */
    int guided = 0;
#if defined(WFC_ENABLE_HIERARCHY) && defined(WFC_USE_FAF)
    guided = w->spectral_guides_choice && w->spectral_enabled && w->spectral_field;
#endif
    cell->entropy = wfc_posterior_entropy(w, cell);
    cell->prior = 0.0f;
#if defined(WFC_USE_FAF)
    if (!guided && w->spectral_enabled && w->spectral_field && cell_idx < w->num_cells)
        cell->entropy += w->spectral_field[cell_idx] * w->spectral_weight;
#endif
    cell->entropy += wfc_tiebreak_noise(w, cell_idx);
    cell->dirty = 0;
}

/* --- Single collapse iteration (shared by wfc_collapse and wfc_step) --- */
static inline wfc_err_t wfc_collapse_one(wfc_world_t* w, uint32_t* out_cell_idx)
{
    /* A retry or an empty heap is not a counted collapse. Callers read
     * this even when we return WFC_OK, so it must not stay uninitialized. */
    if (out_cell_idx) *out_cell_idx = WFC_HEAP_INVALID;
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
        /* Stale key: recompute and put it back if it is no longer the minimum. */
        if (c->dirty) {
            wfc_refresh_entropy(w, cell_idx);
            if (w->heap_size > 0 &&
                w->cells[w->heap[0]].entropy < c->entropy) {
                wfc_heap_insert(w, cell_idx);
                cell_idx = WFC_HEAP_INVALID;
                continue;
            }
        }
        break;
    }

    if (cell_idx == WFC_HEAP_INVALID) return WFC_OK; /* All collapsed */

    wfc_cell_t* cell = &w->cells[cell_idx];

#if defined(WFC_ENABLE_BACKTRACK)
    if (w->backtrack_enabled) wfc_snapshot_push(w);
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

#if defined(WFC_ENABLE_BACKTRACK)
    wfc_trail_push(w, cell_idx);
#endif
    cell->collapsed = pat;
    wfc_mask_clear(cell->mask);
    if (pat < WFC_MAX_PATTERNS)
        cell->mask[pat >> 6] = (UINT64_C(1) << (pat & 63));
    cell->entropy = -1.0f;
    cell->dirty = 0;
    wfc_note_class_placed(w, pat, 1);

    /* Propagate. One queue entry: propagate_from walks every direction. */
    w->queue_head = w->queue_tail = 0;
    memset(w->in_queue, 0, w->num_cells * sizeof(uint8_t));
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
            /* class_placed was snapshotted before the increment. */
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

/* Two pinned cells that share an edge must name a legal pair. Propagation
 * does not rewrite a collapsed neighbor, so the domain pass cannot see this. */
static int wfc_fixed_neighbors_ok(const wfc_world_t* w, uint32_t cell_idx)
{
    const wfc_cell_t* cell = &w->cells[cell_idx];
    uint16_t pat = cell->collapsed;
    if (pat == WFC_PATTERN_INVALID || pat >= WFC_MAX_PATTERNS) return 1;
    int nd = w->is_3d ? 6 : 4;
    for (int d = 0; d < nd; ++d) {
        uint32_t nidx;
        uint16_t np;
        if (!wfc_neighbor_index(w, cell_idx, (wfc_dir_t)d, &nidx)) continue;
        np = w->cells[nidx].collapsed;
        if (np == WFC_PATTERN_INVALID || np >= WFC_MAX_PATTERNS) continue;
        if ((w->compat[d][pat][np >> 6] & (UINT64_C(1) << (np & 63))) == 0)
            return 0;
    }
    return 1;
}

/* Intersect the support, run AC-3 to a fixpoint, then write true entropies.
 * The heap is built by the caller from those entropies. */
static wfc_err_t wfc_prepare_fail(wfc_world_t* w, uint32_t cell_idx)
{
    if (w->on_contradiction) {
        if (cell_idx >= w->num_cells) cell_idx = 0;
        uint32_t x, y, z;
        wfc_cell_coords(w, cell_idx, &x, &y, &z);
        w->on_contradiction(x, y, z, w->contradiction_user_data);
    }
    return WFC_ECONTRADICTION;
}

static wfc_err_t wfc_solver_prepare(wfc_world_t* w)
{
    wfc_prepare_weights(w);
    w->heap_size = 0;
    for (uint32_t i = 0; i < w->num_cells; ++i)
        w->heap_pos[i] = WFC_HEAP_INVALID;

    /* Recount from the grid. wfc_fix_cell increments as it pins, and a
     * second prepare (or a reset) must not add those cells again. */
    memset(w->class_placed, 0, sizeof(w->class_placed));
    for (uint32_t i = 0; i < w->num_cells; ++i) {
        uint16_t pat;
        if (!wfc_cell_logical(w, i)) continue;
        pat = w->cells[i].collapsed;
        if (pat != WFC_PATTERN_INVALID && pat < w->num_patterns)
            wfc_note_class_placed(w, pat, 1);
    }

#if defined(WFC_ENABLE_BACKTRACK)
    w->contradiction = 0;
#endif
    wfc_enforce_class_limits(w);
#if defined(WFC_ENABLE_BACKTRACK)
    if (w->contradiction) return wfc_prepare_fail(w, 0);
#endif

    for (uint32_t i = 0; i < w->num_cells; ++i) {
        if (!wfc_cell_logical(w, i)) continue;
        wfc_cell_t* c = &w->cells[i];
        if (c->collapsed != WFC_PATTERN_INVALID) continue;
        wfc_mask_and(c->mask, w->support_mask);
        if (wfc_mask_empty(c->mask)) return wfc_prepare_fail(w, i);
    }

    w->queue_head = w->queue_tail = 0;
    if (w->in_queue) memset(w->in_queue, 0, (size_t)w->num_cells * sizeof(uint8_t));
    /* Pinned cells are unary constraints. They have to enter the queue or
     * their compat row never reaches an open neighbor. */
    for (uint32_t i = 0; i < w->num_cells; ++i) {
        if (!wfc_cell_logical(w, i)) continue;
        wfc_queue_push(w, i);
    }
    while (w->queue_head != w->queue_tail) {
        uint32_t cur = wfc_queue_pop(w);
        if (cur == WFC_HEAP_INVALID) break;
        wfc_propagate_from(w, cur);
#if defined(WFC_ENABLE_BACKTRACK)
        if (w->contradiction) return wfc_prepare_fail(w, cur);
#endif
    }

    for (uint32_t i = 0; i < w->num_cells; ++i) {
        if (!wfc_cell_logical(w, i)) continue;
        if (w->cells[i].collapsed == WFC_PATTERN_INVALID) continue;
        if (!wfc_fixed_neighbors_ok(w, i)) return wfc_prepare_fail(w, i);
    }

    for (uint32_t i = 0; i < w->num_cells; ++i) {
        if (!wfc_cell_logical(w, i)) continue;
        wfc_cell_t* c = &w->cells[i];
        if (c->collapsed != WFC_PATTERN_INVALID) continue;
        if (wfc_mask_empty(c->mask)) return wfc_prepare_fail(w, i);
        wfc_refresh_entropy(w, i);
    }
    return WFC_OK;
}

static void wfc_heap_build(wfc_world_t* w)
{
    w->heap_size = 0;
    for (uint32_t i = 0; i < w->num_cells; ++i) {
        w->heap_pos[i] = WFC_HEAP_INVALID;
        if (!wfc_cell_logical(w, i)) continue;
        if (w->cells[i].collapsed != WFC_PATTERN_INVALID) continue;
        wfc_heap_insert(w, i);
    }
}

/* --- Core collapse loop --- */
wfc_err_t wfc_collapse(wfc_world_t* w)
{
    if (!w || !w->num_patterns) return WFC_EINVAL;

    w->step_mode = 0;
    w->step_collapsed_count = 0;
    w->step_target_collapses = w->nx * w->ny * w->nz;

    {
        wfc_err_t prep = wfc_solver_prepare(w);
        if (prep != WFC_OK) return prep;
    }
    wfc_heap_build(w);

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

    {
        wfc_err_t prep = wfc_solver_prepare(w);
        if (prep != WFC_OK) return prep;
    }
    wfc_heap_build(w);
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
    w->rng_seed = cfg->seed ? cfg->seed : 0x123456789ABCDEF0ULL;
    w->rng_state = w->rng_seed;

    w->nx_p2 = wfc_next_pow2_u32(w->nx);
    w->ny_p2 = wfc_next_pow2_u32(w->ny);
    w->nz_p2 = wfc_next_pow2_u32(w->nz);
    w->log2_nx = (uint32_t)__builtin_ctz(w->nx_p2);
    w->log2_ny = (uint32_t)__builtin_ctz(w->ny_p2);
    w->log2_nz = (uint32_t)__builtin_ctz(w->nz_p2);

    w->num_cells = w->nx_p2 * w->ny_p2 * w->nz_p2;

    w->cells = (wfc_cell_t*)wfc_alloc_aligned(
        WFC_CELL_ALIGN, (size_t)w->num_cells * sizeof(wfc_cell_t));
    w->heap = (uint32_t*)WFC_MALLOC((size_t)w->num_cells * sizeof(uint32_t));
    w->heap_pos = (uint32_t*)WFC_MALLOC((size_t)w->num_cells * sizeof(uint32_t));
    /* Spare slot: a full ring of num_cells would wrap tail onto head and
     * look empty, so the opening AC-3 would never run. */
    w->queue_cap = w->num_cells << 1;
    w->queue = (uint32_t*)WFC_MALLOC((size_t)w->queue_cap * sizeof(uint32_t));
    w->in_queue = (uint8_t*)WFC_MALLOC((size_t)w->num_cells * sizeof(uint8_t));
    if (!w->cells || !w->heap || !w->heap_pos || !w->queue || !w->in_queue) goto fail;

    /* Initialize cells: logical cells get all patterns valid,
     * padding cells (outside nx×ny×nz) are pre-collapsed as inert. */
    for (uint32_t i = 0; i < w->num_cells; ++i) {
        wfc_cell_t* c = &w->cells[i];
        uint32_t cx, cy, cz;
        if (w->is_3d) {
            uint32_t p2 = w->nx_p2 * w->ny_p2;
            cz = i / p2;
            uint32_t rem = i % p2;
            cy = rem >> w->log2_nx;
            cx = rem & (w->nx_p2 - 1);
        } else {
            cz = 0;
            cy = i >> w->log2_nx;
            cx = i & (w->nx_p2 - 1);
        }
        if (cx >= w->nx || cy >= w->ny || cz >= w->nz) {
            c->collapsed = 0;
            c->entropy = -1.0f;
            c->prior = 0.0f;
            c->dirty = 0;
            memset(c->mask, 0, sizeof(c->mask));
        } else {
            c->collapsed = WFC_PATTERN_INVALID;
            c->entropy = 0.0f;
            c->prior = 0.0f;
            c->dirty = 0;
            for (int b = 0; b < WFC_MASK_U64S; ++b) c->mask[b] = ~0ULL;
        }
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
#if defined(WFC_USE_FAF)
    w->spectral_enabled = cfg->enable_spectral;
    w->spectral_weight = cfg->spectral_weight;
    if (w->spectral_enabled) {
        w->spectral_field = (float*)WFC_MALLOC((size_t)w->num_cells * sizeof(float));
        if (!w->spectral_field) goto fail;
        memset(w->spectral_field, 0, (size_t)w->num_cells * sizeof(float));
    }
    w->faf_row_fwd = NULL;
    w->faf_row_inv = NULL;
    w->faf_col_fwd = NULL;
    w->faf_col_inv = NULL;
    w->faf_workspace = NULL;
    w->faf_col_buf   = NULL;
    w->faf_initialized = 0;
    w->faf_wavelet_family = FAF_WAVELET_CDF97;
    w->faf_wavelet_levels = 0;
    w->faf_live_family = 0;
    w->faf_row_shared = 0;
    w->faf_col_shared = 0;
    w->spectral_guides_choice = 0;
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

    /* Hierarchy */
#if defined(WFC_ENABLE_HIERARCHY)
    w->hierarchy = NULL;
    w->owns_compat = 1;
#endif

    /* Suspenders */
#if defined(WFC_ENABLE_SUSPENDERS)
    w->susp_initialized = 0;
    w->susp_num_workers = 0;
    w->faf_par_scratch = NULL;
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
#if defined(WFC_ENABLE_SUSPENDERS)
    if (w->susp_initialized) wfc_parallel_shutdown(w);
#endif
#if defined(WFC_ENABLE_HIERARCHY)
    if (w->hierarchy) wfc_hierarchy_destroy(w);
#endif
#if defined(WFC_USE_FAF)
    wfc_faf_teardown_transforms(w);
    WFC_FREE(w->spectral_field);
#endif
#if defined(WFC_ENABLE_BACKTRACK)
    WFC_FREE(w->trail);
    WFC_FREE(w->trail_gen);
#endif
    WFC_ALIGNED_FREE(w->cells);
    WFC_FREE(w->heap);
    WFC_FREE(w->heap_pos);
    WFC_FREE(w->queue);
    WFC_FREE(w->in_queue);
    WFC_FREE(w);
}

wfc_err_t wfc_add_pattern(wfc_world_t* w, const wfc_pattern_def_t* pat)
{
    if (!w || !pat || pat->id >= WFC_MAX_PATTERNS) return WFC_EINVAL;
    if (pat->class_id >= WFC_MAX_CLASSES) return WFC_EINVAL;
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
    if (min_count > 0 || max_count < 0xFFFFFFFFu)
        w->class_constrained |= wfc_class_bit(class_id);
    else
        w->class_constrained &= ~wfc_class_bit(class_id);
    return WFC_OK;
}

wfc_err_t wfc_fix_cell(wfc_world_t* w, uint32_t x, uint32_t y, uint32_t z, uint16_t pat)
{
    wfc_cell_t* cell;
    uint32_t idx;
    if (!w || pat >= w->num_patterns || pat >= WFC_MAX_PATTERNS) return WFC_EINVAL;
    if (x >= w->nx || y >= w->ny || z >= w->nz) return WFC_EINVAL;
    idx = wfc_idx(w, x, y, z);
    cell = &w->cells[idx];
    if (cell->collapsed != WFC_PATTERN_INVALID) return WFC_EINVAL;
    wfc_mask_clear(cell->mask);
    cell->mask[pat >> 6] |= (UINT64_C(1) << (pat & 63));
    cell->collapsed = pat;
    cell->entropy = -1.0f;
    cell->dirty = 0;
    wfc_note_class_placed(w, pat, 1);
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

    wfc_prepare_weights(w);

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

    /* Reset cells — mark padding cells as inert, logical cells as uncollapsed */
    for (uint32_t i = 0; i < w->num_cells; ++i) {
        wfc_cell_t* c = &w->cells[i];
        uint32_t cx, cy, cz;
        wfc_cell_coords(w, i, &cx, &cy, &cz);
        if (cx >= w->nx || cy >= w->ny || cz >= w->nz) {
            c->collapsed = 0;
            c->entropy = -1.0f;
            c->prior = 0.0f;
            c->dirty = 0;
            memset(c->mask, 0, sizeof(c->mask));
        } else {
            c->collapsed = WFC_PATTERN_INVALID;
            c->entropy = 0.0f;
            c->prior = 0.0f;
            c->dirty = 0;
            for (int b = 0; b < WFC_MASK_U64S; ++b) c->mask[b] = ~0ULL;
        }
        w->heap_pos[i] = WFC_HEAP_INVALID;
    }

    memset(w->class_placed, 0, sizeof(w->class_placed));
    w->rng_state = w->rng_seed;
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

/* --- Multiscale hierarchy --- */
#if defined(WFC_ENABLE_HIERARCHY)

/* Box-average a padded target onto a coarser scale.
 * Target is stored in the same layout as spectral_field: index = y * nx_p2 + x.
 * Separate x/y ratios handle non-square grids. Only logical (nx, ny) cells
 * contribute, so power-of-two padding does not bias the residual. */
static void wfc_hierarchy_project_target(const wfc_world_t* WFC_RESTRICT w,
                                         const float* WFC_RESTRICT target,
                                         uint32_t out_nx, uint32_t out_ny,
                                         float* WFC_RESTRICT out)
{
    uint32_t rx = (out_nx > 0) ? (w->nx_p2 / out_nx) : 1;
    uint32_t ry = (out_ny > 0) ? (w->ny_p2 / out_ny) : 1;
    if (rx == 0) rx = 1;
    if (ry == 0) ry = 1;

    for (uint32_t j = 0; j < out_ny; ++j) {
        for (uint32_t i = 0; i < out_nx; ++i) {
            float sum = 0.0f;
            uint32_t count = 0;
            uint32_t y0 = j * ry;
            uint32_t x0 = i * rx;
            for (uint32_t dy = 0; dy < ry; ++dy) {
                uint32_t gy = y0 + dy;
                if (gy >= w->ny) break;
                for (uint32_t dx = 0; dx < rx; ++dx) {
                    uint32_t gx = x0 + dx;
                    if (gx >= w->nx) break;
                    sum += target[gy * w->nx_p2 + gx];
                    count++;
                }
            }
            out[j * out_nx + i] = (count > 0) ? (sum / (float)count) : 0.0f;
        }
    }
}

wfc_err_t wfc_hierarchy_init(wfc_world_t* w, uint32_t nlevels)
{
    if (!w || nlevels < 2 || nlevels > 8) return WFC_EINVAL;

    /* Level sizes are successive right-shifts of the padded dimensions.
     * Reject pyramids whose coarsest scale would fall below 4×4 — the clamp
     * below is defensive, but a too-deep stack is a user error. */
    uint32_t coarsest_shift = nlevels - 1;
    if ((w->nx_p2 >> coarsest_shift) < 4 || (w->ny_p2 >> coarsest_shift) < 4)
        return WFC_EINVAL;

    if (w->hierarchy) wfc_hierarchy_destroy(w);

    wfc_hierarchy_t* h = (wfc_hierarchy_t*)WFC_MALLOC(sizeof(wfc_hierarchy_t));
    if (!h) return WFC_ENOMEM;
    memset(h, 0, sizeof(*h));

    h->nlevels = nlevels;
    h->finest_idx = nlevels - 1;
    h->wavelet_family = w->faf_wavelet_family;
    h->wavelet_levels = w->faf_wavelet_levels;
    h->target_field = NULL;

    h->levels = (wfc_scale_t*)WFC_MALLOC(nlevels * sizeof(wfc_scale_t));
    if (!h->levels) { WFC_FREE(h); return WFC_ENOMEM; }
    memset(h->levels, 0, nlevels * sizeof(wfc_scale_t));

    /* Attach early so destroy() can clean up a partial init. */
    w->hierarchy = h;

    for (uint32_t k = 0; k < nlevels; ++k) {
        uint32_t shift = nlevels - 1 - k;
        uint32_t lnx = w->nx_p2 >> shift;
        uint32_t lny = w->ny_p2 >> shift;
        if (lnx < 4) lnx = 4;
        if (lny < 4) lny = 4;

        wfc_scale_t* s = &h->levels[k];
        s->nx = lnx;
        s->ny = lny;
        s->nx_p2 = lnx;
        s->ny_p2 = lny;
        s->ncells = lnx * lny;
        s->weight = ldexpf(1.0f, -(int)shift);

        s->residual_field = (float*)faf_aligned_alloc(s->ncells * sizeof(float));
        s->recon_field    = (float*)faf_aligned_alloc(s->ncells * sizeof(float));
        s->coarse_codes   = (uint16_t*)faf_aligned_alloc(s->ncells * sizeof(uint16_t));
        s->detail_pack    = (float*)faf_aligned_alloc(s->ncells * sizeof(float));
        s->detail_cap     = s->detail_pack ? s->ncells : 0;
        if (!s->residual_field || !s->recon_field || !s->coarse_codes || !s->detail_pack) {
            wfc_hierarchy_destroy(w);
            return WFC_ENOMEM;
        }
        memset(s->residual_field, 0, s->ncells * sizeof(float));
        memset(s->recon_field, 0, s->ncells * sizeof(float));
        memset(s->detail_pack, 0, s->ncells * sizeof(float));
        for (uint32_t i = 0; i < s->ncells; ++i)
            s->coarse_codes[i] = WFC_PATTERN_INVALID;
    }

    return WFC_OK;
}

/* A new family, depth, or target makes the stored Mallat split stale. */
static void wfc_hierarchy_invalidate(wfc_hierarchy_t* h)
{
    uint32_t k;
    if (!h) return;
    h->decomposed = 0;
    if (!h->levels) return;
    for (k = 0; k < h->nlevels; ++k) {
        wfc_scale_t* s = &h->levels[k];
        s->detail_lh = NULL;
        s->detail_hl = NULL;
        s->detail_hh = NULL;
        s->detail_count = 0;
        s->octave_lo = 0;
        s->octave_hi = 0;
    }
}

wfc_err_t wfc_hierarchy_set_wavelet(wfc_world_t* w, int family, uint32_t dwt_levels)
{
    if (!w || !w->hierarchy) return WFC_EUNINIT;
    if (family < 0 || family >= (int)FAF_WAVELET_COUNT) return WFC_EINVAL;
    w->hierarchy->wavelet_family = family;
    w->hierarchy->wavelet_levels = dwt_levels;
    wfc_hierarchy_invalidate(w->hierarchy);

    /* Family drives spectral synthesis and the residual DWT. dwt_levels is
     * the residual depth (0 = every octave). Tear plans down so the next
     * transform is built at that depth instead of keeping the old wavelet. */
    w->faf_wavelet_family = family;
    w->faf_wavelet_levels = dwt_levels;
    if (w->faf_initialized)
        wfc_faf_teardown_transforms(w);
    return WFC_OK;
}

wfc_err_t wfc_hierarchy_set_target(wfc_world_t* w, const float* target, uint32_t count)
{
    if (!w || !w->hierarchy || !target) return WFC_EINVAL;
    if (count != w->num_cells) return WFC_EINVAL;

    if (!w->hierarchy->target_field) {
        w->hierarchy->target_field = (float*)faf_aligned_alloc(count * sizeof(float));
        if (!w->hierarchy->target_field) return WFC_ENOMEM;
    }
    memcpy(w->hierarchy->target_field, target, count * sizeof(float));
    wfc_hierarchy_invalidate(w->hierarchy);
    return WFC_OK;
}

wfc_err_t wfc_hierarchy_set_mode(wfc_world_t* w, int mode)
{
    if (!w || !w->hierarchy) return WFC_EUNINIT;
    if (mode != 0 && mode != 1) return WFC_EINVAL;
    w->hierarchy->mode = mode;
    return WFC_OK;
}

float wfc_hierarchy_residual_energy(const wfc_world_t* w, uint32_t level)
{
    if (!w || !w->hierarchy || level >= w->hierarchy->nlevels) return 0.0f;
    const wfc_scale_t* s = &w->hierarchy->levels[level];
    if (!s->detail_pack || s->detail_count == 0) return 0.0f;
    double acc = 0.0;
    for (uint32_t i = 0; i < s->detail_count; ++i) {
        double v = (double)s->detail_pack[i];
        acc += v * v;
    }
    return (float)sqrt(acc);
}

/* Create a temporary world at a given scale that shares pattern/compat data */
static wfc_world_t* wfc_hierarchy_create_level_world(wfc_world_t* parent, const wfc_scale_t* scale)
{
    wfc_config_t cfg = {0};
    cfg.nx = scale->nx;
    cfg.ny = scale->ny;
    cfg.nz = 1;
    cfg.seed = (uint32_t)(parent->rng_state & 0xFFFFFFFF);
    cfg.enable_spectral = parent->spectral_enabled;
    cfg.spectral_weight = parent->spectral_weight;
#if defined(WFC_ENABLE_COPULA)
    cfg.enable_copula = parent->copula_enabled;
    cfg.copula_weight = parent->copula_weight;
#endif

    wfc_world_t* lw = wfc_world_create(&cfg);
    if (!lw) return NULL;

    lw->faf_wavelet_family = parent->faf_wavelet_family;
    lw->faf_wavelet_levels = parent->faf_wavelet_levels;

    /* Copy pattern catalogue */
    for (uint32_t i = 0; i < parent->num_patterns; ++i) {
        wfc_add_pattern(lw, &parent->patterns[i]);
    }

    /* Copy compat table (shared, read-only) */
    memcpy(lw->compat, parent->compat, sizeof(lw->compat));

    /* Copy class limits */
    memcpy(lw->class_min, parent->class_min, sizeof(lw->class_min));
    memcpy(lw->class_max, parent->class_max, sizeof(lw->class_max));
    lw->class_constrained = parent->class_constrained;
    lw->spectral_guides_choice = 1;

    return lw;
}

/* 1-level inverse DWT: Mallat LL holds `src`, details are zero.
 * This is the single octave between two adjacent hierarchy scales (2×).
 * Residual depth is h->wavelet_levels, not this step. Family shapes the synthesis:
 * Haar is piecewise-constant 2×2, CDF 9/7 interpolates. */
static wfc_err_t wfc_hierarchy_dwt_synthesize(wfc_world_t* lw,
                                              const float* WFC_RESTRICT src,
                                              uint32_t snx, uint32_t sny,
                                              float* WFC_RESTRICT dst,
                                              uint32_t dnx, uint32_t dny)
{
    uint32_t saved = lw->faf_wavelet_levels;
    lw->faf_wavelet_levels = 1;
    if (lw->faf_initialized) wfc_faf_teardown_transforms(lw);
    wfc_err_t err = wfc_faf_ensure_transforms(lw);
    if (err != WFC_OK) {
        lw->faf_wavelet_levels = saved;
        return err;
    }

    uint32_t nx = lw->nx_p2, ny = lw->ny_p2;
    size_t n = 2u * (size_t)nx * (size_t)ny;
    float* buf = (float*)faf_aligned_alloc(n * sizeof(float));
    if (!buf) {
        lw->faf_wavelet_levels = saved;
        return WFC_ENOMEM;
    }
    memset(buf, 0, n * sizeof(float));

    uint32_t hx = nx / 2, hy = ny / 2;
    for (uint32_t y = 0; y < sny && y < hy; ++y)
        for (uint32_t x = 0; x < snx && x < hx; ++x)
            buf[2 * (y * nx + x)] = src[y * snx + x];

    wfc_faf_dwt_2d_inverse(lw, buf);

    memset(dst, 0, (size_t)dnx * (size_t)dny * sizeof(float));
    for (uint32_t y = 0; y < dny && y < ny; ++y)
        for (uint32_t x = 0; x < dnx && x < nx; ++x)
            dst[y * dnx + x] = buf[2 * (y * nx + x)];

    faf_aligned_free(buf);
    lw->faf_wavelet_levels = saved;
    wfc_faf_teardown_transforms(lw);
    return WFC_OK;
}

static uint32_t wfc_hierarchy_full_depth(uint32_t nx, uint32_t ny)
{
    uint32_t m = nx < ny ? nx : ny;
    uint32_t d = 0;
    while (m > 1) { m >>= 1; ++d; }
    return d;
}

static uint32_t wfc_hierarchy_analysis_depth(const wfc_hierarchy_t* h, uint32_t nx, uint32_t ny)
{
    uint32_t full = wfc_hierarchy_full_depth(nx, ny);
    if (full == 0) return 0;
    uint32_t d = (h->wavelet_levels == 0) ? full : h->wavelet_levels;
    if (d > full) d = full;
    return d;
}

static double wfc_hierarchy_octave_ss(const float* coeff, uint32_t nx, uint32_t ny, uint32_t j)
{
    uint32_t hx = (nx >> j) >> 1;
    uint32_t hy = (ny >> j) >> 1;
    double acc = 0.0;
    if (hx == 0 || hy == 0) return 0.0;
    for (uint32_t r = 0; r < hy; ++r) {
        for (uint32_t c = 0; c < hx; ++c) {
            double lh = coeff[(size_t)r * 2u * nx + 2u * (hx + c)];
            double hl = coeff[(size_t)(hy + r) * 2u * nx + 2u * c];
            double hh = coeff[(size_t)(hy + r) * 2u * nx + 2u * (hx + c)];
            acc += lh * lh + hl * hl + hh * hh;
        }
    }
    return acc;
}

static double wfc_hierarchy_range_ss(const double* ss, uint32_t lo, uint32_t hi)
{
    double acc = 0.0;
    for (uint32_t j = lo; j < hi; ++j) acc += ss[j];
    return acc;
}

/* Octave 0 is the finest Mallat band. Level 0 is the coarsest WFC scale.
 * A shallow depth fills the finest levels first. When there are more octaves
 * than levels, extra fine bands start on the finest level and move onto the
 * next coarser level until each level's detail energy is below the one
 * before it. That keeps nlevels in the partition: fewer levels leave a
 * larger pack on the finest scale. */
static void wfc_hierarchy_assign_octaves(uint32_t nlevels, uint32_t depth,
                                         const double* ss, uint32_t* lo, uint32_t* hi)
{
    for (uint32_t level = 0; level < nlevels; ++level) lo[level] = hi[level] = 0;
    if (depth == 0) return;
    if (depth < nlevels) {
        for (uint32_t i = 0; i < depth; ++i) {
            uint32_t level = nlevels - 1u - i;
            lo[level] = i;
            hi[level] = i + 1u;
        }
        return;
    }
    for (uint32_t level = 0; level + 1u < nlevels; ++level) {
        uint32_t octave = (depth - 1u) - level;
        lo[level] = octave;
        hi[level] = octave + 1u;
    }
    lo[nlevels - 1u] = 0;
    hi[nlevels - 1u] = depth - (nlevels - 1u);

    for (;;) {
        int moved = 0;
        for (uint32_t level = nlevels - 1u; level > 0; --level) {
            if (hi[level] <= lo[level] + 1u) continue;
            if (lo[level - 1u] == 0 || lo[level - 1u] != hi[level]) continue;
            if (wfc_hierarchy_range_ss(ss, lo[level], hi[level])
                < wfc_hierarchy_range_ss(ss, lo[level - 1u], hi[level - 1u]))
                continue;
            hi[level] -= 1u;
            lo[level - 1u] -= 1u;
            moved = 1;
        }
        if (!moved) break;
    }
}

static uint32_t wfc_hierarchy_octave_samples(uint32_t nx, uint32_t ny, uint32_t j)
{
    uint32_t bx = nx >> j;
    uint32_t by = ny >> j;
    if (bx < 2 || by < 2) return 0;
    return 3u * (bx >> 1) * (by >> 1);
}

static void wfc_hierarchy_copy_octave(const float* coeff, uint32_t nx, uint32_t ny,
                                      uint32_t j, float* dst)
{
    uint32_t hx = (nx >> j) >> 1;
    uint32_t hy = (ny >> j) >> 1;
    float* lh = dst;
    float* hl = lh + (size_t)hx * hy;
    float* hh = hl + (size_t)hx * hy;
    for (uint32_t r = 0; r < hy; ++r) {
        for (uint32_t c = 0; c < hx; ++c) {
            lh[(size_t)r * hx + c] = coeff[(size_t)r * 2u * nx + 2u * (hx + c)];
            hl[(size_t)r * hx + c] = coeff[(size_t)(hy + r) * 2u * nx + 2u * c];
            hh[(size_t)r * hx + c] = coeff[(size_t)(hy + r) * 2u * nx + 2u * (hx + c)];
        }
    }
}

static void wfc_hierarchy_scatter_octave(float* coeff, uint32_t nx, uint32_t ny,
                                         uint32_t j, const float* src)
{
    uint32_t hx = (nx >> j) >> 1;
    uint32_t hy = (ny >> j) >> 1;
    const float* lh = src;
    const float* hl = lh + (size_t)hx * hy;
    const float* hh = hl + (size_t)hx * hy;
    for (uint32_t r = 0; r < hy; ++r) {
        for (uint32_t c = 0; c < hx; ++c) {
            size_t i_lh = (size_t)r * 2u * nx + 2u * (hx + c);
            size_t i_hl = (size_t)(hy + r) * 2u * nx + 2u * c;
            size_t i_hh = (size_t)(hy + r) * 2u * nx + 2u * (hx + c);
            coeff[i_lh]     = lh[(size_t)r * hx + c];
            coeff[i_lh + 1] = 0.0f;
            coeff[i_hl]     = hl[(size_t)r * hx + c];
            coeff[i_hl + 1] = 0.0f;
            coeff[i_hh]     = hh[(size_t)r * hx + c];
            coeff[i_hh + 1] = 0.0f;
        }
    }
}

static wfc_err_t wfc_hierarchy_grow_pack(wfc_scale_t* s, uint32_t need)
{
    if (need <= s->detail_cap && s->detail_pack) return WFC_OK;
    if (need == 0) return WFC_OK;
    float* p = (float*)faf_aligned_alloc((size_t)need * sizeof(float));
    if (!p) return WFC_ENOMEM;
    faf_aligned_free(s->detail_pack);
    s->detail_pack = p;
    s->detail_cap = need;
    s->detail_lh = s->detail_hl = s->detail_hh = NULL;
    return WFC_OK;
}

static wfc_err_t wfc_hierarchy_extract(wfc_hierarchy_t* h, const float* coeff,
                                       uint32_t nx, uint32_t ny, uint32_t depth)
{
    double* ss = (double*)WFC_MALLOC((size_t)depth * sizeof(double));
    uint32_t* lo = (uint32_t*)WFC_MALLOC((size_t)h->nlevels * sizeof(uint32_t));
    uint32_t* hi = (uint32_t*)WFC_MALLOC((size_t)h->nlevels * sizeof(uint32_t));
    if (!ss || !lo || !hi || depth == 0) {
        WFC_FREE(ss);
        WFC_FREE(lo);
        WFC_FREE(hi);
        return depth == 0 ? WFC_EINVAL : WFC_ENOMEM;
    }
    for (uint32_t j = 0; j < depth; ++j)
        ss[j] = wfc_hierarchy_octave_ss(coeff, nx, ny, j);
    wfc_hierarchy_assign_octaves(h->nlevels, depth, ss, lo, hi);
    WFC_FREE(ss);

    for (uint32_t level = 0; level < h->nlevels; ++level) {
        wfc_scale_t* s = &h->levels[level];
        uint32_t need = 0;
        for (uint32_t j = lo[level]; j < hi[level]; ++j)
            need += wfc_hierarchy_octave_samples(nx, ny, j);
        wfc_err_t err = wfc_hierarchy_grow_pack(s, need);
        if (err != WFC_OK) {
            WFC_FREE(lo);
            WFC_FREE(hi);
            return err;
        }
        s->octave_lo = lo[level];
        s->octave_hi = hi[level];
        s->detail_count = need;
        s->detail_lh = s->detail_hl = s->detail_hh = NULL;
        if (need == 0) continue;
        uint32_t off = 0;
        for (uint32_t j = lo[level]; j < hi[level]; ++j) {
            uint32_t n = wfc_hierarchy_octave_samples(nx, ny, j);
            if (n == 0) continue;
            wfc_hierarchy_copy_octave(coeff, nx, ny, j, s->detail_pack + off);
            /* First owned octave is the view. Its plane is (nx>>j)/2 by (ny>>j)/2,
             * which is this scale's half-resolution band when the level owns that octave. */
            if (!s->detail_lh) {
                uint32_t hx = (nx >> j) >> 1;
                uint32_t hy = (ny >> j) >> 1;
                s->detail_lh = s->detail_pack + off;
                s->detail_hl = s->detail_lh + (size_t)hx * hy;
                s->detail_hh = s->detail_hl + (size_t)hx * hy;
            }
            off += n;
        }
    }
    WFC_FREE(lo);
    WFC_FREE(hi);
    return WFC_OK;
}

static wfc_err_t wfc_hierarchy_bind_depth(wfc_world_t* w, uint32_t depth)
{
    if (w->faf_initialized) wfc_faf_teardown_transforms(w);
    w->faf_wavelet_levels = depth;
    wfc_err_t err = wfc_faf_ensure_transforms(w);
    if (err != WFC_OK) wfc_faf_teardown_transforms(w);
    return err;
}

static void wfc_hierarchy_unbind(wfc_world_t* w, uint32_t saved_levels)
{
    wfc_faf_teardown_transforms(w);
    w->faf_wavelet_levels = saved_levels;
}

/* Inverse of `scale`'s stored detail octaves. LL and every other band stay zero.
 * `w` must already hold a forward/inverse pair built at the analysis depth. */
static wfc_err_t wfc_hierarchy_synth_pack(wfc_world_t* w, const wfc_scale_t* scale,
                                          uint32_t nx, uint32_t ny, float* full_out)
{
    uint32_t nxy = nx * ny;
    if (!scale->detail_pack || scale->detail_count == 0 || scale->octave_lo == scale->octave_hi) {
        memset(full_out, 0, (size_t)nxy * sizeof(float));
        return WFC_OK;
    }
    float* buf = (float*)faf_aligned_alloc(2u * (size_t)nxy * sizeof(float));
    if (!buf) return WFC_ENOMEM;
    memset(buf, 0, 2u * (size_t)nxy * sizeof(float));
    uint32_t off = 0;
    for (uint32_t j = scale->octave_lo; j < scale->octave_hi; ++j) {
        uint32_t n = wfc_hierarchy_octave_samples(nx, ny, j);
        if (n == 0) continue;
        wfc_hierarchy_scatter_octave(buf, nx, ny, j, scale->detail_pack + off);
        off += n;
    }
    wfc_faf_dwt_2d_inverse(w, buf);
    for (uint32_t i = 0; i < nxy; ++i) full_out[i] = buf[2u * i];
    faf_aligned_free(buf);
    return WFC_OK;
}

/* Family-dependent low-pass: inverse of the deepest LL, details discarded
 * from this reconstruction only. The detail samples already live on the scales. */
static wfc_err_t wfc_hierarchy_store_lowpass(wfc_world_t* w, const float* coeff,
                                             uint32_t nx, uint32_t ny, uint32_t depth)
{
    uint32_t nxy = nx * ny;
    float* buf = (float*)faf_aligned_alloc(2u * (size_t)nxy * sizeof(float));
    if (!buf) return WFC_ENOMEM;
    memcpy(buf, coeff, 2u * (size_t)nxy * sizeof(float));
    uint32_t llx = nx >> depth;
    uint32_t lly = ny >> depth;
    if (llx == 0) llx = 1;
    if (lly == 0) lly = 1;
    for (uint32_t y = 0; y < ny; ++y) {
        for (uint32_t x = 0; x < nx; ++x) {
            if (y >= lly || x >= llx) {
                buf[2u * ((size_t)y * nx + x)] = 0.0f;
                buf[2u * ((size_t)y * nx + x) + 1] = 0.0f;
            }
        }
    }
    wfc_faf_dwt_2d_inverse(w, buf);
    float* spatial = (float*)faf_aligned_alloc((size_t)nxy * sizeof(float));
    if (!spatial) { faf_aligned_free(buf); return WFC_ENOMEM; }
    for (uint32_t i = 0; i < nxy; ++i) spatial[i] = buf[2u * i];
    faf_aligned_free(buf);
    wfc_scale_t* coarse = &w->hierarchy->levels[0];
    wfc_hierarchy_project_target(w, spatial, coarse->nx, coarse->ny, coarse->residual_field);
    faf_aligned_free(spatial);
    return WFC_OK;
}

static wfc_err_t wfc_hierarchy_analyze(wfc_world_t* w)
{
    wfc_hierarchy_t* h = w->hierarchy;
    uint32_t nx = w->nx_p2, ny = w->ny_p2;
    uint32_t nxy = nx * ny;
    uint32_t depth = wfc_hierarchy_analysis_depth(h, nx, ny);
    if (depth == 0 || nxy == 0 || w->num_cells < nxy) return WFC_EINVAL;

    if (!h->running) {
        h->running = (float*)faf_aligned_alloc((size_t)w->num_cells * sizeof(float));
        if (!h->running) return WFC_ENOMEM;
    }
    memcpy(h->running, h->target_field, (size_t)w->num_cells * sizeof(float));

    uint32_t saved = w->faf_wavelet_levels;
    w->faf_wavelet_family = h->wavelet_family;
    wfc_err_t err = wfc_hierarchy_bind_depth(w, depth);
    if (err != WFC_OK) {
        wfc_hierarchy_unbind(w, saved);
        return err;
    }

    float* buf = (float*)faf_aligned_alloc(2u * (size_t)nxy * sizeof(float));
    if (!buf) {
        wfc_hierarchy_unbind(w, saved);
        return WFC_ENOMEM;
    }
    memset(buf, 0, 2u * (size_t)nxy * sizeof(float));
    for (uint32_t i = 0; i < nxy; ++i) buf[2u * i] = h->target_field[i];

    wfc_faf_dwt_2d_forward(w, buf);
    err = wfc_hierarchy_extract(h, buf, nx, ny, depth);
    if (err == WFC_OK)
        err = wfc_hierarchy_store_lowpass(w, buf, nx, ny, depth);
    faf_aligned_free(buf);
    wfc_hierarchy_unbind(w, saved);
    return err;
}

/* Snapshot the generated fBm/Matérn field and split it once. Later collapse
 * reuses that Mallat packing and only restores the running residual. */
wfc_err_t wfc_hierarchy_adopt_spectral(wfc_world_t* w)
{
    wfc_err_t err;
    if (!w || !w->hierarchy) return WFC_EUNINIT;
    if (!w->spectral_enabled || !w->spectral_field) return WFC_EUNINIT;
    err = wfc_hierarchy_set_target(w, w->spectral_field, w->num_cells);
    if (err != WFC_OK) return err;
    err = wfc_hierarchy_analyze(w);
    if (err != WFC_OK) return err;
    w->hierarchy->decomposed = 1;
    return WFC_OK;
}

/* 1-level inverse on an interleaved buffer. Serial, so it matches the
 * level-world upsample on grids that do not take the parallel path. */
static wfc_err_t wfc_hierarchy_idwt_1level(int family, float* data, uint32_t nx, uint32_t ny)
{
    faf_transform* row_fwd = NULL;
    faf_transform* row_inv = NULL;
    faf_transform* col_fwd = NULL;
    faf_transform* col_inv = NULL;
    int row_shared = 0, col_shared = 0;
    float* ws = NULL;
    float* col = NULL;
    wfc_err_t err = WFC_OK;
    if (wfc_faf_acquire_plan(family, nx, 1, &row_fwd, &row_inv, &row_shared) != 0)
        return WFC_ENOMEM;
    if (ny == nx) {
        col_fwd = row_fwd;
        col_inv = row_inv;
        col_shared = 1;
    } else if (wfc_faf_acquire_plan(family, ny, 1, &col_fwd, &col_inv, &col_shared) != 0) {
        if (!row_shared) {
            faf_destroy_transform(row_inv);
            faf_destroy_transform(row_fwd);
        }
        return WFC_ENOMEM;
    }

    ws = (float*)faf_aligned_alloc(2u * (size_t)nx * ny * sizeof(float));
    col = (float*)faf_aligned_alloc(2u * (size_t)(nx > ny ? nx : ny) * sizeof(float));
    if (!ws || !col) { err = WFC_ENOMEM; goto done; }

    for (uint32_t i = 0; i < nx; ++i) {
        for (uint32_t j = 0; j < ny; ++j) {
            col[2u * j]     = data[(size_t)j * 2u * nx + 2u * i];
            col[2u * j + 1] = data[(size_t)j * 2u * nx + 2u * i + 1];
        }
        faf_execute_f32(col_inv, ws, col);
        for (uint32_t j = 0; j < ny; ++j) {
            data[(size_t)j * 2u * nx + 2u * i]     = ws[2u * j];
            data[(size_t)j * 2u * nx + 2u * i + 1] = ws[2u * j + 1];
        }
    }
    for (uint32_t j = 0; j < ny; ++j) {
        float* row = data + (size_t)j * 2u * nx;
        faf_execute_f32(row_inv, ws + (size_t)j * 2u * nx, row);
    }
    memcpy(data, ws, 2u * (size_t)nx * ny * sizeof(float));

done:
    faf_aligned_free(ws);
    faf_aligned_free(col);
    if (!row_shared) {
        if (row_inv) faf_destroy_transform(row_inv);
        if (row_fwd) faf_destroy_transform(row_fwd);
    }
    if (!col_shared) {
        if (col_inv) faf_destroy_transform(col_inv);
        if (col_fwd) faf_destroy_transform(col_fwd);
    }
    return err;
}

static wfc_err_t wfc_hierarchy_expand_sig(int family, const float* sig,
                                          uint32_t snx, uint32_t sny,
                                          float* dst, uint32_t dnx, uint32_t dny)
{
    if (snx == 0 || sny == 0 || dnx == 0 || dny == 0) return WFC_EINVAL;
    float* cur = (float*)faf_aligned_alloc((size_t)snx * sny * sizeof(float));
    if (!cur) return WFC_ENOMEM;
    memcpy(cur, sig, (size_t)snx * sny * sizeof(float));
    uint32_t cx = snx, cy = sny;
    while (cx < dnx || cy < dny) {
        uint32_t nx = cx << 1;
        uint32_t ny = cy << 1;
        float* inter;
        float* next;
        wfc_err_t err;
        if (nx > dnx || ny > dny) {
            faf_aligned_free(cur);
            return WFC_EINVAL;
        }
        inter = (float*)faf_aligned_alloc(2u * (size_t)nx * ny * sizeof(float));
        next = (float*)faf_aligned_alloc((size_t)nx * ny * sizeof(float));
        if (!inter || !next) {
            faf_aligned_free(inter);
            faf_aligned_free(next);
            faf_aligned_free(cur);
            return WFC_ENOMEM;
        }
        memset(inter, 0, 2u * (size_t)nx * ny * sizeof(float));
        for (uint32_t y = 0; y < cy; ++y)
            for (uint32_t x = 0; x < cx; ++x)
                inter[2u * ((size_t)y * nx + x)] = cur[(size_t)y * cx + x];
        err = wfc_hierarchy_idwt_1level(family, inter, nx, ny);
        if (err != WFC_OK) {
            faf_aligned_free(inter);
            faf_aligned_free(next);
            faf_aligned_free(cur);
            return err;
        }
        for (uint32_t i = 0; i < nx * ny; ++i) next[i] = inter[2u * i];
        faf_aligned_free(inter);
        faf_aligned_free(cur);
        cur = next;
        cx = nx;
        cy = ny;
    }
    memcpy(dst, cur, (size_t)dnx * dny * sizeof(float));
    faf_aligned_free(cur);
    return WFC_OK;
}

/* z -= R_k. Pure approximation passes a null detail field (details stay in z
 * until the finer level that is biased by them). */
static wfc_err_t wfc_hierarchy_subtract_level(wfc_world_t* parent, wfc_hierarchy_t* h,
                                              uint32_t level, const float* detail_full)
{
    wfc_scale_t* scale = &h->levels[level];
    uint32_t nxy = parent->nx_p2 * parent->ny_p2;
    float* sig;
    float* full;
    wfc_err_t err;
    if (!h->running) return WFC_OK;
    sig = (float*)faf_aligned_alloc((size_t)scale->ncells * sizeof(float));
    full = (float*)faf_aligned_alloc((size_t)nxy * sizeof(float));
    if (!sig || !full) {
        faf_aligned_free(sig);
        faf_aligned_free(full);
        return WFC_ENOMEM;
    }
    for (uint32_t i = 0; i < scale->ncells; ++i)
        sig[i] = wfc_pattern_signature(parent, scale->coarse_codes[i]);
    err = wfc_hierarchy_expand_sig(h->wavelet_family, sig, scale->nx, scale->ny,
                                   full, parent->nx_p2, parent->ny_p2);
    faf_aligned_free(sig);
    if (err != WFC_OK) { faf_aligned_free(full); return err; }
    for (uint32_t i = 0; i < nxy && i < parent->num_cells; ++i)
        h->running[i] -= full[i];
    faf_aligned_free(full);
    if (detail_full) {
        for (uint32_t i = 0; i < nxy && i < parent->num_cells; ++i)
            h->running[i] -= detail_full[i];
    }
    return WFC_OK;
}

static wfc_err_t wfc_hierarchy_solve_scale(wfc_world_t* parent, wfc_scale_t* scale,
                                           int write_back, int guide,
                                           const float* observation)
{
    wfc_world_t* lw = wfc_hierarchy_create_level_world(parent, scale);
    wfc_err_t err;
    if (!lw) return WFC_ENOMEM;
    if (lw->spectral_enabled && lw->spectral_field) {
        if (!guide) lw->spectral_guides_choice = 0;
        else if (observation)
            memcpy(lw->spectral_field, observation, (size_t)scale->ncells * sizeof(float));
    }
    err = wfc_collapse(lw);
    if (err != WFC_OK) { wfc_world_destroy(lw); return err; }
    for (uint32_t y = 0; y < scale->ny; ++y)
        for (uint32_t x = 0; x < scale->nx; ++x)
            scale->coarse_codes[y * scale->nx + x] =
                lw->cells[wfc_idx(lw, x, y, 0)].collapsed;
    if (write_back) {
        for (uint32_t y = 0; y < parent->ny; ++y) {
            for (uint32_t x = 0; x < parent->nx; ++x) {
                uint32_t dst = wfc_idx(parent, x, y, 0);
                uint32_t src = wfc_idx(lw, x, y, 0);
                parent->cells[dst].collapsed = lw->cells[src].collapsed;
                memcpy(parent->cells[dst].mask, lw->cells[src].mask,
                       sizeof(parent->cells[dst].mask));
                parent->cells[dst].entropy = lw->cells[src].entropy;
                parent->cells[dst].dirty = 0;
            }
        }
    }
    parent->rng_state = lw->rng_state;
    wfc_world_destroy(lw);
    return WFC_OK;
}

/* Wavelet 2× upsample of coarse_codes into a float field at fine resolution.
 * Coarse signatures occupy the Mallat LL band; inverse DWT (configured family)
 * synthesizes the fine recon. Haar is piecewise-constant 2×2; CDF 9/7 interpolates. */
static wfc_err_t wfc_hierarchy_upsample(wfc_world_t* fine_lw,
                                        const wfc_world_t* WFC_RESTRICT parent,
                                        const wfc_scale_t* WFC_RESTRICT coarse,
                                        float* WFC_RESTRICT fine_field,
                                        uint32_t fine_nx, uint32_t fine_ny)
{
    float* ll = (float*)faf_aligned_alloc((size_t)coarse->ncells * sizeof(float));
    if (!ll) return WFC_ENOMEM;
    for (uint32_t i = 0; i < coarse->ncells; ++i)
        ll[i] = wfc_pattern_signature(parent, coarse->coarse_codes[i]);
    wfc_err_t err = wfc_hierarchy_dwt_synthesize(fine_lw, ll, coarse->nx, coarse->ny,
                                                 fine_field, fine_nx, fine_ny);
    faf_aligned_free(ll);
    return err;
}

/* Coarse-to-fine. Plans for `depth` are already bound when a target is set.
 *   k = 0: observe the LL low-pass (pure approximation). Owned details are kept.
 *   k > 0: residual = upsample(signatures) + spatial synthesis of this level's
 *          details. Level 1 also adds the coarsest details, which are the
 *          residual the approximation left behind.
 *   Then z -= R_k(signatures, and the details this level just explained).
 *   k = 0 with no target: guides stay off. A zero field would pull every
 *          signature toward 0. Finer levels with no target observe the upsample.
 * Coarse pattern ids are not locked. The finest level writes back to `w`. */
static wfc_err_t wfc_hierarchy_collapse_ctf(wfc_world_t* w)
{
    wfc_hierarchy_t* h = w->hierarchy;
    uint32_t nx = w->nx_p2, ny = w->ny_p2, nxy = nx * ny;

    for (uint32_t k = 0; k < h->nlevels; ++k) {
        wfc_scale_t* scale = &h->levels[k];
        int guide = 0;
        float* detail_full = NULL;
        wfc_err_t err = WFC_OK;

        if (!h->target_field) {
            if (k > 0) {
                wfc_world_t* lw = wfc_hierarchy_create_level_world(w, scale);
                if (!lw) return WFC_ENOMEM;
                err = wfc_hierarchy_upsample(lw, w, &h->levels[k - 1],
                                             scale->recon_field, scale->nx, scale->ny);
                wfc_world_destroy(lw);
                if (err != WFC_OK) return err;
                memcpy(scale->residual_field, scale->recon_field,
                       (size_t)scale->ncells * sizeof(float));
                guide = 1;
            }
        } else if (k == 0) {
            guide = 1;
        } else {
            wfc_world_t* lw = wfc_hierarchy_create_level_world(w, scale);
            float* extra = NULL;
            float* proj = NULL;
            if (!lw) return WFC_ENOMEM;
            err = wfc_hierarchy_upsample(lw, w, &h->levels[k - 1],
                                         scale->recon_field, scale->nx, scale->ny);
            wfc_world_destroy(lw);
            if (err != WFC_OK) return err;

            detail_full = (float*)faf_aligned_alloc((size_t)nxy * sizeof(float));
            extra = (float*)faf_aligned_alloc((size_t)nxy * sizeof(float));
            proj = (float*)faf_aligned_alloc((size_t)scale->ncells * sizeof(float));
            if (!detail_full || !extra || !proj) {
                faf_aligned_free(detail_full);
                faf_aligned_free(extra);
                faf_aligned_free(proj);
                return WFC_ENOMEM;
            }
            err = wfc_hierarchy_synth_pack(w, scale, nx, ny, detail_full);
            if (err == WFC_OK && k == 1)
                err = wfc_hierarchy_synth_pack(w, &h->levels[0], nx, ny, extra);
            else if (err == WFC_OK)
                memset(extra, 0, (size_t)nxy * sizeof(float));
            if (err != WFC_OK) {
                faf_aligned_free(detail_full);
                faf_aligned_free(extra);
                faf_aligned_free(proj);
                return err;
            }
            for (uint32_t i = 0; i < nxy; ++i) detail_full[i] += extra[i];
            faf_aligned_free(extra);
            wfc_hierarchy_project_target(w, detail_full, scale->nx, scale->ny, proj);
            for (uint32_t i = 0; i < scale->ncells; ++i) {
                float obs = scale->recon_field[i] + proj[i];
                if (obs < 0.0f) obs = 0.0f;
                else if (obs > 1.0f) obs = 1.0f;
                scale->residual_field[i] = obs;
            }
            faf_aligned_free(proj);
            guide = 1;
        }

        err = wfc_hierarchy_solve_scale(w, scale, k == h->finest_idx, guide,
                                        guide ? scale->residual_field : NULL);
        if (err != WFC_OK) { faf_aligned_free(detail_full); return err; }
        if (h->target_field) {
            err = wfc_hierarchy_subtract_level(w, h, k, k == 0 ? NULL : detail_full);
            faf_aligned_free(detail_full);
            if (err != WFC_OK) return err;
        }
    }
    return WFC_OK;
}

/* Fine-to-coarse analysis. Each level is biased by its own detail bands.
 * The running residual loses that contribution before the next coarser solve.
 * Only the finest solve writes back. */
static wfc_err_t wfc_hierarchy_collapse_ftc(wfc_world_t* w)
{
    wfc_hierarchy_t* h = w->hierarchy;
    uint32_t nx = w->nx_p2, ny = w->ny_p2, nxy = nx * ny;

    for (int k = (int)h->nlevels - 1; k >= 0; --k) {
        wfc_scale_t* scale = &h->levels[k];
        int guide = 0;
        float* detail_full = NULL;
        wfc_err_t err;

        if (h->target_field && h->running) {
            float* proj = (float*)faf_aligned_alloc((size_t)scale->ncells * sizeof(float));
            if (!proj) return WFC_ENOMEM;
            detail_full = (float*)faf_aligned_alloc((size_t)nxy * sizeof(float));
            if (!detail_full) { faf_aligned_free(proj); return WFC_ENOMEM; }
            wfc_hierarchy_project_target(w, h->running, scale->nx, scale->ny,
                                         scale->residual_field);
            err = wfc_hierarchy_synth_pack(w, scale, nx, ny, detail_full);
            if (err != WFC_OK) {
                faf_aligned_free(proj);
                faf_aligned_free(detail_full);
                return err;
            }
            wfc_hierarchy_project_target(w, detail_full, scale->nx, scale->ny, proj);
            for (uint32_t i = 0; i < scale->ncells; ++i) {
                float obs = scale->residual_field[i] + proj[i];
                if (obs < 0.0f) obs = 0.0f;
                else if (obs > 1.0f) obs = 1.0f;
                scale->residual_field[i] = obs;
            }
            faf_aligned_free(proj);
            guide = 1;
        }

        err = wfc_hierarchy_solve_scale(w, scale, (uint32_t)k == h->finest_idx, guide,
                                        guide ? scale->residual_field : NULL);
        if (err != WFC_OK) { faf_aligned_free(detail_full); return err; }
        if (h->target_field) {
            err = wfc_hierarchy_subtract_level(w, h, (uint32_t)k, detail_full);
            faf_aligned_free(detail_full);
            if (err != WFC_OK) return err;
        }
    }
    return WFC_OK;
}

/* Residual update, coarse to fine (mode 0) unless set_mode selected analysis.
 * z^(0) is the target. Each level keeps its LH/HL/HH samples.
 *   hat z^(k) = hat z^(k-1) + R_k(signatures, details_k)
 *   z^(k)     = z^(k-1) - hat z^(k)
 * R_0 is the pure LL approximation (details are not subtracted there).
 * The next finer observation is the upsampled signatures plus those details.
 * dwt_levels (0 = full) is how many Mallat octaves that split uses.
 * adopt_spectral leaves that split in place. Collapse then only copies the
 * stored target back into the running residual. */
wfc_err_t wfc_hierarchy_collapse(wfc_world_t* w)
{
    wfc_hierarchy_t* h;
    uint32_t saved;
    int bound = 0;
    wfc_err_t err;

    if (!w || !w->hierarchy) return WFC_EUNINIT;

    /* Coupling is the spectral observation. Without it the cascade would be
     * an ordinary flat solve, so refuse instead of pretending. */
    if (!w->spectral_enabled) return WFC_EUNINIT;

    h = w->hierarchy;
    saved = w->faf_wavelet_levels;
    if (h->target_field) {
        if (h->decomposed && h->running) {
            memcpy(h->running, h->target_field, (size_t)w->num_cells * sizeof(float));
        } else {
            err = wfc_hierarchy_analyze(w);
            if (err != WFC_OK) return err;
        }
        w->faf_wavelet_family = h->wavelet_family;
        err = wfc_hierarchy_bind_depth(w, wfc_hierarchy_analysis_depth(h, w->nx_p2, w->ny_p2));
        if (err != WFC_OK) {
            wfc_hierarchy_unbind(w, saved);
            return err;
        }
        bound = 1;
    }

    err = (h->mode == 1) ? wfc_hierarchy_collapse_ftc(w)
                         : wfc_hierarchy_collapse_ctf(w);
    if (bound) wfc_hierarchy_unbind(w, saved);
    return err;
}

void wfc_hierarchy_destroy(wfc_world_t* w)
{
    if (!w || !w->hierarchy) return;
    wfc_hierarchy_t* h = w->hierarchy;
    for (uint32_t k = 0; k < h->nlevels; ++k) {
        h->levels[k].detail_lh = NULL;
        h->levels[k].detail_hl = NULL;
        h->levels[k].detail_hh = NULL;
        faf_aligned_free(h->levels[k].residual_field);
        faf_aligned_free(h->levels[k].recon_field);
        faf_aligned_free(h->levels[k].coarse_codes);
        faf_aligned_free(h->levels[k].detail_pack);
    }
    WFC_FREE(h->levels);
    faf_aligned_free(h->target_field);
    faf_aligned_free(h->running);
    WFC_FREE(h);
    w->hierarchy = NULL;
}

#endif /* WFC_ENABLE_HIERARCHY */

/* ============================================================
 * Suspenders.h coroutine parallelism
 * ============================================================ */
#if defined(WFC_ENABLE_SUSPENDERS)

/* --- Atomic barrier for foreign-thread synchronization --- */

typedef struct {
    wfc_atomic_u32_t remaining;
} wfc_par_sync_t;

static inline void wfc_par_sync_init(wfc_par_sync_t* s, uint32_t count)
{
    WFC_ATOMIC_STORE(&s->remaining, count, release);
}

static inline void wfc_par_sync_done(wfc_par_sync_t* s)
{
    WFC_ATOMIC_FETCH_SUB(&s->remaining, 1, acq_rel);
}

static inline void wfc_par_sync_wait(wfc_par_sync_t* s)
{
    while (WFC_ATOMIC_LOAD(&s->remaining, acquire) > 0)
        sched_yield();
    /* Spawn allocated from this thread's memento heap. Park it once the
     * tasks have finished so a later runtime shutdown, which frees every
     * heap, does not leave the TLS pointer dangling. */
    memento_heap_park();
}

/* --- Task descriptors --- */

typedef struct {
    faf_transform *xform;
    float         *src;
    float         *dst;
    wfc_par_sync_t *sync;
} wfc_par_row_task_t;

static void wfc_par_row_fn(void* arg)
{
    wfc_par_row_task_t* t = (wfc_par_row_task_t*)arg;
    faf_execute_f32(t->xform, t->dst, t->src);
    wfc_par_sync_done(t->sync);
}

typedef struct {
    faf_transform *xform;
    float         *data;        /* full 2D interleaved buffer */
    float         *col_buf;     /* private gather buffer (2*ny) */
    float         *col_out;     /* private output buffer (2*ny) */
    uint32_t       col_idx;
    uint32_t       nx;
    uint32_t       ny;
    wfc_par_sync_t *sync;
} wfc_par_col_task_t;

static void wfc_par_col_fn(void* arg)
{
    wfc_par_col_task_t* t = (wfc_par_col_task_t*)arg;
    uint32_t i = t->col_idx;
    uint32_t nx = t->nx, ny = t->ny;

    /* Gather column from strided 2D layout */
    for (uint32_t j = 0; j < ny; ++j) {
        t->col_buf[2*j]   = t->data[j * 2 * nx + 2*i];
        t->col_buf[2*j+1] = t->data[j * 2 * nx + 2*i+1];
    }

    faf_execute_f32(t->xform, t->col_out, t->col_buf);

    /* Scatter back */
    for (uint32_t j = 0; j < ny; ++j) {
        t->data[j * 2 * nx + 2*i]   = t->col_out[2*j];
        t->data[j * 2 * nx + 2*i+1] = t->col_out[2*j+1];
    }

    wfc_par_sync_done(t->sync);
}

typedef struct {
    float    *buf;
    uint32_t  start;
    uint32_t  end;
    uint64_t  rng_seed;
    wfc_par_sync_t *sync;
} wfc_par_noise_task_t;

static void wfc_par_noise_fn(void* arg)
{
    wfc_par_noise_task_t* t = (wfc_par_noise_task_t*)arg;
    uint64_t rng = t->rng_seed;
    for (uint32_t i = t->start; i < t->end; i += 2) {
        double u1 = ((double)((rng = rng * 6364136223846793005ULL + 1442695040888963407ULL) >> 33) / (double)(1ULL << 31));
        double u2 = ((double)((rng = rng * 6364136223846793005ULL + 1442695040888963407ULL) >> 33) / (double)(1ULL << 31));
        if (u1 < 1e-7) u1 = 1e-7;
        double r = sqrt(-2.0 * log(u1));
        double th = 2.0 * M_PI * u2;
        t->buf[2*i]   = (float)(r * cos(th));
        t->buf[2*i+1] = 0.0f;
        if (i + 1 < t->end) {
            t->buf[2*(i+1)]   = (float)(r * sin(th));
            t->buf[2*(i+1)+1] = 0.0f;
        }
    }
    wfc_par_sync_done(t->sync);
}

/* --- Parallel 2D DWT --- */

static inline void wfc_faf_dwt_2d_forward_par(wfc_world_t* w, float* data)
{
    uint32_t nx = w->nx_p2, ny = w->ny_p2;

    /* Row pass: each row is independent */
    {
        wfc_par_sync_t sync;
        wfc_par_sync_init(&sync, ny);
        wfc_par_row_task_t* tasks = (wfc_par_row_task_t*)WFC_MALLOC(ny * sizeof(wfc_par_row_task_t));
        float* row_out = w->faf_workspace;
        for (uint32_t j = 0; j < ny; ++j) {
            tasks[j].xform = w->faf_row_fwd;
            tasks[j].src   = data + j * 2 * nx;
            tasks[j].dst   = row_out + j * 2 * nx;
            tasks[j].sync  = &sync;
            suspenders_go(wfc_par_row_fn, &tasks[j]);
        }
        wfc_par_sync_wait(&sync);
        memcpy(data, row_out, 2 * (size_t)nx * ny * sizeof(float));
        WFC_FREE(tasks);
    }

    /* Column pass: each column is independent, needs private buffers */
    {
        wfc_par_sync_t sync;
        wfc_par_sync_init(&sync, nx);
        wfc_par_col_task_t* tasks = (wfc_par_col_task_t*)WFC_MALLOC(nx * sizeof(wfc_par_col_task_t));
        float* scratch = w->faf_par_scratch;
        for (uint32_t i = 0; i < nx; ++i) {
            tasks[i].xform   = w->faf_col_fwd;
            tasks[i].data    = data;
            tasks[i].col_buf = scratch + i * 4 * ny;       /* 2*ny for gather */
            tasks[i].col_out = scratch + i * 4 * ny + 2*ny; /* 2*ny for output */
            tasks[i].col_idx = i;
            tasks[i].nx      = nx;
            tasks[i].ny      = ny;
            tasks[i].sync    = &sync;
            suspenders_go(wfc_par_col_fn, &tasks[i]);
        }
        wfc_par_sync_wait(&sync);
        WFC_FREE(tasks);
    }
}

static inline void wfc_faf_dwt_2d_inverse_par(wfc_world_t* w, float* data)
{
    uint32_t nx = w->nx_p2, ny = w->ny_p2;

    /* Inverse column pass first */
    {
        wfc_par_sync_t sync;
        wfc_par_sync_init(&sync, nx);
        wfc_par_col_task_t* tasks = (wfc_par_col_task_t*)WFC_MALLOC(nx * sizeof(wfc_par_col_task_t));
        float* scratch = w->faf_par_scratch;
        for (uint32_t i = 0; i < nx; ++i) {
            tasks[i].xform   = w->faf_col_inv;
            tasks[i].data    = data;
            tasks[i].col_buf = scratch + i * 4 * ny;
            tasks[i].col_out = scratch + i * 4 * ny + 2*ny;
            tasks[i].col_idx = i;
            tasks[i].nx      = nx;
            tasks[i].ny      = ny;
            tasks[i].sync    = &sync;
            suspenders_go(wfc_par_col_fn, &tasks[i]);
        }
        wfc_par_sync_wait(&sync);
        WFC_FREE(tasks);
    }

    /* Inverse row pass */
    {
        wfc_par_sync_t sync;
        wfc_par_sync_init(&sync, ny);
        wfc_par_row_task_t* tasks = (wfc_par_row_task_t*)WFC_MALLOC(ny * sizeof(wfc_par_row_task_t));
        float* row_out = w->faf_workspace;
        for (uint32_t j = 0; j < ny; ++j) {
            tasks[j].xform = w->faf_row_inv;
            tasks[j].src   = data + j * 2 * nx;
            tasks[j].dst   = row_out + j * 2 * nx;
            tasks[j].sync  = &sync;
            suspenders_go(wfc_par_row_fn, &tasks[j]);
        }
        wfc_par_sync_wait(&sync);
        memcpy(data, row_out, 2 * (size_t)nx * ny * sizeof(float));
        WFC_FREE(tasks);
    }
}

/* --- Parallel noise generation --- */

static inline void wfc_fill_noise_par(float* buf, uint32_t n, uint64_t noise_seed,
                                      unsigned num_workers)
{
    uint32_t chunk = (n + num_workers - 1) / num_workers;
    /* Round up to even for Box-Muller pairs */
    if (chunk & 1) chunk++;
    uint32_t ntasks = (n + chunk - 1) / chunk;

    wfc_par_sync_t sync;
    wfc_par_sync_init(&sync, ntasks);
    wfc_par_noise_task_t* tasks = (wfc_par_noise_task_t*)WFC_MALLOC(ntasks * sizeof(wfc_par_noise_task_t));
    for (uint32_t k = 0; k < ntasks; ++k) {
        tasks[k].buf      = buf;
        tasks[k].start    = k * chunk;
        tasks[k].end      = (k + 1) * chunk < n ? (k + 1) * chunk : n;
        tasks[k].rng_seed = noise_seed ^ ((uint64_t)k * 0x9E3779B97F4A7C15ULL);
        tasks[k].sync     = &sync;
        suspenders_go(wfc_par_noise_fn, &tasks[k]);
    }
    wfc_par_sync_wait(&sync);
    WFC_FREE(tasks);
}

/* --- Process-level suspenders runtime ---
 *
 * init, run, and shutdown run on one thread. suspenders_run returns as
 * soon as nothing is parked, so one coroutine stays suspended for the
 * life of the runtime and the foreign threads can submit DWT work.
 * Worlds share that runtime (the first init picks the worker count).
 * The last wfc_parallel_shutdown calls suspenders_request_stop and joins.
 * The run thread then calls suspenders_shutdown, which reaps the parked
 * coroutine. Memento v3.0.3 drops its per-CPU heap slots on that
 * shutdown, so the next init is a clean start.
 */

static pthread_mutex_t wfc_susp_mu = PTHREAD_MUTEX_INITIALIZER;
static int             wfc_susp_refs = 0;
#ifdef __cplusplus
static wfc_atomic_int_t  wfc_susp_ready{0};
static wfc_atomic_int_t  wfc_susp_init_err{0};
#else
static wfc_atomic_int_t  wfc_susp_ready    = 0;
static wfc_atomic_int_t  wfc_susp_init_err = 0;
#endif
static pthread_t   wfc_susp_thread;
static unsigned    wfc_susp_nworkers;

typedef struct {
    unsigned num_workers;
} wfc_par_boot_t;

static void wfc_par_hold_fn(void* arg)
{
    (void)arg;
    for (;;) suspenders_suspend();
}

static void* wfc_par_run_thread_fn(void* arg)
{
    wfc_par_boot_t *boot = (wfc_par_boot_t*)arg;

    int err = suspenders_init(boot->num_workers, 0);
    if (err != SUSPENDERS_OK) {
        WFC_ATOMIC_STORE(&wfc_susp_init_err, 1, release);
        WFC_ATOMIC_STORE(&wfc_susp_ready, 1, release);
        return NULL;
    }

    if (!suspenders_go(wfc_par_hold_fn, NULL)) {
        suspenders_shutdown();
        WFC_ATOMIC_STORE(&wfc_susp_init_err, 1, release);
        WFC_ATOMIC_STORE(&wfc_susp_ready, 1, release);
        return NULL;
    }

    WFC_ATOMIC_STORE(&wfc_susp_init_err, 0, release);
    WFC_ATOMIC_STORE(&wfc_susp_ready, 1, release);

    suspenders_run();
    suspenders_shutdown();
    return NULL;
}

/* Caller holds wfc_susp_mu. */
static int wfc_susp_runtime_start(unsigned num_workers)
{
    static wfc_par_boot_t boot;
    boot.num_workers = num_workers;

    WFC_ATOMIC_STORE(&wfc_susp_ready, 0, release);
    WFC_ATOMIC_STORE(&wfc_susp_init_err, 0, release);

    if (pthread_create(&wfc_susp_thread, NULL, wfc_par_run_thread_fn, &boot) != 0)
        return -1;

    while (!WFC_ATOMIC_LOAD(&wfc_susp_ready, acquire))
        sched_yield();

    if (WFC_ATOMIC_LOAD(&wfc_susp_init_err, acquire)) {
        pthread_join(wfc_susp_thread, NULL);
        return -1;
    }
    wfc_susp_nworkers = num_workers;
    return 0;
}

/* Caller holds wfc_susp_mu. Joining here keeps the next init from
 * starting until this runtime has released Memento. Park first: this
 * thread may still own a heap from an earlier spawn. */
static void wfc_susp_runtime_stop(void)
{
    memento_heap_park();
    suspenders_request_stop();
    pthread_join(wfc_susp_thread, NULL);
}

wfc_err_t wfc_parallel_init(wfc_world_t* w, unsigned num_workers)
{
    if (!w) return WFC_EINVAL;
    if (w->susp_initialized) return WFC_OK;
    if (num_workers == 0) num_workers = 4;

    size_t scratch_sz = 4 * (size_t)w->nx_p2 * w->ny_p2 * sizeof(float);
    float* scratch = (float*)WFC_MALLOC(scratch_sz);
    if (!scratch) return WFC_ENOMEM;

    pthread_mutex_lock(&wfc_susp_mu);
    if (wfc_susp_refs == 0) {
        if (wfc_susp_runtime_start(num_workers) != 0) {
            pthread_mutex_unlock(&wfc_susp_mu);
            WFC_FREE(scratch);
            return WFC_ENOMEM;
        }
    }
    wfc_susp_refs++;
    w->susp_num_workers = wfc_susp_nworkers;
    pthread_mutex_unlock(&wfc_susp_mu);

    w->faf_par_scratch = scratch;
    w->susp_initialized = 1;
    return WFC_OK;
}

void wfc_parallel_shutdown(wfc_world_t* w)
{
    if (!w || !w->susp_initialized) return;
    w->susp_initialized = 0;
    WFC_FREE(w->faf_par_scratch);
    w->faf_par_scratch = NULL;
    w->susp_num_workers = 0;

    pthread_mutex_lock(&wfc_susp_mu);
    if (wfc_susp_refs > 0) wfc_susp_refs--;
    if (wfc_susp_refs == 0) wfc_susp_runtime_stop();
    pthread_mutex_unlock(&wfc_susp_mu);
}

#endif /* WFC_ENABLE_SUSPENDERS */

#endif /* WFC_IMPLEMENTATION */
