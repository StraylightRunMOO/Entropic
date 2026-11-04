/*
 * Wavelet residual hierarchy. These checks fail if detail bands are dropped
 * or if nlevels / the wavelet family do not change the cascade.
 */
#define WFC_IMPLEMENTATION
#ifndef WFC_USE_FAF
#define WFC_USE_FAF
#endif
#ifndef WFC_ENABLE_HIERARCHY
#define WFC_ENABLE_HIERARCHY
#endif
#include "entropic/wfc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

static wfc_world_t* make_world(uint32_t nx, uint32_t ny, uint32_t seed, int npat, float lambda)
{
    wfc_config_t cfg = {0};
    cfg.nx = nx;
    cfg.ny = ny;
    cfg.nz = 1;
    cfg.seed = seed;
    cfg.enable_spectral = 1;
    cfg.spectral_weight = lambda;

    wfc_world_t* w = wfc_world_create(&cfg);
    assert(w);
    for (int i = 0; i < npat; ++i) {
        wfc_pattern_def_t p = {(uint16_t)i, (uint16_t)i, 1.0f};
        assert(wfc_add_pattern(w, &p) == WFC_OK);
    }
    for (int a = 0; a < npat; ++a)
        for (int b = 0; b < npat; ++b)
            for (int d = 0; d < 4; ++d)
                assert(wfc_add_rule(w, (uint16_t)a, (uint16_t)b, (wfc_dir_t)d) == WFC_OK);
    return w;
}

static float* make_target(const wfc_world_t* w, uint32_t salt)
{
    float* t = (float*)malloc((size_t)w->num_cells * sizeof(float));
    assert(t);
    uint64_t rng = 0x9e3779b97f4a7c15ULL ^ salt;
    for (uint32_t y = 0; y < w->ny_p2; ++y) {
        for (uint32_t x = 0; x < w->nx_p2; ++x) {
            rng = rng * 6364136223846793005ULL + 1ULL;
            float noise = (((rng >> 40) / 16777216.0f) - 0.5f) * 0.12f;
            float g = 0.15f
                    + 0.55f * (float)x / (float)(w->nx_p2 - 1)
                    + 0.22f * (float)y / (float)(w->ny_p2 - 1);
            float v = g + noise;
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            t[y * w->nx_p2 + x] = v;
        }
    }
    return t;
}

static uint32_t hamming(const wfc_world_t* a, const wfc_world_t* b)
{
    uint32_t d = 0;
    assert(a->nx == b->nx && a->ny == b->ny);
    for (uint32_t y = 0; y < a->ny; ++y)
        for (uint32_t x = 0; x < a->nx; ++x)
            if (wfc_get_cell_pattern(a, x, y, 0) != wfc_get_cell_pattern(b, x, y, 0))
                ++d;
    return d;
}

static double residual_l2(const wfc_world_t* a, const wfc_world_t* b, uint32_t level)
{
    const wfc_scale_t* sa = &a->hierarchy->levels[level];
    const wfc_scale_t* sb = &b->hierarchy->levels[level];
    double acc = 0.0;
    uint32_t n = sa->ncells < sb->ncells ? sa->ncells : sb->ncells;
    for (uint32_t i = 0; i < n; ++i) {
        double d = (double)sa->residual_field[i] - (double)sb->residual_field[i];
        acc += d * d;
    }
    return sqrt(acc);
}

static double grid_hf(const wfc_world_t* w)
{
    double acc = 0.0;
    for (uint32_t y = 0; y < w->ny; ++y) {
        for (uint32_t x = 0; x < w->nx; ++x) {
            float s = wfc_pattern_signature(w, wfc_get_cell_pattern(w, x, y, 0));
            if (x + 1 < w->nx) {
                float t = wfc_pattern_signature(w, wfc_get_cell_pattern(w, x + 1, y, 0));
                double d = (double)s - (double)t;
                acc += d * d;
            }
            if (y + 1 < w->ny) {
                float t = wfc_pattern_signature(w, wfc_get_cell_pattern(w, x, y + 1, 0));
                double d = (double)s - (double)t;
                acc += d * d;
            }
        }
    }
    return acc;
}

static void test_energy_decreases(void)
{
    printf("  residual energy decreases...");
    wfc_world_t* w = make_world(64, 64, 11, 8, 12.0f);
    float* tgt = make_target(w, 11);
    assert(wfc_hierarchy_init(w, 4) == WFC_OK);
    assert(wfc_hierarchy_set_wavelet(w, FAF_WAVELET_HAAR, 0) == WFC_OK);
    assert(wfc_hierarchy_set_target(w, tgt, w->num_cells) == WFC_OK);
    assert(wfc_hierarchy_collapse(w) == WFC_OK);

    float e[4];
    for (uint32_t k = 0; k < 4; ++k) {
        e[k] = wfc_hierarchy_residual_energy(w, k);
        assert(e[k] > 0.0f && "detail energy is zero; bands were dropped");
        if (k > 0) {
            if (!(e[k] < e[k - 1])) {
                fprintf(stderr, "\n  energies");
                for (uint32_t i = 0; i < 4; ++i) fprintf(stderr, " %.5f", e[i]);
                fprintf(stderr, "\n");
            }
            assert(e[k] < e[k - 1] && "detail energy must fall toward finer levels");
        }
    }
    printf(" %.4f > %.4f > %.4f > %.4f PASS\n", e[0], e[1], e[2], e[3]);
    free(tgt);
    wfc_world_destroy(w);
}

static void test_family_differs(void)
{
    printf("  Haar vs CDF 9/7...");
    wfc_world_t* a = make_world(32, 32, 21, 8, 12.0f);
    wfc_world_t* b = make_world(32, 32, 21, 8, 12.0f);
    float* tgt = make_target(a, 21);
    assert(wfc_hierarchy_init(a, 3) == WFC_OK);
    assert(wfc_hierarchy_init(b, 3) == WFC_OK);
    assert(wfc_hierarchy_set_wavelet(a, FAF_WAVELET_HAAR, 0) == WFC_OK);
    assert(wfc_hierarchy_set_wavelet(b, FAF_WAVELET_CDF97, 0) == WFC_OK);
    assert(wfc_hierarchy_set_target(a, tgt, a->num_cells) == WFC_OK);
    assert(wfc_hierarchy_set_target(b, tgt, b->num_cells) == WFC_OK);
    assert(wfc_hierarchy_collapse(a) == WFC_OK);
    assert(wfc_hierarchy_collapse(b) == WFC_OK);

    uint32_t ham = hamming(a, b);
    double l2 = 0.0;
    for (uint32_t k = 0; k < a->hierarchy->nlevels; ++k)
        l2 += residual_l2(a, b, k);
    assert((ham > 0 || l2 > 0.0) && "family did not change the cascade");
    printf(" Hamming %u, residual L2 %.4f PASS\n", ham, l2);
    free(tgt);
    wfc_world_destroy(a);
    wfc_world_destroy(b);
}

static void test_levels_knob(void)
{
    printf("  nlevels 2 vs 4...");
    wfc_world_t* a = make_world(64, 64, 7, 8, 12.0f);
    wfc_world_t* b = make_world(64, 64, 7, 8, 12.0f);
    float* tgt = make_target(a, 7);
    assert(wfc_hierarchy_init(a, 2) == WFC_OK);
    assert(wfc_hierarchy_init(b, 4) == WFC_OK);
    assert(wfc_hierarchy_set_wavelet(a, FAF_WAVELET_HAAR, 0) == WFC_OK);
    assert(wfc_hierarchy_set_wavelet(b, FAF_WAVELET_HAAR, 0) == WFC_OK);
    assert(wfc_hierarchy_set_target(a, tgt, a->num_cells) == WFC_OK);
    assert(wfc_hierarchy_set_target(b, tgt, b->num_cells) == WFC_OK);
    assert(wfc_hierarchy_collapse(a) == WFC_OK);
    assert(wfc_hierarchy_collapse(b) == WFC_OK);

    float ea = wfc_hierarchy_residual_energy(a, a->hierarchy->finest_idx);
    float eb = wfc_hierarchy_residual_energy(b, b->hierarchy->finest_idx);
    uint32_t ham = hamming(a, b);
    assert(ea != eb && "finest residual energy ignored nlevels");
    assert(ham > 0 && "final grid ignored nlevels");

    /* dwt_levels is the decomposition depth, not a stored flag. */
    wfc_world_t* shallow = make_world(64, 64, 7, 8, 12.0f);
    wfc_world_t* full = make_world(64, 64, 7, 8, 12.0f);
    assert(wfc_hierarchy_init(shallow, 4) == WFC_OK);
    assert(wfc_hierarchy_init(full, 4) == WFC_OK);
    assert(wfc_hierarchy_set_wavelet(shallow, FAF_WAVELET_HAAR, 1) == WFC_OK);
    assert(wfc_hierarchy_set_wavelet(full, FAF_WAVELET_HAAR, 0) == WFC_OK);
    assert(wfc_hierarchy_set_target(shallow, tgt, shallow->num_cells) == WFC_OK);
    assert(wfc_hierarchy_set_target(full, tgt, full->num_cells) == WFC_OK);
    assert(wfc_hierarchy_collapse(shallow) == WFC_OK);
    assert(wfc_hierarchy_collapse(full) == WFC_OK);
    assert(wfc_hierarchy_residual_energy(shallow, 0) == 0.0f);
    assert(wfc_hierarchy_residual_energy(full, 0) > 0.0f);

    printf(" finest E %.4f vs %.4f, Hamming %u PASS\n", ea, eb, ham);
    free(tgt);
    wfc_world_destroy(a);
    wfc_world_destroy(b);
    wfc_world_destroy(shallow);
    wfc_world_destroy(full);
}

static void test_coherence(void)
{
    printf("  hierarchy vs single-scale...");
    wfc_world_t* hier = make_world(32, 32, 9, 8, 4.0f);
    wfc_world_t* flat = make_world(32, 32, 9, 8, 4.0f);
    float* tgt = make_target(hier, 9);
    assert(wfc_hierarchy_init(hier, 3) == WFC_OK);
    assert(wfc_hierarchy_set_wavelet(hier, FAF_WAVELET_CDF97, 0) == WFC_OK);
    assert(wfc_hierarchy_set_target(hier, tgt, hier->num_cells) == WFC_OK);
    assert(wfc_hierarchy_collapse(hier) == WFC_OK);

    memcpy(flat->spectral_field, tgt, (size_t)flat->num_cells * sizeof(float));
    assert(wfc_collapse(flat) == WFC_OK);

    double hf_h = grid_hf(hier);
    double hf_f = grid_hf(flat);
    assert(hf_h < hf_f && "hierarchy is not smoother than the single-scale prior");
    printf(" HF %.4f < %.4f PASS\n", hf_h, hf_f);
    free(tgt);
    wfc_world_destroy(hier);
    wfc_world_destroy(flat);
}

static void test_detail_retained(void)
{
    printf("  detail energy non-zero on every level...");
    wfc_world_t* w = make_world(64, 64, 13, 8, 8.0f);
    float* tgt = make_target(w, 13);
    assert(wfc_hierarchy_init(w, 4) == WFC_OK);
    assert(wfc_hierarchy_set_target(w, tgt, w->num_cells) == WFC_OK);
    assert(wfc_hierarchy_collapse(w) == WFC_OK);
    for (uint32_t k = 0; k < 4; ++k) {
        float e = wfc_hierarchy_residual_energy(w, k);
        assert(e > 0.0f);
        assert(w->hierarchy->levels[k].detail_lh != NULL);
        assert(w->hierarchy->levels[k].detail_hl != NULL);
        assert(w->hierarchy->levels[k].detail_hh != NULL);
    }
    free(tgt);
    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_determinism(void)
{
    printf("  determinism...");
    wfc_world_t* a = make_world(32, 32, 42, 8, 6.0f);
    wfc_world_t* b = make_world(32, 32, 42, 8, 6.0f);
    float* tgt = make_target(a, 42);
    assert(wfc_hierarchy_init(a, 3) == WFC_OK);
    assert(wfc_hierarchy_init(b, 3) == WFC_OK);
    assert(wfc_hierarchy_set_wavelet(a, FAF_WAVELET_CDF97, 0) == WFC_OK);
    assert(wfc_hierarchy_set_wavelet(b, FAF_WAVELET_CDF97, 0) == WFC_OK);
    assert(wfc_hierarchy_set_target(a, tgt, a->num_cells) == WFC_OK);
    assert(wfc_hierarchy_set_target(b, tgt, b->num_cells) == WFC_OK);
    assert(wfc_hierarchy_collapse(a) == WFC_OK);
    assert(wfc_hierarchy_collapse(b) == WFC_OK);
    assert(hamming(a, b) == 0);
    free(tgt);
    wfc_world_destroy(a);
    wfc_world_destroy(b);
    printf(" PASS\n");
}

static int energy_differs(const wfc_world_t* a, const wfc_world_t* b)
{
    uint32_t n = a->hierarchy->nlevels < b->hierarchy->nlevels
               ? a->hierarchy->nlevels : b->hierarchy->nlevels;
    for (uint32_t k = 0; k < n; ++k) {
        float ea = wfc_hierarchy_residual_energy(a, k);
        float eb = wfc_hierarchy_residual_energy(b, k);
        if (ea != eb) return 1;
    }
    return 0;
}

static float energy_sum(const wfc_world_t* w)
{
    float s = 0.0f;
    for (uint32_t k = 0; k < w->hierarchy->nlevels; ++k)
        s += wfc_hierarchy_residual_energy(w, k);
    return s;
}

/* Section 2.5: one split of a generated fBm/Matérn, not a second projection.
 * The same spatial field under Haar and CDF 9/7 must not share detail energy.
 * Collapse reuses that split, so the energies stay bit-exact. */
static void test_adopt_spectral(void)
{
    printf("  adopt spectral...");
    wfc_world_t* bare = make_world(32, 32, 4, 4, 1.0f);
    assert(wfc_hierarchy_adopt_spectral(bare) == WFC_EUNINIT);

    wfc_world_t* a = make_world(32, 32, 4, 4, 1.0f);
    wfc_world_t* b = make_world(32, 32, 4, 4, 1.0f);
    wfc_world_t* m = make_world(32, 32, 4, 4, 1.0f);
    assert(wfc_hierarchy_init(a, 3) == WFC_OK);
    assert(wfc_hierarchy_init(b, 3) == WFC_OK);
    assert(wfc_hierarchy_init(m, 3) == WFC_OK);

    /* Generate once, then copy. set_wavelet before the field would change it. */
    assert(wfc_set_gumbel_field(a, 0.7, 12345) == WFC_OK);
    memcpy(b->spectral_field, a->spectral_field, (size_t)a->num_cells * sizeof(float));
    assert(wfc_hierarchy_set_wavelet(a, FAF_WAVELET_HAAR, 0) == WFC_OK);
    assert(wfc_hierarchy_set_wavelet(b, FAF_WAVELET_CDF97, 0) == WFC_OK);
    assert(wfc_hierarchy_adopt_spectral(a) == WFC_OK);
    assert(wfc_hierarchy_adopt_spectral(b) == WFC_OK);
    assert(energy_sum(a) > 0.0f);
    assert(energy_sum(b) > 0.0f);
    assert(energy_differs(a, b) && "Haar vs CDF97 detail energy matched");

    float before = wfc_hierarchy_residual_energy(a, 0);
    assert(wfc_hierarchy_collapse(a) == WFC_OK);
    assert(wfc_hierarchy_collapse(b) == WFC_OK);
    assert(wfc_hierarchy_residual_energy(a, 0) == before);
    for (uint32_t y = 0; y < a->ny; ++y)
        for (uint32_t x = 0; x < a->nx; ++x)
            assert(wfc_get_cell_pattern(a, x, y, 0) < 4);

    assert(wfc_set_matern_field(m, 1.0, 1.5, 99) == WFC_OK);
    assert(wfc_hierarchy_adopt_spectral(m) == WFC_OK);
    assert(energy_sum(m) > 0.0f);
    float mb = wfc_hierarchy_residual_energy(m, 0);
    assert(wfc_hierarchy_collapse(m) == WFC_OK);
    assert(wfc_hierarchy_residual_energy(m, 0) == mb);

    /* No target: the live spectral field is not decomposed on its own. */
    wfc_world_t* plain = make_world(16, 16, 4, 2, 1.0f);
    assert(wfc_hierarchy_init(plain, 2) == WFC_OK);
    assert(wfc_set_gumbel_field(plain, 0.5, 3) == WFC_OK);
    assert(wfc_hierarchy_collapse(plain) == WFC_OK);
    assert(wfc_hierarchy_residual_energy(plain, 0) == 0.0f);
    assert(wfc_hierarchy_residual_energy(plain, 1) == 0.0f);

    wfc_config_t cfg = {0};
    cfg.nx = 16;
    cfg.ny = 16;
    cfg.nz = 1;
    cfg.seed = 1;
    cfg.enable_spectral = 0;
    wfc_world_t* off = wfc_world_create(&cfg);
    assert(off);
    wfc_pattern_def_t p0 = {0, 0, 1.0f};
    assert(wfc_add_pattern(off, &p0) == WFC_OK);
    assert(wfc_hierarchy_init(off, 2) == WFC_OK);
    assert(wfc_hierarchy_adopt_spectral(off) == WFC_EUNINIT);

    printf(" Haar %.4f CDF97 %.4f Matérn %.4f PASS\n",
           before, wfc_hierarchy_residual_energy(b, 0), mb);

    wfc_world_destroy(bare);
    wfc_world_destroy(a);
    wfc_world_destroy(b);
    wfc_world_destroy(m);
    wfc_world_destroy(plain);
    wfc_world_destroy(off);
}

static void test_errors_and_mode(void)
{
    printf("  errors and analysis mode...");
    wfc_world_t* w = make_world(32, 32, 3, 4, 1.0f);
    assert(wfc_hierarchy_init(w, 1) == WFC_EINVAL);
    assert(wfc_hierarchy_set_mode(w, 0) == WFC_EUNINIT);
    assert(wfc_hierarchy_set_target(w, NULL, w->num_cells) == WFC_EINVAL);
    assert(wfc_hierarchy_residual_energy(w, 0) == 0.0f);

    assert(wfc_hierarchy_init(w, 2) == WFC_OK);
    assert(wfc_hierarchy_set_mode(w, 2) == WFC_EINVAL);
    assert(wfc_hierarchy_set_mode(w, 1) == WFC_OK);
    assert(wfc_hierarchy_set_wavelet(w, 99, 1) == WFC_EINVAL);

    float* tgt = make_target(w, 3);
    assert(wfc_hierarchy_set_target(w, tgt, w->num_cells) == WFC_OK);
    assert(wfc_hierarchy_collapse(w) == WFC_OK);
    for (uint32_t y = 0; y < w->ny; ++y)
        for (uint32_t x = 0; x < w->nx; ++x) {
            uint16_t p = wfc_get_cell_pattern(w, x, y, 0);
            assert(p < 4);
        }
    free(tgt);
    wfc_world_destroy(w);

    wfc_config_t cfg = {0};
    cfg.nx = 16;
    cfg.ny = 16;
    cfg.nz = 1;
    cfg.seed = 1;
    cfg.enable_spectral = 0;
    wfc_world_t* bare = wfc_world_create(&cfg);
    assert(bare);
    wfc_pattern_def_t p0 = {0, 0, 1.0f};
    wfc_add_pattern(bare, &p0);
    for (int d = 0; d < 4; ++d) wfc_add_rule(bare, 0, 0, d);
    assert(wfc_hierarchy_init(bare, 2) == WFC_OK);
    assert(wfc_hierarchy_collapse(bare) == WFC_EUNINIT);
    wfc_world_destroy(bare);
    printf(" PASS\n");
}

int main(void)
{
    printf("=== Hierarchy Wavelet Tests ===\n\n");
    test_energy_decreases();
    test_family_differs();
    test_levels_knob();
    test_coherence();
    test_detail_retained();
    test_determinism();
    test_adopt_spectral();
    test_errors_and_mode();
    printf("\nAll hierarchy wavelet tests passed.\n");
    return 0;
}
