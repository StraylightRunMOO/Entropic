/*
 * Tests for the multiscale hierarchy.
 */
#define WFC_IMPLEMENTATION
#ifndef WFC_ENABLE_COPULA
#define WFC_ENABLE_COPULA
#endif
#ifndef WFC_ENABLE_BACKTRACK
#define WFC_ENABLE_BACKTRACK
#endif
#ifndef WFC_USE_FAF
#define WFC_USE_FAF
#endif
#ifndef WFC_ENABLE_HIERARCHY
#define WFC_ENABLE_HIERARCHY
#endif
#include "entropic/wfc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static wfc_world_t* make_world(uint32_t nx, uint32_t ny, uint32_t seed)
{
    wfc_config_t cfg = {0};
    cfg.nx = nx;
    cfg.ny = ny;
    cfg.nz = 1;
    cfg.seed = seed;
    cfg.enable_spectral = 1;
    cfg.spectral_weight = 1.0f;

    wfc_world_t* w = wfc_world_create(&cfg);
    assert(w && "world create failed");

    wfc_pattern_def_t p0 = {0, 0, 5.0f};
    wfc_pattern_def_t p1 = {1, 1, 3.0f};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);

    for (int d = 0; d < 4; ++d) {
        wfc_add_rule(w, 0, 0, d);
        wfc_add_rule(w, 0, 1, d);
        wfc_add_rule(w, 1, 0, d);
        wfc_add_rule(w, 1, 1, d);
    }
    return w;
}

static void test_init_destroy(void)
{
    printf("  init/destroy...");
    wfc_world_t* w = make_world(32, 32, 1);
    wfc_err_t err = wfc_hierarchy_init(w, 3);
    assert(err == WFC_OK);
    assert(w->hierarchy != NULL);
    assert(w->hierarchy->nlevels == 3);
    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_level_dimensions(void)
{
    printf("  level dimensions...");
    wfc_world_t* w = make_world(64, 64, 2);
    wfc_err_t err = wfc_hierarchy_init(w, 3);
    assert(err == WFC_OK);

    /* 3 levels for 64x64: 16x16, 32x32, 64x64 */
    assert(w->hierarchy->levels[0].nx == 16);
    assert(w->hierarchy->levels[0].ny == 16);
    assert(w->hierarchy->levels[1].nx == 32);
    assert(w->hierarchy->levels[1].ny == 32);
    assert(w->hierarchy->levels[2].nx == 64);
    assert(w->hierarchy->levels[2].ny == 64);

    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_two_level_collapse(void)
{
    printf("  2-level collapse...");
    wfc_world_t* w = make_world(16, 16, 3);
    wfc_err_t err = wfc_hierarchy_init(w, 2);
    assert(err == WFC_OK);

    err = wfc_hierarchy_collapse(w);
    assert(err == WFC_OK);

    /* All logical cells should be collapsed */
    for (uint32_t y = 0; y < 16; ++y)
        for (uint32_t x = 0; x < 16; ++x) {
            uint16_t pat = wfc_get_cell_pattern(w, x, y, 0);
            assert(pat == 0 || pat == 1);
        }

    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_three_level_collapse(void)
{
    printf("  3-level collapse...");
    wfc_world_t* w = make_world(32, 32, 4);
    wfc_err_t err = wfc_hierarchy_init(w, 3);
    assert(err == WFC_OK);

    err = wfc_hierarchy_collapse(w);
    assert(err == WFC_OK);

    for (uint32_t y = 0; y < 32; ++y)
        for (uint32_t x = 0; x < 32; ++x) {
            uint16_t pat = wfc_get_cell_pattern(w, x, y, 0);
            assert(pat == 0 || pat == 1);
        }

    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_set_wavelet(void)
{
    printf("  set_wavelet...");
    wfc_world_t* w = make_world(32, 32, 5);
    wfc_err_t err = wfc_hierarchy_init(w, 2);
    assert(err == WFC_OK);

    err = wfc_hierarchy_set_wavelet(w, FAF_WAVELET_HAAR, 3);
    assert(err == WFC_OK);
    assert(w->hierarchy->wavelet_family == FAF_WAVELET_HAAR);
    assert(w->hierarchy->wavelet_levels == 3);

    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_with_target(void)
{
    printf("  collapse with target...");
    wfc_world_t* w = make_world(32, 32, 6);
    wfc_err_t err = wfc_hierarchy_init(w, 3);
    assert(err == WFC_OK);

    err = wfc_set_gumbel_field(w, 0.7, 99);
    assert(err == WFC_OK);

    err = wfc_hierarchy_set_target(w, w->spectral_field, w->num_cells);
    assert(err == WFC_OK);

    err = wfc_hierarchy_collapse(w);
    assert(err == WFC_OK);

    for (uint32_t y = 0; y < 32; ++y)
        for (uint32_t x = 0; x < 32; ++x) {
            uint16_t pat = wfc_get_cell_pattern(w, x, y, 0);
            assert(pat == 0 || pat == 1);
        }

    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_residual_nonzero(void)
{
    printf("  residual non-zero with target...");
    wfc_world_t* w = make_world(32, 32, 8);
    wfc_err_t err = wfc_hierarchy_init(w, 3);
    assert(err == WFC_OK);

    /* Strong spatial ramp so the box-averaged residual cannot vanish. */
    float* tgt = (float*)malloc(w->num_cells * sizeof(float));
    assert(tgt);
    for (uint32_t y = 0; y < w->ny_p2; ++y)
        for (uint32_t x = 0; x < w->nx_p2; ++x)
            tgt[y * w->nx_p2 + x] = (float)x + 2.0f * (float)y;

    err = wfc_hierarchy_set_target(w, tgt, w->num_cells);
    assert(err == WFC_OK);
    err = wfc_hierarchy_collapse(w);
    assert(err == WFC_OK);

    int nonzero = 0;
    for (uint32_t k = 0; k < w->hierarchy->nlevels; ++k) {
        wfc_scale_t* s = &w->hierarchy->levels[k];
        for (uint32_t i = 0; i < s->ncells; ++i) {
            if (s->residual_field[i] != 0.0f) { nonzero = 1; break; }
        }
        if (nonzero) break;
    }
    assert(nonzero && "expected non-zero residual after targeted collapse");

    free(tgt);
    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_finest_writes_back(void)
{
    printf("  finest level writes back...");
    wfc_world_t* w = make_world(32, 32, 9);
    wfc_err_t err = wfc_hierarchy_init(w, 3);
    assert(err == WFC_OK);

    err = wfc_hierarchy_collapse(w);
    assert(err == WFC_OK);

    wfc_scale_t* finest = &w->hierarchy->levels[w->hierarchy->finest_idx];
    assert(finest->nx == w->nx_p2);
    assert(finest->ny == w->ny_p2);
    for (uint32_t y = 0; y < w->ny; ++y) {
        for (uint32_t x = 0; x < w->nx; ++x) {
            uint16_t from_world = wfc_get_cell_pattern(w, x, y, 0);
            uint16_t from_scale = finest->coarse_codes[y * finest->nx + x];
            assert(from_world == from_scale);
            assert(from_world == 0 || from_world == 1);
        }
    }

    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_nonsquare_levels(void)
{
    printf("  non-square / non-p2 grid...");
    wfc_world_t* w = make_world(48, 32, 10);
    wfc_err_t err = wfc_hierarchy_init(w, 3);
    assert(err == WFC_OK);
    /* 48×32 pads to 64×32; 3 levels → 16×8, 32×16, 64×32 */
    assert(w->hierarchy->levels[0].nx == 16);
    assert(w->hierarchy->levels[0].ny == 8);
    assert(w->hierarchy->levels[1].nx == 32);
    assert(w->hierarchy->levels[1].ny == 16);
    assert(w->hierarchy->levels[2].nx == 64);
    assert(w->hierarchy->levels[2].ny == 32);

    err = wfc_hierarchy_collapse(w);
    assert(err == WFC_OK);
    for (uint32_t y = 0; y < 32; ++y)
        for (uint32_t x = 0; x < 48; ++x) {
            uint16_t pat = wfc_get_cell_pattern(w, x, y, 0);
            assert(pat == 0 || pat == 1);
        }

    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_invalid_args(void)
{
    printf("  invalid args...");
    wfc_world_t* w = make_world(16, 16, 7);

    /* nlevels < 2 */
    assert(wfc_hierarchy_init(w, 1) == WFC_EINVAL);
    /* nlevels > 8 */
    assert(wfc_hierarchy_init(w, 9) == WFC_EINVAL);
    /* coarsest scale would be < 4 (16 >> 4 == 1) */
    assert(wfc_hierarchy_init(w, 5) == WFC_EINVAL);

    /* set_wavelet before init */
    assert(wfc_hierarchy_set_wavelet(w, FAF_WAVELET_HAAR, 3) == WFC_EUNINIT);

    wfc_world_destroy(w);
    printf(" PASS\n");
}

int main(void)
{
    printf("=== Hierarchy Tests ===\n\n");

    test_init_destroy();
    test_level_dimensions();
    test_two_level_collapse();
    test_three_level_collapse();
    test_set_wavelet();
    test_with_target();
    test_residual_nonzero();
    test_finest_writes_back();
    test_nonsquare_levels();
    test_invalid_args();

    printf("\nAll hierarchy tests passed.\n");
    return 0;
}
