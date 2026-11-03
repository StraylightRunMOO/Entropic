/*
 * Tests for suspenders.h coroutine parallelism integration.
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
#ifndef WFC_ENABLE_SUSPENDERS
#define WFC_ENABLE_SUSPENDERS
#endif
#include "entropic/wfc.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

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

static void test_lifecycle(void)
{
    printf("  lifecycle...");
    wfc_world_t* w = make_world(32, 32, 1);
    wfc_err_t err = wfc_parallel_init(w, 4);
    assert(err == WFC_OK);
    assert(w->susp_initialized == 1);
    assert(w->susp_num_workers == 4);
    assert(w->faf_par_scratch != NULL);
    wfc_parallel_shutdown(w);
    assert(w->susp_initialized == 0);
    assert(w->faf_par_scratch == NULL);
    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_double_init(void)
{
    printf("  double init...");
    wfc_world_t* w = make_world(32, 32, 2);
    wfc_err_t err = wfc_parallel_init(w, 4);
    assert(err == WFC_OK);
    err = wfc_parallel_init(w, 4);
    assert(err == WFC_OK);
    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_gumbel_parallel(void)
{
    printf("  gumbel field (parallel DWT)...");
    wfc_world_t* w = make_world(64, 64, 3);
    wfc_err_t err = wfc_parallel_init(w, 4);
    assert(err == WFC_OK);

    err = wfc_set_gumbel_field(w, 0.7, 12345);
    assert(err == WFC_OK);

    /* Verify the field is non-trivial */
    float min_v = 1e9f, max_v = -1e9f;
    for (uint32_t i = 0; i < w->num_cells; ++i) {
        float v = w->spectral_field[i];
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }
    assert(max_v - min_v > 0.01f && "spectral field should have variation");

    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_matern_parallel(void)
{
    printf("  matern field (parallel DWT)...");
    wfc_world_t* w = make_world(64, 64, 4);
    wfc_err_t err = wfc_parallel_init(w, 4);
    assert(err == WFC_OK);

    err = wfc_set_matern_field(w, 1.0, 1.5, 54321);
    assert(err == WFC_OK);

    float min_v = 1e9f, max_v = -1e9f;
    for (uint32_t i = 0; i < w->num_cells; ++i) {
        float v = w->spectral_field[i];
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }
    assert(max_v - min_v > 0.001f && "matern field should have variation");

    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_collapse_with_parallel(void)
{
    printf("  collapse with parallel DWT...");
    wfc_world_t* w = make_world(64, 64, 5);
    wfc_err_t err = wfc_parallel_init(w, 4);
    assert(err == WFC_OK);

    err = wfc_set_gumbel_field(w, 0.7, 99);
    assert(err == WFC_OK);

    err = wfc_collapse(w);
    assert(err == WFC_OK);

    for (uint32_t y = 0; y < 64; ++y)
        for (uint32_t x = 0; x < 64; ++x) {
            uint16_t pat = wfc_get_cell_pattern(w, x, y, 0);
            assert(pat == 0 || pat == 1);
        }

    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_hierarchy_with_parallel(void)
{
    printf("  hierarchy + parallel...");
    wfc_world_t* w = make_world(32, 32, 6);
    wfc_err_t err = wfc_parallel_init(w, 4);
    assert(err == WFC_OK);

    err = wfc_hierarchy_init(w, 3);
    assert(err == WFC_OK);

    err = wfc_set_gumbel_field(w, 0.7, 42);
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

static void test_single_worker(void)
{
    printf("  single worker...");
    wfc_world_t* w = make_world(64, 64, 7);
    wfc_err_t err = wfc_parallel_init(w, 1);
    assert(err == WFC_OK);

    err = wfc_set_gumbel_field(w, 0.5, 777);
    assert(err == WFC_OK);

    err = wfc_collapse(w);
    assert(err == WFC_OK);

    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_multiple_fields(void)
{
    printf("  multiple field generations...");
    wfc_world_t* w = make_world(64, 64, 8);
    wfc_err_t err = wfc_parallel_init(w, 4);
    assert(err == WFC_OK);

    err = wfc_set_gumbel_field(w, 0.7, 100);
    assert(err == WFC_OK);
    err = wfc_set_gumbel_field(w, 0.3, 200);
    assert(err == WFC_OK);
    err = wfc_set_matern_field(w, 2.0, 1.0, 300);
    assert(err == WFC_OK);

    wfc_world_destroy(w);
    printf(" PASS\n");
}

static void test_small_grid_serial_fallback(void)
{
    printf("  small grid (serial fallback)...");
    wfc_world_t* w = make_world(8, 8, 9);
    wfc_err_t err = wfc_parallel_init(w, 4);
    assert(err == WFC_OK);

    err = wfc_set_gumbel_field(w, 0.7, 555);
    assert(err == WFC_OK);

    err = wfc_collapse(w);
    assert(err == WFC_OK);

    wfc_world_destroy(w);
    printf(" PASS\n");
}

int main(void)
{
    printf("=== Suspenders Parallelism Tests ===\n\n");

    test_lifecycle();
    test_double_init();
    test_gumbel_parallel();
    test_matern_parallel();
    test_collapse_with_parallel();
    test_hierarchy_with_parallel();
    test_single_worker();
    test_multiple_fields();
    test_small_grid_serial_fallback();

    printf("\nAll suspenders tests passed.\n");
    return 0;
}
