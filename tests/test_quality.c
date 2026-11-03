/* test_quality.c — generation quality: constraints, weights, determinism */
#undef NDEBUG
#define WFC_IMPLEMENTATION
#include "entropic/wfc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) static void test_##name(void)
#define RUN(name) do { test_count++; printf("  [%2d] " #name " ... ", test_count); fflush(stdout); test_##name(); pass_count++; printf("PASS\n"); } while(0)

static void add_full(wfc_world_t* w, int npat, int ndirs) {
    for (int i = 0; i < npat; i++)
        for (int j = 0; j < npat; j++)
            for (int d = 0; d < ndirs; d++)
                wfc_add_rule(w, (uint16_t)i, (uint16_t)j, (wfc_dir_t)d);
}

static int neighbor_ok(const wfc_world_t* w, uint32_t x, uint32_t y, uint32_t z,
                       uint32_t nx, uint32_t ny, uint32_t nz, wfc_dir_t dir) {
    uint16_t a = wfc_get_cell_pattern(w, x, y, z);
    uint16_t b = wfc_get_cell_pattern(w, nx, ny, nz);
    if (a == WFC_PATTERN_INVALID || b == WFC_PATTERN_INVALID) return 0;
    uint64_t bit = 1ULL << (b & 63);
    return (w->compat[dir][a][b >> 6] & bit) != 0;
}

static int grid_adjacency_holds(const wfc_world_t* w) {
    for (uint32_t z = 0; z < w->nz; z++) {
        for (uint32_t y = 0; y < w->ny; y++) {
            for (uint32_t x = 0; x < w->nx; x++) {
                if (x + 1 < w->nx && !neighbor_ok(w, x, y, z, x + 1, y, z, WFC_DIR_E))
                    return 0;
                if (y + 1 < w->ny && !neighbor_ok(w, x, y, z, x, y + 1, z, WFC_DIR_S))
                    return 0;
                if (w->is_3d && z + 1 < w->nz &&
                    !neighbor_ok(w, x, y, z, x, y, z + 1, WFC_DIR_D))
                    return 0;
                if (w->seamless) {
                    uint32_t wx = (x + 1) % w->nx;
                    uint32_t wy = (y + 1) % w->ny;
                    if (x + 1 == w->nx && !neighbor_ok(w, x, y, z, wx, y, z, WFC_DIR_E))
                        return 0;
                    if (y + 1 == w->ny && !neighbor_ok(w, x, y, z, x, wy, z, WFC_DIR_S))
                        return 0;
                }
            }
        }
    }
    return 1;
}

TEST(adjacency_soundness) {
    wfc_config_t cfg = {8, 8, 1, 99, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t p2 = {2, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    wfc_add_pattern(w, &p2);
    /* Not fully connected: 0-1 and 1-2, no 0-2 */
    for (int d = 0; d < 4; d++) {
        wfc_add_rule(w, 0, 0, (wfc_dir_t)d);
        wfc_add_rule(w, 1, 1, (wfc_dir_t)d);
        wfc_add_rule(w, 2, 2, (wfc_dir_t)d);
        wfc_add_rule(w, 0, 1, (wfc_dir_t)d);
        wfc_add_rule(w, 1, 0, (wfc_dir_t)d);
        wfc_add_rule(w, 1, 2, (wfc_dir_t)d);
        wfc_add_rule(w, 2, 1, (wfc_dir_t)d);
    }
    assert(wfc_collapse(w) == WFC_OK);
    assert(grid_adjacency_holds(w));
    wfc_world_destroy(w);
}

TEST(learned_checkerboard) {
    uint16_t sample[16] = {
        0,1,0,1,
        1,0,1,0,
        0,1,0,1,
        1,0,1,0
    };
    wfc_config_t cfg = {8, 8, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 0.0f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 0, 0.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    assert(wfc_learn_from_sample(w, sample, 4, 4, 1) == WFC_OK);
    assert(wfc_collapse(w) == WFC_OK);
    int same = 0, edges = 0;
    for (uint32_t y = 0; y < 8; y++) {
        for (uint32_t x = 0; x < 8; x++) {
            uint16_t p = wfc_get_cell_pattern(w, x, y, 0);
            assert(p == 0 || p == 1);
            if (x + 1 < 8) {
                edges++;
                if (p == wfc_get_cell_pattern(w, x + 1, y, 0)) same++;
            }
            if (y + 1 < 8) {
                edges++;
                if (p == wfc_get_cell_pattern(w, x, y + 1, 0)) same++;
            }
        }
    }
    assert(same == 0);
    assert(edges > 0);
    wfc_world_destroy(w);
}

TEST(weight_histogram) {
    const int N = 16;
    const int runs = 8;
    int count0 = 0, total = 0;
    for (int r = 0; r < runs; r++) {
        wfc_config_t cfg = {(uint32_t)N, (uint32_t)N, 1, 1000u + (uint32_t)r,
                            0, 0, 0, 0, 1.0f, 1.0f};
        wfc_world_t* w = wfc_world_create(&cfg);
        wfc_pattern_def_t p0 = {0, 0, 9.0f, 0, 0, {0}};
        wfc_pattern_def_t p1 = {1, 0, 1.0f, 0, 0, {0}};
        wfc_add_pattern(w, &p0);
        wfc_add_pattern(w, &p1);
        add_full(w, 2, 4);
        assert(wfc_collapse(w) == WFC_OK);
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                total++;
                if (wfc_get_cell_pattern(w, x, y, 0) == 0) count0++;
            }
        wfc_world_destroy(w);
    }
    double frac = (double)count0 / (double)total;
    /* 9:1 weights → expect ~0.9; allow a wide band for RNG variance. */
    assert(frac > 0.75 && frac < 0.98);
    printf("(p0=%.3f) ", frac);
}

TEST(seed_determinism) {
    uint16_t a[64], b[64];
    wfc_config_t cfg = {8, 8, 1, 12345, 0, 0, 0, 0, 1.0f, 1.0f};
    for (int pass = 0; pass < 2; pass++) {
        wfc_world_t* w = wfc_world_create(&cfg);
        wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
        wfc_pattern_def_t p1 = {1, 0, 1.0f, 0, 0, {0}};
        wfc_add_pattern(w, &p0);
        wfc_add_pattern(w, &p1);
        add_full(w, 2, 4);
        assert(wfc_collapse(w) == WFC_OK);
        assert(wfc_export_grid(w, pass ? b : a, 64) == WFC_OK);
        wfc_world_destroy(w);
    }
    assert(memcmp(a, b, sizeof(a)) == 0);
}

TEST(class_max_quality) {
    wfc_config_t cfg = {8, 8, 1, 7, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 1, 5.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    add_full(w, 2, 4);
    wfc_set_class_limit(w, 1, 0, 8);
    assert(wfc_collapse(w) == WFC_OK);
    int n1 = 0;
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 8; x++)
            if (wfc_get_cell_pattern(w, x, y, 0) == 1) n1++;
    assert(n1 <= 8);
    wfc_world_destroy(w);
}

#if defined(WFC_ENABLE_COPULA)
TEST(copula_path) {
    wfc_config_t cfg = {8, 8, 1, 11, 0, 0, 1, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 1, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    add_full(w, 2, 4);
    w->copula_z[0] = wfc_inv_cdf_normal(0.5f);
    w->copula_z[1] = wfc_inv_cdf_normal(0.5f);
    for (int d = 0; d < 4; d++) {
        w->copula_rho[d][0][0] = 0.85f;
        w->copula_rho[d][1][1] = 0.85f;
        w->copula_rho[d][0][1] = -0.4f;
        w->copula_rho[d][1][0] = -0.4f;
    }
    assert(wfc_collapse(w) == WFC_OK);
    assert(grid_adjacency_holds(w));
    wfc_world_destroy(w);
}
#endif

int main(void) {
    printf("========================================\n");
    printf("WFC Generation Quality Suite\n");
    printf("========================================\n\n");
    RUN(adjacency_soundness);
    RUN(learned_checkerboard);
    RUN(weight_histogram);
    RUN(seed_determinism);
    RUN(class_max_quality);
#if defined(WFC_ENABLE_COPULA)
    RUN(copula_path);
#endif
    printf("\n========================================\n");
    printf("Results: %d/%d passed\n", pass_count, test_count);
    printf("========================================\n");
    return (pass_count == test_count) ? 0 : 1;
}
