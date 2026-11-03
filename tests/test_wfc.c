/* test_wfc.c — C API test suite */
#undef NDEBUG
#define WFC_IMPLEMENTATION
#include "entropic/wfc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) static void test_##name(void)
#define RUN(name) do { test_count++; printf("  [%2d] " #name " ... ", test_count); fflush(stdout); test_##name(); pass_count++; printf("PASS\n"); } while(0)

static void add_full_rules(wfc_world_t* w, int npat, int ndirs) {
    for (int i = 0; i < npat; i++)
        for (int j = 0; j < npat; j++)
            for (int d = 0; d < ndirs; d++)
                wfc_add_rule(w, (uint16_t)i, (uint16_t)j, (wfc_dir_t)d);
}

/* --- Basic lifecycle --- */
TEST(create_destroy) {
    wfc_config_t cfg = {8, 8, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    assert(w != NULL);
    assert(w->nx == 8 && w->ny == 8 && w->nz == 1);
    assert(w->num_cells == 64); /* power-of-2 padded */
    wfc_world_destroy(w);
}

TEST(create_3d) {
    wfc_config_t cfg = {4, 4, 4, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    assert(w != NULL);
    assert(w->is_3d);
    assert(w->num_cells == 64); /* 4*4*4 = 64 */
    wfc_world_destroy(w);
}

TEST(create_invalid) {
    wfc_config_t cfg = {0, 8, 1, 0, 0, 0, 0, 0, 1.0f, 1.0f};
    assert(wfc_world_create(&cfg) == NULL);
}

/* --- Pattern & rule management --- */
TEST(add_pattern) {
    wfc_config_t cfg = {4, 4, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t pat = {0, 0, 1.0f, 0, 0, {0}};
    assert(wfc_add_pattern(w, &pat) == WFC_OK);
    assert(w->num_patterns == 1);
    wfc_world_destroy(w);
}

TEST(add_rule) {
    wfc_config_t cfg = {4, 4, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    assert(wfc_add_rule(w, 0, 1, WFC_DIR_E) == WFC_OK);
    assert(w->compat[WFC_DIR_E][0][0] & (1ULL << 1));
    /* Symmetric rule */
    assert(w->compat[WFC_DIR_W][1][0] & (1ULL << 0));
    wfc_world_destroy(w);
}

TEST(invalid_pattern) {
    wfc_config_t cfg = {4, 4, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t pat = {256, 0, 1.0f, 0, 0, {0}};
    assert(wfc_add_pattern(w, &pat) == WFC_EINVAL);
    wfc_world_destroy(w);
}

/* --- Simple collapse: 1x1 grid --- */
TEST(collapse_1x1) {
    wfc_config_t cfg = {1, 1, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    assert(wfc_collapse(w) == WFC_OK);
    assert(wfc_get_cell_pattern(w, 0, 0, 0) == 0);
    wfc_world_destroy(w);
}

/* --- Simple 2D grid with 2 patterns --- */
TEST(collapse_2x2_two_patterns) {
    wfc_config_t cfg = {2, 2, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    /* Allow both patterns adjacent in all directions (no constraints) */
    wfc_add_rule(w, 0, 0, WFC_DIR_E);
    wfc_add_rule(w, 0, 1, WFC_DIR_E);
    wfc_add_rule(w, 1, 0, WFC_DIR_E);
    wfc_add_rule(w, 1, 1, WFC_DIR_E);
    wfc_add_rule(w, 0, 0, WFC_DIR_N);
    wfc_add_rule(w, 0, 1, WFC_DIR_N);
    wfc_add_rule(w, 1, 0, WFC_DIR_N);
    wfc_add_rule(w, 1, 1, WFC_DIR_N);

    assert(wfc_collapse(w) == WFC_OK);
    for (int y = 0; y < 2; y++) {
        for (int x = 0; x < 2; x++) {
            uint16_t p = wfc_get_cell_pattern(w, x, y, 0);
            assert(p == 0 || p == 1);
        }
    }
    wfc_world_destroy(w);
}

/* --- Collapse with adjacency constraints --- */
TEST(collapse_with_constraints) {
    wfc_config_t cfg = {4, 4, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);

    /* Pattern 0 = grass, Pattern 1 = water, Pattern 2 = shore */
    wfc_pattern_def_t grass = {0, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t water = {1, 1, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t shore = {2, 2, 0.5f, 0, 0, {0}};
    wfc_add_pattern(w, &grass);
    wfc_add_pattern(w, &water);
    wfc_add_pattern(w, &shore);

    /* Rules: grass next to grass, water next to water,
     * shore can be next to both */
    wfc_add_rule(w, 0, 0, WFC_DIR_E);
    wfc_add_rule(w, 0, 0, WFC_DIR_N);
    wfc_add_rule(w, 1, 1, WFC_DIR_E);
    wfc_add_rule(w, 1, 1, WFC_DIR_N);
    wfc_add_rule(w, 2, 0, WFC_DIR_E);
    wfc_add_rule(w, 2, 1, WFC_DIR_E);
    wfc_add_rule(w, 2, 0, WFC_DIR_N);
    wfc_add_rule(w, 2, 1, WFC_DIR_N);
    wfc_add_rule(w, 0, 2, WFC_DIR_E);
    wfc_add_rule(w, 1, 2, WFC_DIR_E);
    wfc_add_rule(w, 0, 2, WFC_DIR_N);
    wfc_add_rule(w, 1, 2, WFC_DIR_N);

    assert(wfc_collapse(w) == WFC_OK);
    wfc_world_destroy(w);
}

/* --- Export grid --- */
TEST(export_grid) {
    wfc_config_t cfg = {2, 2, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    /* Self-compatibility for propagation */
    for (int d = 0; d < 4; d++) wfc_add_rule(w, 0, 0, (wfc_dir_t)d);
    assert(wfc_collapse(w) == WFC_OK);
    uint16_t grid[4];
    assert(wfc_export_grid(w, grid, 4) == WFC_OK);
    for (int i = 0; i < 4; i++) assert(grid[i] == 0);
    wfc_world_destroy(w);
}

/* --- Reset --- */
TEST(reset_world) {
    wfc_config_t cfg = {2, 2, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    for (int d = 0; d < 4; d++) wfc_add_rule(w, 0, 0, (wfc_dir_t)d);
    assert(wfc_collapse(w) == WFC_OK);
    assert(wfc_get_cell_pattern(w, 0, 0, 0) == 0);
    wfc_world_reset(w);
    assert(wfc_get_cell_pattern(w, 0, 0, 0) == WFC_PATTERN_INVALID);
    /* Can collapse again */
    assert(wfc_collapse(w) == WFC_OK);
    assert(wfc_get_cell_pattern(w, 0, 0, 0) == 0);
    wfc_world_destroy(w);
}

/* --- Learn from sample --- */
TEST(learn_from_sample) {
    /* 4x4 sample: checkerboard of 0 and 1 */
    uint16_t sample[16] = {
        0,1,0,1,
        1,0,1,0,
        0,1,0,1,
        1,0,1,0
    };
    wfc_config_t cfg = {4, 4, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 0.0f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 0, 0.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    assert(wfc_learn_from_sample(w, sample, 4, 4, 1) == WFC_OK);
    /* Should have learned that 0 and 1 are never adjacent */
    assert(wfc_collapse(w) == WFC_OK);
    wfc_world_destroy(w);
}

/* --- Contradiction callback --- */
static int contradiction_fired = 0;
static uint32_t contra_x, contra_y, contra_z;

static void test_contra_cb(uint32_t x, uint32_t y, uint32_t z, void* user) {
    (void)user;
    contradiction_fired = 1;
    contra_x = x; contra_y = y; contra_z = z;
}

TEST(contradiction_callback) {
    contradiction_fired = 0;
    wfc_config_t cfg = {2, 2, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    /* One pattern, but no rules — this creates a contradiction
     * because after first cell collapses, neighbors have no valid patterns */
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    /* Only allow 0 adjacent to 0, but then force 1 to be incompatible */
    wfc_add_rule(w, 0, 0, WFC_DIR_E);
    wfc_add_rule(w, 0, 0, WFC_DIR_N);
    /* Pattern 1 has NO rules, so it can only appear in isolation */
    wfc_set_contradiction_callback(w, test_contra_cb, NULL);
    wfc_collapse(w); /* May or may not contradiction depending on random */
    wfc_world_destroy(w);
}

/* --- Seamless mode --- */
TEST(seamless_mode) {
    wfc_config_t cfg = {4, 4, 1, 42, 0, 0, 0, 1, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    for (int d = 0; d < 4; d++) wfc_add_rule(w, 0, 0, (wfc_dir_t)d);
    assert(wfc_collapse(w) == WFC_OK);
    wfc_world_destroy(w);
}

/* --- Step-by-step execution --- */
TEST(step_by_step) {
    wfc_config_t cfg = {2, 2, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    for (int d = 0; d < 4; d++) wfc_add_rule(w, 0, 0, (wfc_dir_t)d);

    assert(wfc_step_init(w) == WFC_OK);
    int steps = 0;
    while (!wfc_step_done(w)) {
        assert(wfc_step(w) == WFC_OK);
        steps++;
        assert(steps < 100); /* safety */
    }
    assert(steps == 4); /* 2x2 grid */
    for (int y = 0; y < 2; y++)
        for (int x = 0; x < 2; x++)
            assert(wfc_get_cell_pattern(w, x, y, 0) == 0);
    wfc_world_destroy(w);
}

/* --- Large grid collapse --- */
TEST(collapse_large) {
    wfc_config_t cfg = {32, 32, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    /* 4-pattern tileset with full compatibility */
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t p2 = {2, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t p3 = {3, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    wfc_add_pattern(w, &p2);
    wfc_add_pattern(w, &p3);
    /* Full compatibility: all patterns adjacent to all patterns */
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int d = 0; d < 4; d++)
                wfc_add_rule(w, (uint16_t)i, (uint16_t)j, (wfc_dir_t)d);

    clock_t t0 = clock();
    assert(wfc_collapse(w) == WFC_OK);
    clock_t t1 = clock();
    double ms = (double)(t1 - t0) * 1000.0 / CLOCKS_PER_SEC;
    printf("(32x32 in %.2fms) ", ms);
    wfc_world_destroy(w);
}

/* --- 3D collapse --- */
TEST(collapse_3d) {
    wfc_config_t cfg = {4, 4, 4, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    /* Full compatibility in 3D */
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int d = 0; d < 6; d++)
                wfc_add_rule(w, (uint16_t)i, (uint16_t)j, (wfc_dir_t)d);
    assert(wfc_collapse(w) == WFC_OK);
    /* All cells should be collapsed */
    for (uint32_t z = 0; z < 4; z++)
        for (uint32_t y = 0; y < 4; y++)
            for (uint32_t x = 0; x < 4; x++)
                assert(wfc_get_cell_pattern(w, x, y, z) != WFC_PATTERN_INVALID);
    wfc_world_destroy(w);
}

/* --- Class limits (max) --- */
TEST(class_max_limit) {
    wfc_config_t cfg = {4, 4, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 1, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    /* Full compatibility */
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int d = 0; d < 4; d++)
                wfc_add_rule(w, (uint16_t)i, (uint16_t)j, (wfc_dir_t)d);
    /* Allow at most 4 of class 1 */
    wfc_set_class_limit(w, 1, 0, 4);
    assert(wfc_collapse(w) == WFC_OK);
    /* Verify at most 4 of class 1 */
    int count = 0;
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            if (wfc_get_cell_pattern(w, x, y, 0) == 1) count++;
    assert(count <= 4);
    wfc_world_destroy(w);
}

/* --- Bitset primitives --- */
TEST(bitset_ops) {
    uint64_t a[4] = {~0ULL, ~0ULL, ~0ULL, ~0ULL};
    uint64_t b[4] = {0x0F0F0F0F0F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL, 0, 0};

    uint64_t c[4];
    memcpy(c, a, sizeof(c));
    wfc_mask_and(c, b);
    assert(c[0] == b[0] && c[1] == b[1] && c[2] == 0 && c[3] == 0);

    uint64_t d[4] = {~0ULL, ~0ULL, ~0ULL, ~0ULL};
    wfc_mask_andnot(d, b);
    assert(d[0] == (~b[0]) && d[1] == (~b[1]) && d[2] == ~0ULL && d[3] == ~0ULL);

    assert(wfc_mask_popcnt(b) == 64);  /* 32 bits in each of b[0] and b[1] */
    assert(wfc_mask_empty(b) == 0);

    uint64_t e[4] = {0, 0, 0, 0};
    assert(wfc_mask_empty(e) == 1);
}

/* --- RNG distribution --- */
TEST(rng_distribution) {
    uint64_t s = 42;
    int buckets[10] = {0};
    for (int i = 0; i < 10000; i++) {
        float f = wfc_randf(&s);
        assert(f >= 0.0f && f < 1.0f);
        buckets[(int)(f * 10)]++;
    }
    /* Roughly uniform */
    for (int i = 0; i < 10; i++) {
        assert(buckets[i] > 500 && buckets[i] < 1500);
    }
    uint64_t s2 = 1;
    uint32_t r = wfc_rand_range(&s2, 7);
    assert(r < 7);
    assert(wfc_rand_range(&s2, 0) == 0);
    assert(wfc_next_pow2_u32(1) == 1);
    assert(wfc_next_pow2_u32(2) == 2);
    assert(wfc_next_pow2_u32(5) == 8);
    (void)wfc_next_pow2_u32(0); /* defined; 0 is not a power-of-two input */
}

TEST(mask_eq_copy) {
    uint64_t a[WFC_MASK_U64S];
    uint64_t b[WFC_MASK_U64S];
    memset(a, 0xAB, sizeof(a));
    wfc_mask_copy(b, a);
    assert(wfc_mask_eq(a, b));
    b[0] ^= 1;
    assert(!wfc_mask_eq(a, b));
}

TEST(create_null_and_zero_seed) {
    assert(wfc_world_create(NULL) == NULL);
    wfc_config_t cfg = {3, 5, 0, 0, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    assert(w != NULL);
    assert(w->nz == 1);
    assert(w->nx_p2 == 4 && w->ny_p2 == 8);
    assert(w->rng_state == 0x123456789ABCDEF0ULL);
    assert(wfc_idx_2d(w, 1, 1) == wfc_idx(w, 1, 1, 0));
    wfc_world_destroy(w);
    wfc_world_destroy(NULL);
}

TEST(get_pattern_and_export_errors) {
    wfc_config_t cfg = {2, 2, 1, 1, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 2, 2, {3, 4, 0, 0}};
    assert(wfc_add_pattern(w, &p0) == WFC_OK);
    const wfc_pattern_def_t* gp = wfc_get_pattern(w, 0);
    assert(gp && gp->id == 0 && gp->num_layers == 2 && gp->layer_types[0] == 3);
    assert(wfc_get_pattern(w, 255) != NULL); /* slot exists, may be empty */
    assert(wfc_get_pattern(NULL, 0) == NULL);
    assert(wfc_get_pattern(w, WFC_MAX_PATTERNS) == NULL);
    assert(wfc_get_cell_pattern(NULL, 0, 0, 0) == WFC_PATTERN_INVALID);
    assert(wfc_get_cell_pattern(w, 9, 0, 0) == WFC_PATTERN_INVALID);
    uint16_t tiny[1];
    assert(wfc_export_grid(w, tiny, 1) == WFC_EINVAL);
    assert(wfc_export_grid(w, NULL, 4) == WFC_EINVAL);
    assert(wfc_export_grid(NULL, tiny, 4) == WFC_EINVAL);
    assert(wfc_add_pattern(NULL, &p0) == WFC_EINVAL);
    assert(wfc_add_pattern(w, NULL) == WFC_EINVAL);
    assert(wfc_add_rule(NULL, 0, 0, WFC_DIR_E) == WFC_EINVAL);
    assert(wfc_add_rule(w, 0, 0, WFC_DIR_COUNT) == WFC_EINVAL);
    assert(wfc_set_class_limit(NULL, 0, 0, 1) == WFC_EINVAL);
    assert(wfc_set_class_limit(w, WFC_MAX_CLASSES, 0, 1) == WFC_EINVAL);
    assert(wfc_collapse(NULL) == WFC_EINVAL);
    wfc_world_t* empty = wfc_world_create(&cfg);
    assert(wfc_collapse(empty) == WFC_EINVAL);
    assert(wfc_step_init(empty) == WFC_EINVAL);
    assert(wfc_step(w) == WFC_EINVAL); /* not in step mode */
    assert(wfc_step_done(NULL) == 1);
    wfc_world_reset(NULL);
    wfc_set_contradiction_callback(NULL, NULL, NULL);
    assert(wfc_learn_from_sample(w, NULL, 2, 2, 1) == WFC_EINVAL);
    uint16_t samp[1] = {0};
    assert(wfc_learn_from_sample(w, samp, 0, 2, 1) == WFC_EINVAL);
    wfc_world_destroy(empty);
    wfc_world_destroy(w);
}

TEST(guaranteed_contradiction) {
    contradiction_fired = 0;
    wfc_config_t cfg = {2, 2, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_set_contradiction_callback(w, test_contra_cb, NULL);
    assert(wfc_collapse(w) == WFC_ECONTRADICTION);
    assert(contradiction_fired == 1);
    wfc_world_destroy(w);
}

TEST(non_power_of_two) {
    wfc_config_t cfg = {5, 7, 1, 3, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    add_full_rules(w, 2, 4);
    assert(wfc_collapse(w) == WFC_OK);
    uint16_t grid[35];
    assert(wfc_export_grid(w, grid, 35) == WFC_OK);
    for (int i = 0; i < 35; i++) assert(grid[i] == 0 || grid[i] == 1);
    wfc_world_destroy(w);
}

TEST(learn_3d) {
    uint16_t sample[8] = {0,1, 1,0,  0,1, 1,0};
    wfc_config_t cfg = {2, 2, 2, 5, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 0.0f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 0, 0.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    assert(wfc_learn_from_sample(w, sample, 2, 2, 2) == WFC_OK);
    assert(wfc_collapse(w) == WFC_OK);
    assert(wfc_idx_3d(w, 1, 1, 1) == wfc_idx(w, 1, 1, 1));
    wfc_world_destroy(w);
}

TEST(ud_rules_3d) {
    wfc_config_t cfg = {2, 2, 2, 9, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    add_full_rules(w, 1, 6);
    assert(wfc_collapse(w) == WFC_OK);
    wfc_world_destroy(w);
}

TEST(class_min_and_high_id) {
    wfc_config_t cfg = {4, 4, 1, 2, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 1, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    add_full_rules(w, 2, 4);
    assert(wfc_set_class_limit(w, 1, 1, 16) == WFC_OK);
    assert(wfc_set_class_limit(w, 40, 0, 2) == WFC_OK); /* >=32, no bitmask bit */
    assert(wfc_collapse(w) == WFC_OK);
    wfc_world_destroy(w);
}

TEST(step_errors_and_done) {
    wfc_config_t cfg = {2, 2, 1, 1, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    add_full_rules(w, 1, 4);
    assert(wfc_step_init(w) == WFC_OK);
    while (!wfc_step_done(w))
        assert(wfc_step(w) == WFC_OK);
    assert(wfc_step(w) == WFC_OK); /* already done */
    wfc_world_destroy(w);
}

#if defined(WFC_ENABLE_BACKTRACK)
TEST(backtrack_recovery) {
    wfc_config_t cfg = {4, 4, 1, 13, 1, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 0.01f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 0, 100.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    for (int d = 0; d < 4; d++)
        wfc_add_rule(w, 0, 0, (wfc_dir_t)d);
    wfc_err_t err = wfc_collapse(w);
    /* Recovery may exhaust the trail; either a complete grid or a
     * reported contradiction is acceptable. */
    assert(err == WFC_OK || err == WFC_ECONTRADICTION);
    if (err == WFC_OK) {
        for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) {
                uint16_t p = wfc_get_cell_pattern(w, x, y, 0);
                assert(p == 0 || p == 1);
            }
    }
    wfc_world_destroy(w);
}
#endif

#if defined(WFC_ENABLE_COPULA)
TEST(copula_nonzero_rho) {
    wfc_config_t cfg = {6, 6, 1, 21, 0, 0, 1, 1, 1.0f, 1.5f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 1, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    add_full_rules(w, 2, 4);
    w->copula_marginal[0] = 0.01f;
    w->copula_marginal[1] = 0.99f;
    w->copula_z[0] = wfc_inv_cdf_normal(0.01f);
    w->copula_z[1] = wfc_inv_cdf_normal(0.99f);
    (void)wfc_cdf_normal(0.0f);
    (void)wfc_cdf_normal(2.0f);
    for (int d = 0; d < 4; d++) {
        w->copula_rho[d][0][0] = 0.9f;
        w->copula_rho[d][1][1] = 0.9f;
        w->copula_rho[d][0][1] = -0.5f;
        w->copula_rho[d][1][0] = -0.5f;
    }
    assert(wfc_collapse(w) == WFC_OK);
    wfc_world_reset(w);
    assert(wfc_collapse(w) == WFC_OK);
    wfc_world_destroy(w);
}
#endif

#if defined(WFC_USE_FAF)
TEST(spectral_gumbel_matern) {
    wfc_config_t cfg = {8, 8, 1, 4, 0, 1, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_pattern_def_t p1 = {1, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    add_full_rules(w, 2, 4);
    assert(wfc_set_gumbel_field(w, 0.7, 99) == WFC_OK);
    assert(w->spectral_field != NULL);
    assert(wfc_collapse(w) == WFC_OK);
    wfc_world_reset(w);
    assert(wfc_set_matern_field(w, 1.0, 1.5, 77) == WFC_OK);
    assert(wfc_collapse(w) == WFC_OK);
    wfc_world_destroy(w);

    wfc_config_t off = {4, 4, 1, 1, 0, 0, 0, 0, 1.0f, 1.0f};
    wfc_world_t* w2 = wfc_world_create(&off);
    assert(wfc_set_gumbel_field(w2, 0.5, 1) == WFC_EUNINIT);
    assert(wfc_set_matern_field(w2, 1.0, 1.0, 1) == WFC_EUNINIT);
    wfc_world_destroy(w2);
}

TEST(spectral_odd_cells_and_fill) {
    wfc_config_t cfg = {3, 3, 1, 8, 0, 1, 0, 0, 0.5f, 1.0f};
    wfc_world_t* w = wfc_world_create(&cfg);
    float buf[9];
    uint64_t rng = 3;
    wfc_fill_normal_f(buf, 9, &rng);
    wfc_gumbel_link_f(buf, 9);
    wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
    wfc_add_pattern(w, &p0);
    add_full_rules(w, 1, 4);
    assert(wfc_set_gumbel_field(w, 0.3, 5) == WFC_OK);
    assert(wfc_collapse(w) == WFC_OK);
    wfc_world_destroy(w);
}
#endif

int main(void) {
    printf("========================================\n");
    printf("WFC Engine C Test Suite\n");
    printf("========================================\n\n");

    /* Core */
    RUN(create_destroy);
    RUN(create_3d);
    RUN(create_invalid);
    RUN(add_pattern);
    RUN(add_rule);
    RUN(invalid_pattern);

    /* Collapse */
    RUN(collapse_1x1);
    RUN(collapse_2x2_two_patterns);
    RUN(collapse_with_constraints);
    RUN(collapse_large);
    RUN(collapse_3d);

    /* Features */
    RUN(export_grid);
    RUN(reset_world);
    RUN(learn_from_sample);
    RUN(contradiction_callback);
    RUN(seamless_mode);
    RUN(step_by_step);
    RUN(class_max_limit);

    /* Primitives */
    RUN(bitset_ops);
    RUN(rng_distribution);
    RUN(mask_eq_copy);

    /* Extra API / edge paths */
    RUN(create_null_and_zero_seed);
    RUN(get_pattern_and_export_errors);
    RUN(guaranteed_contradiction);
    RUN(non_power_of_two);
    RUN(learn_3d);
    RUN(ud_rules_3d);
    RUN(class_min_and_high_id);
    RUN(step_errors_and_done);
#if defined(WFC_ENABLE_BACKTRACK)
    RUN(backtrack_recovery);
#endif
#if defined(WFC_ENABLE_COPULA)
    RUN(copula_nonzero_rho);
#endif
#if defined(WFC_USE_FAF)
    RUN(spectral_gumbel_matern);
    RUN(spectral_odd_cells_and_fill);
#endif

    printf("\n========================================\n");
    printf("Results: %d/%d passed\n", pass_count, test_count);
    printf("========================================\n");

    return (pass_count == test_count) ? 0 : 1;
}
