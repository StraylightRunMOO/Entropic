/* example_stepwise.c — Step-by-step collapse with class limits,
 * learn_from_sample, contradiction callback, reset, and grid export.
 * Demonstrates API surface not covered by the other examples. */
#define WFC_IMPLEMENTATION
#define WFC_ENABLE_COPULA 1
#define WFC_ENABLE_BACKTRACK 1

#include "entropic/wfc.h"
#include <stdio.h>
#include <stdlib.h>

enum { GRASS = 0, WATER = 1, SAND = 2, ROCK = 3 };

static const char* TILE_NAME[] = {"grass", "water", "sand", "rock"};
static int contradiction_count = 0;

static void on_contradiction(uint32_t x, uint32_t y, uint32_t z, void* ctx) {
    (void)ctx;
    contradiction_count++;
    printf("  contradiction at (%u,%u,%u)\n", x, y, z);
}

static void print_grid(const uint16_t* grid, uint32_t nx, uint32_t ny) {
    const char sym[] = {'.', '~', ',', '#'};
    for (uint32_t y = 0; y < ny; y++) {
        for (uint32_t x = 0; x < nx; x++) {
            uint16_t t = grid[y * nx + x];
            putchar(t < 4 ? sym[t] : '?');
        }
        putchar('\n');
    }
}

static void add_all_rules(wfc_world_t* w) {
    for (uint16_t a = 0; a < 4; a++)
        for (uint16_t b = 0; b < 4; b++)
            for (int d = 0; d < 4; d++)
                wfc_add_rule(w, a, b, (wfc_dir_t)d);
}

int main(void) {
    uint32_t nx = 16, ny = 16;
    size_t total = nx * ny;

    wfc_config_t cfg = {
        .nx = nx, .ny = ny, .nz = 1,
        .seed = 42,
        .seamless = 0
    };
    wfc_world_t* w = wfc_world_create(&cfg);
    if (!w) { printf("Failed to create world\n"); return 1; }

    wfc_pattern_def_t patterns[] = {
        {GRASS, 0, 5.0f},
        {WATER, 1, 2.0f},
        {SAND,  2, 1.5f},
        {ROCK,  3, 1.0f},
    };
    for (int i = 0; i < 4; i++) wfc_add_pattern(w, &patterns[i]);

    /* --- 1. Learn additional rules from a hand-crafted sample --- */
    printf("=== Phase 1: learn_from_sample ===\n");
    uint16_t sample[] = {
        0,0,2,1,1,2,0,0,
        0,2,1,1,1,1,2,0,
        0,2,1,1,1,1,2,0,
        0,0,2,1,1,2,0,0,
        0,0,0,2,2,0,0,3,
        0,0,0,0,0,0,3,3,
        0,0,0,0,0,3,3,3,
        0,0,0,0,0,0,3,0,
    };
    add_all_rules(w);
    wfc_learn_from_sample(w, sample, 8, 8, 1);
    printf("Rules learned from 8x8 sample (supplemented with full adjacency)\n");

    /* --- 2. Class limits: cap rock tiles --- */
    printf("\n=== Phase 2: class limits ===\n");
    wfc_set_class_limit(w, 3, 0, 30);
    printf("Rock (class 3) capped at 30 tiles max\n");

    /* --- 3. Contradiction callback --- */
    printf("\n=== Phase 3: contradiction callback ===\n");
    wfc_set_contradiction_callback(w, on_contradiction, NULL);

    /* --- 4. Step-by-step collapse --- */
    printf("\n=== Phase 4: step-by-step collapse ===\n");
    wfc_step_init(w);
    int steps = 0;
    while (!wfc_step_done(w)) {
        wfc_err_t err = wfc_step(w);
        steps++;
        if (err != WFC_OK) {
            printf("Contradiction at step %d\n", steps);
            break;
        }
    }
    printf("Collapsed in %d steps (contradictions: %d)\n", steps, contradiction_count);

    /* --- 5. Export grid and print --- */
    printf("\n=== Phase 5: export grid ===\n");
    uint16_t* grid = malloc(total * sizeof(uint16_t));
    wfc_export_grid(w, grid, total);
    print_grid(grid, nx, ny);

    int rock_count = 0;
    for (size_t i = 0; i < total; i++)
        if (grid[i] == ROCK) rock_count++;
    printf("Rock tiles: %d (limit: 30)\n", rock_count);

    /* --- 6. Get pattern info --- */
    printf("\n=== Phase 6: pattern metadata ===\n");
    for (int i = 0; i < 4; i++) {
        const wfc_pattern_def_t* p = wfc_get_pattern(w, (uint16_t)i);
        printf("  %s: id=%u class=%u weight=%.2f\n",
               TILE_NAME[i], p->id, p->class_id, p->weight);
    }

    /* --- 7. Reset and re-collapse --- */
    printf("\n=== Phase 7: reset + full collapse ===\n");
    wfc_world_reset(w);
    contradiction_count = 0;
    wfc_err_t err = wfc_collapse(w);
    if (err == WFC_OK) {
        wfc_export_grid(w, grid, total);
        print_grid(grid, nx, ny);
        printf("Second collapse succeeded\n");
    } else {
        printf("Second collapse failed (err %d)\n", err);
    }

    free(grid);
    wfc_world_destroy(w);
    printf("\nDone.\n");
    return 0;
}
