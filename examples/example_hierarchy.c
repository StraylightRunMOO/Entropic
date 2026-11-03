/*
 * Multiscale hierarchy example: coarse-to-fine WFC collapse.
 * Demonstrates 3-level hierarchy with CDF 9/7 wavelets.
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

static void print_grid(const wfc_world_t* w, uint32_t nx, uint32_t ny)
{
    for (uint32_t y = 0; y < ny; ++y) {
        for (uint32_t x = 0; x < nx; ++x) {
            uint16_t pat = wfc_get_cell_pattern(w, x, y, 0);
            printf("%c", pat == 0 ? '.' : (pat == 1 ? '#' : (pat == 2 ? '~' : '?')));
        }
        printf("\n");
    }
}

int main(void)
{
    printf("=== WFC Hierarchy Example (3-level) ===\n\n");

    wfc_config_t cfg = {0};
    cfg.nx = 32;
    cfg.ny = 32;
    cfg.nz = 1;
    cfg.seed = 42;
    cfg.enable_spectral = 1;
    cfg.spectral_weight = 1.5f;

    wfc_world_t* w = wfc_world_create(&cfg);
    if (!w) { fprintf(stderr, "Failed to create world\n"); return 1; }

    /* Three tile types: grass, water, sand */
    wfc_pattern_def_t grass = {0, 0, 5.0f};
    wfc_pattern_def_t water = {1, 1, 3.0f};
    wfc_pattern_def_t sand  = {2, 2, 2.0f};
    wfc_add_pattern(w, &grass);
    wfc_add_pattern(w, &water);
    wfc_add_pattern(w, &sand);

    /* Adjacency: everything can be next to everything */
    for (int d = 0; d < 4; ++d) {
        for (uint16_t a = 0; a < 3; ++a)
            for (uint16_t b = 0; b < 3; ++b)
                wfc_add_rule(w, a, b, d);
    }

    /* init → set_wavelet → set_target → collapse */
    wfc_err_t err = wfc_hierarchy_init(w, 3);          /* 8×8 → 16×16 → 32×32 */
    if (err != WFC_OK) { fprintf(stderr, "hierarchy_init failed: %d\n", err); return 1; }

    err = wfc_hierarchy_set_wavelet(w, FAF_WAVELET_CDF97, 0);
    if (err != WFC_OK) { fprintf(stderr, "set_wavelet failed: %d\n", err); return 1; }

    err = wfc_set_gumbel_field(w, 0.7, 12345);
    if (err != WFC_OK) { fprintf(stderr, "set_gumbel_field failed: %d\n", err); return 1; }

    err = wfc_hierarchy_set_target(w, w->spectral_field, w->num_cells);
    if (err != WFC_OK) { fprintf(stderr, "set_target failed: %d\n", err); return 1; }

    err = wfc_hierarchy_collapse(w);
    if (err != WFC_OK) { fprintf(stderr, "hierarchy_collapse failed: %d\n", err); return 1; }

    printf("Hierarchy collapse result:\n");
    print_grid(w, cfg.nx, cfg.ny);

    uint16_t* grid = (uint16_t*)malloc(cfg.nx * cfg.ny * sizeof(uint16_t));
    if (grid) {
        wfc_export_grid(w, grid, cfg.nx * cfg.ny);
        free(grid);
    }

    /* Print scale info */
    printf("\nScale levels:\n");
    for (uint32_t k = 0; k < w->hierarchy->nlevels; ++k) {
        wfc_scale_t* s = &w->hierarchy->levels[k];
        printf("  Level %u: %u x %u (weight=%.3f)\n", k, s->nx, s->ny, s->weight);
    }

    wfc_world_destroy(w);
    printf("\nDone!\n");
    return 0;
}
