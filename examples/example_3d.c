/* example_3d.c — 3D WFC: Simple cube world */
#define WFC_IMPLEMENTATION
#include "entropic/wfc.h"
#include <stdio.h>

int main(void) {
    printf("=== WFC 3D Example ===\n\n");

    /* 8x8x8 3D world */
    wfc_config_t cfg = {
        .nx = 8, .ny = 8, .nz = 8,
        .seed = 123
    };
    wfc_world_t* w = wfc_world_create(&cfg);
    if (!w) { printf("Failed to create world\n"); return 1; }

    /* 2 materials: stone(0) and air(1) */
    wfc_pattern_def_t stone = {.id = 0, .class_id = 0, .weight = 1.0f};
    wfc_pattern_def_t air   = {.id = 1, .class_id = 1, .weight = 1.0f};
    wfc_add_pattern(w, &stone);
    wfc_add_pattern(w, &air);

    /* Full compatibility in all 6 directions */
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int d = 0; d < 6; d++)
                wfc_add_rule(w, (uint16_t)i, (uint16_t)j, (wfc_dir_t)d);

    printf("Collapsing 8x8x8 cube...\n");
    wfc_err_t err = wfc_collapse(w);
    if (err != WFC_OK) {
        printf("Collapse failed: %d\n", err);
        wfc_world_destroy(w);
        return 1;
    }

    /* Print layer by layer */
    printf("\nResult (layers from bottom to top):\n");
    for (uint32_t z = 0; z < w->nz; z++) {
        printf("\nLayer z=%d:\n", z);
        for (uint32_t y = 0; y < w->ny; y++) {
            for (uint32_t x = 0; x < w->nx; x++) {
                uint16_t p = wfc_get_cell_pattern(w, x, y, z);
                printf("%c", p == 0 ? '#' : '.');
            }
            printf("\n");
        }
    }

    /* Stats */
    int stone_count = 0, air_count = 0;
    for (uint32_t i = 0; i < w->num_cells; i++) {
        if (w->cells[i].collapsed == 0) stone_count++;
        else if (w->cells[i].collapsed == 1) air_count++;
    }
    printf("\nStone: %d, Air: %d (total: %d)\n", stone_count, air_count, stone_count + air_count);

    wfc_world_destroy(w);
    printf("\nDone!\n");
    return 0;
}
