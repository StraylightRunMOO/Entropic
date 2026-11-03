/* example_basic.c — Simple 2D WFC with 3 tile types */
#define WFC_IMPLEMENTATION
#include "entropic/wfc.h"
#include <stdio.h>

/* Visual tiles */
static const char* tile_names[] = {"grass", "water", "shore"};
static const char tile_chars[] = {'G', 'W', '~'};

int main(void) {
    printf("=== WFC Basic Example ===\n\n");

    /* Create a 16x16 2D world */
    wfc_config_t cfg = {
        .nx = 16, .ny = 16, .nz = 1,
        .seed = 42,
        .seamless = 0
    };
    wfc_world_t* w = wfc_world_create(&cfg);
    if (!w) { printf("Failed to create world\n"); return 1; }

    /* Define 3 patterns: grass, water, shore */
    wfc_pattern_def_t grass = {.id = 0, .class_id = 0, .weight = 5.0f};
    wfc_pattern_def_t water = {.id = 1, .class_id = 1, .weight = 3.0f};
    wfc_pattern_def_t shore = {.id = 2, .class_id = 2, .weight = 1.0f};
    wfc_add_pattern(w, &grass);
    wfc_add_pattern(w, &water);
    wfc_add_pattern(w, &shore);

    /* Adjacency rules:
     *   grass <-> grass, grass <-> shore
     *   water <-> water, water <-> shore
     *   shore <-> grass, shore <-> water
     */
    wfc_dir_t dirs[4] = {WFC_DIR_N, WFC_DIR_S, WFC_DIR_E, WFC_DIR_W};
    for (int i = 0; i < 4; i++) {
        wfc_dir_t d = dirs[i];
        wfc_add_rule(w, 0, 0, d);  /* grass next to grass */
        wfc_add_rule(w, 1, 1, d);  /* water next to water */
        wfc_add_rule(w, 0, 2, d);  /* grass next to shore */
        wfc_add_rule(w, 2, 0, d);
        wfc_add_rule(w, 1, 2, d);  /* water next to shore */
        wfc_add_rule(w, 2, 1, d);
        wfc_add_rule(w, 2, 2, d);  /* shore next to shore */
    }

    /* Collapse */
    printf("Collapsing 16x16 grid...\n");
    wfc_err_t err = wfc_collapse(w);
    if (err != WFC_OK) {
        printf("Collapse failed: %d\n", err);
        wfc_world_destroy(w);
        return 1;
    }

    /* Print result */
    printf("\nResult:\n");
    for (uint32_t y = 0; y < w->ny; y++) {
        for (uint32_t x = 0; x < w->nx; x++) {
            uint16_t p = wfc_get_cell_pattern(w, x, y, 0);
            if (p < 3) printf("%c ", tile_chars[p]);
            else printf("? ");
        }
        printf("\n");
    }

    /* Stats */
    int counts[3] = {0};
    for (uint32_t y = 0; y < w->ny; y++)
        for (uint32_t x = 0; x < w->nx; x++)
            counts[wfc_get_cell_pattern(w, x, y, 0)]++;

    printf("\nTile counts:\n");
    for (int i = 0; i < 3; i++)
        printf("  %s (%c): %d\n", tile_names[i], tile_chars[i], counts[i]);

    wfc_world_destroy(w);
    printf("\nDone!\n");
    return 0;
}
