/* example_seamless.c — Toroidal (wrap-around) boundary conditions */
#define WFC_IMPLEMENTATION
#include "entropic/wfc.h"
#include <stdio.h>

int main(void)
{
    printf("=== WFC Seamless (Torus) Example ===\n\n");

    wfc_config_t cfg = {
        .nx = 16, .ny = 16, .nz = 1,
        .seed = 8,
        .seamless = 1
    };
    wfc_world_t* w = wfc_world_create(&cfg);
    if (!w) { printf("Failed to create world\n"); return 1; }

    wfc_pattern_def_t a = {.id = 0, .class_id = 0, .weight = 1.0f};
    wfc_pattern_def_t b = {.id = 1, .class_id = 0, .weight = 1.0f};
    wfc_add_pattern(w, &a);
    wfc_add_pattern(w, &b);

    /* Directed rules plus the reverse direction so wrap-around edges match. */
    for (int d = 0; d < 4; d++) {
        wfc_add_rule(w, 0, 0, (wfc_dir_t)d);
        wfc_add_rule(w, 1, 1, (wfc_dir_t)d);
        wfc_add_rule(w, 0, 1, (wfc_dir_t)d);
        wfc_add_rule(w, 1, 0, (wfc_dir_t)d);
    }

    printf("Collapsing 16x16 torus (edges wrap)...\n");
    wfc_err_t err = wfc_collapse(w);
    if (err != WFC_OK) {
        printf("Collapse failed: %d\n", err);
        wfc_world_destroy(w);
        return 1;
    }

    printf("\nResult (. / #). First and last rows/cols are neighbours:\n");
    for (uint32_t y = 0; y < w->ny; y++) {
        for (uint32_t x = 0; x < w->nx; x++) {
            uint16_t p = wfc_get_cell_pattern(w, x, y, 0);
            putchar(p == 0 ? '.' : '#');
        }
        putchar('\n');
    }

    uint16_t nw = wfc_get_cell_pattern(w, 0, 0, 0);
    uint16_t ne = wfc_get_cell_pattern(w, w->nx - 1, 0, 0);
    uint16_t sw = wfc_get_cell_pattern(w, 0, w->ny - 1, 0);
    printf("\nCorner samples: (0,0)=%u (nx-1,0)=%u (0,ny-1)=%u\n", nw, ne, sw);
    printf("seamless flag: %d\n", w->seamless);

    wfc_world_destroy(w);
    printf("\nDone!\n");
    return 0;
}
