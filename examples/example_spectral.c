/* example_spectral.c — Spectral field WFC (requires FFTW) */
#define WFC_IMPLEMENTATION
#include "entropic/wfc.h"
#include <stdio.h>

int main(void) {
    printf("=== WFC Spectral Field Example ===\n");
    printf("(Requires FFTW support compiled in)\n\n");

    wfc_config_t cfg = {
        .nx = 32, .ny = 32, .nz = 1,
        .seed = 42,
        .enable_spectral = 1,
        .spectral_weight = 0.5f,
        .seamless = 0
    };
    wfc_world_t* w = wfc_world_create(&cfg);
    if (!w) { printf("Failed to create world\n"); return 1; }

    /* Two patterns */
    wfc_pattern_def_t p0 = {.id = 0, .class_id = 0, .weight = 1.0f};
    wfc_pattern_def_t p1 = {.id = 1, .class_id = 1, .weight = 1.0f};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    for (int d = 0; d < 4; d++) {
        wfc_add_rule(w, 0, 0, (wfc_dir_t)d);
        wfc_add_rule(w, 1, 1, (wfc_dir_t)d);
	wfc_add_rule(w, 0, 1, (wfc_dir_t)d);
    }

    /* Generate fractional Brownian field (H=0.7) */
    printf("Generating fBm field (H=0.7)...\n");
    wfc_err_t err = wfc_set_gumbel_field(w, 0.7, 12345);
    if (err != WFC_OK) {
        printf("Spectral field generation failed: %d\n", err);
        wfc_world_destroy(w);
        return 1;
    }

    printf("Collapsing with spectral bias...\n");
    err = wfc_collapse(w);
    if (err != WFC_OK) {
        printf("Collapse failed: %d\n", err);
        wfc_world_destroy(w);
        return 1;
    }

    /* Print */
    printf("\nResult (spectral bias creates clustering):\n");
    for (uint32_t y = 0; y < w->ny; y++) {
        for (uint32_t x = 0; x < w->nx; x++) {
            printf("%c", wfc_get_cell_pattern(w, x, y, 0) == 0 ? '#' : '.');
        }
        printf("\n");
    }

    wfc_world_destroy(w);
    printf("\nDone!\n");
    return 0;
}
