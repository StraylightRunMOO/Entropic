/* example_parallel.c — suspenders.h workers for spectral DWT (requires FAF) */
#define WFC_IMPLEMENTATION
#include "entropic/wfc.h"
#include <stdio.h>

int main(void)
{
    printf("=== WFC Parallel Spectral Example ===\n");
    printf("(Requires FastAndFourier + suspenders.h)\n\n");

    /* Grids below WFC_PAR_MIN_DIM (32) stay on the serial path. */
    wfc_config_t cfg = {
        .nx = 64, .ny = 64, .nz = 1,
        .seed = 42,
        .enable_spectral = 1,
        .spectral_weight = 0.8f,
        .seamless = 0
    };
    wfc_world_t* w = wfc_world_create(&cfg);
    if (!w) { printf("Failed to create world\n"); return 1; }

    wfc_pattern_def_t p0 = {.id = 0, .class_id = 0, .weight = 1.0f};
    wfc_pattern_def_t p1 = {.id = 1, .class_id = 1, .weight = 1.0f};
    wfc_add_pattern(w, &p0);
    wfc_add_pattern(w, &p1);
    for (int d = 0; d < 4; d++) {
        wfc_add_rule(w, 0, 0, (wfc_dir_t)d);
        wfc_add_rule(w, 1, 1, (wfc_dir_t)d);
        wfc_add_rule(w, 0, 1, (wfc_dir_t)d);
        wfc_add_rule(w, 1, 0, (wfc_dir_t)d);
    }

    printf("Starting parallel runtime (4 workers)...\n");
    wfc_err_t err = wfc_parallel_init(w, 4);
    if (err != WFC_OK) {
        printf("parallel_init failed: %d\n", err);
        wfc_world_destroy(w);
        return 1;
    }

    printf("Generating fBm field on worker pool (H=0.7)...\n");
    err = wfc_set_gumbel_field(w, 0.7, 12345);
    if (err != WFC_OK) {
        printf("set_gumbel_field failed: %d\n", err);
        wfc_parallel_shutdown(w);
        wfc_world_destroy(w);
        return 1;
    }

    printf("Collapsing 64x64...\n");
    err = wfc_collapse(w);
    if (err != WFC_OK) {
        printf("Collapse failed: %d\n", err);
        wfc_parallel_shutdown(w);
        wfc_world_destroy(w);
        return 1;
    }

    printf("\nDownsampled preview (every 4th cell):\n");
    for (uint32_t y = 0; y < w->ny; y += 4) {
        for (uint32_t x = 0; x < w->nx; x += 4) {
            putchar(wfc_get_cell_pattern(w, x, y, 0) == 0 ? '#' : '.');
        }
        putchar('\n');
    }

    wfc_parallel_shutdown(w);
    wfc_world_destroy(w);
    printf("\nDone!\n");
    return 0;
}
