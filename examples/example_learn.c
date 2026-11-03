/* example_learn.c — Infer adjacency and weights from a sample grid */
#define WFC_IMPLEMENTATION
#include "entropic/wfc.h"
#include <stdio.h>

static const char GLYPH[] = {'.', '#', '~'};

static void print_grid(const wfc_world_t* w)
{
    for (uint32_t y = 0; y < w->ny; y++) {
        for (uint32_t x = 0; x < w->nx; x++) {
            uint16_t p = wfc_get_cell_pattern(w, x, y, 0);
            putchar(p < 3 ? GLYPH[p] : '?');
        }
        putchar('\n');
    }
}

int main(void)
{
    printf("=== WFC Learn-from-Sample Example ===\n\n");

    wfc_config_t cfg = {
        .nx = 16, .ny = 16, .nz = 1,
        .seed = 7,
        .seamless = 0
    };
    wfc_world_t* w = wfc_world_create(&cfg);
    if (!w) { printf("Failed to create world\n"); return 1; }

    /* Register tiles with dummy weights — learn_from_sample overwrites them. */
    wfc_pattern_def_t land  = {.id = 0, .class_id = 0, .weight = 0.0f};
    wfc_pattern_def_t wall  = {.id = 1, .class_id = 1, .weight = 0.0f};
    wfc_pattern_def_t water = {.id = 2, .class_id = 2, .weight = 0.0f};
    wfc_add_pattern(w, &land);
    wfc_add_pattern(w, &wall);
    wfc_add_pattern(w, &water);

    /* Hand-authored 8x8: a walled pond. Adjacency and frequencies come from this. */
    uint16_t sample[] = {
        0,0,0,0,0,0,0,0,
        0,1,1,1,1,1,1,0,
        0,1,2,2,2,2,1,0,
        0,1,2,2,2,2,1,0,
        0,1,2,2,2,2,1,0,
        0,1,2,2,2,2,1,0,
        0,1,1,1,1,1,1,0,
        0,0,0,0,0,0,0,0,
    };

    printf("Learning rules from 8x8 pond sample...\n");
    wfc_err_t err = wfc_learn_from_sample(w, sample, 8, 8, 1);
    if (err != WFC_OK) {
        printf("learn_from_sample failed: %d\n", err);
        wfc_world_destroy(w);
        return 1;
    }

    printf("Pattern weights after learning:\n");
    for (uint16_t i = 0; i < 3; i++) {
        const wfc_pattern_def_t* p = wfc_get_pattern(w, i);
        printf("  id=%u class=%u weight=%.3f freq=%u\n",
               p->id, p->class_id, p->weight, p->freq);
    }

    printf("\nCollapsing 16x16 with learned rules...\n");
    err = wfc_collapse(w);
    if (err != WFC_OK) {
        printf("Collapse failed: %d\n", err);
        wfc_world_destroy(w);
        return 1;
    }

    printf("\nResult (.=land #=wall ~=water):\n");
    print_grid(w);

    wfc_world_destroy(w);
    printf("\nDone!\n");
    return 0;
}
