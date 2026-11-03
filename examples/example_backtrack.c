/* example_backtrack.c — Delta-trail backtracking on a tight constraint set */
#define WFC_IMPLEMENTATION
#include "entropic/wfc.h"
#include <stdio.h>

static int contradictions = 0;

static void on_contradiction(uint32_t x, uint32_t y, uint32_t z, void* ctx)
{
    (void)x; (void)y; (void)z; (void)ctx;
    contradictions++;
}

int main(void)
{
    printf("=== WFC Backtracking Example ===\n\n");

    /* Heavy weight on pattern 1, but only pattern 0 is allowed to sit next to
     * itself. The solver will try the popular tile, contradict, then rewind. */
    wfc_config_t cfg = {
        .nx = 8, .ny = 8, .nz = 1,
        .seed = 13,
        .enable_backtracking = 1,
        .seamless = 0
    };
    wfc_world_t* w = wfc_world_create(&cfg);
    if (!w) { printf("Failed to create world\n"); return 1; }

    wfc_pattern_def_t rare = {.id = 0, .class_id = 0, .weight = 0.05f};
    wfc_pattern_def_t common = {.id = 1, .class_id = 0, .weight = 20.0f};
    wfc_add_pattern(w, &rare);
    wfc_add_pattern(w, &common);

    /* Only pattern 0 is compatible with anything. Pattern 1 has no rules, so
     * collapsing a neighbour of a 1-cell empties the mask. */
    for (int d = 0; d < 4; d++) {
        wfc_add_rule(w, 0, 0, (wfc_dir_t)d);
        wfc_add_rule(w, 0, 1, (wfc_dir_t)d);
        wfc_add_rule(w, 1, 0, (wfc_dir_t)d);
    }

    wfc_set_contradiction_callback(w, on_contradiction, NULL);

    printf("Collapsing 8x8 with backtracking enabled...\n");
    wfc_err_t err = wfc_collapse(w);
    if (err == WFC_OK) {
        printf("Collapse succeeded (callback fires: %d)\n", contradictions);
        printf("\nResult (0=rare, 1=common):\n");
        for (uint32_t y = 0; y < w->ny; y++) {
            for (uint32_t x = 0; x < w->nx; x++) {
                uint16_t p = wfc_get_cell_pattern(w, x, y, 0);
                putchar(p == 0 ? '.' : (p == 1 ? '#' : '?'));
            }
            putchar('\n');
        }
    } else if (err == WFC_ECONTRADICTION) {
        printf("Trail exhausted; remaining contradiction (err %d)\n", err);
    } else {
        printf("Collapse failed: %d\n", err);
        wfc_world_destroy(w);
        return 1;
    }

    wfc_world_destroy(w);
    printf("\nDone!\n");
    return 0;
}
