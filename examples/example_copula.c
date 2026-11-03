/* example_copula.c — Gaussian copula entropy biasing */
#define WFC_IMPLEMENTATION
#include "entropic/wfc.h"
#include <stdio.h>

int main(void) {
    printf("=== WFC Copula Example ===\n\n");

    wfc_config_t cfg = {
        .nx = 16, .ny = 16, .nz = 1,
        .seed = 1337,
        .enable_copula = 1,
        .copula_weight = 1.0f,
        .seamless = 0
    };
    wfc_world_t* w = wfc_world_create(&cfg);
    if (!w) { printf("Failed to create world\n"); return 1; }

    /* Three tile classes that should cluster */
    wfc_pattern_def_t forest[4] = {
        {.id = 0, .class_id = 0, .weight = 1.0f},
        {.id = 1, .class_id = 0, .weight = 1.0f},
        {.id = 2, .class_id = 0, .weight = 1.0f},
        {.id = 3, .class_id = 0, .weight = 1.0f},
    };
    wfc_pattern_def_t desert[4] = {
        {.id = 4, .class_id = 1, .weight = 1.0f},
        {.id = 5, .class_id = 1, .weight = 1.0f},
        {.id = 6, .class_id = 1, .weight = 1.0f},
        {.id = 7, .class_id = 1, .weight = 1.0f},
    };
    wfc_pattern_def_t water[2] = {
        {.id = 8,  .class_id = 2, .weight = 0.5f},
        {.id = 9,  .class_id = 2, .weight = 0.5f},
    };

    for (int i = 0; i < 4; i++) wfc_add_pattern(w, &forest[i]);
    for (int i = 0; i < 4; i++) wfc_add_pattern(w, &desert[i]);
    for (int i = 0; i < 2; i++) wfc_add_pattern(w, &water[i]);

    /* Within-class rules: anything of same class can be adjacent */
    for (int d = 0; d < 4; d++) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                wfc_add_rule(w, (uint16_t)i, (uint16_t)j, (wfc_dir_t)d);
        for (int i = 4; i < 8; i++)
            for (int j = 4; j < 8; j++)
                wfc_add_rule(w, (uint16_t)i, (uint16_t)j, (wfc_dir_t)d);
        for (int i = 8; i < 10; i++)
            for (int j = 8; j < 10; j++)
                wfc_add_rule(w, (uint16_t)i, (uint16_t)j, (wfc_dir_t)d);
    }

    for (int d = 0; d < 4; d++) {
        for(int i = 0; i < 4; i++) {
	   for(int j = 4; j < 10; j++) {
	        wfc_add_rule(w, (uint16_t)j, (uint16_t)i, (wfc_dir_t)d);
	        wfc_add_rule(w, (uint16_t)i, (uint16_t)j, (wfc_dir_t)d);
           }
        }
    }

    /* learn_from_sample never writes rho. Set class correlation by hand:
     * same-class neighbours attract, cross-class pairs mildly repel. */
#ifdef WFC_ENABLE_COPULA
    w->copula_marginal[0] = 0.4f;
    w->copula_marginal[1] = 0.4f;
    w->copula_marginal[2] = 0.2f;
    w->copula_z[0] = wfc_inv_cdf_normal(w->copula_marginal[0]);
    w->copula_z[1] = wfc_inv_cdf_normal(w->copula_marginal[1]);
    w->copula_z[2] = wfc_inv_cdf_normal(w->copula_marginal[2]);
    for (int d = 0; d < 4; d++) {
        w->copula_rho[d][0][0] = 0.85f;
        w->copula_rho[d][1][1] = 0.85f;
        w->copula_rho[d][2][2] = 0.70f;
        w->copula_rho[d][0][1] = -0.35f;
        w->copula_rho[d][1][0] = -0.35f;
        w->copula_rho[d][0][2] = -0.20f;
        w->copula_rho[d][2][0] = -0.20f;
        w->copula_rho[d][1][2] = -0.20f;
        w->copula_rho[d][2][1] = -0.20f;
    }
#endif

    printf("Collapsing 16x16 with copula biasing...\n");
    wfc_err_t err = wfc_collapse(w);
    if (err != WFC_OK) {
        printf("Collapse failed: %d\n", err);
        wfc_world_destroy(w);
        return 1;
    }

    /* Print with class-based coloring */
    printf("\nResult (F=forest, D=desert, W=water):\n");
    for (uint32_t y = 0; y < w->ny; y++) {
        for (uint32_t x = 0; x < w->nx; x++) {
            uint16_t p = wfc_get_cell_pattern(w, x, y, 0);
            char c = (p < 4) ? 'F' : (p < 8) ? 'D' : 'W';
            printf("%c ", c);
        }
        printf("\n");
    }

    wfc_world_destroy(w);
    printf("\nDone!\n");
    return 0;
}
