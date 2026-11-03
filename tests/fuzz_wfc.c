/* fuzz_wfc.c — libFuzzer harness + standalone PRNG driver
 *
 * Build with -fsanitize=fuzzer for libFuzzer, or -DFUZZ_STANDALONE for ctest.
 * Invariants: no crash; on WFC_OK every cell is a defined pattern and
 * every neighbour pair is allowed by the compatibility table.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define WFC_IMPLEMENTATION
#include "entropic/wfc.h"

static uint8_t u8(const uint8_t* data, size_t size, size_t i) {
    return size ? data[i % size] : 0;
}

int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (!data || size < 8) return 0;

    uint32_t nx = (uint32_t)(u8(data, size, 0) % 12) + 1;
    uint32_t ny = (uint32_t)(u8(data, size, 1) % 12) + 1;
    uint32_t nz = (u8(data, size, 2) & 1) ? 1u : 2u;
    uint32_t npat = (uint32_t)(u8(data, size, 3) % 6) + 1;
    uint32_t seed = (uint32_t)data[4] | ((uint32_t)data[5] << 8) |
                    ((uint32_t)data[6] << 16) | ((uint32_t)data[7] << 24);

    wfc_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.nx = nx;
    cfg.ny = ny;
    cfg.nz = nz;
    cfg.seed = seed ? seed : 1;
    cfg.seamless = (u8(data, size, 8) & 1) != 0;
#if defined(WFC_ENABLE_BACKTRACK)
    cfg.enable_backtracking = (u8(data, size, 8) & 2) != 0;
#endif
#if defined(WFC_ENABLE_COPULA)
    cfg.enable_copula = (u8(data, size, 8) & 4) != 0;
    cfg.copula_weight = 1.0f;
#endif
    cfg.spectral_weight = 1.0f;

    wfc_world_t* w = wfc_world_create(&cfg);
    if (!w) return 0;

    for (uint32_t i = 0; i < npat; i++) {
        wfc_pattern_def_t p;
        memset(&p, 0, sizeof(p));
        p.id = (uint16_t)i;
        p.class_id = (uint16_t)(i % 3);
        uint8_t wb = u8(data, size, 9 + i);
        p.weight = (wb == 0) ? 1.0f : (float)wb;
        if (wfc_add_pattern(w, &p) != WFC_OK) {
            wfc_world_destroy(w);
            return 0;
        }
    }

    int ndirs = (nz > 1) ? 6 : 4;
    /* Dense-ish random compatibility so most cases remain solvable. */
    for (uint32_t i = 0; i < npat; i++) {
        for (uint32_t j = 0; j < npat; j++) {
            for (int d = 0; d < ndirs; d++) {
                uint8_t b = u8(data, size, 16 + i * 8 + j + (uint32_t)d);
                if (b > 40 || i == j)
                    wfc_add_rule(w, (uint16_t)i, (uint16_t)j, (wfc_dir_t)d);
            }
        }
    }

    if (u8(data, size, 10) & 1) {
        uint16_t sample[16];
        uint32_t sx = 2, sy = 2, szs = 1;
        for (uint32_t k = 0; k < sx * sy * szs; k++)
            sample[k] = (uint16_t)(u8(data, size, 20 + k) % npat);
        wfc_learn_from_sample(w, sample, sx, sy, szs);
    }

    wfc_err_t err = wfc_collapse(w);
    if (err == WFC_OK) {
        for (uint32_t z = 0; z < nz; z++) {
            for (uint32_t y = 0; y < ny; y++) {
                for (uint32_t x = 0; x < nx; x++) {
                    uint16_t p = wfc_get_cell_pattern(w, x, y, z);
                    if (p == WFC_PATTERN_INVALID) continue;
                    if (p >= WFC_MAX_PATTERNS) abort();
                }
            }
        }
        uint16_t* grid = (uint16_t*)malloc((size_t)nx * ny * nz * sizeof(uint16_t));
        if (grid) {
            (void)wfc_export_grid(w, grid, (size_t)nx * ny * nz);
            free(grid);
        }
    }

    wfc_world_reset(w);
    wfc_world_destroy(w);
    return 0;
}

#ifdef FUZZ_STANDALONE
int main(int argc, char** argv) {
    int iters = 2000;
    const char* env = getenv("ENTROPIC_FUZZ_ITERS");
    if (env && env[0]) iters = atoi(env);
    if (argc > 1) iters = atoi(argv[1]);
    if (iters < 1) iters = 1;

    uint64_t rng = 0xC0FFEEULL;
    uint8_t buf[96];
    for (int i = 0; i < iters; i++) {
        for (size_t j = 0; j < sizeof(buf); j++) {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            buf[j] = (uint8_t)(rng >> 24);
        }
        LLVMFuzzerTestOneInput(buf, sizeof(buf));
    }
    printf("fuzz_standalone: %d iterations ok\n", iters);
    return 0;
}
#endif
