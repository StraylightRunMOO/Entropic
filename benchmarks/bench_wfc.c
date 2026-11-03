/* bench_wfc.c — Performance benchmarks for wfc.h */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define WFC_IMPLEMENTATION
#include "entropic/wfc.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void bench_collapse(const char* name, uint32_t nx, uint32_t ny, uint32_t nz,
                           int n_patterns, int n_runs) {
    wfc_config_t cfg = {nx, ny, nz, 42, 0, 0, 0, 0, 1.0f, 1.0f};

    double total_ms = 0;
    double min_ms = 1e9, max_ms = 0;
    int successes = 0;

    for (int r = 0; r < n_runs; r++) {
        wfc_world_t* w = wfc_world_create(&cfg);
        if (!w) continue;

        /* Add patterns */
        for (int i = 0; i < n_patterns; i++) {
            wfc_pattern_def_t p = {(uint16_t)i, (uint16_t)(i % 4), 1.0f, 0, 0, {0}};
            wfc_add_pattern(w, &p);
        }

        /* Fully connected adjacency (no constraints = fastest case) */
        for (int i = 0; i < n_patterns; i++) {
            for (int j = 0; j < n_patterns; j++) {
                int nd = (nz > 1) ? 6 : 4;
                for (int d = 0; d < nd; d++)
                    wfc_add_rule(w, (uint16_t)i, (uint16_t)j, (wfc_dir_t)d);
            }
        }

        double t0 = now_ms();
        wfc_err_t err = wfc_collapse(w);
        double t1 = now_ms();
        double ms = t1 - t0;

        if (err == WFC_OK) {
            successes++;
            total_ms += ms;
            if (ms < min_ms) min_ms = ms;
            if (ms > max_ms) max_ms = ms;
        }

        wfc_world_destroy(w);
    }

    if (successes > 0) {
        printf("%-20s %4dx%-4dx%-4d  %3d pat  %3d/%3d ok  "
               "avg=%7.2fms  min=%7.2fms  max=%7.2fms  %.1f cells/us\n",
               name, nx, ny, nz, n_patterns,
               successes, n_runs, total_ms / successes, min_ms, max_ms,
               (nx * ny * nz) / (total_ms / successes * 1000.0));
    } else {
        printf("%-20s FAILED (0/%d)\n", name, n_runs);
    }
}

static void bench_constrained(const char* name, uint32_t nx, uint32_t ny, int n_runs) {
    /* 4-pattern with partial constraints */
    wfc_config_t cfg = {nx, ny, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};

    double total_ms = 0;
    int successes = 0;

    for (int r = 0; r < n_runs; r++) {
        wfc_world_t* w = wfc_world_create(&cfg);
        if (!w) continue;

        wfc_pattern_def_t p[4];
        for (int i = 0; i < 4; i++) {
            p[i] = (wfc_pattern_def_t){(uint16_t)i, 0, 1.0f, 0, 0, {0}};
            wfc_add_pattern(w, &p[i]);
        }
        /* All patterns compatible - solvable but with more entropy */
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                for (int d = 0; d < 4; d++)
                    wfc_add_rule(w, (uint16_t)i, (uint16_t)j, (wfc_dir_t)d);

        double t0 = now_ms();
        wfc_err_t err = wfc_collapse(w);
        double t1 = now_ms();

        if (err == WFC_OK) {
            successes++;
            total_ms += (t1 - t0);
        }
        wfc_world_destroy(w);
    }

    if (successes > 0)
        printf("%-20s %4dx%-4d        4-pattern full %3d ok  avg=%.2fms\n",
               name, nx, ny, successes, total_ms / successes);
    else
        printf("%-20s FAILED\n", name);
}

static void bench_step_mode(uint32_t nx, uint32_t ny, int n_runs) {
    wfc_config_t cfg = {nx, ny, 1, 42, 0, 0, 0, 0, 1.0f, 1.0f};

    double total_ms = 0;
    for (int r = 0; r < n_runs; r++) {
        wfc_world_t* w = wfc_world_create(&cfg);
        if (!w) continue;

        wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
        wfc_add_pattern(w, &p0);

        double t0 = now_ms();
        wfc_step_init(w);
        while (!wfc_step_done(w)) wfc_step(w);
        double t1 = now_ms();
        total_ms += (t1 - t0);

        wfc_world_destroy(w);
    }

    printf("step_mode            %4dx%-4d        single pat     %3d runs avg=%.2fms\n",
           nx, ny, n_runs, total_ms / n_runs);
}

int main(void) {
    printf("========================================\n");
    printf("WFC Engine Benchmark Suite v2.0\n");
    printf("========================================\n\n");

    printf("%-20s %-18s  %-8s  %-8s  %-20s\n",
           "Benchmark", "Size", "Patterns", "Success", "Timing");
    printf("-----------------------------------------------------------------------\n");

    /* 2D benchmarks */
    bench_collapse("2d_small",   8,   8, 1,   2, 100);
    bench_collapse("2d_medium",  32,  32, 1,   4, 50);
    bench_collapse("2d_large",   64,  64, 1,   4, 20);
    bench_collapse("2d_xlarge",  128, 128, 1,  4, 10);
    bench_collapse("2d_huge",    256, 256, 1,  2, 5);

    printf("\n");

    /* 3D benchmarks */
    bench_collapse("3d_small",   8,  8,  8,   2, 20);
    bench_collapse("3d_medium",  16, 16, 16,  4, 10);
    bench_collapse("3d_large",   32, 32, 32,  4, 5);

    printf("\n");

    /* Constrained benchmarks */
    bench_constrained("constrained_32",  32,  32, 50);
    bench_constrained("constrained_64",  64,  64, 20);
    bench_constrained("constrained_128", 128, 128, 10);

    printf("\n");

    /* Step mode */
    bench_step_mode(16, 16, 20);
    bench_step_mode(32, 32, 10);

    printf("\n========================================\n");
    printf("Benchmarks complete.\n");
    printf("========================================\n");
    return 0;
}
