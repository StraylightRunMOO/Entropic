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

#if defined(WFC_ENABLE_HIERARCHY)
static void bench_hierarchy(const char* name, uint32_t nx, uint32_t ny, int n_runs)
{
    wfc_config_t cfg = {nx, ny, 1, 42, 0, 1, 0, 0, 1.0f, 1.0f};

    double total_ms = 0;
    int successes = 0;

    /* r == -1 builds the n >= 128 JIT kernel and is not timed. */
    for (int r = -1; r < n_runs; r++) {
        wfc_world_t* w = wfc_world_create(&cfg);
        if (!w) continue;

        for (int i = 0; i < 4; i++) {
            wfc_pattern_def_t p = {(uint16_t)i, (uint16_t)(i % 4), 1.0f, 0, 0, {0}};
            wfc_add_pattern(w, &p);
        }
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                for (int d = 0; d < 4; d++)
                    wfc_add_rule(w, (uint16_t)i, (uint16_t)j, (wfc_dir_t)d);

        if (wfc_hierarchy_init(w, 3) != WFC_OK) {
            wfc_world_destroy(w);
            continue;
        }

        double t0 = now_ms();
        wfc_err_t err = wfc_hierarchy_collapse(w);
        double t1 = now_ms();
        if (r < 0) {
            wfc_world_destroy(w);
            continue;
        }

        if (err == WFC_OK) {
            successes++;
            total_ms += (t1 - t0);
        }
        wfc_world_destroy(w);
    }

    double single_ms = 0.0;
    int single_ok = 0;
    for (int r = 0; r < n_runs; r++) {
        wfc_world_t* w = wfc_world_create(&cfg);
        if (!w) continue;
        for (int i = 0; i < 4; i++) {
            wfc_pattern_def_t p = {(uint16_t)i, (uint16_t)(i % 4), 1.0f, 0, 0, {0}};
            wfc_add_pattern(w, &p);
        }
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                for (int d = 0; d < 4; d++)
                    wfc_add_rule(w, (uint16_t)i, (uint16_t)j, (wfc_dir_t)d);
        double t0 = now_ms();
        wfc_err_t err = wfc_collapse(w);
        double t1 = now_ms();
        if (err == WFC_OK) { single_ok++; single_ms += (t1 - t0); }
        wfc_world_destroy(w);
    }

    if (successes > 0 && single_ok > 0) {
        double hms = total_ms / successes;
        double sms = single_ms / single_ok;
        printf("%-20s %4dx%-4d  hierarchy %.1f cells/us   single-scale %.1f cells/us\n",
               name, nx, ny,
               (nx * ny) / (hms * 1000.0),
               (nx * ny) / (sms * 1000.0));
    } else {
        printf("%-20s FAILED\n", name);
    }
}
#endif

static void add_full(wfc_world_t* w, int npat, int ndirs) {
    for (int i = 0; i < npat; i++)
        for (int j = 0; j < npat; j++)
            for (int d = 0; d < ndirs; d++)
                wfc_add_rule(w, (uint16_t)i, (uint16_t)j, (wfc_dir_t)d);
}

static void bench_learn(uint32_t nx, uint32_t ny, int n_runs) {
    uint16_t sample[16] = {
        0,1,0,1, 1,0,1,0, 0,1,0,1, 1,0,1,0
    };
    double total_ms = 0;
    int successes = 0;
    for (int r = 0; r < n_runs; r++) {
        wfc_config_t cfg = {nx, ny, 1, 42u + (uint32_t)r, 0, 0, 0, 0, 1.0f, 1.0f};
        wfc_world_t* w = wfc_world_create(&cfg);
        if (!w) continue;
        wfc_pattern_def_t p0 = {0, 0, 0.0f, 0, 0, {0}};
        wfc_pattern_def_t p1 = {1, 0, 0.0f, 0, 0, {0}};
        wfc_add_pattern(w, &p0);
        wfc_add_pattern(w, &p1);
        wfc_learn_from_sample(w, sample, 4, 4, 1);
        double t0 = now_ms();
        wfc_err_t err = wfc_collapse(w);
        double t1 = now_ms();
        if (err == WFC_OK) { successes++; total_ms += (t1 - t0); }
        wfc_world_destroy(w);
    }
    if (successes > 0)
        printf("%-20s %4dx%-4d        learned        %3d ok  avg=%.2fms\n",
               "learn_checker", nx, ny, successes, total_ms / successes);
    else
        printf("%-20s FAILED\n", "learn_checker");
}

static void bench_non_p2(int n_runs) {
    bench_collapse("2d_nonsquare", 48, 32, 1, 4, n_runs);
}

#if defined(WFC_ENABLE_BACKTRACK)
static void bench_backtrack(uint32_t nx, uint32_t ny, int n_runs) {
    double total_ms = 0;
    int successes = 0;
    for (int r = 0; r < n_runs; r++) {
        wfc_config_t cfg = {nx, ny, 1, 7u + (uint32_t)r, 1, 0, 0, 0, 1.0f, 1.0f};
        wfc_world_t* w = wfc_world_create(&cfg);
        if (!w) continue;
        wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
        wfc_pattern_def_t p1 = {1, 0, 1.0f, 0, 0, {0}};
        wfc_add_pattern(w, &p0);
        wfc_add_pattern(w, &p1);
        add_full(w, 2, 4);
        double t0 = now_ms();
        wfc_err_t err = wfc_collapse(w);
        double t1 = now_ms();
        if (err == WFC_OK) { successes++; total_ms += (t1 - t0); }
        wfc_world_destroy(w);
    }
    if (successes > 0)
        printf("%-20s %4dx%-4d        backtrack      %3d ok  avg=%.2fms\n",
               "backtrack", nx, ny, successes, total_ms / successes);
    else
        printf("%-20s FAILED\n", "backtrack");
}
#endif

#if defined(WFC_ENABLE_COPULA)
static void bench_copula(uint32_t nx, uint32_t ny, int n_runs) {
    double total_ms = 0;
    int successes = 0;
    for (int r = 0; r < n_runs; r++) {
        wfc_config_t cfg = {nx, ny, 1, 11u + (uint32_t)r, 0, 0, 1, 0, 1.0f, 1.0f};
        wfc_world_t* w = wfc_world_create(&cfg);
        if (!w) continue;
        wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
        wfc_pattern_def_t p1 = {1, 1, 1.0f, 0, 0, {0}};
        wfc_add_pattern(w, &p0);
        wfc_add_pattern(w, &p1);
        add_full(w, 2, 4);
        w->copula_z[0] = 0.0f;
        w->copula_z[1] = 0.0f;
        for (int d = 0; d < 4; d++) {
            w->copula_rho[d][0][0] = 0.8f;
            w->copula_rho[d][1][1] = 0.8f;
        }
        double t0 = now_ms();
        wfc_err_t err = wfc_collapse(w);
        double t1 = now_ms();
        if (err == WFC_OK) { successes++; total_ms += (t1 - t0); }
        wfc_world_destroy(w);
    }
    if (successes > 0)
        printf("%-20s %4dx%-4d        copula         %3d ok  avg=%.2fms\n",
               "copula", nx, ny, successes, total_ms / successes);
    else
        printf("%-20s FAILED\n", "copula");
}
#endif

#if defined(WFC_USE_FAF)
static void bench_spectral(uint32_t nx, uint32_t ny, int n_runs) {
    double total_ms = 0;
    int successes = 0;
    for (int r = 0; r < n_runs; r++) {
        wfc_config_t cfg = {nx, ny, 1, 5u + (uint32_t)r, 0, 1, 0, 0, 1.0f, 1.0f};
        wfc_world_t* w = wfc_world_create(&cfg);
        if (!w) continue;
        wfc_pattern_def_t p0 = {0, 0, 1.0f, 0, 0, {0}};
        wfc_pattern_def_t p1 = {1, 0, 1.0f, 0, 0, {0}};
        wfc_add_pattern(w, &p0);
        wfc_add_pattern(w, &p1);
        add_full(w, 2, 4);
        if (wfc_set_gumbel_field(w, 0.7, 99) != WFC_OK) {
            wfc_world_destroy(w);
            continue;
        }
        double t0 = now_ms();
        wfc_err_t err = wfc_collapse(w);
        double t1 = now_ms();
        if (err == WFC_OK) { successes++; total_ms += (t1 - t0); }
        wfc_world_destroy(w);
    }
    if (successes > 0)
        printf("%-20s %4dx%-4d        gumbel         %3d ok  avg=%.2fms\n",
               "spectral_gumbel", nx, ny, successes, total_ms / successes);
    else
        printf("%-20s FAILED\n", "spectral_gumbel");
}
#endif

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

#if defined(WFC_ENABLE_HIERARCHY)
    printf("\n");
    bench_hierarchy("hierarchy_64",  64,  64, 10);
    bench_hierarchy("hierarchy_128", 128, 128, 5);
#endif

    printf("\n");
    bench_collapse("2d_8pat_64", 64, 64, 1, 8, 10);
    bench_collapse("seamless_64", 64, 64, 1, 4, 10);
    bench_non_p2(15);
    bench_learn(32, 32, 20);
    bench_learn(64, 64, 10);

#if defined(WFC_ENABLE_BACKTRACK)
    bench_backtrack(32, 32, 20);
    bench_backtrack(64, 64, 10);
#endif
#if defined(WFC_ENABLE_COPULA)
    bench_copula(32, 32, 20);
    bench_copula(64, 64, 10);
#endif
#if defined(WFC_USE_FAF)
    bench_spectral(32, 32, 10);
    bench_spectral(64, 64, 5);
#endif

    printf("\n========================================\n");
    printf("Benchmarks complete.\n");
    printf("========================================\n");
    return 0;
}
