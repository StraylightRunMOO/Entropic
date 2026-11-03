/* example_gray_sobol.c — Standalone Gray-code and Sobol utilities.
 * These headers are independent of the WFC solver: no WFC_IMPLEMENTATION. */
#include "entropic/gray_codes.h"
#include "entropic/sobol.h"
#include <stdio.h>
#include <stdint.h>

int main(void)
{
    printf("=== Gray Codes + Sobol Example ===\n\n");

    printf("4-bit binary reflected Gray code:\n");
    uint64_t codes[16];
    size_t n = gc_generate_brgc(4, codes);
    for (size_t i = 0; i < n; i++) {
        printf("  rank %2zu  binary %2llu  gray %2llu\n",
               i, (unsigned long long)gc_decode(codes[i]),
               (unsigned long long)codes[i]);
    }
    printf("sequence valid: %s\n", gc_validate_sequence(codes, n) ? "yes" : "no");

    printf("\nRound-trip encode/decode of 0..15:\n");
    for (uint64_t i = 0; i < 16; i++) {
        uint64_t g = gc_encode(i);
        if (gc_decode(g) != i) {
            printf("  mismatch at %llu\n", (unsigned long long)i);
            return 1;
        }
    }
    printf("  ok\n");

    printf("\nFirst 8 Sobol points in 2D (unit square):\n");
    sob31_generator_t gen;
    if (!sob31_init(&gen, 2)) {
        printf("sob31_init failed\n");
        return 1;
    }
    for (int i = 0; i < 8; i++) {
        double x = sob31_to_double(sob31_next(&gen, 0));
        double y = sob31_to_double(sob31_next(&gen, 1));
        gen.count++; /* next() updates coordinates; caller advances the index */
        printf("  %d: (%.5f, %.5f)\n", i, x, y);
    }
    sob31_skip_ahead(&gen, 0, 4);
    printf("skipped 4 more on dim 0; next x=%.5f\n",
           sob31_to_double(sob31_next(&gen, 0)));
    sob31_free(&gen);

    printf("\nDone!\n");
    return 0;
}
