/* test_gray_sobol.c — Gray codes and Sobol sequence utilities */
#undef NDEBUG
#include "entropic/gray_codes.h"
#include "entropic/sobol.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) static void test_##name(void)
#define RUN(name) do { test_count++; printf("  [%2d] " #name " ... ", test_count); fflush(stdout); test_##name(); pass_count++; printf("PASS\n"); } while(0)

TEST(bit_utils) {
    assert(gc_popcount64(0) == 0);
    assert(gc_popcount64(0xFULL) == 4);
    assert(gc_popcount64(~0ULL) == 64);
    assert(gc_parity64(1) == 1);
    assert(gc_parity64(3) == 0);
    assert(gc_log2_64(1) == 0);
    assert(gc_log2_64(8) == 3);
    assert(gc_ctz64(8) == 3);
    assert(gc_rotate_left_u64(1, 1) == 2);
    assert(gc_rotate_right_u64(2, 1) == 1);
    assert(gc_hamming_distance(0, 7) == 3);
}

TEST(brgc_roundtrip) {
    for (uint64_t i = 0; i < 1024; i++) {
        uint64_t g = gc_encode(i);
        assert(gc_decode(g) == i);
        assert(gc_to_binary(g) == i);
        assert(gc_from_binary(i) == g);
        assert(gc_rank(g) == (size_t)i);
        assert(gc_kth(10, (size_t)i) == g);
    }
    uint64_t codes[16];
    assert(gc_generate_brgc(4, codes) == 16);
    assert(gc_validate_sequence(codes, 16));
    assert(gc_generate_brgc(4, NULL) == 0);
    assert(gc_generate(GC_TYPE_BINARY_REFLECTED, 4, 0, codes) == 16);
}

TEST(nary_and_variants) {
    uint64_t codes[256];
    assert(gc_generate_balanced(4, codes) == 16);
    assert(gc_generate_monotone(4, codes) == 16);
    assert(gc_generate_single_track(4, codes) == 16);
    assert(gc_generate_nary(3, 3, codes) == 27);
    (void)gc_nary_encode(11, 3);
    (void)gc_nary_decode(5, 3);
    size_t n = gc_generate_constant_weight(5, 2, codes);
    assert(n > 0);
    for (size_t i = 0; i < n; i++)
        assert(gc_popcount64(gc_decode(codes[i])) <= 5);
    assert(gc_generate_beckett(4, codes) > 0);
    assert(gc_generate_beckett(3, codes) > 0);
    assert(gc_generate_rll(4, 2, codes) > 0);
    assert(gc_generate_perm_gray(3, codes) == 6);
    assert(gc_generate(GC_TYPE_NARY, 2, 3, codes) == 9);
    assert(gc_generate(GC_TYPE_COMBINATION, 4, 2, codes) > 0);
    assert(gc_generate((gc_type_t)99, 4, 0, codes) == 0);
    assert(gc_validate_sequence(NULL, 4) == 0);
}

TEST(sobol_basic) {
    sob31_generator_t gen;
    assert(sob31_init(NULL, 2) == 0);
    assert(sob31_init(&gen, 0) == 0);
    assert(sob31_init(&gen, 99) == 0);
    assert(sob31_init(&gen, 4) == 1);
    uint64_t a = sob31_next(&gen, 0);
    uint64_t b = sob31_next(&gen, 1);
    (void)a; (void)b;
    double u = sob31_to_double(sob31_next(&gen, 0));
    assert(u >= 0.0 && u < 1.0);
    uint64_t batch[32];
    assert(sob31_generate(&gen, 0, 32, batch) == 32);
    sob31_skip_ahead(&gen, 0, 4);
    sob31_free(&gen);
    sob31_free(NULL);
}

int main(void) {
    printf("========================================\n");
    printf("Gray / Sobol Test Suite\n");
    printf("========================================\n\n");
    RUN(bit_utils);
    RUN(brgc_roundtrip);
    RUN(nary_and_variants);
    RUN(sobol_basic);
    printf("\n========================================\n");
    printf("Results: %d/%d passed\n", pass_count, test_count);
    printf("========================================\n");
    return (pass_count == test_count) ? 0 : 1;
}
