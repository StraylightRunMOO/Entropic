/*
 * Smoke test for FastAndFourier DWT integration.
 * Verifies round-trip accuracy for CDF 9/7, Haar, and Daubechies-4.
 */
#include <fastandfourier.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

static void test_roundtrip(int family, const char* name, size_t n, size_t levels)
{
    faf_config cfg = faf_config_init(n);
    cfg.family = family;
    cfg.levels = levels;
    cfg.dir = FAF_DIR_FORWARD;
    cfg.layout = FAF_LAYOUT_INTERLEAVED;

    faf_transform* fwd = faf_create_dwt(&cfg);
    if (!fwd) {
        const char* err = faf_get_error();
        fprintf(stderr, "faf_create_dwt failed for %s n=%zu: %s\n",
                name, n, err ? err : "unknown");
    }
    assert(fwd && "faf_create_dwt forward failed");

    faf_transform* inv = faf_create_inverse(fwd);
    assert(inv && "faf_create_inverse failed");

    /* Interleaved: [re0, im0, re1, im1, ...] — 2*n floats */
    float* input  = (float*)faf_aligned_alloc(2 * n * sizeof(float));
    float* coeffs = (float*)faf_aligned_alloc(2 * n * sizeof(float));
    float* output = (float*)faf_aligned_alloc(2 * n * sizeof(float));
    assert(input && coeffs && output);

    /* Fill real parts with a ramp + sine signal; im = 0 */
    for (size_t i = 0; i < n; i++) {
        input[2*i]   = (float)i / (float)n
                        + 0.5f * sinf(2.0f * 3.14159265f * 4.0f * (float)i / (float)n);
        input[2*i+1] = 0.0f;
    }

    int rc = faf_execute_f32(fwd, coeffs, input);
    assert(rc == 0 && "forward DWT failed");

    rc = faf_execute_f32(inv, output, coeffs);
    assert(rc == 0 && "inverse DWT failed");

    /* Check round-trip error (real parts only) */
    float max_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float err = fabsf(output[2*i] - input[2*i]);
        if (err > max_err) max_err = err;
    }

    int ok = max_err < 1e-4f;
    printf("  %-12s n=%4zu levels=%zu  max_err=%.2e  %s\n",
           name, n, levels, max_err, ok ? "PASS" : "FAIL");
    fflush(stdout);
    assert(ok && "round-trip error too large");

    faf_aligned_free(input);
    faf_aligned_free(coeffs);
    faf_aligned_free(output);
    faf_destroy_transform(fwd);
    faf_destroy_transform(inv);
}

int main(void)
{
    printf("=== FastAndFourier DWT Smoke Test ===\n\n");

    faf_init();

    test_roundtrip(FAF_WAVELET_CDF97, "CDF 9/7", 256, 3);
    test_roundtrip(FAF_WAVELET_CDF97, "CDF 9/7", 1024, 5);
    test_roundtrip(FAF_WAVELET_HAAR,  "Haar",    256, 3);
    test_roundtrip(FAF_WAVELET_HAAR,  "Haar",    64,  2);
    test_roundtrip(FAF_WAVELET_D4,    "Daub-4",  256, 3);
    test_roundtrip(FAF_WAVELET_D4,    "Daub-4",  512, 4);
    test_roundtrip(FAF_WAVELET_CDF53, "CDF 5/3", 256, 3);
    test_roundtrip(FAF_WAVELET_SYM4,  "Symlet-4", 256, 3);

    faf_cleanup();

    printf("\nAll DWT smoke tests passed.\n");
    return 0;
}
