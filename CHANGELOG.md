# Changelog

## Unreleased

Expanded test coverage toward 90% of `wfc.h`, plus fuzz, generation-quality,
and Python regression suites. Examples catalog now covers the C, C++, and
Python API surface.

- C tests for copula (nonzero rho), delta-trail backtrack recovery, spectral
  Gumbel/Matérn fields, non-power-of-two grids, 3D learn, and EINVAL paths
- `test_gray_sobol`, `test_quality`, standalone + libFuzzer `fuzz_wfc`
- C benches: learned checkerboard, copula, backtrack, spectral, 8-pattern
- Python pytest (`python/tests`) and JSON benches (`python/benchmarks`)
- `scripts/run_coverage.sh` — clang llvm-cov, fail if `wfc.h` lines < 90%
- CMake: `ENTROPIC_ENABLE_FUZZ`, `ENTROPIC_ENABLE_COVERAGE`
- C examples: learn, backtrack, seamless, Gray/Sobol, parallel DWT; copula
  sets `copula_rho`; spectral also runs Matérn
- C++ examples: 3D, API tour (`c_ptr`, move, CellView), wavelets
- Python examples under `python/examples/`
- 64-byte aligned `wfc_cell_t` allocation (`posix_memalign` / `_aligned_malloc`);
  type-level `alignas`/`__attribute__((aligned))` plus explicit padding so C and
  C++ agree on sizeof (GCC previously discarded a lone member `_Alignas`)
- C++ `CellView` uses logical indices; `class_constrained` is a `uint64_t` bitset
  with `1ULL << c` (class ids 32–63 are defined and enforced)
- `wfc_world_reset` restores `rng_state` from the original seed
- Hierarchy residual: 1-level inverse DWT upsample of pattern signatures;
  residual Gumbel-tilts pattern choice, not only observe order
- FAF Debug no longer bakes ASan/UBSan into consumers; C++ suspenders TUs
  include `<stdint.h>` before `std::atomic<uint_fast32_t>`
- ASan/UBSan apply whenever `ENTROPIC_ENABLE_SANITIZERS=ON`, including RelWithDebInfo

## v1.1.0

Optional residual multiscale hierarchy and FastAndFourier wavelet spectral
priors. `wfc_hierarchy_init` / `set_wavelet` / `set_target` / `collapse`
run a coarse-to-fine cascade: level sizes are successive right-shifts of
the padded grid (minimum 4), the residual is the target minus the
upsampled coarser solution, and the finest level writes back into the
main world. Coupling is spectral-bias only. Hierarchy and FAF are
compile-time gates (`ENTROPIC_ENABLE_HIERARCHY`, `ENTROPIC_USE_FAF`) and
off by default. C++ (`wfcpp::World`) and Python bindings expose the same
four calls. FastAndFourier is fetched at a pinned commit; its JIT wants
a system compiler at runtime, with the VM as fallback.

## v1.0.0

Initial release: SIMD WFC solver, copula biasing, delta-trail
backtracking, C++ wrapper, and Python bindings.
