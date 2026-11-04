# Changelog

## Unreleased

- The shared suspenders runtime stops when the last parallel world shuts down. CMake pins Memento v3.0.3 and suspenders v1.2.1.
- Observe uses Shannon entropy of the masked weights (`log2 Z − Σ w log2 w / Z`). A singleton is 0. Stale heap keys are recomputed and pushed back when they are no longer the minimum.
- Propagation is AC-3: a neighbor keeps the OR of `compat` over every pattern still alive, run to a fixpoint before the first observe. Seamless wraps on the logical rectangle. The queue ring is twice the cell count so a full frontier does not look empty.
- Pattern choice samples that posterior. Hierarchy multiplies by a Gaussian likelihood on the signature. Detail subbands are kept and steer the next finer level; the running residual subtracts each synthesized contribution. With no target the root solve is unbiased and finer levels observe the upsample. The copula multiplies by the Gaussian copula density.
- Backtrack records the observed cell. Class counts restore from the pre-observe snapshot.
- Bitset OR/AND/eq/empty use AVX2 or NEON when the TU was compiled with that ISA. Popcount is the hardware instruction. The mask is 256 bits, so AVX-512 is not used.
- Spectral fields are standardized before the Gumbel link. A 3D world copies the 2D plane per slice instead of reading past the buffer. Column DWTs on grids ≥32 transpose to sequential access.

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
- Hierarchy residual: detail subbands retained from the Mallat split;
  inverse DWT upsample of pattern signatures; residual is a Gaussian
  likelihood on pattern choice, and the heap uses the entropy of that
  posterior
- FAF Debug no longer bakes ASan/UBSan into consumers; C++ suspenders TUs
  include `<stdint.h>` before `std::atomic<uint_fast32_t>`
- ASan/UBSan apply whenever `ENTROPIC_ENABLE_SANITIZERS=ON`, including RelWithDebInfo

## v1.1.0

Optional residual multiscale hierarchy and FastAndFourier wavelet spectral
priors. `wfc_hierarchy_init` / `set_wavelet` / `set_target` / `set_mode` /
`collapse` run a coarse-to-fine cascade: level sizes are successive
right-shifts of the padded grid (minimum 4), and the finest level writes
back into the main world.

- True multi-resolution residual hierarchy: detail subbands retained, residual energy decreases across levels, levels knob controls real decomposition depth.
- `wfc_hierarchy_adopt_spectral` decomposes a generated fBm or Matérn field once into the hierarchy detail buffers. The cascade starts from that split. A spectral field is not a target until this call or `set_target`.
- `wfc_fix_cell` pins a cell before the solve, including a zero-weight pattern. The opening AC-3 propagates from pinned cells and rejects a pinned pair that breaks a rule.

Coupling is the Gaussian likelihood on pattern signatures. Hierarchy and
FAF are compile-time gates (`ENTROPIC_ENABLE_HIERARCHY`, `ENTROPIC_USE_FAF`)
and off by default. C++ (`wfcpp::World`) and Python bindings expose the
same calls, including `hierarchy_residual_energy`. FastAndFourier is
fetched at a pinned commit; its JIT wants a system compiler at runtime,
with the VM as fallback.

## v1.0.0

Initial release: SIMD WFC solver, copula biasing, delta-trail
backtracking, C++ wrapper, and Python bindings.
