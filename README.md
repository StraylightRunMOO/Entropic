# Entropic

A single-header C11 Wave Function Collapse engine for people who want
constrained procedural generation at HPC speed. SIMD bitsets, spectral
priors, Gaussian copula biasing, optional residual multiscale hierarchy
and wavelet spectral priors (FastAndFourier), and a step-by-step API
you can visualise.

```c
#define WFC_IMPLEMENTATION
#include "entropic/wfc.h"

wfc_config_t cfg = {.nx = 64, .ny = 64, .nz = 1, .seed = 42};
wfc_world_t* w = wfc_world_create(&cfg);

wfc_pattern_def_t grass = {0, 0, 5.0f};
wfc_pattern_def_t water = {1, 1, 3.0f};
wfc_add_pattern(w, &grass);
wfc_add_pattern(w, &water);

for (int d = 0; d < 4; d++) {
    wfc_add_rule(w, 0, 0, d);  wfc_add_rule(w, 1, 1, d);
    wfc_add_rule(w, 0, 1, d);  wfc_add_rule(w, 1, 0, d);
}

wfc_collapse(w);                              /* full solve */
uint16_t pat = wfc_get_cell_pattern(w, 0, 0, 0);
wfc_world_destroy(w);
```

That is a complete program. Two tiles, four-way adjacency, a 64×64 grid.
The solver observes the minimum-entropy cell, collapses it to one pattern
with weighted randomness, propagates constraints via AC-3, and repeats.
The whole thing runs off a single `uint64_t` PRNG and a pre-allocated
binary heap — zero heap traffic on the hot path.

Version 1.1.0. MIT, see `LICENSE`.

Part of the [StraylightRunMOO](https://github.com/StraylightRunMOO)
ecosystem alongside
[SIGMA](https://github.com/StraylightRunMOO/SIGMA) (statistical computing),
[noize.h](https://github.com/StraylightRunMOO/noize.h) (noise synthesis),
[Memento](https://github.com/StraylightRunMOO/memento) (allocator), and
[Suspenders](https://github.com/StraylightRunMOO/suspenders.h) (coroutines).

## What's in the box

| Piece | What it is |
|---|---|
| `include/entropic/wfc.h` | The solver. One header, `#define WFC_IMPLEMENTATION` in one TU |
| `include/entropic/wfcpp.hpp` | C++17 wrapper: RAII, move semantics, exceptions, STL containers |
| `include/entropic/entropic.h` | Umbrella header — includes everything, C++ wrapper gated on `__cplusplus` |
| `include/entropic/gray_codes.h` | Standalone Gray code library (binary reflected, balanced, monotone, Beckett, …) |
| `include/entropic/sobol.h` | Sobol quasi-random sequences (Joe–Kuo, 10 dimensions) |
| `python/` | nanobind Python bindings — NumPy arrays out, GIL released during collapse |
| `examples/` | C and C++ demos covering the solver, wrapper, Gray/Sobol, and gated FAF features |
| `python/examples/` | Python scripts covering the nanobind API (`hasattr` for optional methods) |
| `benchmarks/` | Microbench suite: 2D/3D, learned, spectral, copula, backtrack, hierarchy |
| `python/tests/` | pytest API + generation-quality suite |
| `python/benchmarks/` | Python collapse/learn/step JSON benches |

## How it works (the five-minute version)

**The grid.** Dimensions are rounded up to powers of two (`nx_p2`, `ny_p2`,
`nz_p2`); index arithmetic uses shifts. Each cell is 64-byte aligned and
holds a `uint64_t mask[]` bitset of valid patterns, a collapsed ID, cached
entropy, an optional spectral/copula prior, and a dirty flag.

**Observe.** A binary min-heap tracks cells by entropy. The cheapest cell
is popped in O(log n) and collapsed to one pattern by weighted random
selection with a SplitMix64 PRNG. Tie-breaking uses deterministic
micro-noise to prevent directional artifacts.

**Propagate (AC-3).** Neighbours of the collapsed cell are enqueued in a
ring buffer. For each queued cell, the mask is ANDed against the precomputed
compatibility table `compat[dir][pat][block]`. If the mask changed, the
cell's neighbours are enqueued. If the mask is empty, that is a
contradiction.

**Bitset SIMD.** Mask operations dispatch at compile time:

| Operation | Scalar | AVX2 | NEON |
|---|---|---|---|
| mask_and | loop | 256-bit | 2×128-bit |
| mask_andnot | loop | 256-bit | 2×128-bit |
| mask_empty | loop | testz | vceq+and |
| mask_popcnt | Harley-Seal | Harley-Seal | vcnt+vaddv |

**Spectral priors** (`WFC_USE_FAF`). An fBm or Matérn random field is
generated via FastAndFourier's DWT (CDF 9/7 wavelets by default) and
stored as per-cell entropy bias. Cells in high-variance regions of the
field collapse first, giving large-scale spatial coherence to the output
without touching the constraint rules.

**Residual multiscale hierarchy** (`WFC_ENABLE_HIERARCHY`, requires
`WFC_USE_FAF`). A coarse-to-fine cascade where the grid is solved at
progressively finer resolutions. Each level creates a temporary world
at that scale, biased by the residual between a target field and the
upsampled coarser solution. The finest level copies its result back to
the main world.

### How the residual hierarchy works

Level sizes are derived by successive right-shifts of the padded
dimensions (`nx_p2`, `ny_p2`), with a minimum of 4 on each axis.
`wfc_hierarchy_init` returns `WFC_EINVAL` if `nlevels` is outside
`[2, 8]` or if the coarsest scale would fall below 4×4. Weights scale
as `1 / 2^shift`. The residual at level *k* is the difference between
the target (or, with no target, zero) projected onto that scale by
box-averaging logical cells, and the 2× upsample of the coarser
level's **pattern signatures** (class bin plus a unique pattern-id
sub-quantum — weights do not enter). Coupling between levels is only
through the spectral bias field — there is no direct pattern locking
from coarse codes.

```c
wfc_hierarchy_init(w, 3);                         /* 8×8 → 16×16 → 32×32 */
wfc_hierarchy_set_wavelet(w, FAF_WAVELET_CDF97, 0); /* 0 = full DWT depth */
wfc_hierarchy_set_target(w, field, w->num_cells); /* optional */
wfc_hierarchy_collapse(w);                        /* writes back to w */
```

See `examples/example_hierarchy.c` and `python/examples/wavelets.py`.

**Gaussian copula** (`WFC_ENABLE_COPULA`). Class-aware correlation via
C-vine factorisation with precomputed Z-values. Patterns of the same class
cluster together — "desert next to desert" — without adding explicit
adjacency rules for every pair.

**Delta-trail backtracking** (`WFC_ENABLE_BACKTRACK`). On contradiction,
the solver rewinds to the last branch point by replaying a stack of cell
mask diffs. No full grid copies.

**Class limits.** Hard min/max constraints per pattern class. The solver
tracks class population during collapse and masks out patterns whose class
has hit its ceiling.

## What it costs

Measured on a Jetson Orin (Cortex-A78AE, 1.73 GHz, aarch64 NEON, Clang 22,
`-O3 -march=native`). Treat the absolutes as machine-specific and the
ordering as the finding.

| Grid | Patterns | Avg time | Throughput |
|---|---|---|---|
| 8×8 2D | 2 | 0.06 ms | 1.1 cells/µs |
| 32×32 2D | 4 | 0.96 ms | 1.1 cells/µs |
| 64×64 2D | 4 | 4.2 ms | 1.0 cells/µs |
| 128×128 2D | 4 | 20.5 ms | 0.8 cells/µs |
| 256×256 2D | 2 | 142 ms | 0.5 cells/µs |
| 8×8×8 3D | 2 | 0.47 ms | 1.1 cells/µs |
| 16×16×16 3D | 4 | 4.2 ms | 1.0 cells/µs |
| 32×32×32 3D | 4 | 48.9 ms | 0.7 cells/µs |
| 64×64 2D hierarchy (3-level) | 4 | 4.7 ms | 0.9 cells/µs |
| 128×128 2D hierarchy (3-level) | 4 | 22.3 ms | 0.7 cells/µs |

The 256×256 row is the heap rebuild cost asserting itself — lazy entropy
updates batch well when most neighbours haven't changed, less well when the
entire frontier is dirty. The 3D throughput drop at 32³ is cache pressure:
32K cells × 64 bytes is 2 MiB, which exceeds L1.

Run it yourself: `./build/bench_wfc`.

## How to build

```bash
mkdir build && cd build
cmake .. -DENTROPIC_BUILD_TESTS=ON -DENTROPIC_BUILD_EXAMPLES=ON
cmake --build .
ctest                    # three test suites
```

The Makefile (`make all test`) builds the C tests without CMake but does
not build the C++ or gtest suites.

### Build options

| Option | Default | What it does |
|---|---|---|
| `ENTROPIC_BUILD_TESTS` | ON | Three test suites: C assert, Google Test, C++ inline |
| `ENTROPIC_BUILD_BENCHMARKS` | ON | `bench_wfc` microbench binary |
| `ENTROPIC_BUILD_EXAMPLES` | ON | C/C++ example binaries (catalog in `examples/README.md`) |
| `ENTROPIC_ENABLE_AVX2` | OFF | AVX2 bitset operations (x86_64) |
| `ENTROPIC_ENABLE_NEON` | OFF | ARM NEON bitset operations (aarch64) |
| `ENTROPIC_USE_FAF` | OFF | Wavelet spectral backend via FastAndFourier (FetchContent, pinned commit) |
| `ENTROPIC_ENABLE_HIERARCHY` | OFF | Residual multiscale hierarchy (requires FAF) |
| `ENTROPIC_ENABLE_SUSPENDERS` | OFF | Coroutine parallelism for DWT/noise (requires FAF) |
| `ENTROPIC_ENABLE_COPULA` | ON | Gaussian copula biasing |
| `ENTROPIC_ENABLE_BACKTRACK` | ON | Delta-trail backtracking |
| `ENTROPIC_ENABLE_SANITIZERS` | OFF | ASan + UBSan (Debug builds only) |
| `ENTROPIC_BUILD_PYTHON` | OFF | nanobind Python extension module |

### Compile-time knobs

These are `#define`s set before including the header:

| Knob | Default | What it does |
|---|---|---|
| `WFC_MAX_PATTERNS` | 64 | Maximum pattern count; must be a power of two, ≤ 256 |
| `WFC_MAX_CLASSES` | 16 | Maximum class count |
| `WFC_MAX_LAYERS` | 4 | Maximum layers per pattern |
| `WFC_MAX_BACKTRACK_DEPTH` | 1024 | Maximum trail entries |
| `WFC_MALLOC` / `WFC_FREE` / `WFC_REALLOC` | stdlib | Custom allocator hooks |

### Python bindings

```bash
pip install .                          # from repo root; pulls nanobind + scikit-build-core
python -c "import entropic; print(entropic.World(entropic.Config(nx=8, ny=8)))"
```

Or from the top-level CMake: `cmake .. -DENTROPIC_BUILD_PYTHON=ON`.
Worked scripts live in `python/examples/` (`PYTHONPATH=build/python`).

## How to call it

### C API

Define `WFC_IMPLEMENTATION` in exactly one translation unit. Every other
file includes the header and defines nothing.

```c
#define WFC_IMPLEMENTATION
#include "entropic/wfc.h"
```

| Function | What it does |
|---|---|
| `wfc_world_create` | Allocate and initialise world from config |
| `wfc_world_destroy` | Free all resources |
| `wfc_world_reset` | Reset for re-solving; preserves patterns and rules |
| `wfc_add_pattern` | Register a pattern with weight and class |
| `wfc_add_rule` | Add directional adjacency (A beside B going dir) |
| `wfc_set_class_limit` | Hard min/max constraints per class |
| `wfc_learn_from_sample` | Infer rules and frequencies from an example grid |
| `wfc_set_gumbel_field` | Generate fBm spectral prior (requires FAF) |
| `wfc_set_matern_field` | Generate Matérn spectral prior (requires FAF) |
| `wfc_hierarchy_init` | Set up N-level coarse-to-fine hierarchy (requires FAF + hierarchy) |
| `wfc_hierarchy_set_wavelet` | Choose wavelet family and decomposition levels |
| `wfc_hierarchy_set_target` | Provide a target field for residual biasing |
| `wfc_hierarchy_collapse` | Solve via coarse-to-fine cascade |
| `wfc_collapse` | Full WFC solve |
| `wfc_step_init` / `wfc_step` / `wfc_step_done` | Step-by-step mode for visualisation |
| `wfc_set_contradiction_callback` | Register a callback fired on contradiction |
| `wfc_get_cell_pattern` | Query collapsed pattern at (x, y, z) |
| `wfc_get_pattern` | Look up a pattern definition by ID |
| `wfc_export_grid` | Copy the collapsed grid into a user buffer |

### C++17 wrapper

```cpp
#include <entropic/wfcpp.hpp>

wfcpp::World w({.nx = 64, .ny = 64, .seed = 42});
w.add_pattern(wfcpp::Pattern(0, 0, 5.0f));
w.add_pattern(wfcpp::Pattern(1, 1, 3.0f));
for (auto d : {wfcpp::Dir::N, wfcpp::Dir::E, wfcpp::Dir::S, wfcpp::Dir::W}) {
    w.add_rule(0, 0, d);  w.add_rule(0, 1, d);
    w.add_rule(1, 0, d);  w.add_rule(1, 1, d);
}
w.collapse();
auto grid = w.export_grid();   // std::vector<uint16_t>
```

The wrapper handles lifecycle (RAII, move, no copy), throws on error
(`wfcpp::Contradiction`, `wfcpp::InvalidArgument`, `wfcpp::OutOfMemory`),
and provides `CellView` for per-cell inspection, `grid_to_string` for
debug output, and `idx_to_coords` / `coords_to_idx` for index conversion.
`wfcpp.hpp` defines `WFC_IMPLEMENTATION` internally — don't define it
yourself when using the C++ header. With hierarchy enabled the wrapper
forwards `hierarchy_init`, `hierarchy_set_wavelet`, `hierarchy_set_target`
(`std::vector<float>` or pointer + count), and `hierarchy_collapse`.

### Python

```python
import entropic
w = entropic.World(entropic.Config(nx=64, ny=64, seed=42))
w.add_pattern(entropic.Pattern(0, 0, 5.0))
w.add_pattern(entropic.Pattern(1, 1, 3.0))
for d in [entropic.Dir.N, entropic.Dir.E, entropic.Dir.S, entropic.Dir.W]:
    w.add_rule(0, 0, d);  w.add_rule(0, 1, d)
    w.add_rule(1, 0, d);  w.add_rule(1, 1, d)
w.collapse()
grid = w.export_grid()   # numpy uint16 array, shape (nz, ny, nx)
```

With hierarchy enabled (`-DENTROPIC_USE_FAF=ON -DENTROPIC_ENABLE_HIERARCHY=ON`):

```python
w.hierarchy_init(3)
w.hierarchy_set_wavelet(3, 0)          # FAF_WAVELET_CDF97, full depth
w.hierarchy_set_target(field)          # 1-D float32 NumPy array, length num_cells
w.hierarchy_collapse()
```

The GIL is released during `collapse()` and `hierarchy_collapse()`.
Contradiction callbacks re-acquire it.

## The small print (a.k.a. honest limitations)

- **The solver is single-threaded.** Parallelism lives in the SIMD bitset
  operations, not in the collapse loop. AC-3 propagation has data
  dependencies that make lock-free parallelism worse than serial on grids
  this size. If your grid is 4096² and you need threads, this is not the
  library.
- **Power-of-two padding wastes memory.** A 33×33 grid allocates as
  64×64. The solver only collapses the logical cells and fences off the
  padding, but the memory is still allocated. The shifts buy fast index
  arithmetic; the waste is the price.
- **`WFC_MAX_PATTERNS` caps at 256.** The bitset is `uint64_t` words.
  256 patterns is 4 words per cell, 32 bytes per mask. Raise it and the
  bitset operations, the compatibility table, and the per-cell size all
  grow. 64 is the default because most tile sets are small.
- **Spectral fields need FastAndFourier.** Build with
  `-DENTROPIC_USE_FAF=ON` to enable `wfc_set_gumbel_field` and
  `wfc_set_matern_field`. Without it the solver works fine; you just
  lose the spatial bias. The multiscale hierarchy also requires FAF.
  FastAndFourier is pulled via FetchContent at a pinned commit. The
  FAF JIT wants a system C compiler (`cc`) at runtime; if none is
  available the interpreter VM is used, which is the right fallback
  for locked-down targets.
- **Hierarchy coupling is spectral-only.** Coarse pattern IDs bias the
  finer entropy field; they do not lock finer cells to matching
  patterns. Hierarchy is 2D (the finest level writes `z = 0` back to
  the main world).
- **Copula calibration is manual.** The library provides the machinery
  (`wfc_set_class_limit`, copula weight in config) but not a way to
  estimate correlation parameters from data. Bring your own Z-values, or
  use [SIGMA](https://github.com/StraylightRunMOO/SIGMA)'s `sigma_qnorm_d`.
- **Backtracking is bounded.** `WFC_MAX_BACKTRACK_DEPTH` defaults to 1024
  trail entries. A contradiction deeper than that is a hard failure. In
  practice, well-constrained tile sets rarely hit it.
- **`wfc_learn_from_sample` learns adjacency, not weights.** It scans the
  sample for which patterns appear beside which, in which direction, and
  adds those rules. Pattern weights come from the `wfc_pattern_def_t` you
  registered; the sample does not adjust them.

## Ecosystem integration

Entropic is designed to interoperate with the other StraylightRunMOO
libraries. The short version:

**SIGMA** provides distributions, a SIMD xoshiro256** PRNG, and online
statistics. Use it to sample pattern weights from a beta or Dirichlet
distribution, compute inverse-CDF Z-values for copula calibration
(`sigma_qnorm_d`), or monitor entropy convergence with Welford's algorithm.

**noize.h** provides spectral noise, spatial noise, and quasi-random
sequences. Use `noize_matern()` or `noize_fbm()` for richer spectral
priors than the built-in pair, `noize_perlin2()` to generate continuous
fields that bias cell patterns before collapse, or `noize_poisson_disk()`
to place initial constraints with guaranteed spacing.

**The three-library pipeline:**

```
SIGMA (distributions) → noize.h (spatial fields) → Entropic (constrained collapse)
```

Sample a correlation matrix with SIGMA. Generate a Matérn field and
Poisson-disk seed points with noize.h. Load the field as entropy bias,
pin the seeds as initial constraints, configure the copula, and collapse.

## Bundled utilities

`gray_codes.h` and `sobol.h` are standalone headers not included by
`wfc.h`. Gray codes cover binary reflected, balanced, monotone,
single-track, m-ary, constant-weight, Beckett, RLL, and permutation
variants. Sobol covers Joe–Kuo direction numbers for 10 dimensions.
noize.h provides a more complete Sobol implementation (2049 dimensions)
if you need it.

## Testing

C/C++ suites, each its own binary so `WFC_IMPLEMENTATION` is defined once:

| Target | What it covers |
|---|---|
| `test_wfc_c` | C API, edge paths, copula/backtrack/spectral when enabled |
| `test_wfc_gtest` | C++ wrapper via Google Test 1.14.0 |
| `test_wfc_cpp` | C++ wrapper, inline assert framework |
| `test_gray_sobol` | `gray_codes.h` / `sobol.h` |
| `test_quality` | Constraint fidelity, learned checkerboard, weight histogram, seed determinism |
| `test_faf_smoke` | FastAndFourier DWT (`ENTROPIC_USE_FAF`) |
| `test_hierarchy` | Residual cascade (`ENTROPIC_ENABLE_HIERARCHY`) |
| `test_suspenders` | Coroutine parallelism (`ENTROPIC_ENABLE_SUSPENDERS`) |
| `fuzz_wfc_standalone` | Bounded PRNG fuzz (`ENTROPIC_ENABLE_FUZZ`); `fuzz_wfc` is the libFuzzer binary |

Python (with `-DENTROPIC_BUILD_PYTHON=ON`):

```bash
ctest -R python_pytest
# or
PYTHONPATH=build/python python -m pytest python/tests -q
PYTHONPATH=build/python python python/benchmarks/bench_entropic.py
```

ASan/UBSan: `-DENTROPIC_ENABLE_SANITIZERS=ON` (Debug only). Coverage
(clang source-based, fail under 90% lines in `wfc.h`):

```bash
./scripts/run_coverage.sh
```

Fuzz with libFuzzer (clang):

```bash
cmake -B build -DENTROPIC_ENABLE_FUZZ=ON
cmake --build build --target fuzz_wfc
./build/fuzz_wfc -max_total_time=30
```

## Roadmap

Things we'd take patches for, in rough order of interest:

- Overlapping-pattern model (NxN tiles extracted from bitmaps)
- Constraint painting API for interactive editors
- GPU compute path (Vulkan or CUDA) for grids past 512²
- noize.h integration as a first-class spectral backend
- SIGMA PRNG as a pluggable replacement for SplitMix64
- WebAssembly build and JS bindings

## License

MIT. Do what you like; tell us if it eats your tileset, and send the
seed.
