# Examples

Worked programs for the C solver, the C++ wrapper, and the Python bindings.
Each one is a complete, printable demo: title banner, ASCII grid, `Done!`.

Build the C/C++ binaries with `-DENTROPIC_BUILD_EXAMPLES=ON` (the default):

```bash
cmake -S . -B build -DENTROPIC_BUILD_EXAMPLES=ON \
    -DENTROPIC_USE_FAF=ON -DENTROPIC_ENABLE_HIERARCHY=ON \
    -DENTROPIC_ENABLE_SUSPENDERS=ON
cmake --build build
./build/example_basic
```

Python examples look for `build/python` on `sys.path`. After a CMake Python
build (`-DENTROPIC_BUILD_PYTHON=ON`):

```bash
python python/examples/basic.py
# or
PYTHONPATH=build/python python python/examples/basic.py
```

Optional APIs (spectral, hierarchy, suspenders) are compiled only when the
matching CMake option is on. Python `wavelets.py` skips missing methods
with `hasattr`.

## C

| Binary | Source | What it shows |
|---|---|---|
| `example_basic` | `example_basic.c` | 16×16 grass/water/shore, directed rules |
| `example_3d` | `example_3d.c` | 8×8×8 stone/air, six directions |
| `example_learn` | `example_learn.c` | `wfc_learn_from_sample`, weights, `wfc_get_pattern` |
| `example_stepwise` | `example_stepwise.c` | class limits, callback, `step`/`reset`/`export_grid` |
| `example_seamless` | `example_seamless.c` | toroidal `seamless` wrap |
| `example_backtrack` | `example_backtrack.c` | `enable_backtracking` + contradiction callback |
| `example_copula` | `example_copula.c` | Gaussian copula (`copula_rho` / `copula_z` set by hand) |
| `example_gray_sobol` | `example_gray_sobol.c` | `gray_codes.h` + `sobol.h` (no `WFC_IMPLEMENTATION`) |
| `example_kowloon` | `kowloon.c` | larger tile-set demo using Sobol |
| `example_spectral` | `example_spectral.c` | fBm Gumbel + Matérn priors (**FAF**) |
| `example_hierarchy` | `example_hierarchy.c` | 3-level CDF 9/7 cascade (**FAF + hierarchy**) |
| `example_parallel` | `example_parallel.c` | `wfc_parallel_init` on 64×64 (**FAF + suspenders**) |

C sources that call the solver `#define WFC_IMPLEMENTATION` in that file.
`example_gray_sobol.c` does not — Gray/Sobol are standalone headers.

Copula: `wfc_learn_from_sample` never writes `copula_rho`. Set
`w->copula_rho` and `w->copula_z` on the world after create.

## C++

| Binary | Source | What it shows |
|---|---|---|
| `example_cpp` | `example_cpp.cpp` | RAII `World`, callback, `collapse`, `reset`, `grid_to_string` |
| `example_cpp_advanced` | `example_cpp_advanced.cpp` | learn, class limits, `step`, `CellView` |
| `example_cpp_3d` | `example_cpp_3d.cpp` | 3D + `add_symmetric_rule` on U/D |
| `example_cpp_tour` | `example_cpp_tour.cpp` | seamless, copula via `c_ptr()`, backtrack, move, metadata |
| `example_cpp_wavelets` | `example_cpp_wavelets.cpp` | Gumbel/Matérn, hierarchy, `parallel_init` (**FAF**) |

`wfcpp.hpp` defines `WFC_IMPLEMENTATION` internally. Do not define it in
the `.cpp` that includes the wrapper.

## Python (`python/examples/`)

| Script | What it shows |
|---|---|
| `basic.py` | Config, Pattern, Dir, `opposite`, collapse, export, reset |
| `learn.py` | `learn_from_sample` from a list and a NumPy buffer |
| `stepwise.py` | `step_init` / `step` / `step_done`, `CellView`, callback/`None` |
| `three_d.py` | 3D export shape `(nz, ny, nx)`, `is_3d`, U/D |
| `class_limits.py` | `set_class_limit`, `add_symmetric_rule`, seamless torus |
| `wavelets.py` | Gumbel/Matérn, hierarchy, `parallel_init` if present |

`export_grid()` is a NumPy `uint16` array with shape `(nz, ny, nx)`.
