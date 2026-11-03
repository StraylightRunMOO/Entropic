"""Entropic — Wave Function Collapse HPC Engine for Python.

Ultra-performant WFC solver with SIMD acceleration, spectral priors,
Gaussian copula biasing, and delta-trail backtracking.

    import entropic
    w = entropic.World(entropic.Config(nx=32, ny=32, seed=42))
    w.add_pattern(entropic.Pattern(0, 0, 1.0))
    w.add_pattern(entropic.Pattern(1, 0, 1.0))
    for d in [entropic.Dir.N, entropic.Dir.S, entropic.Dir.E, entropic.Dir.W]:
        w.add_rule(0, 1, d)
        w.add_rule(1, 0, d)
        w.add_rule(0, 0, d)
        w.add_rule(1, 1, d)
    w.collapse()
    grid = w.export_grid()  # numpy array, shape (nz, ny, nx)
"""

from ._entropic import (
    Dir,
    opposite,
    Pattern,
    Config,
    CellView,
    World,
)

__version__ = "1.1.0"

__all__ = [
    "Dir",
    "opposite",
    "Pattern",
    "Config",
    "CellView",
    "World",
]
