#!/usr/bin/env python3
"""Python-side microbenchmarks. Prints JSON for regression tracking.

    PYTHONPATH=build/python python python/benchmarks/bench_entropic.py
"""
from __future__ import annotations

import json
import sys
import time

try:
    import numpy as np
    import entropic
except ImportError as e:
    print(json.dumps({"error": str(e), "ok": False}))
    sys.exit(1)


def _full(world, npat, dirs=None):
    if dirs is None:
        dirs = [entropic.Dir.N, entropic.Dir.S, entropic.Dir.E, entropic.Dir.W]
    for d in dirs:
        for i in range(npat):
            for j in range(npat):
                world.add_rule(i, j, d)


def _bench(name, fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "name": name,
        "repeats": repeats,
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def collapse_case(nx, ny, npat, seed=42):
    def run():
        w = entropic.World(entropic.Config(nx=nx, ny=ny, seed=seed))
        for i in range(npat):
            w.add_pattern(entropic.Pattern(i, 0, 1.0))
        _full(w, npat)
        w.collapse()
    return run


def learn_case():
    sample = np.array(
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
        dtype=np.uint16,
    )

    def run():
        w = entropic.World(entropic.Config(nx=16, ny=16, seed=7))
        w.add_pattern(entropic.Pattern(0, 0, 0.0))
        w.add_pattern(entropic.Pattern(1, 0, 0.0))
        w.learn_from_sample(sample, 4, 4, 1)
        w.collapse()
    return run


def step_case():
    def run():
        w = entropic.World(entropic.Config(nx=16, ny=16, seed=3))
        w.add_pattern(entropic.Pattern(0, 0))
        _full(w, 1)
        w.step_init()
        while not w.step_done():
            w.step()
    return run


def main():
    results = [
        _bench("py_collapse_8", collapse_case(8, 8, 2), 40),
        _bench("py_collapse_16", collapse_case(16, 16, 4), 20),
        _bench("py_collapse_32", collapse_case(32, 32, 4), 10),
        _bench("py_collapse_64", collapse_case(64, 64, 2), 5),
        _bench("py_learn_16", learn_case(), 10),
        _bench("py_step_16", step_case(), 10),
    ]
    if hasattr(entropic.World, "hierarchy_init"):
        def hier():
            cfg = entropic.Config(nx=16, ny=16, seed=2)
            cfg.enable_spectral = True
            w = entropic.World(cfg)
            w.add_pattern(entropic.Pattern(0, 0, 5.0))
            w.add_pattern(entropic.Pattern(1, 1, 3.0))
            _full(w, 2)
            w.hierarchy_init(2)
            w.hierarchy_set_target(np.ones(w.num_cells(), dtype=np.float32))
            w.hierarchy_collapse()
        results.append(_bench("py_hierarchy_16", hier, 5))

    payload = {
        "ok": True,
        "version": getattr(entropic, "__version__", "unknown"),
        "benchmarks": results,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
