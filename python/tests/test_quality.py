"""Generation quality checks on the Python bindings."""
import numpy as np
import pytest

entropic = pytest.importorskip("entropic")

DIRS2D = [entropic.Dir.N, entropic.Dir.S, entropic.Dir.E, entropic.Dir.W]


def _full(world, npat=2):
    for d in DIRS2D:
        for i in range(npat):
            for j in range(npat):
                world.add_rule(i, j, d)


def test_seed_determinism():
    def run():
        w = entropic.World(entropic.Config(nx=8, ny=8, seed=12345))
        w.add_pattern(entropic.Pattern(0, 0))
        w.add_pattern(entropic.Pattern(1, 0))
        _full(w)
        w.collapse()
        return np.array(w.export_grid())

    assert np.array_equal(run(), run())


def test_learned_checkerboard():
    sample = np.array(
        [0, 1, 0, 1,
         1, 0, 1, 0,
         0, 1, 0, 1,
         1, 0, 1, 0],
        dtype=np.uint16,
    )
    w = entropic.World(entropic.Config(nx=8, ny=8, seed=42))
    w.add_pattern(entropic.Pattern(0, 0, 0.0))
    w.add_pattern(entropic.Pattern(1, 0, 0.0))
    w.learn_from_sample(sample, 4, 4, 1)
    w.collapse()
    g = w.export_grid()[0]
    same = 0
    same += int(np.sum(g[:, :-1] == g[:, 1:]))
    same += int(np.sum(g[:-1, :] == g[1:, :]))
    assert same == 0


def test_weight_histogram():
    counts = np.zeros(2, dtype=np.int64)
    for seed in range(6):
        w = entropic.World(entropic.Config(nx=12, ny=12, seed=200 + seed))
        w.add_pattern(entropic.Pattern(0, 0, 9.0))
        w.add_pattern(entropic.Pattern(1, 0, 1.0))
        _full(w)
        w.collapse()
        g = w.export_grid()
        counts[0] += int(np.sum(g == 0))
        counts[1] += int(np.sum(g == 1))
    frac = counts[0] / counts.sum()
    assert 0.75 < frac < 0.98


def test_class_max():
    w = entropic.World(entropic.Config(nx=8, ny=8, seed=9))
    w.add_pattern(entropic.Pattern(0, 0, 1.0))
    w.add_pattern(entropic.Pattern(1, 1, 5.0))
    _full(w)
    w.set_class_limit(1, 0, 8)
    w.collapse()
    assert int(np.sum(w.export_grid() == 1)) <= 8
