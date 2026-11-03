"""Python API regression tests for entropic."""
import numpy as np
import pytest

entropic = pytest.importorskip("entropic")


def _full_rules(world, npat=2, dirs=None):
    if dirs is None:
        dirs = [entropic.Dir.N, entropic.Dir.S, entropic.Dir.E, entropic.Dir.W]
    for d in dirs:
        for i in range(npat):
            for j in range(npat):
                world.add_rule(i, j, d)


def test_version():
    assert entropic.__version__ == "1.1.0"


def test_opposite():
    assert entropic.opposite(entropic.Dir.N) == entropic.Dir.S
    assert entropic.opposite(entropic.Dir.E) == entropic.Dir.W
    assert entropic.opposite(entropic.Dir.U) == entropic.Dir.D


def test_config_and_pattern_repr():
    cfg = entropic.Config(nx=8, ny=8, seed=42)
    assert cfg.nx == 8
    assert "Config" in repr(cfg)
    p = entropic.Pattern(1, 2, 0.5)
    assert p.id == 1 and p.class_id == 2
    assert "Pattern" in repr(p)


def test_world_collapse_export():
    w = entropic.World(entropic.Config(nx=4, ny=4, seed=42))
    w.add_pattern(entropic.Pattern(0, 0, 1.0))
    w.add_pattern(entropic.Pattern(1, 0, 1.0))
    _full_rules(w)
    w.collapse()
    grid = w.export_grid()
    assert grid.shape == (1, 4, 4)
    assert set(np.unique(grid)).issubset({0, 1})
    assert w.get_pattern(0, 0) in (0, 1)
    cell = w.cell(0)
    assert cell.collapsed()
    assert "World" in repr(w)
    assert "CellView" in repr(cell)


def test_step_mode():
    w = entropic.World(entropic.Config(nx=2, ny=2, seed=1))
    w.add_pattern(entropic.Pattern(0, 0))
    _full_rules(w, npat=1)
    w.step_init()
    n = 0
    while not w.step_done():
        w.step()
        n += 1
        assert n < 20
    assert n == 4


def test_reset_and_learn():
    w = entropic.World(entropic.Config(nx=4, ny=4, seed=7))
    w.add_pattern(entropic.Pattern(0, 0, 0.0))
    w.add_pattern(entropic.Pattern(1, 0, 0.0))
    sample = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0], dtype=np.uint16)
    w.learn_from_sample(sample, 4, 4, 1)
    w.collapse()
    w.reset()
    assert w.num_patterns() == 2


def test_learn_from_list():
    w = entropic.World(entropic.Config(nx=2, ny=2, seed=3))
    w.add_pattern(entropic.Pattern(0, 0))
    w.add_pattern(entropic.Pattern(1, 0))
    w.learn_from_sample([0, 1, 1, 0], 2, 2, 1)
    w.collapse()
    assert w.export_grid().shape == (1, 2, 2)


def test_3d_export():
    w = entropic.World(entropic.Config(nx=2, ny=2, nz=2, seed=4))
    w.add_pattern(entropic.Pattern(0, 0))
    dirs = [entropic.Dir.N, entropic.Dir.S, entropic.Dir.E,
            entropic.Dir.W, entropic.Dir.U, entropic.Dir.D]
    _full_rules(w, npat=1, dirs=dirs)
    w.collapse()
    grid = w.export_grid()
    assert grid.shape == (2, 2, 2)
    assert w.is_3d()


def test_invalid_pattern():
    w = entropic.World(entropic.Config(nx=2, ny=2, seed=1))
    with pytest.raises(Exception):
        w.add_pattern(entropic.Pattern(256, 0, 1.0))


def test_contradiction():
    w = entropic.World(entropic.Config(nx=2, ny=2, seed=1))
    w.add_pattern(entropic.Pattern(0, 0))
    with pytest.raises(Exception):
        w.collapse()


def test_class_limit():
    w = entropic.World(entropic.Config(nx=4, ny=4, seed=5))
    w.add_pattern(entropic.Pattern(0, 0))
    w.add_pattern(entropic.Pattern(1, 1))
    _full_rules(w)
    w.set_class_limit(1, 0, 4)
    w.collapse()
    grid = w.export_grid()
    assert int(np.sum(grid == 1)) <= 4


def test_seamless_and_symmetric():
    w = entropic.World(entropic.Config(nx=4, ny=4, seed=8, seamless=True))
    w.add_pattern(entropic.Pattern(0, 0))
    w.add_pattern(entropic.Pattern(1, 0))
    w.add_symmetric_rule(0, 0, entropic.Dir.E)
    w.add_symmetric_rule(0, 0, entropic.Dir.N)
    w.add_symmetric_rule(1, 1, entropic.Dir.E)
    w.add_symmetric_rule(1, 1, entropic.Dir.N)
    w.add_symmetric_rule(0, 1, entropic.Dir.E)
    w.add_symmetric_rule(0, 1, entropic.Dir.N)
    w.add_symmetric_rule(1, 0, entropic.Dir.E)
    w.add_symmetric_rule(1, 0, entropic.Dir.N)
    w.collapse()
    assert w.nx() == 4


def test_callback_none():
    w = entropic.World(entropic.Config(nx=2, ny=2, seed=1))
    w.add_pattern(entropic.Pattern(0, 0))
    _full_rules(w, npat=1)
    w.set_contradiction_callback(None)
    w.collapse()


def test_callback_fires():
    w = entropic.World(entropic.Config(nx=2, ny=2, seed=1))
    w.add_pattern(entropic.Pattern(0, 0))
    seen = []
    w.set_contradiction_callback(lambda x, y, z: seen.append((x, y, z)))
    with pytest.raises(Exception):
        w.collapse()
    assert seen


@pytest.mark.skipif(not hasattr(entropic.World, "set_gumbel_field"),
                    reason="FAF not enabled in this build")
def test_spectral():
    cfg = entropic.Config(nx=8, ny=8, seed=2)
    cfg.enable_spectral = True
    w = entropic.World(cfg)
    w.add_pattern(entropic.Pattern(0, 0))
    _full_rules(w, npat=1)
    w.set_gumbel_field(0.5, 9)
    w.collapse()
    w.reset()
    w.set_matern_field(1.0, 1.5, 10)
    w.collapse()


@pytest.mark.skipif(not hasattr(entropic.World, "hierarchy_init"),
                    reason="hierarchy not enabled in this build")
def test_hierarchy():
    cfg = entropic.Config(nx=16, ny=16, seed=3)
    cfg.enable_spectral = True
    w = entropic.World(cfg)
    w.add_pattern(entropic.Pattern(0, 0, 5.0))
    w.add_pattern(entropic.Pattern(1, 1, 3.0))
    _full_rules(w)
    w.hierarchy_init(2)
    target = np.ones((w.num_cells(),), dtype=np.float32)
    w.hierarchy_set_target(target)
    w.hierarchy_collapse()
    grid = w.export_grid()
    assert set(np.unique(grid)).issubset({0, 1})
