"""Optional FAF / hierarchy / suspenders APIs. Skips missing methods."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_STAGE = _ROOT / "build" / "python"
if _STAGE.exists():
    sys.path.insert(0, str(_STAGE))

import numpy as np
import entropic


def _two_tiles(w):
    w.add_pattern(entropic.Pattern(0, 0, 1.0))
    w.add_pattern(entropic.Pattern(1, 1, 1.0))
    for d in (entropic.Dir.N, entropic.Dir.S, entropic.Dir.E, entropic.Dir.W):
        w.add_rule(0, 0, d)
        w.add_rule(1, 1, d)
        w.add_symmetric_rule(0, 1, d)


def print_preview(grid, stride=1):
    layer = grid[0]
    for y in range(0, layer.shape[0], stride):
        print("".join("#" if int(layer[y, x]) == 0 else "."
                      for x in range(0, layer.shape[1], stride)))


def main():
    print("=== Entropic Python Wavelets Example ===\n")

    if hasattr(entropic.World, "set_gumbel_field"):
        print("--- Gumbel fBm then Matérn ---")
        cfg = entropic.Config(nx=16, ny=16, seed=42)
        cfg.enable_spectral = True
        cfg.spectral_weight = 0.6
        w = entropic.World(cfg)
        _two_tiles(w)
        w.set_gumbel_field(0.7, 12345)
        w.collapse()
        print("fBm:")
        print_preview(w.export_grid())
        w.reset()
        w.set_matern_field(1.0, 1.5, 99)
        w.collapse()
        print("\nMatérn:")
        print_preview(w.export_grid())
    else:
        print("set_gumbel_field not in this build (need ENTROPIC_USE_FAF=ON)")

    if hasattr(entropic.World, "hierarchy_init"):
        print("\n--- 3-level hierarchy (CDF 9/7 = family 3) ---")
        cfg = entropic.Config(nx=32, ny=32, seed=7)
        cfg.enable_spectral = True
        cfg.spectral_weight = 1.2
        w = entropic.World(cfg)
        _two_tiles(w)
        w.hierarchy_init(3)
        w.hierarchy_set_wavelet(3, 0)
        target = np.ones((w.num_cells(),), dtype=np.float32)
        w.hierarchy_set_target(target)
        w.hierarchy_collapse()
        print_preview(w.export_grid(), stride=2)
    else:
        print("\nhierarchy_init not in this build "
              "(need ENTROPIC_ENABLE_HIERARCHY=ON)")

    if hasattr(entropic.World, "parallel_init"):
        print("\n--- Parallel DWT (64x64, 4 workers) ---")
        cfg = entropic.Config(nx=64, ny=64, seed=3)
        cfg.enable_spectral = True
        cfg.spectral_weight = 0.8
        w = entropic.World(cfg)
        _two_tiles(w)
        w.parallel_init(4)
        if hasattr(w, "set_gumbel_field"):
            w.set_gumbel_field(0.65, 5)
        w.collapse()
        print_preview(w.export_grid(), stride=4)
        w.parallel_shutdown()
    else:
        print("\nparallel_init not in this build "
              "(need ENTROPIC_ENABLE_SUSPENDERS=ON)")

    print("\nDone!")


if __name__ == "__main__":
    main()
