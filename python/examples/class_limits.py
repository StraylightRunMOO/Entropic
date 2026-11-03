"""Class caps, symmetric rules, seamless torus, and backtracking flag."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_STAGE = _ROOT / "build" / "python"
if _STAGE.exists():
    sys.path.insert(0, str(_STAGE))

import entropic

GLYPH = ".#O"


def print_layer(grid, glyphs=GLYPH):
    for row in grid[0]:
        print("".join(glyphs[int(p)] if int(p) < len(glyphs) else "?" for p in row))


def main():
    print("=== Entropic Python Class Limits + Seamless Example ===\n")

    cfg = entropic.Config(nx=16, ny=16, seed=5, seamless=True)
    w = entropic.World(cfg)

    w.add_pattern(entropic.Pattern(0, 0, 5.0))   # floor
    w.add_pattern(entropic.Pattern(1, 1, 2.0))   # wall
    w.add_pattern(entropic.Pattern(2, 2, 0.4))   # pillar

    for d in (entropic.Dir.N, entropic.Dir.E):
        for a in range(3):
            for b in range(3):
                w.add_symmetric_rule(a, b, d)

    w.set_class_limit(2, 0, 12)
    print("Pillar class capped at 12; torus wrap-around on")

    w.collapse()
    grid = w.export_grid()
    print("\nResult (.=floor #=wall O=pillar):")
    print_layer(grid)
    pillars = int((grid == 2).sum())
    print(f"\nPillar tiles: {pillars} (limit 12)")
    print(repr(cfg))
    print("\nDone!")


if __name__ == "__main__":
    main()
