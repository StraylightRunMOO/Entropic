"""2D collapse: Config, Pattern, Dir, add_rule, export_grid, reset."""
import sys
from pathlib import Path

# Allow `python python/examples/basic.py` against an in-tree build.
_ROOT = Path(__file__).resolve().parents[2]
_STAGE = _ROOT / "build" / "python"
if _STAGE.exists():
    sys.path.insert(0, str(_STAGE))

import entropic

GLYPH = ".#~"


def print_layer(grid, z=0, glyphs=GLYPH):
    layer = grid[z]
    for row in layer:
        print("".join(glyphs[int(p)] if int(p) < len(glyphs) else "?" for p in row))


def main():
    print("=== Entropic Python Basic Example ===\n")
    print(f"opposite(Dir.N) = {entropic.opposite(entropic.Dir.N)}")

    cfg = entropic.Config(nx=16, ny=16, seed=42)
    w = entropic.World(cfg)
    w.add_pattern(entropic.Pattern(0, 0, 5.0))   # grass
    w.add_pattern(entropic.Pattern(1, 1, 3.0))   # water
    w.add_pattern(entropic.Pattern(2, 2, 1.0))   # shore

    for d in (entropic.Dir.N, entropic.Dir.S, entropic.Dir.E, entropic.Dir.W):
        w.add_rule(0, 0, d)
        w.add_rule(1, 1, d)
        w.add_rule(2, 2, d)
        w.add_rule(0, 2, d)
        w.add_rule(2, 0, d)
        w.add_rule(1, 2, d)
        w.add_rule(2, 1, d)

    print(f"Collapsing {w.nx()}x{w.ny()} ({w.num_patterns()} patterns)...")
    w.collapse()
    grid = w.export_grid()
    print(f"export_grid shape={grid.shape} dtype={grid.dtype}")
    print("\nResult (.=grass #=water ~=shore):")
    print_layer(grid)
    print(f"get_pattern(0,0) = {w.get_pattern(0, 0)}")

    print("\nReset + re-collapse...")
    w.reset()
    w.collapse()
    print_layer(w.export_grid())
    print("\nDone!")


if __name__ == "__main__":
    main()
