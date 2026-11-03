"""Step-by-step collapse, CellView, and contradiction callback."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_STAGE = _ROOT / "build" / "python"
if _STAGE.exists():
    sys.path.insert(0, str(_STAGE))

import entropic

GLYPH = ".#+"


def print_layer(grid, glyphs=GLYPH):
    for row in grid[0]:
        print("".join(glyphs[int(p)] if int(p) < len(glyphs) else "?" for p in row))


def main():
    print("=== Entropic Python Stepwise Example ===\n")

    w = entropic.World(entropic.Config(nx=12, ny=12, seed=11, enable_backtracking=True))
    w.add_pattern(entropic.Pattern(0, 0, 4.0))
    w.add_pattern(entropic.Pattern(1, 1, 2.0))
    w.add_pattern(entropic.Pattern(2, 1, 0.5))
    for d in (entropic.Dir.N, entropic.Dir.S, entropic.Dir.E, entropic.Dir.W):
        for a in range(3):
            for b in range(3):
                w.add_rule(a, b, d)

    seen = []
    w.set_contradiction_callback(lambda x, y, z: seen.append((x, y, z)))

    print("Stepping until done...")
    w.step_init()
    steps = 0
    while not w.step_done():
        w.step()
        steps += 1
    print(f"Collapsed in {steps} steps (callback hits: {len(seen)})")
    print(w)

    cell = w.cell(0)
    print(f"cell(0): collapsed={cell.collapsed()} pattern={cell.pattern_id()} "
          f"entropy={cell.entropy():.3f} index={cell.index()}")
    print(cell)

    print("\nResult:")
    print_layer(w.export_grid())

    print("\nClearing callback with None, then reset + collapse()...")
    w.set_contradiction_callback(None)
    w.reset()
    w.collapse()
    print_layer(w.export_grid())
    print("\nDone!")


if __name__ == "__main__":
    main()
