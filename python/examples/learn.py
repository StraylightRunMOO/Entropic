"""Learn adjacency and weights from a sample (list or NumPy)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_STAGE = _ROOT / "build" / "python"
if _STAGE.exists():
    sys.path.insert(0, str(_STAGE))

import numpy as np
import entropic

GLYPH = ".#~"


def print_layer(grid, glyphs=GLYPH):
    for row in grid[0]:
        print("".join(glyphs[int(p)] if int(p) < len(glyphs) else "?" for p in row))


def main():
    print("=== Entropic Python Learn-from-Sample Example ===\n")

    w = entropic.World(entropic.Config(nx=16, ny=16, seed=7))
    w.add_pattern(entropic.Pattern(0, 0, 0.0))
    w.add_pattern(entropic.Pattern(1, 1, 0.0))
    w.add_pattern(entropic.Pattern(2, 2, 0.0))

    sample = [
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 2, 2, 2, 2, 1, 0,
        0, 1, 2, 2, 2, 2, 1, 0,
        0, 1, 2, 2, 2, 2, 1, 0,
        0, 1, 2, 2, 2, 2, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ]
    print("Learning from 8x8 pond (Python list)...")
    w.learn_from_sample(sample, 8, 8, 1)

    print("Collapsing 16x16...")
    w.collapse()
    print("\nResult (.=land #=wall ~=water):")
    print_layer(w.export_grid())

    print("\nReset and re-learn from a NumPy uint16 buffer...")
    w.reset()
    arr = np.array(sample, dtype=np.uint16)
    w.learn_from_sample(arr, 8, 8, 1)
    w.collapse()
    print_layer(w.export_grid())
    print("\nDone!")


if __name__ == "__main__":
    main()
