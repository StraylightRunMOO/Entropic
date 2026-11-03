"""3D world: U/D directions, is_3d, export_grid shape (nz, ny, nx)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_STAGE = _ROOT / "build" / "python"
if _STAGE.exists():
    sys.path.insert(0, str(_STAGE))

import entropic


def main():
    print("=== Entropic Python 3D Example ===\n")

    w = entropic.World(entropic.Config(nx=6, ny=6, nz=4, seed=123))
    w.add_pattern(entropic.Pattern(0, 0, 2.0))  # stone
    w.add_pattern(entropic.Pattern(1, 1, 1.0))  # air

    dirs = [entropic.Dir.N, entropic.Dir.S, entropic.Dir.E,
            entropic.Dir.W, entropic.Dir.U, entropic.Dir.D]
    for d in dirs:
        w.add_rule(0, 0, d)
        w.add_rule(1, 1, d)
        w.add_symmetric_rule(0, 1, d)

    print(f"Collapsing {w.nx()}x{w.ny()}x{w.nz()} is_3d={w.is_3d()} "
          f"cells={w.num_cells()}...")
    w.collapse()
    grid = w.export_grid()
    print(f"export_grid shape={grid.shape}")

    for z in range(w.nz()):
        print(f"\nLayer z={z}:")
        for y in range(w.ny()):
            print("".join("#" if int(grid[z, y, x]) == 0 else "."
                          for x in range(w.nx())))

    stone = int((grid == 0).sum())
    air = int((grid == 1).sum())
    print(f"\nStone: {stone}  Air: {air}")
    print(f"get_pattern(0,0,0) = {w.get_pattern(0, 0, 0)}")
    print("\nDone!")


if __name__ == "__main__":
    main()
