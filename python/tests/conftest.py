"""Locate the in-tree _entropic extension for pytest."""
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
_candidates = [
    Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]) if os.environ.get("PYTHONPATH") else None,
    _root / "build" / "python",
    _root / "build-coverage" / "python",
    Path(__file__).resolve().parents[1],
]
for cand in _candidates:
    if cand is None:
        continue
    pkg = cand / "entropic"
    if pkg.is_dir() and (pkg / "__init__.py").exists():
        p = str(cand)
        if p not in sys.path:
            sys.path.insert(0, p)
        break
