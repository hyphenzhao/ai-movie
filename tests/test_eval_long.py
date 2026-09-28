"""eval_long helpers: cue coverage and orphan distance (no GPU, no film needed).

    .venv/bin/python tests/test_eval_long.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("eval_long", ROOT / "scripts" / "eval_long.py")
el = importlib.util.module_from_spec(spec); spec.loader.exec_module(el)     # type: ignore[union-attr]


def test_coverage_needs_real_overlap():
    cues = [(10.0, 12.0), (20.0, 21.0), (30.0, 31.0)]
    segs = [(11.5, 13.0), (21.05, 22.0), (29.0, 30.05)]
    assert el.coverage(cues, segs) == [True, False, False]          # 0.5 s, 0 s, 0.05 s of overlap
    assert el.coverage(cues, []) == [False, False, False]


def test_cue_distance():
    cues = [(10.0, 12.0), (20.0, 21.0)]
    assert el.cue_distance(cues, 11.0, 11.5) == 0.0
    assert abs(el.cue_distance(cues, 14.0, 15.0) - 2.0) < 1e-9
    assert abs(el.cue_distance(cues, 25.0, 26.0) - 4.0) < 1e-9
    assert el.cue_distance([], 1.0, 2.0) == float("inf")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
