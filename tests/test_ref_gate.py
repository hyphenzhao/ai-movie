"""Reference-clip gate: no-pitch clips never win, ratios rank (no GPU).

    .venv/bin/python tests/test_ref_gate.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("auto_select_refs", ROOT / "scripts" / "auto_select_refs.py")
asr_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(asr_mod)   # type: ignore[union-attr]


def test_candidate_gate():
    ok = asr_mod.candidate_ok
    assert ok(3.0, 150, 250.0, "female") is None
    assert ok(1.5, 150, 250.0, "female") == "short"
    assert ok(4.4, 0, None, "male") == "no_pitch"                       # p05's pick
    assert ok(3.1, 5, 120.0, "male").startswith("voiced_frames")        # p01's male pick
    assert ok(8.0, 45, 240.0, "female").startswith("voiced_ratio")      # 45 of 500 frames = 0.09
    assert ok(6.5, 45, 228.0, "female") is None                          # p14: 45 of 406 = 0.11, sparse but real
    assert ok(3.0, 150, 118.0, "female").startswith("f0_")


def test_qualify_never_picks_unmeasurable_and_ranks_by_ratio():
    rows = [
        {"path": "none", "f0": None, "voiced": 0, "out_f0": [None, None], "ratio": None},
        {"path": "one", "f0": 250.0, "voiced": 90, "out_f0": [245.0, None], "ratio": 0.98},
        {"path": "good", "f0": 250.0, "voiced": 80, "out_f0": [245.0, 240.0], "ratio": 0.97},
        {"path": "fast", "f0": 250.0, "voiced": 95, "out_f0": [300.0, 295.0], "ratio": 1.19},
        {"path": "octave", "f0": 232.0, "voiced": 99, "out_f0": [117.0, 119.0], "ratio": 0.51},
    ]
    ok = asr_mod.qualify(rows, "female")
    assert [r["path"] for r in ok] == ["good", "fast"]
    # same 0.1 ratio band → the clip with more voiced frames wins
    tie = [{"path": "short", "f0": 250.0, "voiced": 60, "out_f0": [262.0, 258.0], "ratio": 1.05},
           {"path": "long", "f0": 250.0, "voiced": 94, "out_f0": [234.0, 240.0], "ratio": 0.94}]
    assert [r["path"] for r in asr_mod.qualify(tie, "female")] == ["long", "short"]
    assert rows[0]["reject"].startswith("measurable_0")
    assert rows[1]["reject"].startswith("measurable_1")
    assert "outside" in rows[4]["reject"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
