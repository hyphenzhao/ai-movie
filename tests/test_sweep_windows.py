"""asr._sweep_windows: complement of VAD spans, energy floor, long-gap cutting (no GPU).

    .venv/bin/python tests/test_sweep_windows.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie.asr import _frame_db, _sweep_windows      # noqa: E402

SR = 16000


def test_gaps_between_spans_padded_and_floor_applied():
    total = 60.0
    energy = np.full(int(total * 50), -30.0, dtype=np.float32)      # 20 ms frames, audible everywhere
    energy[int(40 * 50):int(60 * 50)] = -70.0                        # silence after 40 s
    spans = [{"start": 5.0, "end": 8.0}, {"start": 8.5, "end": 12.0}, {"start": 30.0, "end": 35.0}]
    wins = _sweep_windows(spans, int(total * SR), energy, min_gap=1.0, max_win=20.0, floor_db=-50.0)
    starts = [w["start"] for w in wins]
    assert starts[0] == 0.0 and abs(wins[0]["end"] - 5.3) < 1e-6          # 0 → first span (+0.3 pad)
    assert not any(abs(w["start"] - 8.0) < 0.5 for w in wins)              # 0.5 s gap < min_gap: skipped
    assert any(abs(w["start"] - 11.7) < 1e-6 and abs(w["end"] - 30.3) < 1e-6 for w in wins)  # the 18.6 s gap, whole
    assert not any(w["start"] >= 40.5 for w in wins)                        # the silent tail yields no window
    assert max(w["end"] for w in wins) < 50.0                               # only the piece that still holds sound


def test_long_gap_is_cut_at_quietest_point():
    total = 80.0
    energy = np.full(int(total * 50), -30.0, dtype=np.float32)
    energy[int(37 * 50):int(38 * 50)] = -60.0                        # a quiet second inside the middle 40 %
    wins = _sweep_windows([], int(total * SR), energy, min_gap=1.0, max_win=50.0, floor_db=-50.0)
    assert len(wins) == 2 and 37.0 <= wins[0]["end"] <= 38.0 and wins[1]["start"] == wins[0]["end"]


def test_frame_db():
    y = np.zeros(SR, dtype=np.float32)
    y[: SR // 2] = 0.1
    db = _frame_db(y)
    assert len(db) == 50 and db[0] > -21 and db[-1] < -100


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
