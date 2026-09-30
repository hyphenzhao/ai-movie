"""vc_guard.judge_line: lost voicing, octave collapse and out-of-band outputs fall back (no GPU).

    .venv/bin/python tests/test_vc_guard.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie.vc_guard import guard_lines, judge_line     # noqa: E402

SR = 16000


def voiced(sec: float, hz: float, seed: int = 0) -> np.ndarray:
    """A pulse train with harmonics — pYIN sees it as confidently voiced at *hz*."""
    t = np.arange(int(sec * SR)) / SR
    y = sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 8))
    return (0.3 * y / np.abs(y).max()).astype("float32")


def noise(sec: float) -> np.ndarray:
    return (0.05 * np.random.default_rng(1).standard_normal(int(sec * SR))).astype("float32")


def wav(d: Path, name: str, y: np.ndarray) -> str:
    p = d / name
    sf.write(p, y, SR)
    return str(p)


def test_judge_line():
    d = Path(tempfile.mkdtemp(prefix="vcguard_"))
    v1 = wav(d, "v1.wav", voiced(2.0, 220))
    assert judge_line(wav(d, "ok.wav", voiced(2.1, 230)), v1, "female")["ok"]
    r = judge_line(wav(d, "noise.wav", noise(2.0)), v1, "female")
    assert not r["ok"] and r["reason"].startswith(("lost_voicing", "unvoiced"))
    r = judge_line(wav(d, "octave.wav", voiced(2.0, 110)), v1, "female")
    assert not r["ok"] and (r["reason"].startswith("band_female") or r["reason"].startswith("pitch_jump"))
    r = judge_line(wav(d, "up.wav", voiced(2.0, 310)), v1, "female")     # 1.41× = a jump even inside the band
    assert not r["ok"] and r["reason"].startswith("pitch_jump")
    assert judge_line(wav(d, "male.wav", voiced(2.0, 120)), wav(d, "v1m.wav", voiced(2.0, 125)), "male")["ok"]


def test_short_baseline_is_not_judged_and_chunks_drop_whole():
    d = Path(tempfile.mkdtemp(prefix="vcguard_"))
    v1_short = wav(d, "v1s.wav", voiced(0.25, 220))           # ~15 voiced frames: below the judging floor
    r = judge_line(wav(d, "cs.wav", noise(0.3)), v1_short, "female")
    assert r["ok"] and not r["judged"]
    v1 = [{"audio_fit": wav(d, f"v1_{i}.wav", voiced(1.5, 220))} for i in range(4)]
    segs = [{"gender": "female"} for _ in range(4)]
    items = {0: {"audio": wav(d, "c0.wav", voiced(1.5, 225)), "vc": True, "chunk": [0, 1]},
             1: {"audio": wav(d, "c1.wav", noise(1.5)), "vc": True, "chunk": [0, 1]},
             2: {"audio": wav(d, "c2.wav", voiced(1.5, 225)), "vc": True},
             3: {"audio": wav(d, "c3.wav", voiced(1.5, 110)), "vc": True}}
    st = guard_lines(segs, items, v1)
    assert st["rejected"] == 2 and st["dropped"] == 3                  # 1 fails → its chunk mate 0 goes too; 3 fails alone
    assert items[0]["vc"] is False and items[0]["guard"] == "chunk_member" and items[2]["vc"] is True
    assert list(st["reasons"]) and all(k in ("lost", "unvoiced", "band", "pitch") for k in st["reasons"])


def test_guard_lines_rewrites_items():
    d = Path(tempfile.mkdtemp(prefix="vcguard_"))
    v1 = [{"audio_fit": wav(d, f"v1_{i}.wav", voiced(1.5, 220))} for i in range(3)]
    segs = [{"gender": "female"} for _ in range(3)]
    items = {0: {"audio": wav(d, "c0.wav", voiced(1.5, 225)), "vc": True},
             1: {"audio": wav(d, "c1.wav", noise(1.5)), "vc": True},
             2: {"audio": v1[2]["audio_fit"], "vc": False}}
    st = guard_lines(segs, items, v1)
    assert st["checked"] == 2 and st["rejected"] == 1
    assert items[1]["vc"] is False and items[1]["audio"] == v1[1]["audio_fit"] and items[1]["guard"]
    assert items[0]["vc"] is True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
