"""Chunked voice conversion: grouping, chunk layout and split-back (no GPU).

    .venv/bin/python tests/test_vc_chunks.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie import tts as T                   # noqa: E402

SR = 22050


def tone(sec: float, hz: float) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype("float32")


def test_groups_join_short_lines_to_same_speaker_neighbours():
    segs = [{"start": 0, "end": 0.4, "speaker": "S0"},
            {"start": 0.5, "end": 1.4, "speaker": "S0"},
            {"start": 1.5, "end": 2.0, "speaker": "S0"},
            {"start": 2.1, "end": 2.4, "speaker": "S1"},
            {"start": 9.0, "end": 9.3, "speaker": "S1"}]
    conv = {i: ("x", "refA" if i < 3 else "refB", d)
            for i, d in enumerate([0.4, 0.9, 0.5, 0.3, 0.3])}
    assert T._vc_chunks(segs, conv, min_seconds=0.7, target_seconds=1.5) == [[0, 1, 2], [3, 4]]
    # a long line stays alone unless its neighbour is too short to convert
    conv2 = {0: ("x", "r", 2.0), 1: ("x", "r", 1.8), 2: ("x", "r", 0.4)}
    segs2 = [{"start": i, "end": i + 1, "speaker": "S0"} for i in range(3)]
    assert T._vc_chunks(segs2, conv2, min_seconds=0.7, target_seconds=1.5) == [[0], [1, 2]]
    # never across a speaker change or a different reference
    conv3 = {0: ("x", "rA", 0.3), 1: ("x", "rB", 0.3)}
    segs3 = [{"start": 0, "end": 0.3, "speaker": "S0"}, {"start": 0.4, "end": 0.7, "speaker": "S0"}]
    assert T._vc_chunks(segs3, conv3, min_seconds=0.7, target_seconds=1.5) == [[0], [1]]


def test_chunk_roundtrip_splits_at_inserted_silence():
    d = Path(tempfile.mkdtemp(prefix="vcchunk_"))
    lens = [0.4, 0.9, 0.5]
    srcs = []
    for n, (sec, hz) in enumerate(zip(lens, (220, 330, 440))):
        p = d / f"src{n}.wav"
        sf.write(p, tone(sec, hz), SR)
        srcs.append(str(p))
    layout = T._write_vc_chunk(srcs, d / "chunk.wav")
    assert layout and len(layout["bounds"]) == 3
    # pretend the converter resampled to 24 kHz and stretched by 3 %
    y, _ = sf.read(d / "chunk.wav", dtype="float32")
    n_out = int(len(y) * 24000 / SR * 1.03)
    y2 = y[(np.arange(n_out) * (len(y) - 1) / (n_out - 1)).astype(int)]
    sf.write(d / "conv.wav", y2, 24000)
    pieces = T._split_vc_chunk(str(d / "conv.wav"), {"members": [10, 11, 12], **layout}, d)
    assert all(pieces)
    for p, sec in zip(pieces, lens):
        got = sf.info(p).duration
        assert abs(got - sec * 1.03) < 0.08, (p, got, sec)
        assert sf.info(p).samplerate == 24000


def test_mixed_sample_rates_refuse_chunking():
    d = Path(tempfile.mkdtemp(prefix="vcchunk_"))
    a, b = d / "a.wav", d / "b.wav"
    sf.write(a, tone(0.5, 220), 22050)
    sf.write(b, tone(0.5, 220), 24000)
    assert T._write_vc_chunk([str(a), str(b)], d / "chunk.wav") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
