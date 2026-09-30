"""composer._restore_original_ranges: original vocals return only inside kept ranges (no GPU).

    .venv/bin/python tests/test_mix_restore.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie.composer import _restore_original_ranges     # noqa: E402


def test_restore_ranges_and_mask():
    sr = 48000
    d = Path(tempfile.mkdtemp(prefix="restore_"))
    voc = (0.5 * np.sin(2 * np.pi * 220 * np.arange(10 * 16000) / 16000)).astype("float32")  # 16 kHz mono
    sf.write(d / "voc.wav", voc, 16000)
    total = 10 * sr
    r = _restore_original_ranges([(2.0, 3.0), (6.0, 6.5)], d / "voc.wav", total, sr, 2, pad_ms=100)
    assert r is not None and r["audio"].shape == (total, 2)
    m = r["mask"]
    assert m[int(2.5 * sr)] == 1 and m[int(6.2 * sr)] == 1 and m[int(4.5 * sr)] == 0 and m[int(1.0 * sr)] == 0
    assert m[int(1.95 * sr)] == 1 and m[int(1.85 * sr)] == 0                      # 100 ms pad
    assert np.abs(r["audio"][int(2.5 * sr):int(2.51 * sr)]).max() > 0.3 and np.abs(r["audio"][int(4.5 * sr):int(4.51 * sr)]).max() == 0
    # fades: the first sample of a range is (near) silent, the middle is not
    assert abs(r["audio"][int(1.9 * sr)][0]) < 0.01
    assert _restore_original_ranges([], d / "voc.wav", total, sr, 2) is None
    assert _restore_original_ranges([(1, 2)], d / "missing.wav", total, sr, 2) is None


def test_adjacent_ranges_restore_once():
    """Two kept-original lines closer than 2 × pad used to get the vocals summed over the shared
    pad (+6 dB, peak 1.0 for a 0.5 sine) with a fade-out/fade-in crossing in the middle of a moan."""
    sr = 48000
    d = Path(tempfile.mkdtemp(prefix="restore_"))
    voc = (0.5 * np.sin(2 * np.pi * 220 * np.arange(10 * sr) / sr)).astype("float32")
    sf.write(d / "voc.wav", voc, sr)
    total = 10 * sr
    for ranges in ([(2.0, 3.0), (3.0, 4.0)], [(3.0, 4.0), (2.0, 3.0)], [(2.0, 3.0), (3.2, 4.0)], [(2.0, 3.5), (2.5, 4.0)]):
        r = _restore_original_ranges(ranges, d / "voc.wav", total, sr, 1, pad_ms=150)
        a = np.abs(r["audio"][:, 0])
        assert a[int(2.8 * sr):int(3.3 * sr)].max() <= 0.5 + 1e-3, (ranges, a[int(2.8 * sr):int(3.3 * sr)].max())
        assert a[int(2.9 * sr):int(3.2 * sr)].max() >= 0.45, ranges          # no fade dip inside the merged span
        assert r["mask"][int(1.9 * sr)] == 1 and r["mask"][int(4.1 * sr)] == 1 and r["mask"][int(1.8 * sr)] == 0
    # ranges further apart than 2 × pad stay separate, with silence between them
    r = _restore_original_ranges([(2.0, 3.0), (3.5, 4.0)], d / "voc.wav", total, sr, 1, pad_ms=150)
    assert r["mask"][int(3.25 * sr)] == 0 and np.abs(r["audio"][int(3.2 * sr):int(3.3 * sr)]).max() == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
