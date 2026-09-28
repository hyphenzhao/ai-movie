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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
