"""Pitch measurement and the F0-ratio gate for reference / clone selection.

Why this exists (Documentation/vc-gate-result.md): the first shipped female
reference made every converted line come out an octave low (232 Hz → 117 Hz,
reads as male) while scoring a *higher* ECAPA similarity than references
that worked.  ECAPA is largely pitch-invariant, so similarity alone cannot
see octave collapse.  The only measure that did was the ratio of the
output's median F0 to the reference's — 0.97–1.01 for good references,
0.49–0.60 for the collapsed one.

The pyin parameters here are the ones the finding was established with
(``scripts/auto_select_refs.py``), deliberately *not* ``diarize.pitch_track``'s
(2048-sample frames, 0.25 voiced probability), so numbers stay comparable
with the gate document.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_SR = 16000


def f0_median(audio: "np.ndarray | str | Path", *, sr: int = _SR,
              min_frames: int = 6) -> tuple[float | None, int]:
    """Median F0 (Hz) of confidently-voiced frames and their count.

    Accepts a path (loaded as 16 kHz mono) or a 16 kHz mono array.
    Returns ``(None, n)`` when fewer than *min_frames* frames are voiced.
    """
    import librosa
    if isinstance(audio, (str, Path)):
        from ai_movie.diarize import _load_mono16k
        a = _load_mono16k(audio)
        sr = _SR
    else:
        a = np.asarray(audio, dtype=np.float32)
    if a.size < 1024:
        return None, 0
    f0, vf, vp = librosa.pyin(a, fmin=60, fmax=500, sr=sr,
                              frame_length=1024, hop_length=256)
    ok = vf & (vp > 0.5) & ~np.isnan(f0)
    n = int(ok.sum())
    if n < min_frames:
        return None, n
    return float(np.median(f0[ok])), n


def gender_band(gender: str | None) -> tuple[float, float] | None:
    from ai_movie.config import TTS_GENDER_HZ
    if not gender:
        return None
    return TTS_GENDER_HZ.get(gender)


def gate(ref_f0: float | None, out_f0: float | None, gender: str | None,
         *, ratio_range: tuple[float, float] | None = None) -> dict:
    """Decide whether a probe output is pitch-consistent with its reference.

    ``ok`` requires the output F0 to be measurable, inside the gender's
    band when the gender is known, and — when the reference F0 is
    measurable — an output/reference ratio inside *ratio_range*
    (default ``TTS_F0_RATIO_RANGE``: 0.8–1.25, i.e. no octave jump).
    """
    from ai_movie.config import TTS_F0_RATIO_RANGE
    lo_r, hi_r = ratio_range or TTS_F0_RATIO_RANGE
    res = {"ok": False, "ratio": None, "in_band": None, "reason": ""}
    if out_f0 is None:
        res["reason"] = "output_f0_unmeasurable"
        return res
    band = gender_band(gender)
    if band is not None:
        res["in_band"] = bool(band[0] <= out_f0 <= band[1])
        if not res["in_band"]:
            res["reason"] = f"output_{out_f0:.0f}Hz_outside_{gender}_band"
            return res
    if ref_f0:
        ratio = out_f0 / ref_f0
        res["ratio"] = round(ratio, 3)
        if not (lo_r <= ratio <= hi_r):
            res["reason"] = f"ratio_{ratio:.2f}_outside_{lo_r}-{hi_r}"
            return res
    res["ok"] = True
    return res
