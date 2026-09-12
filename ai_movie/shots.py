"""Explicit shot-cut detection (ffmpeg ``scdet``), cached per video.

Until v3 the only shot-boundary awareness was a geometric heuristic ("the
face box jumped more than 0.6× its size") duplicated in ``faces`` and
``face_restore``.  That misses a cut between two similar framings and
fires on fast head motion.  ``scdet`` scores every frame's difference
against its predecessor; frames above ``threshold`` are cuts.  Track
interpolation, box resolution and the occlusion gate all refuse to bridge
one, and the QC report counts cuts inside each segment.
"""

from __future__ import annotations

import bisect
import json
import re
import subprocess
from pathlib import Path

_TIME_RE = re.compile(r"lavfi\.scd\.time:\s*([0-9.]+)")
_SCORE_RE = re.compile(r"lavfi\.scd\.score:\s*([0-9.]+)")


def _first_pts(video: Path) -> float:
    """pts of the first decoded frame (scdet reports absolute pts_time)."""
    try:
        out = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=pts_time", "-read_intervals", "%+#1",
            "-of", "csv=p=0", str(video),
        ], capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out.splitlines()[0].split(",")[0])
    except Exception:                                   # noqa: BLE001
        return 0.0


def _fps(video: Path) -> float:
    try:
        out = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video),
        ], capture_output=True, text=True, timeout=60).stdout.strip()
        num, den = out.split("/")
        return float(num) / float(den)
    except Exception:                                   # noqa: BLE001
        return 25.0


def _cache_key(video: Path, threshold: float) -> str:
    st = video.stat()
    return f"{video.resolve()}|{st.st_size}|{st.st_mtime_ns}|scdet={threshold}"


def detect_cuts(video: str | Path, *, threshold: float | None = None,
                cache: str | Path | None = None) -> list[int]:
    """Frame indices at which a new shot starts (sorted, unique).

    A cut at index ``c`` means frame ``c`` is the first frame of the new
    shot, so interpolation between frames ``a < c <= b`` must be refused.
    """
    from ai_movie.config import SHOT_SCDET_THRESHOLD
    video = Path(video)
    threshold = SHOT_SCDET_THRESHOLD if threshold is None else float(threshold)
    key = _cache_key(video, threshold)
    cache_p = Path(cache) if cache else None
    if cache_p and cache_p.exists():
        try:
            d = json.loads(cache_p.read_text(encoding="utf-8"))
            if d.get("key") == key:
                return [int(c) for c in d.get("cuts", [])]
        except Exception:                               # noqa: BLE001
            pass

    fps = _fps(video)
    t0 = _first_pts(video)
    res = subprocess.run([
        "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "verbose",
        "-i", str(video), "-an", "-vf", f"scdet=threshold={threshold}",
        "-f", "null", "-",
    ], capture_output=True, text=True, timeout=3600)
    cuts: set[int] = set()
    for line in res.stderr.splitlines():
        m = _TIME_RE.search(line)
        if not m:
            continue
        t = float(m.group(1))
        idx = int(round((t - t0) * fps))
        if idx > 0:
            cuts.add(idx)
    out = sorted(cuts)
    if cache_p:
        try:
            cache_p.parent.mkdir(parents=True, exist_ok=True)
            cache_p.write_text(json.dumps({"key": key, "threshold": threshold,
                                           "fps": fps, "t0": t0, "cuts": out}),
                               encoding="utf-8")
        except Exception:                               # noqa: BLE001
            pass
    return out


def shot_id_for(frame: int, cuts: list[int]) -> int:
    """0 for frames before the first cut, k for frames after the k-th cut."""
    return bisect.bisect_right(cuts, frame)


def crosses_cut(a: int, b: int, cuts) -> bool:
    """True if a cut lies in ``(a, b]`` — i.e. a and b are in different shots."""
    if not cuts:
        return False
    if isinstance(cuts, (set, frozenset)):
        return any(a < c <= b for c in cuts)
    i = bisect.bisect_right(cuts, a)
    return i < len(cuts) and cuts[i] <= b


def local_cuts(cuts, base: int, n: int) -> set[int]:
    """Cuts re-based to a clip that starts at global frame *base* (n frames)."""
    return {int(c) - base for c in (cuts or []) if base < int(c) < base + n}
