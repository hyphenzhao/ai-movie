"""Overlapped-speech detection (OSD) — pyannote/segmentation-3.0 on CPU.

Why: the remaining diarization errors on the reference film are not
misclassified voices but *two people in one window* (one talks, the other
laughs or interjects).  Pitch, channel features and ECAPA all measure a
mixture there.  Knowing where the overlap is lets the pipeline (a) keep
those units out of the gender classifier's seed set, (b) refuse them as
timbre references, and (c) flag the segments for review instead of
pretending they are clean.

The model is gated on HuggingFace and pyannote drags in a dependency tree
we do not want in the ROCm venv, so inference runs in an isolated CPU venv
(``vendor/osd_venv``, built by ``scripts/setup_osd_env.sh``) through
``ai_movie/osd_worker.py`` — the same pattern as ``tts_worker``.  The token
is passed through the environment only and never logged.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_WORKER = Path(__file__).resolve().parent / "osd_worker.py"
_avail_cache: bool | None = None


def hf_token() -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(var)
        if v:
            return v.strip()
    for p in (Path.home() / ".cache" / "huggingface" / "token",
              _ROOT / "asset" / "hf_token"):
        try:
            if p.exists():
                t = p.read_text(encoding="utf-8").strip()
                if t:
                    return t
        except Exception:                               # noqa: BLE001
            pass
    return None


def worker_python() -> Path | None:
    from ai_movie.config import OSD_VENV
    p = Path(OSD_VENV) / "bin" / "python"
    return p if p.exists() else None


def available() -> bool:
    """Isolated venv present and pyannote importable there (cached)."""
    global _avail_cache
    if _avail_cache is not None:
        return _avail_cache
    py = worker_python()
    if py is None:
        _avail_cache = False
        return False
    try:
        r = subprocess.run([str(py), "-c", "import pyannote.audio"],
                           capture_output=True, text=True, timeout=120)
        _avail_cache = r.returncode == 0
    except Exception:                                   # noqa: BLE001
        _avail_cache = False
    return _avail_cache


def overlap_ratio(regions, a: float, b: float) -> float:
    """Fraction of ``[a, b]`` covered by overlap regions."""
    if b <= a or not regions:
        return 0.0
    cov = 0.0
    for s, e in regions:
        cov += max(0.0, min(b, float(e)) - max(a, float(s)))
    return min(1.0, cov / (b - a))


def unit_overlap(regions, units) -> list[float]:
    return [overlap_ratio(regions, float(s), float(e)) for s, e in units]


def run_osd(audio_path: str | Path, out_json: str | Path | None = None, *,
            token: str | None = None, model: str | None = None,
            device: str | None = None, timeout: int = 3600,
            log_cb=None) -> dict:
    """Run OSD on a 16 kHz mono wav; returns the overlap document.

    ``{"available": bool, "regions": [[start, end], …], "total_overlap_s",
    "model", "reason"}``.  Never raises for a missing environment or token —
    the pipeline degrades to "no overlap information" and QC says so.
    """
    from ai_movie.config import OSD_DEVICE, OSD_MODEL
    model = model or OSD_MODEL
    device = device or OSD_DEVICE
    doc = {"available": False, "regions": [], "total_overlap_s": 0.0,
           "model": model, "reason": ""}
    if not available():
        doc["reason"] = "osd venv missing (run scripts/setup_osd_env.sh)"
        return _finish(doc, out_json)
    token = token or hf_token()
    if not token:
        doc["reason"] = "no HuggingFace token (HF_TOKEN / ~/.cache/huggingface/token)"
        return _finish(doc, out_json)

    job = {"audio": str(Path(audio_path).resolve()), "model": model, "device": device}
    env = {**os.environ, "HF_TOKEN": token, "PYTHONUNBUFFERED": "1",
           "OMP_NUM_THREADS": str(max(1, (os.cpu_count() or 8) // 2))}
    py = worker_python()
    try:
        r = subprocess.run([str(py), str(_WORKER)], input=json.dumps(job),
                           capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        doc["reason"] = f"worker timed out after {timeout}s"
        return _finish(doc, out_json)
    line = ""
    for ln in reversed(r.stdout.splitlines()):
        if ln.startswith("{"):
            line = ln
            break
    if r.returncode != 0 or not line:
        tail = (r.stderr or "")[-600:]
        doc["reason"] = f"worker failed (rc={r.returncode}): {tail}"
        if log_cb:
            log_cb(f"[osd] {doc['reason']}")
        return _finish(doc, out_json)
    res = json.loads(line)
    if res.get("error"):
        doc["reason"] = str(res["error"])
        return _finish(doc, out_json)
    regions = [[round(float(s), 3), round(float(e), 3)] for s, e in res.get("regions", [])]
    doc.update({"available": True, "regions": regions,
                "total_overlap_s": round(sum(e - s for s, e in regions), 2),
                "model": res.get("model", model), "reason": "",
                "worker": {k: v for k, v in res.items() if k not in ("regions",)}})
    return _finish(doc, out_json)


def _finish(doc: dict, out_json) -> dict:
    if out_json:
        p = Path(out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        doc["json"] = str(p)
    return doc
