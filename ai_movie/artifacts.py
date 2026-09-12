"""Per-step deliverable export (SRT / CSV / demo audio / reports).

The GUI keeps everything inside one ``.aimovie.json`` project file, which is
fine for resuming work but useless for *reviewing* a run.  These helpers
write human-checkable artifacts next to the workspace so each pipeline stage
can be verified on its own: subtitles, a speaker table, a few seconds of each
detected speaker's voice, translation comparisons, and CSV reports.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ai_movie.utils import ensure_dir


# ── time formatting ────────────────────────────────────────────────

def srt_timestamp(seconds: float) -> str:
    """Format *seconds* as ``HH:MM:SS,mmm``."""
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ── subtitles ──────────────────────────────────────────────────────

def export_srt(
    segments: list[dict],
    path: str | Path,
    *,
    text_key: str = "text",
    speaker_prefix: bool = False,
) -> Path:
    """Write *segments* as an SRT file.

    ``speaker_prefix`` prepends ``[S0♀] `` so a reviewer can see the
    diarization result directly in a subtitle player.
    """
    path = Path(path)
    ensure_dir(path.parent)
    lines: list[str] = []
    n = 0
    for seg in segments:
        text = (seg.get(text_key) or "").strip()
        if not text:
            continue
        n += 1
        if speaker_prefix and seg.get("speaker"):
            mark = {"female": "♀", "male": "♂"}.get(
                seg.get("gender") or seg.get("tts_gender") or "", "?")
            text = f"[{seg['speaker']}{mark}] {text}"
        lines.append(str(n))
        lines.append(f"{srt_timestamp(float(seg.get('start', 0.0)))} --> "
                     f"{srt_timestamp(float(seg.get('end', 0.0)))}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ── tabular reports ────────────────────────────────────────────────

def export_csv(rows: list[dict], path: str | Path,
               fields: list[str] | None = None) -> Path:
    """Write *rows* as UTF-8-BOM CSV (BOM so Excel opens CJK correctly)."""
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return path
    if fields is None:
        fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def export_speaker_csv(segments: list[dict], path: str | Path) -> Path:
    """One row per segment: timing, speaker, gender, confidence, text."""
    rows = []
    for i, seg in enumerate(segments):
        rows.append({
            "idx": i,
            "start": round(float(seg.get("start", 0.0)), 2),
            "end": round(float(seg.get("end", 0.0)), 2),
            "dur": round(float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)), 2),
            "speaker": seg.get("speaker", ""),
            "gender": seg.get("gender") or seg.get("tts_gender") or "",
            "spk_conf": round(float(seg.get("speaker_conf", 0.0)), 2),
            "asr_conf": round(float(seg.get("asr_conf", 0.0)), 2),
            "overlap": seg.get("overlap", ""),
            "text": (seg.get("text") or "").strip(),
            "text_translated": (seg.get("text_translated") or "").strip(),
        })
    return export_csv(rows, path)


# ── audio demos ────────────────────────────────────────────────────

def export_speaker_demo_wavs(
    segments: list[dict],
    audio_path: str | Path,
    out_dir: str | Path,
    *,
    max_seconds: float = 25.0,
    gap_seconds: float = 0.25,
    prefix: str = "spk",
) -> dict[str, str]:
    """Concatenate a few seconds of each speaker into one demo WAV each.

    Lets a human confirm the male/female labelling in half a minute instead
    of scrubbing the whole video.  Returns ``{speaker: wav_path}``.
    """
    import numpy as np
    import soundfile as sf

    out_dir = ensure_dir(Path(out_dir))
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    by_spk: dict[str, list[dict]] = {}
    for seg in segments:
        spk = seg.get("speaker") or ""
        if spk:
            by_spk.setdefault(spk, []).append(seg)

    gap = np.zeros(int(sr * gap_seconds), dtype=np.float32)
    written: dict[str, str] = {}
    for spk, segs in sorted(by_spk.items()):
        # Longest segments first — they carry the clearest voice evidence.
        segs = sorted(segs, key=lambda s: float(s.get("end", 0)) - float(s.get("start", 0)),
                      reverse=True)
        chunks: list[np.ndarray] = []
        total = 0.0
        for seg in segs:
            if total >= max_seconds:
                break
            a = int(float(seg.get("start", 0.0)) * sr)
            b = int(float(seg.get("end", 0.0)) * sr)
            a, b = max(0, a), min(len(audio), b)
            if b - a < int(sr * 0.3):
                continue
            chunks.append(audio[a:b])
            chunks.append(gap)
            total += (b - a) / sr
        if not chunks:
            continue
        gender = ""
        for seg in segs:
            gender = seg.get("gender") or seg.get("tts_gender") or ""
            if gender:
                break
        dst = out_dir / f"{prefix}_{spk}_{gender or 'unknown'}_demo.wav"
        sf.write(str(dst), np.concatenate(chunks), sr)
        written[spk] = str(dst)
    return written


def export_ab_wav(
    a_path: str | Path,
    b_path: str | Path,
    dst: str | Path,
    *,
    gap_seconds: float = 0.5,
) -> Path:
    """Concatenate two clips with a gap — "original then clone" A/B demo."""
    import numpy as np
    import soundfile as sf
    import librosa

    dst = Path(dst)
    ensure_dir(dst.parent)
    a, sr_a = sf.read(str(a_path), dtype="float32")
    b, sr_b = sf.read(str(b_path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if b.ndim > 1:
        b = b.mean(axis=1)
    sr = max(sr_a, sr_b)
    if sr_a != sr:
        a = librosa.resample(a, orig_sr=sr_a, target_sr=sr)
    if sr_b != sr:
        b = librosa.resample(b, orig_sr=sr_b, target_sr=sr)
    gap = np.zeros(int(sr * gap_seconds), dtype=np.float32)
    sf.write(str(dst), np.concatenate([a, gap, b]), sr)
    return dst


# ── translation comparison ─────────────────────────────────────────

def export_translation_compare(
    segments: list[dict],
    variants: dict[str, list[str]],
    path: str | Path,
    *,
    title: str = "翻译方案对照",
) -> Path:
    """Markdown table: source line vs every engine's rendering."""
    path = Path(path)
    ensure_dir(path.parent)
    names = list(variants.keys())

    out = [f"# {title}", ""]
    out.append(f"共 {len(segments)} 句，对照 {len(names)} 条路线：" + "、".join(names))
    out.append("")
    for i, seg in enumerate(segments):
        spk = seg.get("speaker", "")
        gender = seg.get("gender") or seg.get("tts_gender") or ""
        mark = {"female": "♀", "male": "♂"}.get(gender, "")
        head = (f"### [{i}] {float(seg.get('start', 0)):.2f}–"
                f"{float(seg.get('end', 0)):.2f}s")
        if spk:
            head += f"  `{spk}{mark}`"
        out.append(head)
        out.append("")
        out.append(f"- **原文**：{(seg.get('text') or '').strip()}")
        for name in names:
            vals = variants[name]
            val = vals[i] if i < len(vals) else ""
            out.append(f"- **{name}**：{val}")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def export_json(data, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path
