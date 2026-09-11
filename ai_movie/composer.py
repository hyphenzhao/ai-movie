"""Audio processing: voice-background separation and mixing.

Supports two backends:
- ``"demucs"`` — htdemucs on GPU (reliable, default)
- ``"uvr"``    — Mel-Band RoiFormer via audio-separator (higher quality,
                 requires network for first-time model download)
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch

from ai_movie.utils import ensure_dir

# ── UVR singleton ───────────────────────────────────────────────────
_uvr_separator = None
_UVR_VOCALS_MODEL = "vocals_mel_band_roformer.ckpt"


def separate_vocals(
    audio_path: Path,
    output_dir: Path | None = None,
    backend: str = "demucs",
) -> dict:
    """Separate a mixed audio file into vocals and background.

    Parameters
    ----------
    audio_path:
        Path to input audio file (any format FFmpeg can read).
    output_dir:
        Where to write vocals.wav and background.wav.
    backend:
        ``"demucs"`` — htdemucs on GPU (default, reliable).
        ``"uvr"``    — Mel-Band RoiFormer on GPU (needs model download).

    Returns
    -------
    dict with keys ``vocals``, ``background`` (both Path).
    """
    if output_dir is None:
        output_dir = audio_path.parent
    output_dir = ensure_dir(output_dir)

    vocals_path = output_dir / "vocals.wav"
    background_path = output_dir / "background.wav"

    # Only use cache if files have actual audio content AND match input
    cache_valid = False
    if vocals_path.exists() and background_path.exists():
        try:
            v_data, _ = sf.read(str(vocals_path), frames=1000, dtype="float64")
            b_data, _ = sf.read(str(background_path), frames=1000, dtype="float64")
            if np.any(np.abs(v_data) > 1e-8) and np.any(np.abs(b_data) > 1e-8):
                # Verify cache is for THIS audio file (check duration match)
                import soundfile as _sf
                src_info = _sf.info(str(audio_path))
                cache_info = _sf.info(str(vocals_path))
                if abs(src_info.duration - cache_info.duration) < 0.5:
                    cache_valid = True
                    return {"vocals": vocals_path, "background": background_path}
                else:
                    print(f"[composer] Cache mismatch: src={src_info.duration:.1f}s "
                          f"cache={cache_info.duration:.1f}s — re-separating",
                          file=sys.stderr)
            vocals_path.unlink(missing_ok=True)
            background_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Dispatch
    if backend == "uvr":
        try:
            return _separate_uvr(audio_path, output_dir, vocals_path, background_path)
        except Exception as exc:
            print(f"[composer] UVR failed ({exc}), falling back to Demucs…",
                  file=sys.stderr)
            # fall through to Demucs
    return _separate_demucs(audio_path, output_dir, vocals_path, background_path)


# ── Demucs backend (htdemucs, GPU) ─────────────────────────────────

def _separate_demucs(
    audio_path: Path,
    output_dir: Path,
    vocals_path: Path,
    background_path: Path,
) -> dict:
    """Run Demucs htdemucs separation on GPU (or CPU fallback)."""
    from demucs.pretrained import get_model
    from demucs.separate import apply_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("[composer] Demucs running on GPU (ROCm)", file=sys.stderr)

    model = get_model("htdemucs")
    model.to(device).eval()

    audio_np, sr = sf.read(str(audio_path))
    if audio_np.ndim == 1:
        audio_np = np.stack([audio_np, audio_np], axis=1)
    elif audio_np.ndim == 2 and audio_np.shape[1] == 1:
        audio_np = np.tile(audio_np, (1, 2))
    audio_tensor = torch.from_numpy(audio_np.T).unsqueeze(0).float()

    with torch.no_grad():
        sources = apply_model(
            model, audio_tensor, device=device,
            split=True, overlap=0.25, progress=True,
        )
    vocals_np = sources[0, 3].numpy().T
    background_np = sources[0, 0:3].sum(dim=0).numpy().T

    sf.write(str(vocals_path), vocals_np, sr)
    sf.write(str(background_path), background_np, sr)

    return {"vocals": vocals_path, "background": background_path}


# ── UVR backend (Mel-Band RoiFormer, GPU) ──────────────────────────

def _separate_uvr(
    audio_path: Path,
    output_dir: Path,
    vocals_path: Path,
    background_path: Path,
) -> dict:
    """Run UVR Mel-Band RoiFormer separation (GPU via audio-separator)."""
    global _uvr_separator

    from audio_separator.separator import Separator

    if _uvr_separator is None:
        from ai_movie.config import UVR_MODEL_FILE_DIR
        os.makedirs(UVR_MODEL_FILE_DIR, exist_ok=True)
        _uvr_separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=UVR_MODEL_FILE_DIR,
            output_dir=str(output_dir),
            output_format="WAV",
        )
        try:
            _uvr_separator.load_model(_UVR_VOCALS_MODEL)
        except Exception:
            _uvr_separator = None
            raise

    tmp_out = ensure_dir(output_dir / ".uvr_tmp")
    _uvr_separator.output_dir = str(tmp_out)

    output_files = _uvr_separator.separate(str(audio_path))

    vocals_src = None
    bg_src = None
    for f in output_files:
        fpath = Path(f)
        fname = fpath.name.lower()
        if "(vocals)" in fname or "vocals" in fname:
            vocals_src = fpath
        elif "(instrumental)" in fname or "no_vocals" in fname or "instrumental" in fname:
            bg_src = fpath

    if vocals_src is None or bg_src is None:
        output_files.sort(key=lambda f: Path(f).stat().st_size)
        if len(output_files) >= 2:
            vocals_src = Path(output_files[0])
            bg_src = Path(output_files[1])

    if vocals_src is None or bg_src is None:
        raise RuntimeError(
            f"UVR produced unexpected output: {output_files}")

    shutil.move(str(vocals_src), str(vocals_path))
    shutil.move(str(bg_src), str(background_path))
    shutil.rmtree(str(tmp_out), ignore_errors=True)

    return {"vocals": vocals_path, "background": background_path}


# ── duration fitting (TTS → timeline slot) ─────────────────────────
#
# Mandarin renders most Japanese lines longer than the original, and until
# now nothing re-timed the result: mix_audio simply stamped each wav at
# seg["start"] and summed it, so an overlong line bled into — and additively
# mixed with — the next one.  That wrecks both the dub and the lip-sync,
# which is driven by the same track.

def _has_rubberband() -> bool:
    """Whether this ffmpeg build exposes the rubberband filter."""
    global _RUBBERBAND
    if _RUBBERBAND is None:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                                 capture_output=True, text=True, timeout=30)
            _RUBBERBAND = " rubberband " in out.stdout
        except Exception:                               # noqa: BLE001
            _RUBBERBAND = False
    return _RUBBERBAND


_RUBBERBAND: bool | None = None


def _atempo_chain(ratio: float) -> str:
    """ffmpeg atempo only accepts 0.5–2.0 per instance; chain for the rest."""
    parts, r = [], ratio
    while r > 2.0:
        parts.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        parts.append("atempo=0.5")
        r /= 0.5
    parts.append(f"atempo={r:.6f}")
    return ",".join(parts)


def stretch_audio(src: Path, dst: Path, ratio: float,
                  backend: str = "rubberband") -> Path:
    """Time-stretch *src* by *ratio* (>1 = faster) preserving pitch."""
    if abs(ratio - 1.0) < 1e-3:
        if src != dst:
            shutil.copy2(str(src), str(dst))
        return dst

    if backend == "rubberband" and _has_rubberband():
        af = f"rubberband=tempo={ratio:.6f}:pitchq=quality"
    else:
        af = _atempo_chain(ratio)

    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-af", af, "-c:a", "pcm_s16le", str(dst),
    ], check=True, capture_output=True)
    return dst


def fit_audio_to_slot(
    src_wav: Path,
    target_dur: float,
    dst_wav: Path,
    *,
    max_speedup: float | None = None,
    min_speedup: float | None = None,
    tail_tolerance: float | None = None,
    backend: str | None = None,
) -> tuple[Path, float]:
    """Fit one synthesized clip into ``target_dur`` seconds.

    The policy is deliberately asymmetric.  Over-compression is the audible
    failure mode — above ~1.3x Mandarin sounds rushed and the lip-sync mouth
    turns mushy — so we cap the speed-up and let the remainder spill into the
    following gap instead.  We never slow speech *down*; a short clip simply
    leaves silence, which sounds natural.

    Returns ``(path, applied_ratio)``.
    """
    from ai_movie.config import (
        TTS_FIT_BACKEND, TTS_FIT_MAX_SPEEDUP, TTS_FIT_MIN_SPEEDUP,
        TTS_FIT_TAIL_TOLERANCE,
    )

    max_speedup = TTS_FIT_MAX_SPEEDUP if max_speedup is None else max_speedup
    min_speedup = TTS_FIT_MIN_SPEEDUP if min_speedup is None else min_speedup
    tail_tolerance = (TTS_FIT_TAIL_TOLERANCE if tail_tolerance is None
                      else tail_tolerance)
    backend = backend or TTS_FIT_BACKEND

    info = sf.info(str(src_wav))
    cur = float(info.duration)
    if cur <= 0 or target_dur <= 0:
        if src_wav != dst_wav:
            shutil.copy2(str(src_wav), str(dst_wav))
        return dst_wav, 1.0

    ratio = cur / target_dur
    if ratio <= 1.0 + tail_tolerance:
        # Fits, or overruns by a tolerable tail — leave it alone.
        if src_wav != dst_wav:
            shutil.copy2(str(src_wav), str(dst_wav))
        return dst_wav, 1.0

    applied = min(ratio, max_speedup)
    applied = max(applied, min_speedup)
    stretch_audio(Path(src_wav), Path(dst_wav), applied, backend=backend)
    return dst_wav, applied


def _visible_chars(text: str) -> int:
    """Characters that take time to say (drops spaces and punctuation)."""
    return sum(1 for ch in (text or "")
               if not ch.isspace() and ch not in "。、，,．.！!？?…‥・「」『』（）()～~-—")


def estimate_rate_correction(
    segments: list[dict],
    *,
    text_key: str = "text_translated",
) -> dict[str, float]:
    """Per-speaker tempo factor that brings cloned speech to a natural rate.

    Returns ``{speaker: factor}`` where a factor above 1 means "speed this
    speaker up by this much".  Computed from the median seconds-per-character
    of that speaker's synthesized audio, so one odd segment cannot skew it.
    """
    import soundfile as _sf
    from ai_movie.config import (
        TTS_NATURAL_SEC_PER_CHAR, TTS_RATE_MAX_CORRECTION,
        TTS_RATE_MIN_CORRECTION,
    )

    by_spk: dict[str, list[float]] = {}
    for seg in segments:
        path = seg.get("audio")
        n = _visible_chars(seg.get(text_key) or "")
        if not path or not Path(path).exists() or n < 4:
            continue
        try:
            dur = float(_sf.info(str(path)).duration)
        except Exception:                               # noqa: BLE001
            continue
        if dur <= 0:
            continue
        by_spk.setdefault(seg.get("speaker") or "", []).append(dur / n)

    out: dict[str, float] = {}
    for spk, rates in by_spk.items():
        med = float(np.median(rates))
        factor = med / TTS_NATURAL_SEC_PER_CHAR
        if factor < TTS_RATE_MIN_CORRECTION:
            continue
        out[spk] = round(min(factor, TTS_RATE_MAX_CORRECTION), 3)
    return out


def fit_segments_to_timeline(
    segments: list[dict],
    *,
    out_dir: Path | str,
    min_gap: float | None = None,
    max_tail: float | None = None,
    max_speedup: float | None = None,
    rate_correct: bool = True,
    progress_cb: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Fit every synthesized segment into the gap before the next one.

    Sets ``audio_fit`` (the wav actually mixed), ``fit_ratio`` and
    ``fit_end`` on each segment, in place, and returns the list.
    """
    from ai_movie.config import (
        TTS_FIT_MAX_SPEEDUP, TTS_FIT_MAX_SPEEDUP_HARD, TTS_FIT_MAX_TAIL,
        TTS_FIT_MAX_TRUNCATE, TTS_FIT_MIN_GAP, TTS_FIT_TAIL_TOLERANCE,
    )

    min_gap = TTS_FIT_MIN_GAP if min_gap is None else min_gap
    max_tail = TTS_FIT_MAX_TAIL if max_tail is None else max_tail
    max_speedup = TTS_FIT_MAX_SPEEDUP if max_speedup is None else max_speedup
    hard_speedup = TTS_FIT_MAX_SPEEDUP_HARD
    max_truncate = TTS_FIT_MAX_TRUNCATE
    tail_tol = TTS_FIT_TAIL_TOLERANCE

    out_dir = ensure_dir(Path(out_dir))
    ordered = sorted(range(len(segments)),
                     key=lambda i: float(segments[i].get("start", 0.0)))

    # Correct a systematically slow clone before fitting individual slots.
    rates = estimate_rate_correction(segments) if rate_correct else {}
    if rates:
        print(f"[composer] speaking-rate correction per speaker: {rates}",
              file=sys.stderr)

    for n, i in enumerate(ordered):
        seg = segments[i]
        src = seg.get("audio")
        seg.pop("audio_fit", None)
        seg.pop("fit_ratio", None)
        seg.pop("rate_factor", None)
        if not src or not Path(src).exists():
            continue
        base = rates.get(seg.get("speaker") or "", 1.0)

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        # The slot may extend into the following silence, but must stop
        # short of the next segment.
        limit = end + max_tail
        if n + 1 < len(ordered):
            nxt = segments[ordered[n + 1]]
            limit = min(limit, float(nxt.get("start", limit)) - min_gap)
        slot = max(0.2, limit - start)

        dst = out_dir / (Path(src).stem + ".fit.wav")
        try:
            work = Path(src)
            if base > 1.001:
                rated = out_dir / (Path(src).stem + ".rate.wav")
                stretch_audio(work, rated, base)
                work = rated

            # The per-segment cap applies *on top of* the rate correction.
            # Escalate it only when the normal cap would cut real words off
            # the end: the rate correction is a median, so a speaker's slow
            # outliers still overrun, and truncating them loses content.
            cap = max_speedup
            try:
                cur = float(sf.info(str(work)).duration)
                if cur > slot * (1.0 + tail_tol):
                    needed = cur / max(slot, 1e-6)
                    would_cut = cur / min(needed, max_speedup) - slot
                    if would_cut > max_truncate:
                        cap = min(max(needed, max_speedup), hard_speedup)
            except Exception:                           # noqa: BLE001
                pass

            path, ratio = fit_audio_to_slot(work, slot, dst,
                                            max_speedup=cap)
        except Exception as exc:                        # noqa: BLE001
            print(f"[composer] fit failed for {src}: {exc}", file=sys.stderr)
            continue

        seg["audio_fit"] = str(path)
        seg["rate_factor"] = round(base, 3)
        seg["fit_ratio"] = round(base * ratio, 3)
        try:
            dur = float(sf.info(str(path)).duration)
            natural_end = start + dur
            # Even at the maximum speed-up some lines cannot fit a very short
            # slot (a 0.6 s slot cannot hold 9 Chinese characters).  The mixer
            # truncates those at the next segment's start with a fade, so
            # report the *placed* end plus how much had to be cut, rather than
            # a fit_end that implies an overlap which will not actually occur.
            hard_limit = float("inf")
            if n + 1 < len(ordered):
                nxt = segments[ordered[n + 1]]
                hard_limit = max(float(seg.get("end", 0.0)),
                                 float(nxt.get("start", float("inf"))))
            seg["fit_end"] = round(min(natural_end, hard_limit), 2)
            over = max(0.0, natural_end - hard_limit)
            if over > 0.01:
                seg["overrun"] = round(over, 2)
        except Exception:                               # noqa: BLE001
            pass
        if progress_cb:
            progress_cb(n + 1, len(ordered))

    return segments


def _seg_audio(seg: dict) -> str | None:
    """The wav to actually place on the timeline (fitted version wins)."""
    for key in ("audio_fit", "audio"):
        p = seg.get(key)
        if p and Path(p).exists():
            return p
    return None


def _placements(segments: list[dict]) -> list[tuple[dict, float]]:
    """Pair each segment with the time it must stop by.

    The limit is the next segment's start (never trimming below the
    segment's own ``end``, so a naturally short slot can't clip speech that
    fits).  Returned in timeline order.
    """
    order = sorted(range(len(segments)),
                   key=lambda i: float(segments[i].get("start", 0.0)))
    out: list[tuple[dict, float]] = []
    for n, i in enumerate(order):
        seg = segments[i]
        limit = float("inf")
        if n + 1 < len(order):
            nxt = segments[order[n + 1]]
            limit = max(float(seg.get("end", 0.0)),
                        float(nxt.get("start", float("inf"))))
        out.append((seg, limit))
    return out


def mix_audio(
    segments: list[dict],
    background_path: Path,
    output_path: Path,
    speech_gain: float = 0.85,
    bg_gain_speech: float = 0.25,
    bg_gain_silence: float = 1.0,
    fade_ms: int = 20,
) -> Path:
    """Mix TTS speech segments into background audio at their original timestamps.

    Each segment is placed at ``seg['start']`` seconds in the timeline so
    the dubbed speech stays in sync with the original video.  Background
    audio is ducked (reduced) wherever speech is present.

    Parameters
    ----------
    segments:
        List of segment dicts with keys ``audio``, ``start``, ``end``.
        Segments without an ``audio`` value are skipped (silence retained).
    background_path:
        Demucs-separated background (no vocals) WAV.
    output_path:
        Destination WAV path.
    speech_gain:
        Peak volume for the speech track (0-1).
    bg_gain_speech:
        Background volume where speech is active.
    bg_gain_silence:
        Background volume where there is no speech.
    fade_ms:
        Fade-in/out duration in milliseconds to avoid clicks.
    """
    import librosa as _librosa

    # Load background (authoritative sample-rate and length)
    bg, sr = sf.read(str(background_path))
    if bg.ndim > 1:
        bg = bg.mean(axis=1)
    bg = bg.astype(np.float32)

    # Total output length: at least as long as the background
    last_end = max((max(float(seg.get("end", 0) or 0),
                        float(seg.get("fit_end", 0) or 0))
                    for seg in segments), default=0)
    total = max(len(bg), int(last_end * sr) + sr)   # +1 s buffer

    speech_track = np.zeros(total, dtype=np.float32)
    speech_mask  = np.zeros(total, dtype=np.float32)

    for seg, limit in _placements(segments):
        audio_path = _seg_audio(seg)
        if not audio_path:
            continue
        wav, wav_sr = sf.read(str(audio_path))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        if wav_sr != sr:
            wav = _librosa.resample(wav, orig_sr=wav_sr, target_sr=sr)

        start_s = max(0, int(seg.get("start", 0) * sr))
        # Hard stop before the next segment: overrunning speech used to be
        # summed on top of the following line.  The last segment has no
        # successor, so its limit is +inf — clamp before converting.
        limit_s = total if limit == float("inf") else int(limit * sr)
        end_s   = min(total, start_s + len(wav), limit_s)
        wav     = wav[: max(0, end_s - start_s)]
        if len(wav) == 0:
            continue

        # Short fade-in / fade-out to suppress clicks
        fade = min(int(fade_ms * sr / 1000), max(1, len(wav) // 4))
        wav[:fade]  *= np.linspace(0, 1, fade, dtype=np.float32)
        wav[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)

        speech_track[start_s:end_s] += wav
        speech_mask[start_s:end_s]   = 1.0

    # Normalize speech track
    peak = np.abs(speech_track).max()
    if peak > 1e-8:
        speech_track = speech_track / peak * speech_gain

    # Pad / trim background
    if len(bg) < total:
        bg = np.pad(bg, (0, total - len(bg)))
    else:
        bg = bg[:total]

    # Duck background under speech
    bg_gain = speech_mask * bg_gain_speech + (1 - speech_mask) * bg_gain_silence
    mixed = speech_track + bg * bg_gain

    # Final peak-normalize to prevent clipping
    peak = np.abs(mixed).max()
    if peak > 1.0:
        mixed = mixed / peak

    sf.write(str(output_path), mixed.astype(np.float32), sr)
    return output_path


def build_speech_track(
    segments: list[dict],
    output_path: Path,
    speech_gain: float = 0.85,
    fade_ms: int = 20,
    sample_rate: int = 24000,
) -> Path:
    """Build a clean speech-only audio track from TTS segments.

    Places each TTS segment at its ``start`` timestamp so the resulting
    audio follows the original video timeline.  No background audio is
    mixed in — this is the **clean TTS vocals** intended as the driving
    signal for Wav2Lip lip-sync.

    Parameters
    ----------
    segments:
        List of segment dicts with keys ``audio``, ``start``, ``end``.
        Segments without an ``audio`` value are skipped (silence retained).
    output_path:
        Destination WAV path.
    speech_gain:
        Peak volume for the speech track (0-1).
    fade_ms:
        Fade-in/out duration in milliseconds to avoid clicks.
    sample_rate:
        Output sample rate in Hz.  Default 24000 matches CosyVoice output.

    Returns
    -------
    ``output_path``
    """
    import librosa as _librosa

    # Determine total length from the latest segment end
    last_end = max((max(float(seg.get("end", 0) or 0),
                        float(seg.get("fit_end", 0) or 0))
                    for seg in segments), default=0)
    total = int(last_end * sample_rate) + sample_rate  # +1 s buffer

    speech_track = np.zeros(total, dtype=np.float32)

    for seg, limit in _placements(segments):
        audio_path = _seg_audio(seg)
        if not audio_path:
            continue
        wav, wav_sr = sf.read(str(audio_path))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        if wav_sr != sample_rate:
            wav = _librosa.resample(wav, orig_sr=wav_sr, target_sr=sample_rate)

        start_s = max(0, int(seg.get("start", 0) * sample_rate))
        limit_s = total if limit == float("inf") else int(limit * sample_rate)
        end_s = min(total, start_s + len(wav), limit_s)
        wav = wav[: max(0, end_s - start_s)]
        if len(wav) == 0:
            continue

        # Short fade-in / fade-out to suppress clicks
        fade = min(int(fade_ms * sample_rate / 1000), max(1, len(wav) // 4))
        wav[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        wav[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)

        speech_track[start_s:end_s] += wav

    # Normalize
    peak = np.abs(speech_track).max()
    if peak > 1e-8:
        speech_track = speech_track / peak * speech_gain

    sf.write(str(output_path), speech_track.astype(np.float32), sample_rate)
    return output_path


def compose_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    progress_cb=None,
) -> Path:
    """Replace the audio track of *video_path* with *audio_path*.

    Copies the video stream without re-encoding; re-encodes audio to AAC 192k.
    *progress_cb* is called with a status string at key steps.
    """
    if progress_cb:
        progress_cb("FFmpeg 合成中…")
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        str(output_path),
    ], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace")[-500:])
    return output_path
