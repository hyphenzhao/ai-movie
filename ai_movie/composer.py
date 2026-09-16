"""Audio processing: voice-background separation and mixing.

Supports two backends:
- ``"demucs"`` — htdemucs on GPU (reliable, default)
- ``"uvr"``    — Mel-Band RoiFormer via audio-separator (higher quality,
                 requires network for first-time model download)
"""

import logging
import os
import shutil
import re
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
    *,
    stem_suffix: str = "",
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
    stem_suffix:
        Inserted before ``.wav`` (``"_full"`` → ``vocals_full.wav``) so one
        directory can hold both the analysis and the production stems.

    Returns
    -------
    dict with keys ``vocals``, ``background`` (both Path).
    """
    if output_dir is None:
        output_dir = audio_path.parent
    output_dir = ensure_dir(output_dir)

    vocals_path = output_dir / f"vocals{stem_suffix}.wav"
    background_path = output_dir / f"background{stem_suffix}.wav"

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


def _downmix_16k_mono(src: Path, dst: Path) -> Path:
    """Analysis copy of a full-rate stem: 16 kHz mono, same length."""
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst),
    ], check=True, capture_output=True)
    return dst


def separate_for_pipeline(
    audio16k: Path,
    audio_full: Path | None,
    output_dir: Path,
    backend: str = "uvr",
    *,
    derive_analysis: bool | None = None,
) -> dict:
    """Separate once at full rate; derive the 16 kHz analysis stems from it.

    Returns ``vocals`` / ``background`` (16 kHz mono, what ASR, diarization,
    reference extraction and the GUI have always consumed) plus
    ``vocals_full`` / ``background_full`` (source rate, stereo — the
    production bed for :func:`mix_audio`) and ``bed_sr`` / ``bed_channels``.

    When *audio_full* is missing (a workspace demuxed before v3) the 16 kHz
    file is separated as before and no full-rate stems are produced.
    Existing valid 16 kHz stems are always kept, so a re-run never changes
    the analysis input under a cached ASR.
    """
    from ai_movie.config import SEPARATE_ANALYSIS_FROM_FULL
    if derive_analysis is None:
        derive_analysis = SEPARATE_ANALYSIS_FROM_FULL
    output_dir = ensure_dir(Path(output_dir))
    out: dict = {}

    have_full = audio_full is not None and Path(audio_full).exists()
    if have_full:
        full = separate_vocals(Path(audio_full), output_dir, backend=backend,
                               stem_suffix="_full")
        out["vocals_full"] = full["vocals"]
        out["background_full"] = full["background"]
        try:
            info = sf.info(str(full["background"]))
            out["bed_sr"], out["bed_channels"] = info.samplerate, info.channels
        except Exception:                               # noqa: BLE001
            pass

    v16, b16 = output_dir / "vocals.wav", output_dir / "background.wav"
    if v16.exists() and b16.exists():
        out["vocals"], out["background"] = v16, b16
    elif have_full and derive_analysis:
        out["vocals"] = _downmix_16k_mono(out["vocals_full"], v16)
        out["background"] = _downmix_16k_mono(out["background_full"], b16)
    else:
        a = separate_vocals(Path(audio16k), output_dir, backend=backend)
        out["vocals"], out["background"] = a["vocals"], a["background"]
    return out


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


def segment_slots(
    segments: list[dict],
    *,
    min_gap: float | None = None,
    max_tail: float | None = None,
) -> dict[int, float]:
    """Seconds available to each segment: ``{index: slot}``.

    The slot may extend ``max_tail`` into the following silence but must stop
    ``min_gap`` short of the next segment's start.  Shared by the fit stage
    and the compact stage so both agree on what "fits".
    """
    from ai_movie.config import TTS_FIT_MAX_TAIL, TTS_FIT_MIN_GAP
    min_gap = TTS_FIT_MIN_GAP if min_gap is None else min_gap
    max_tail = TTS_FIT_MAX_TAIL if max_tail is None else max_tail
    ordered = sorted(range(len(segments)),
                     key=lambda i: float(segments[i].get("start", 0.0)))
    slots: dict[int, float] = {}
    for n, i in enumerate(ordered):
        seg = segments[i]
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        limit = end + max_tail
        if n + 1 < len(ordered):
            nxt = segments[ordered[n + 1]]
            limit = min(limit, float(nxt.get("start", limit)) - min_gap)
        slots[i] = max(0.2, limit - start)
    return slots


def speaker_sec_per_char(
    segments: list[dict],
    *,
    text_key: str = "text_translated",
    audio_key: str = "audio",
    default: float | None = None,
) -> dict[str, float]:
    """Median measured seconds-per-visible-character per speaker.

    Unlike :func:`estimate_rate_correction` this returns the raw rate for
    *every* speaker (with *default* for speakers that have no measurable
    line), so a character budget can be derived from a slot length.
    """
    from ai_movie.config import TTS_COMPACT_SEC_PER_CHAR_DEFAULT
    default = TTS_COMPACT_SEC_PER_CHAR_DEFAULT if default is None else default
    by_spk: dict[str, list[float]] = {}
    for seg in segments:
        path = seg.get(audio_key)
        n = _visible_chars(seg.get(text_key) or "")
        if not path or not Path(path).exists() or n < 4:
            continue
        try:
            dur = float(sf.info(str(path)).duration)
        except Exception:                               # noqa: BLE001
            continue
        if dur > 0:
            by_spk.setdefault(seg.get("speaker") or "", []).append(dur / n)
    out = {spk: float(np.median(v)) for spk, v in by_spk.items() if v}
    for seg in segments:
        out.setdefault(seg.get("speaker") or "", default)
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
        TTS_FIT_MAX_SPEEDUP, TTS_FIT_MAX_SPEEDUP_HARD,
        TTS_FIT_MAX_TRUNCATE, TTS_FIT_TAIL_TOLERANCE,
    )

    max_speedup = TTS_FIT_MAX_SPEEDUP if max_speedup is None else max_speedup
    hard_speedup = TTS_FIT_MAX_SPEEDUP_HARD
    max_truncate = TTS_FIT_MAX_TRUNCATE
    tail_tol = TTS_FIT_TAIL_TOLERANCE

    out_dir = ensure_dir(Path(out_dir))
    ordered = sorted(range(len(segments)),
                     key=lambda i: float(segments[i].get("start", 0.0)))
    slots = segment_slots(segments, min_gap=min_gap, max_tail=max_tail)

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
        slot = slots[i]

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


def _duck_envelope(mask: np.ndarray, sr: int, attack_ms: float,
                   release_ms: float, ctrl_hz: int = 200) -> np.ndarray:
    """Smooth 0/1 speech mask → [0,1] envelope with attack/release.

    Runs the one-pole follower at a low control rate (200 Hz) and
    interpolates back, so a nine-minute film costs ~100k Python iterations
    rather than tens of millions.
    """
    hop = max(1, sr // ctrl_hz)
    n_ctrl = (len(mask) + hop - 1) // hop
    ctrl_in = np.zeros(n_ctrl, dtype=np.float32)
    # A control block is "speech" if any sample in it is.
    padded = np.zeros(n_ctrl * hop, dtype=np.float32)
    padded[:len(mask)] = mask
    ctrl_in = padded.reshape(n_ctrl, hop).max(axis=1)

    a_up = float(np.exp(-hop / (sr * max(attack_ms, 1e-3) / 1000.0)))
    a_dn = float(np.exp(-hop / (sr * max(release_ms, 1e-3) / 1000.0)))
    env = np.zeros(n_ctrl, dtype=np.float32)
    y = 0.0
    for i in range(n_ctrl):
        x = float(ctrl_in[i])
        a = a_up if x > y else a_dn
        y = a * y + (1.0 - a) * x
        env[i] = y
    xs = np.arange(n_ctrl) * hop
    return np.interp(np.arange(len(mask)), xs, env).astype(np.float32)


def _rms_db(x: np.ndarray, sr: int, floor_db: float = -45.0,
            frame_ms: float = 20.0) -> float | None:
    """RMS (dBFS) over frames louder than *floor_db*; None if nothing is."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = int(sr * frame_ms / 1000)
    if n <= 0 or len(x) < n:
        return None
    m = len(x) // n
    frames = x[:m * n].reshape(m, n).astype(np.float64)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    keep = db > floor_db
    if not keep.any():
        return None
    return float(20 * np.log10(np.sqrt((rms[keep] ** 2).mean())))


def loudnorm_two_pass(src: Path, dst: Path, *, target_lufs: float = -16.0,
                      true_peak_db: float = -1.0, lra: float = 11.0,
                      sample_rate: int | None = None) -> bool:
    """EBU R128 normalisation with ffmpeg's loudnorm (measure, then apply).

    Returns False (and writes nothing) when ffmpeg fails, so the caller can
    fall back to a plain peak normalisation.
    """
    base = f"loudnorm=I={target_lufs}:TP={true_peak_db}:LRA={lra}"
    try:
        p1 = subprocess.run([
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(src),
            "-af", base + ":print_format=json", "-f", "null", "-",
        ], capture_output=True, text=True, timeout=1800)
        err = p1.stderr
        j0, j1 = err.rfind("{"), err.rfind("}")
        if j0 < 0 or j1 < 0:
            return False
        import json as _json
        m = _json.loads(err[j0:j1 + 1])
        af = (base + f":measured_I={m['input_i']}:measured_TP={m['input_tp']}"
              f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
              f":offset={m['target_offset']}:linear=true")
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-af", af]
        if sample_rate:
            cmd += ["-ar", str(sample_rate)]
        cmd += ["-c:a", "pcm_s24le", str(dst)]
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
        return True
    except Exception as exc:                            # noqa: BLE001
        print(f"[composer] loudnorm failed ({exc}); using peak normalisation",
              file=sys.stderr)
        return False


def mix_audio(
    segments: list[dict],
    background_path: Path,
    output_path: Path,
    speech_gain: float = 0.85,
    bg_gain_speech: float = 0.25,
    bg_gain_silence: float = 1.0,
    fade_ms: int = 20,
    *,
    orig_vocals_path: Path | str | None = None,
    duck_db: float | None = None,
    duck_attack_ms: float | None = None,
    duck_release_ms: float | None = None,
    match_loudness: bool | None = None,
    match_clamp_db: float | None = None,
    target_lufs: float | None = None,
    true_peak_db: float | None = None,
    loudnorm: bool = True,
    stats: dict | None = None,
) -> Path:
    """Mix TTS speech segments into the background bed at their timestamps.

    Each segment is placed at ``seg['start']`` so the dub stays in sync with
    the picture.  v3 changes, all keyword-only so older callers behave as
    before:

    * the bed keeps its channel count and sample rate (a 48 kHz stereo bed
      yields a 48 kHz stereo mix; speech is placed identically on every
      channel);
    * the bed is ducked by ``duck_db`` under speech through a smooth
      attack/release envelope rather than a per-sample gate;
    * with ``orig_vocals_path`` (the separated original dialogue, any rate)
      every line is gain-matched to the original dialogue in its own slot,
      bounded to ±``match_clamp_db``, so the dub sits where the actors did
      relative to the music — the old global peak normalisation is then
      skipped;
    * the result is loudness-normalised to ``target_lufs`` / ``true_peak_db``
      with a two-pass ffmpeg loudnorm.

    ``stats`` (if given) receives ``{"gain_db": {idx: dB}, "sr", "channels"}``.
    """
    import librosa as _librosa
    from ai_movie.config import (
        MIX_DUCK_ATTACK_MS, MIX_DUCK_DB, MIX_DUCK_RELEASE_MS, MIX_MATCH_CLAMP_DB,
        MIX_MATCH_LOUDNESS, MIX_TARGET_LUFS, MIX_TRUE_PEAK_DB,
    )
    duck_db = MIX_DUCK_DB if duck_db is None else duck_db
    duck_attack_ms = MIX_DUCK_ATTACK_MS if duck_attack_ms is None else duck_attack_ms
    duck_release_ms = MIX_DUCK_RELEASE_MS if duck_release_ms is None else duck_release_ms
    match_clamp_db = MIX_MATCH_CLAMP_DB if match_clamp_db is None else match_clamp_db
    target_lufs = MIX_TARGET_LUFS if target_lufs is None else target_lufs
    true_peak_db = MIX_TRUE_PEAK_DB if true_peak_db is None else true_peak_db
    if match_loudness is None:
        match_loudness = MIX_MATCH_LOUDNESS
    match_loudness = bool(match_loudness and orig_vocals_path
                          and Path(orig_vocals_path).exists())

    # Load background (authoritative sample-rate, channels and length)
    bg, sr = sf.read(str(background_path), dtype="float32")
    if bg.ndim == 1:
        bg = bg[:, None]
    n_ch = bg.shape[1]

    # Original dialogue for loudness matching (mono, resampled lazily).
    orig = None
    orig_sr = 0
    if match_loudness:
        try:
            orig, orig_sr = sf.read(str(orig_vocals_path), dtype="float32")
            if orig.ndim > 1:
                orig = orig.mean(axis=1)
        except Exception:                               # noqa: BLE001
            orig, match_loudness = None, False

    # Total output length: at least as long as the background
    last_end = max((max(float(seg.get("end", 0) or 0),
                        float(seg.get("fit_end", 0) or 0))
                    for seg in segments), default=0)
    total = max(len(bg), int(last_end * sr) + sr)   # +1 s buffer

    speech_track = np.zeros(total, dtype=np.float32)
    speech_mask  = np.zeros(total, dtype=np.float32)
    gains: dict[int, float] = {}
    index_of = {id(s): i for i, s in enumerate(segments)}

    # First pass: measure every line and its original slot.
    placed: list[tuple[dict, float, np.ndarray, float | None, float | None]] = []
    for seg, limit in _placements(segments):
        audio_path = _seg_audio(seg)
        if not audio_path:
            continue
        wav, wav_sr = sf.read(str(audio_path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if wav_sr != sr:
            wav = _librosa.resample(wav, orig_sr=wav_sr, target_sr=sr)
        tts_db = _rms_db(wav, sr) if match_loudness else None
        orig_db = None
        if match_loudness and orig is not None:
            a = max(0, int(float(seg.get("start", 0)) * orig_sr))
            b = min(len(orig), int(float(seg.get("end", 0)) * orig_sr))
            if b > a:
                orig_db = _rms_db(orig[a:b], orig_sr)
        placed.append((seg, limit, wav, tts_db, orig_db))

    # TTS output sits near full scale while the separated dialogue is at the
    # film's natural level, so the bulk of the correction is one global
    # offset (median over all measurable lines, unclamped).  Only the
    # per-line *residual* is clamped, so a mis-measured slot cannot blow a
    # single line up while the systematic difference is still removed.
    global_off = 0.0
    if match_loudness:
        diffs = [o - t for _, _, _, t, o in placed
                 if o is not None and t is not None]
        global_off = float(np.median(diffs)) if diffs else 0.0

    for seg, limit, wav, tts_db, orig_db in placed:
        gain_db = 0.0
        if match_loudness and tts_db is not None:
            gain_db = global_off
            if orig_db is not None:
                resid = (orig_db - tts_db) - global_off
                gain_db += float(np.clip(resid, -match_clamp_db, match_clamp_db))
        if match_loudness:
            wav = wav * float(10 ** (gain_db / 20))
        i = index_of.get(id(seg))
        if i is not None:
            gains[i] = round(gain_db, 2)
            seg["mix_gain_db"] = round(gain_db, 2)

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

    if not match_loudness:
        # Legacy behaviour: peak-normalise the speech bus.
        peak = np.abs(speech_track).max()
        if peak > 1e-8:
            speech_track = speech_track / peak * speech_gain

    # Pad / trim background
    if len(bg) < total:
        bg = np.pad(bg, ((0, total - len(bg)), (0, 0)))
    else:
        bg = bg[:total]

    # Duck background under speech (smooth envelope)
    env = _duck_envelope(speech_mask, sr, duck_attack_ms, duck_release_ms)
    duck_lin = float(10 ** (duck_db / 20)) if match_loudness else bg_gain_speech
    bg_gain = env * duck_lin + (1.0 - env) * bg_gain_silence
    mixed = bg * bg_gain[:, None] + speech_track[:, None]

    if stats is not None:
        stats.update({"gain_db": gains, "sr": int(sr), "channels": int(n_ch),
                      "match_loudness": match_loudness,
                      "global_offset_db": round(global_off, 2)})

    if loudnorm:
        tmp = Path(str(output_path) + ".premix.wav")
        sf.write(str(tmp), mixed.astype(np.float32), sr, subtype="FLOAT")
        ok = loudnorm_two_pass(tmp, Path(output_path), target_lufs=target_lufs,
                               true_peak_db=true_peak_db, sample_rate=int(sr))
        tmp.unlink(missing_ok=True)
        if ok:
            return output_path

    # Fallback: peak-normalise to prevent clipping
    peak = np.abs(mixed).max()
    if peak > 1.0:
        mixed = mixed / peak
    sf.write(str(output_path), mixed.astype(np.float32), sr, subtype="PCM_24")
    return output_path


def mix_for_state(state: dict, segments: list[dict], output_path: Path,
                  *, stats: dict | None = None) -> Path:
    """Mix using whatever bed the workspace has (full-rate stereo preferred)."""
    sep = state.get("separate") or {}
    bg = sep.get("background_full") or sep.get("background")
    if bg and Path(bg).exists():
        return mix_audio(segments, Path(bg), output_path,
                         orig_vocals_path=sep.get("vocals"), stats=stats)
    return build_speech_track(segments, output_path)


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


def encoded_true_peak(path: Path) -> float | None:
    """True peak (dBTP) of the first audio stream of an encoded file."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                        "-map", "0:a:0", "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    peaks = re.findall(r"Peak:\s+(-?[0-9.]+) dBFS", r.stderr)
    return float(peaks[-1]) if peaks else None


def compose_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    progress_cb=None,
) -> Path:
    """Replace the audio track of *video_path* with *audio_path*.

    Copies the video stream without re-encoding; re-encodes audio to AAC 192k.
    *progress_cb* is called with a status string at key steps.

    The WAV is already normalised to ``MIX_TRUE_PEAK_DB``, but a lossy
    encoder reconstructs peaks slightly higher than the samples it was given
    — measured on v3.1.0, AAC 192k added 0.6 dB on one film and 2.0 dB on
    another, pushing the delivered track to +0.2 dBTP.  Lowering the mix
    target for everyone would cost loudness on material that never clips, so
    instead the encoded result is measured and, only when it overshoots,
    re-encoded with exactly the attenuation it needs.
    """
    from ai_movie.config import MIX_TRUE_PEAK_DB

    if progress_cb:
        progress_cb("FFmpeg 合成中…")
    ceiling = max(MIX_TRUE_PEAK_DB, -1.0)      # delivery limit, not the mix target
    gain_db = 0.0
    for attempt in range(3):
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-i", str(audio_path),
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
        if gain_db:
            cmd += ["-af", f"volume={gain_db:.2f}dB"]
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-shortest", str(output_path)]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace")[-500:])
        tp = encoded_true_peak(output_path)
        if tp is None or tp <= ceiling:
            return output_path
        gain_db += round(ceiling - 0.3 - tp, 2)
        if progress_cb:
            progress_cb(f"编码后真峰值 {tp:+.1f} dBTP 超出 {ceiling:.1f}，"
                        f"以 {gain_db:.2f} dB 重新编码")
    return output_path
