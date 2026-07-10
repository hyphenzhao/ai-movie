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
    last_end = max((seg.get("end", 0) for seg in segments), default=0)
    total = max(len(bg), int(last_end * sr) + sr)   # +1 s buffer

    speech_track = np.zeros(total, dtype=np.float32)
    speech_mask  = np.zeros(total, dtype=np.float32)

    for seg in segments:
        audio_path = seg.get("audio")
        if not audio_path or not Path(audio_path).exists():
            continue
        wav, wav_sr = sf.read(str(audio_path))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        if wav_sr != sr:
            wav = _librosa.resample(wav, orig_sr=wav_sr, target_sr=sr)

        start_s = max(0, int(seg.get("start", 0) * sr))
        end_s   = min(total, start_s + len(wav))
        wav     = wav[: end_s - start_s]

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
    last_end = max((seg.get("end", 0) for seg in segments), default=0)
    total = int(last_end * sample_rate) + sample_rate  # +1 s buffer

    speech_track = np.zeros(total, dtype=np.float32)

    for seg in segments:
        audio_path = seg.get("audio")
        if not audio_path or not Path(audio_path).exists():
            continue
        wav, wav_sr = sf.read(str(audio_path))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        if wav_sr != sample_rate:
            wav = _librosa.resample(wav, orig_sr=wav_sr, target_sr=sample_rate)

        start_s = max(0, int(seg.get("start", 0) * sample_rate))
        end_s = min(total, start_s + len(wav))
        wav = wav[: end_s - start_s]

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
