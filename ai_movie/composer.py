"""Audio processing: voice-background separation and mixing.

Uses Demucs (htdemucs) for source separation and FFmpeg for mixing.
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from ai_movie.utils import ensure_dir


def separate_vocals(
    audio_path: Path,
    output_dir: Path | None = None,
) -> dict:
    """Separate a mixed audio file into vocals and background.

    Uses the Demucs htdemucs model (best for speech).

    Parameters
    ----------
    audio_path:
        Path to input audio file (any format FFmpeg can read).
    output_dir:
        Where to write vocals.wav and background.wav.

    Returns
    -------
    dict with keys ``vocals``, ``background`` (both Path).
    """
    from demucs.pretrained import get_model
    from demucs.separate import apply_model

    if output_dir is None:
        output_dir = audio_path.parent
    output_dir = ensure_dir(output_dir)

    vocals_path = output_dir / "vocals.wav"
    background_path = output_dir / "background.wav"

    if vocals_path.exists() and background_path.exists():
        return {"vocals": vocals_path, "background": background_path}

    # Load model (cached after first call by demucs)
    model = get_model("htdemucs")
    model.to("cpu").eval()

    # Load audio via soundfile (supports many formats)
    audio_np, sr = sf.read(str(audio_path))
    # Convert to stereo if needed, then to tensor (1, 2, samples)
    if audio_np.ndim == 1:
        audio_np = np.stack([audio_np, audio_np], axis=1)
    elif audio_np.ndim == 2 and audio_np.shape[1] == 1:
        audio_np = np.tile(audio_np, (1, 2))
    audio_tensor = torch.from_numpy(audio_np.T).unsqueeze(0).float()  # (1, 2, samples)

    with torch.no_grad():
        sources = apply_model(
            model, audio_tensor, device="cpu",
            split=True, overlap=0.25, progress=True,
        )
    # sources shape: (1, 4, 2, samples) → [drums, bass, other, vocals]
    vocals_np = sources[0, 3].numpy().T           # (samples, 2)
    background_np = sources[0, 0:3].sum(dim=0).numpy().T  # (samples, 2)

    sf.write(str(vocals_path), vocals_np, sr)
    sf.write(str(background_path), background_np, sr)

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
