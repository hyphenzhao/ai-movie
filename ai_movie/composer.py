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
    speech_paths: list[Path],
    background_path: Path,
    output_path: Path,
    sample_rate: int = 24000,
) -> Path:
    """Mix generated speech segments with background audio.

    Parameters
    ----------
    speech_paths:
        Ordered list of per-segment generated speech WAV files.
    background_path:
        Background audio (without vocals).
    output_path:
        Where to write the mixed audio.
    sample_rate:
        Output sample rate.

    Returns
    -------
    *output_path*
    """
    # Concatenate speech segments with silence gaps
    speech_parts = []
    for p in speech_paths:
        if p is not None and Path(p).exists():
            audio, sr = sf.read(str(p))
            if audio.ndim == 2:
                audio = audio.mean(axis=1)  # stereo → mono
            speech_parts.append(audio)
        else:
            speech_parts.append(np.array([], dtype=np.float32))

    full_speech = np.concatenate(speech_parts) if speech_parts else np.array([], dtype=np.float32)

    # Load background
    bg, bg_sr = sf.read(str(background_path))
    if bg.ndim == 2:
        bg = bg.mean(axis=1)

    # Match lengths: pad or trim background to match speech
    if len(full_speech) > len(bg):
        bg = np.pad(bg, (0, len(full_speech) - len(bg)))
    else:
        bg = bg[:len(full_speech)]

    # Normalize and mix
    if full_speech.size > 0:
        full_speech = full_speech / (np.abs(full_speech).max() + 1e-8)
    if bg.size > 0:
        bg = bg / (np.abs(bg).max() + 1e-8)

    mixed = (full_speech * 0.7 + bg * 0.3) if full_speech.size > 0 else bg

    sf.write(str(output_path), mixed.astype(np.float32), sample_rate)
    return output_path


def compose_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> Path:
    """Replace the audio track of *video_path* with *audio_path*.

    Returns *output_path*.
    """
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        str(output_path),
    ], check=True, capture_output=True)
    return output_path
