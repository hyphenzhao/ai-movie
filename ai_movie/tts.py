"""Text-to-speech synthesis using CosyVoice 2.

Two synthesis modes (selectable per-run):
  "gender"  — detect speaker gender from the source vocals, then use the
               bundled high-quality male/female reference voice for synthesis.
               Best clarity; voice is same gender as original but not the same
               person.
  "clone"   — extract the highest-energy 8 s speech segment from the source
               vocals and use it for cross-lingual voice cloning.
               Attempts to preserve the original speaker timbre; quality
               depends on how clean the Demucs-separated vocals are.

IMPORTANT: CosyVoice uses internal threading for LLM inference and
must be called from the main thread.
"""

import sys
import threading
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import soundfile as sf

from ai_movie.config import WORKSPACE_DIR
from ai_movie.utils import ensure_dir

# ── model cache ──────────────────────────────────────────────────
_model = None
_lock = threading.Lock()

_COSYVOICE_SRC = Path(__file__).parent.parent / "models" / "CosyVoice"
_MATCHA_SRC = _COSYVOICE_SRC / "third_party" / "Matcha-TTS"
_MODEL_DIR = Path(__file__).parent.parent / "models" / "CosyVoice2-0.5B"

# ── bundled clean reference voices (shipped with CosyVoice) ──────
_ASSET_DIR = _COSYVOICE_SRC / "asset"
# Female: 274 Hz, 3.5 s, Mandarin — known reference text available
_FEMALE_REF_AUDIO = _ASSET_DIR / "zero_shot_prompt.wav"
_FEMALE_REF_TEXT = "希望你以后能够做的比我还好呦。"
# Male: 148 Hz, 13.7 s, English — use cross-lingual (no reference text needed)
_MALE_REF_AUDIO = _ASSET_DIR / "cross_lingual_prompt.wav"

# F0 threshold (Hz) separating male from female
_GENDER_THRESHOLD_HZ = 165.0


def _load_model():
    global _model
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        if str(_COSYVOICE_SRC) not in sys.path:
            sys.path.insert(0, str(_COSYVOICE_SRC))
        if str(_MATCHA_SRC) not in sys.path:
            sys.path.insert(0, str(_MATCHA_SRC))
        from cosyvoice.cli.cosyvoice import AutoModel
        _model = AutoModel(model_dir=str(_MODEL_DIR))


# ── public helpers ────────────────────────────────────────────────

def detect_gender(audio_path: str | Path) -> str:
    """Estimate speaker gender by fundamental frequency (F0).

    Analyses up to 20 s of audio around the midpoint and returns
    ``'female'`` if median F0 >= 165 Hz, else ``'male'``.
    Falls back to ``'female'`` when no voiced frames are found.
    """
    import librosa

    a, sr = sf.read(str(audio_path))
    if a.ndim > 1:
        a = a.mean(axis=1)
    # Analyse a centred 20 s window for a representative estimate
    clip_len = min(len(a), sr * 20)
    mid = len(a) // 2
    start = max(0, mid - clip_len // 2)
    clip = a[start: start + clip_len].astype(np.float32)

    f0, voiced_flag, _ = librosa.pyin(
        clip,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
    )
    valid = f0[voiced_flag]
    if len(valid) == 0:
        return "female"
    return "female" if float(np.median(valid)) >= _GENDER_THRESHOLD_HZ else "male"


def extract_best_speech_segment(
    audio_path: str | Path,
    out_path: str | Path,
    duration: int = 8,
) -> str:
    """Extract the highest-RMS speech segment as a clean reference clip.

    Slides a ``duration``-second window across the audio and picks the
    window with the highest RMS energy (loudest clear speech).
    Writes the result to *out_path* and returns its path string.
    """
    a, sr = sf.read(str(audio_path))
    if a.ndim > 1:
        a = a.mean(axis=1)

    target_len = sr * duration
    if len(a) <= target_len:
        sf.write(str(out_path), a, sr)
        return str(out_path)

    hop = max(1, sr // 2)  # 0.5 s steps
    best_rms, best_start = -1.0, 0
    for start in range(0, len(a) - target_len, hop):
        rms = float(np.sqrt(np.mean(a[start: start + target_len] ** 2)))
        if rms > best_rms:
            best_rms, best_start = rms, start

    sf.write(str(out_path), a[best_start: best_start + target_len], sr)
    return str(out_path)


def trim_reference_audio(reference_audio: str | Path, max_seconds: int = 10) -> str:
    """Trim reference audio to at most *max_seconds* (≤ 30 s CosyVoice limit).

    Writes a ``*.ref10s.wav`` file next to the source and returns its path.
    Falls back to the original path on any error.
    """
    src = Path(reference_audio)
    dst = src.with_name(src.stem + ".ref10s.wav")
    try:
        audio_np, sr = sf.read(str(src))
        clip_len = min(len(audio_np), sr * max_seconds)
        sf.write(str(dst), audio_np[:clip_len], sr)
        return str(dst)
    except Exception:
        return str(src)


def prepare_reference(
    vocals_path: str | Path,
    mode: Literal["gender", "clone"] = "gender",
    cache_dir: str | Path | None = None,
) -> tuple[str, str | None, str]:
    """Derive the reference audio and synthesis method for one session.

    Parameters
    ----------
    vocals_path:
        Demucs-separated vocals file for the current project.
    mode:
        ``'gender'``  — detect gender, use bundled clean reference.
        ``'clone'``   — extract the best 8 s segment from *vocals_path*.
    cache_dir:
        Directory for temporary files; defaults to WORKSPACE_DIR/synthesized.

    Returns
    -------
    (ref_audio_path, ref_text_or_None, synthesis_method)

    *synthesis_method* is ``'zero_shot'`` or ``'cross_lingual'``.
    """
    cache_dir = ensure_dir(Path(cache_dir) if cache_dir else WORKSPACE_DIR / "synthesized")

    if mode == "gender":
        gender = detect_gender(vocals_path)
        if gender == "female" and _FEMALE_REF_AUDIO.exists():
            return str(_FEMALE_REF_AUDIO), _FEMALE_REF_TEXT, "zero_shot"
        if _MALE_REF_AUDIO.exists():
            return str(_MALE_REF_AUDIO), None, "cross_lingual"
        # Fallback: use trimmed vocals if bundled files missing
        return trim_reference_audio(vocals_path), None, "cross_lingual"

    # mode == "clone"
    best_seg = str(cache_dir / "ref_best_segment.wav")
    extract_best_speech_segment(vocals_path, best_seg)
    return best_seg, None, "cross_lingual"


def call_tts(
    model,
    text: str,
    ref_audio: str,
    ref_text: str | None,
    method: str,
) -> np.ndarray:
    """Run one CosyVoice2 inference call and return a float32 numpy array.

    Parameters
    ----------
    model:
        Loaded CosyVoice2 AutoModel instance.
    text:
        Chinese text to synthesise.
    ref_audio:
        Path to reference audio (≤ 30 s).
    ref_text:
        Text spoken in *ref_audio*; required for ``'zero_shot'`` method.
    method:
        ``'zero_shot'`` or ``'cross_lingual'``.
    """
    chunks = []
    if method == "zero_shot" and ref_text:
        for gen in model.inference_zero_shot(text, ref_text, ref_audio, stream=False):
            chunks.append(gen["tts_speech"].squeeze(0).cpu().numpy())
    else:
        for gen in model.inference_cross_lingual(f"<|zh|>{text}", ref_audio, stream=False):
            chunks.append(gen["tts_speech"].squeeze(0).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def synthesize(
    segments: list[dict],
    reference_audio: str | Path,
    output_dir: str | Path | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    mode: Literal["gender", "clone"] = "gender",
) -> list[dict]:
    """Generate Chinese speech for translated segments.

    Must be called from the **main thread** (CosyVoice internal threading).

    Parameters
    ----------
    mode:
        ``'gender'`` — clear bundled voice matched to detected gender (default).
        ``'clone'``  — voice cloning from the best segment of *reference_audio*.
    """
    _load_model()

    if output_dir is None:
        output_dir = ensure_dir(WORKSPACE_DIR / "synthesized")
    else:
        output_dir = ensure_dir(Path(output_dir))

    ref_audio, ref_text, method = prepare_reference(
        reference_audio, mode=mode, cache_dir=output_dir
    )

    total = len(segments)
    results: list[dict] = list(segments)

    for i, seg in enumerate(results):
        if cancel_check and cancel_check():
            break

        text = seg.get("text_translated", "").strip()
        if not text:
            results[i]["audio"] = None
            if progress_cb:
                progress_cb(i + 1, total)
            continue

        try:
            audio_np = call_tts(_model, text, ref_audio, ref_text, method)
            out_path = str(output_dir / f"seg_{i + 1:04d}.wav")
            sf.write(out_path, audio_np, _model.sample_rate)
            results[i]["audio"] = out_path
        except Exception as exc:
            results[i]["audio"] = None
            results[i]["tts_error"] = str(exc)

        if progress_cb:
            progress_cb(i + 1, total)

    return results
