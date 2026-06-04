"""Text-to-speech synthesis using CosyVoice.

Two synthesis modes (selectable per-run):
  "gender"  — detect speaker gender from the source vocals, then use the
               matching CosyVoice-300M-SFT built-in speaker ("中文女" / "中文男").
               Requires CosyVoice-300M-SFT in models/; falls back to
               CosyVoice2-0.5B zero-shot if SFT model is absent.
  "clone"   — extract the highest-energy 8 s speech segment from the source
               vocals and use it for cross-lingual voice cloning via
               CosyVoice2-0.5B.  Quality depends on Demucs separation.

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

# ── model paths ──────────────────────────────────────────────────
_COSYVOICE_SRC = Path(__file__).parent.parent / "models" / "CosyVoice"
_MATCHA_SRC    = _COSYVOICE_SRC / "third_party" / "Matcha-TTS"
_SFT_MODEL_DIR = Path(__file__).parent.parent / "models" / "CosyVoice-300M-SFT"
_ZS_MODEL_DIR  = Path(__file__).parent.parent / "models" / "CosyVoice2-0.5B"

# ── SFT speaker IDs ───────────────────────────────────────────────
_SFT_FEMALE_SPK = "中文女"
_SFT_MALE_SPK   = "中文男"

# ── fallback reference voices (CosyVoice2 zero-shot) ─────────────
_ASSET_DIR       = _COSYVOICE_SRC / "asset"
_FEMALE_REF_WAV  = _ASSET_DIR / "zero_shot_prompt.wav"
_FEMALE_REF_TEXT = "希望你以后能够做的比我还好呦。"
_MALE_REF_WAV    = _ASSET_DIR / "cross_lingual_prompt.wav"

# F0 threshold (Hz) separating male from female
_GENDER_THRESHOLD_HZ = 165.0

# ── model cache ───────────────────────────────────────────────────
_model    = None
_is_sft   = False          # True when CosyVoice-300M-SFT is loaded
_lock     = threading.Lock()


def _load_model():
    global _model, _is_sft
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
        if _SFT_MODEL_DIR.exists() and (_SFT_MODEL_DIR / "llm.pt").exists():
            _model  = AutoModel(model_dir=str(_SFT_MODEL_DIR))
            _is_sft = True
        else:
            _model  = AutoModel(model_dir=str(_ZS_MODEL_DIR))
            _is_sft = False


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
    """Derive the speaker / reference and synthesis method for one session.

    Returns
    -------
    (spk_or_ref, ref_text_or_None, method)

    When *method* is ``'sft'``, *spk_or_ref* is a speaker ID string
    (e.g. ``'中文女'``) and *ref_text* is None.
    When *method* is ``'zero_shot'`` or ``'cross_lingual'``, *spk_or_ref*
    is a file path to reference audio.
    """
    cache_dir = ensure_dir(Path(cache_dir) if cache_dir else WORKSPACE_DIR / "synthesized")
    gender    = detect_gender(vocals_path)

    if mode == "gender":
        if _is_sft:
            # Best path: SFT built-in speaker, no reference audio needed
            spk = _SFT_FEMALE_SPK if gender == "female" else _SFT_MALE_SPK
            return spk, None, "sft"
        # Fallback: zero-shot / cross-lingual with bundled reference audio
        if gender == "female" and _FEMALE_REF_WAV.exists():
            return str(_FEMALE_REF_WAV), _FEMALE_REF_TEXT, "zero_shot"
        if _MALE_REF_WAV.exists():
            return str(_MALE_REF_WAV), None, "cross_lingual"
        return trim_reference_audio(vocals_path), None, "cross_lingual"

    # mode == "clone" — extract best speech segment from source vocals
    best_seg = str(cache_dir / "ref_best_segment.wav")
    extract_best_speech_segment(vocals_path, best_seg)
    return best_seg, None, "cross_lingual"


def call_tts(
    model,
    text: str,
    spk_or_ref: str,
    ref_text: str | None,
    method: str,
) -> np.ndarray:
    """Run one CosyVoice inference call and return a float32 numpy array.

    Parameters
    ----------
    method:
        ``'sft'``           — *spk_or_ref* is a speaker ID; uses inference_sft.
        ``'zero_shot'``     — *spk_or_ref* is a reference audio path.
        ``'cross_lingual'`` — *spk_or_ref* is a reference audio path.
    """
    chunks = []
    if method == "sft":
        for gen in model.inference_sft(text, spk_or_ref, stream=False):
            chunks.append(gen["tts_speech"].squeeze(0).cpu().numpy())
    elif method == "zero_shot" and ref_text:
        for gen in model.inference_zero_shot(text, ref_text, spk_or_ref, stream=False):
            chunks.append(gen["tts_speech"].squeeze(0).cpu().numpy())
    else:
        for gen in model.inference_cross_lingual(text, spk_or_ref, stream=False):
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
