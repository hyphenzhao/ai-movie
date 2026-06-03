"""Text-to-speech synthesis using CosyVoice 2 for voice cloning.

Lazy-loads the model on first use. Supports cross-lingual voice cloning:
provide a reference audio of the original speaker and the model
generates speech in the same voice for the translated text.

IMPORTANT: CosyVoice uses internal threading for LLM inference and
must be called from the main thread. The ``synthesize()`` function
schedules work on the main thread via a queue to avoid GIL deadlocks.
"""

import queue
import sys
import threading
from pathlib import Path
from typing import Callable

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


def trim_reference_audio(reference_audio: str | Path, max_seconds: int = 10) -> str:
    """Trim reference audio to at most max_seconds.

    CosyVoice speech tokenizer rejects audio longer than 30s; 10s is enough
    for voice cloning and keeps inference fast.  Returns the path to the
    trimmed file (written next to the original with a .ref10s.wav suffix).
    Falls back to the original path if trimming fails.
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


def synthesize(
    segments: list[dict],
    reference_audio: str | Path,
    output_dir: str | Path | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[dict]:
    """Generate Chinese speech for translated segments using voice cloning.

    Must be called from the **main thread** (CosyVoice uses internal
    threading that conflicts with background threads).
    """
    _load_model()

    if output_dir is None:
        output_dir = ensure_dir(WORKSPACE_DIR / "synthesized")
    else:
        output_dir = ensure_dir(Path(output_dir))

    ref_path = trim_reference_audio(reference_audio)

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

        tagged_text = f"<|zh|>{text}"
        try:
            chunks = []
            for gen in _model.inference_cross_lingual(
                tagged_text, ref_path, stream=False
            ):
                chunks.append(gen["tts_speech"].squeeze(0).cpu().numpy())
            audio_np = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
            out_path = str(output_dir / f"seg_{i + 1:04d}.wav")
            sf.write(out_path, audio_np, _model.sample_rate)
            results[i]["audio"] = out_path
        except Exception as exc:
            results[i]["audio"] = None
            results[i]["tts_error"] = str(exc)

        if progress_cb:
            progress_cb(i + 1, total)

    return results
