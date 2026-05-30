"""Speech-to-text using faster-whisper.

Model download
--------------
On first use, faster-whisper downloads the model from HuggingFace
(``Systran/faster-whisper-large-v3``, ~3 GB).

If HuggingFace is unreachable, download the model manually:

1. Visit (VPN required from some regions):
   https://huggingface.co/Systran/faster-whisper-large-v3

2. Download all files into a local folder, e.g. ``C:/models/faster-whisper-large-v3``

3. Set the path in ``ai_movie/config.py``::

       ASR_MODEL_SIZE = "C:/models/faster-whisper-large-v3"

   Or use ``hf-mirror.com``::

       set HF_ENDPOINT=https://hf-mirror.com
"""

from pathlib import Path
from typing import Callable

LANGUAGES = {
    "日本語": "ja",
    "English": "en",
    "中文":     "zh",
    "한국어":   "ko",
}
LANG_LABELS = list(LANGUAGES.keys())

_MODEL_DOWNLOAD_HELP = (
    "模型下载失败。请参考 ai_movie/asr.py 顶部的注释说明，"
    "手动下载模型后设置 ASR_MODEL_SIZE 为本地路径。"
)


def _load_model(model_size: str):
    """Return a WhisperModel, trying CUDA first then CPU."""
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(model_size, device="cuda", compute_type="float16")
    except Exception:
        try:
            return WhisperModel(model_size, device="cpu", compute_type="int8")
        except Exception as e:
            if "LocalEntryNotFoundError" in type(e).__name__ or "ConnectTimeout" in str(e):
                raise RuntimeError(_MODEL_DOWNLOAD_HELP) from e
            raise


def transcribe_all(
    audio_paths: list[Path],
    language: str = "ja",
    model_size: str | None = None,
    segment_cb: Callable[[int, dict], None] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[dict]:
    """Transcribe a list of audio files, streaming segments in real time.

    Parameters
    ----------
    segment_cb:
        Called from a worker thread as each sentence is recognised:
        ``segment_cb(file_index, {"start": 0.0, "end": 2.5, "text": "…"})``
    progress_cb:
        ``progress_cb(current_file_index, total_files)`` — called once per file.

    Returns a list of dicts::

        {
            "source": str,
            "language": str,
            "segments": [{"start": 0.0, "end": 2.5, "text": "…"}, …],
        }
    """
    if model_size is None:
        from ai_movie.config import ASR_MODEL_SIZE
        model_size = ASR_MODEL_SIZE

    model = _load_model(model_size)

    all_results: list[dict] = []
    for i, p in enumerate(audio_paths):
        if cancel_check and cancel_check():
            break

        try:
            segments_iter, info = model.transcribe(
                str(p), language=language,
                beam_size=5, vad_filter=True,
            )
        except Exception as exc:
            all_results.append({"source": str(p), "error": str(exc)})
            if progress_cb:
                progress_cb(i + 1, len(audio_paths))
            continue

        segs: list[dict] = []
        for seg in segments_iter:
            if cancel_check and cancel_check():
                break
            d = {"start": round(seg.start, 2),
                  "end": round(seg.end, 2),
                  "text": seg.text.strip(),
                  "source": str(p)}
            segs.append(d)
            if segment_cb:
                segment_cb(i, d)

        all_results.append({
            "source": str(p),
            "language": info.language,
            "segments": segs,
        })

        if progress_cb:
            progress_cb(i + 1, len(audio_paths))

    return all_results
