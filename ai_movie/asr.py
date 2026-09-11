"""Speech-to-text with automatic GPU / CPU backend selection.

Architecture (priority order)
-----------------------------
**Linux / macOS:**
  1. openai-whisper + PyTorch GPU (ROCm / CUDA) + Silero VAD pre-segmentation
  2. faster-whisper (CTranslate2) + built-in Silero VAD — CPU/GPU fallback

**Windows:**
  1. WSL + ROCm: launches ``wsl.exe python3 asr_wsl.py`` subprocess
  2. DirectML GPU: launches subprocess with torch-directml + openai-whisper
  3. CPU fallback: faster-whisper (CTranslate2, int8 quantized)

All GPU paths now use Silero VAD to prevent hallucination loops
(especially critical for Japanese) and to produce sentence boundaries
aligned with natural conversational pauses.

Call ``transcribe_all()`` — it auto-selects the best available backend.
"""

import functools
import json
import os
import shutil
import subprocess
import sys
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

_WSL_VENV_PYTHON = "~/ai-movie-venv/bin/python3"
_WSL_BRIDGE = "ai_movie/asr_wsl.py"


# ── WSL path mapping ───────────────────────────────────────────

def _win_to_wsl_path(win_path: str) -> str:
    """Convert ``C:\\foo\\bar`` to ``/mnt/c/foo/bar``."""
    if len(win_path) >= 2 and win_path[1] == ":":
        drive = win_path[0].lower()
        rest = win_path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return win_path.replace("\\", "/")


# ── WSL + ROCm detection ──────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _wsl_available() -> bool:
    return shutil.which("wsl.exe") is not None


@functools.lru_cache(maxsize=1)
def wsl_rocm_available() -> bool:
    """Check WSL+ROCm availability (cached per process)."""
    if not _wsl_available():
        return False

    # Fast check: marker file (avoids WSL Python startup for negative case)
    try:
        marker = subprocess.run(
            ["wsl.exe", "bash", "-c",
             "test -f ~/.config/ai-movie-wsl-rocm && echo '1' || echo '0'"],
            capture_output=True, text=True, timeout=10,
        )
        if marker.returncode != 0 or "1" not in marker.stdout:
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

    # Deep probe: run asr_wsl.py --probe in WSL venv
    try:
        result = subprocess.run(
            ["wsl.exe", _WSL_VENV_PYTHON,
             _win_to_wsl_path(str(Path(__file__).parent / "asr_wsl.py")),
             "--probe"],
            capture_output=True, text=True, timeout=60,
        )
        for line in result.stdout.splitlines():
            try:
                msg = json.loads(line)
                if msg.get("type") == "probe_result":
                    return msg.get("available", False)
            except json.JSONDecodeError:
                continue
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False


# ── shared subprocess JSON-lines parser ───────────────────────

def _run_asr_subprocess(
    proc: subprocess.Popen,
    audio_paths: list[Path],
    segment_cb: Callable[[int, dict], None] | None,
    progress_cb: Callable[[int, int], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> list[dict]:
    """Read JSON-lines from *proc*.stdout, fire callbacks, return results."""
    source_to_idx = {str(p): i for i, p in enumerate(audio_paths)}
    all_results: list[dict] = []
    current_file = 0

    for line in proc.stdout:
        if cancel_check and cancel_check():
            proc.kill()
            break

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        t = msg.get("type")

        if t == "segment":
            source = msg.get("source", "")
            idx = source_to_idx.get(source, 0)
            seg = {"start": msg["start"], "end": msg["end"],
                   "text": msg["text"], "source": source}
            if segment_cb:
                segment_cb(idx, seg)

        elif t == "file_done":
            current_file += 1
            if progress_cb:
                progress_cb(current_file, len(audio_paths))

        elif t == "all_done":
            all_results = msg.get("results", [])
            break

        elif t == "error":
            stderr_tail = ""
            try:
                proc.wait(timeout=2)
                stderr_tail = proc.stderr.read()
            except Exception:
                proc.kill()
            raise RuntimeError(
                f"Subprocess transcription failed: {msg.get('message', '')}"
                + (f"\n{stderr_tail}" if stderr_tail else "")
            )

    proc.wait(timeout=10)
    return all_results


# ── backend detection ──────────────────────────────────────────

def _gpu_venv_python() -> Path | None:
    """Return the Python 3.12 venv executable, or None."""
    candidates = [
        # Windows: bundled venv with torch-directml
        Path(__file__).parent.parent / "venv312" / "Scripts" / "python.exe",
        # Linux: bundled venv with PyTorch CUDA/ROCm
        Path(__file__).parent.parent / "venv312" / "bin" / "python3",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _linux_gpu_available() -> bool:
    """Check for native GPU support on Linux (CUDA / ROCm)."""
    if sys.platform == "win32":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def gpu_available() -> bool:
    """True if a GPU backend (DirectML or native CUDA/ROCm) is available."""
    if sys.platform != "win32":
        return _linux_gpu_available()
    return _gpu_venv_python() is not None


# ── GPU backend ────────────────────────────────────────────────

def _transcribe_gpu(
    audio_paths: list[Path],
    language: str,
    model_size: str,
    segment_cb: Callable[[int, dict], None] | None,
    progress_cb: Callable[[int, int], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> list[dict]:
    """Launch the DirectML GPU subprocess and stream results."""
    venv_py = _gpu_venv_python()
    if venv_py is None:
        raise RuntimeError("GPU venv not found")

    bridge = Path(__file__).parent / "asr_gpu.py"
    proc = subprocess.Popen(
        [str(venv_py), str(bridge)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    job = {
        "audio_paths": [str(p) for p in audio_paths],
        "language": language,
        "model_size": model_size,
    }
    try:
        proc.stdin.write(json.dumps(job, ensure_ascii=False) + "\n")
        proc.stdin.flush()
    except Exception:
        proc.kill()
        raise RuntimeError("Failed to communicate with GPU process")

    return _run_asr_subprocess(
        proc, audio_paths, segment_cb, progress_cb, cancel_check,
    )


# ── WSL + ROCm backend ─────────────────────────────────────────

def _transcribe_wsl(
    audio_paths: list[Path],
    language: str,
    model_size: str,
    segment_cb: Callable[[int, dict], None] | None,
    progress_cb: Callable[[int, int], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> list[dict]:
    """Launch WSL subprocess running asr_wsl.py with ROCm."""
    bridge_wsl = _win_to_wsl_path(
        str(Path(__file__).parent / "asr_wsl.py")
    )
    audio_paths_wsl = [_win_to_wsl_path(str(p)) for p in audio_paths]

    proc = subprocess.Popen(
        ["wsl.exe", _WSL_VENV_PYTHON, bridge_wsl],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    job = {
        "audio_paths": audio_paths_wsl,
        "language": language,
        "model_size": model_size,
    }
    try:
        proc.stdin.write(json.dumps(job, ensure_ascii=False) + "\n")
        proc.stdin.flush()
    except Exception:
        proc.kill()
        raise RuntimeError("Failed to communicate with WSL process")

    return _run_asr_subprocess(
        proc, audio_paths, segment_cb, progress_cb, cancel_check,
    )


# ── CPU backend (faster-whisper) ───────────────────────────────

def _load_cpu_model(model_size: str):
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


def _transcribe_cpu(
    audio_paths: list[Path],
    language: str,
    model_size: str,
    segment_cb: Callable[[int, dict], None] | None,
    progress_cb: Callable[[int, int], None] | None,
    file_start_cb: Callable[[int, str], None] | None,
    file_progress_cb: Callable[[int, int], None] | None,
    cancel_check: Callable[[], bool] | None,
    diarize: bool = False,
    num_speakers: int | None = None,
    vocals_path: str | Path | None = None,
    max_duration: float | None = None,
    max_chars: int | None = None,
    status_cb: Callable[[str], None] | None = None,
) -> list[dict]:
    from ai_movie.config import ASR_CONDITION_ON_PREVIOUS, ASR_WORD_TIMESTAMPS

    model = _load_cpu_model(model_size)
    all_results: list[dict] = []

    for i, p in enumerate(audio_paths):
        if cancel_check and cancel_check():
            break

        if file_start_cb:
            file_start_cb(i, p.name)

        duration = _get_audio_duration(p)

        try:
            segments_iter, info = model.transcribe(
                str(p), language=language,
                beam_size=5, vad_filter=True,
                word_timestamps=ASR_WORD_TIMESTAMPS,
                condition_on_previous_text=ASR_CONDITION_ON_PREVIOUS,
            )
        except Exception as exc:
            all_results.append({"source": str(p), "error": str(exc)})
            if progress_cb:
                progress_cb(i + 1, len(audio_paths))
            continue

        raw_segs: list[dict] = []
        words: list[dict] = []
        for seg in segments_iter:
            if cancel_check and cancel_check():
                break
            raw_segs.append({"start": round(seg.start, 2),
                             "end": round(seg.end, 2),
                             "text": seg.text.strip()})
            for w in (getattr(seg, "words", None) or []):
                words.append({"w": w.word, "s": round(w.start, 3),
                              "e": round(w.end, 3),
                              "p": round(float(getattr(w, "probability", 0.0)), 3)})
            # Per-file progress from segment end timestamp
            if duration > 0 and file_progress_cb:
                pct = min(int(seg.end / duration * 100), 99)
                file_progress_cb(i, pct)

        diar = None
        if diarize:
            diar = _run_diarization(
                p, None, num_speakers=num_speakers,
                vocals_path=vocals_path, status_cb=status_cb,
            )

        segs = _finalize_segments(
            raw_segs, words, source=str(p), diarization=diar,
            max_duration=max_duration, max_chars=max_chars,
            segment_cb=(lambda d, _i=i: segment_cb(_i, d)) if segment_cb else None,
        )

        if file_progress_cb:
            file_progress_cb(i, 100)

        entry = {
            "source": str(p),
            "language": info.language,
            "segments": segs,
            "words": words,
        }
        if diar:
            entry["diarization"] = diar
        all_results.append(entry)

        if progress_cb:
            progress_cb(i + 1, len(audio_paths))

    return all_results


# ── openai-whisper GPU backend (Linux ROCm / CUDA) ───────────────



def _get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds (fast ffprobe)."""
    import subprocess
    result = subprocess.run([
        "ffprobe", "-v", "quiet", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ], capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return float(result.stdout.strip())
    return 0.0


# ── VAD (Voice Activity Detection) ─────────────────────────────

_VAD_CACHE: dict = {}


def _load_silero_vad():
    """Lazy-load Silero VAD model (thread-safe, cached).

    Silero VAD is a lightweight ONNX model (~1.7 MB) that detects
    speech vs. silence with high accuracy across 100+ languages.

    Returns
    -------
    ``(model, utils)`` tuple on success, ``(None, None)`` if VAD is
    unavailable (the caller should fall back to whole-file transcription).
    """
    if "model" in _VAD_CACHE:
        return _VAD_CACHE["model"], _VAD_CACHE["utils"]

    try:
        import torch

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        _VAD_CACHE["model"] = model
        _VAD_CACHE["utils"] = utils
        return model, utils
    except Exception as exc:
        print(
            f"[ASR] Silero VAD 加载失败，将回退到整文件转录: {exc}",
            file=sys.stderr,
        )
        _VAD_CACHE["model"] = None
        _VAD_CACHE["utils"] = None
        return None, None


def _vad_detect(
    audio,  # 1-D float tensor, 16 kHz mono
    threshold: float = 0.35,
    min_silence_duration_ms: int = 500,
    min_speech_duration_ms: int = 150,
    speech_pad_ms: int = 200,
) -> list[dict]:
    """Run Silero VAD and return speech-segment timestamps.

    Parameters
    ----------
    audio:
        1-D float tensor, **must be 16 kHz mono** (as loaded by
        ``whisper.load_audio()``).
    threshold:
        Speech probability threshold (0.0–1.0).  Lower = more sensitive.
        Default 0.35 is tuned for Japanese conversational speech.
    min_silence_duration_ms:
        How many ms of silence before marking a segment boundary.
    min_speech_duration_ms:
        Segments shorter than this are treated as noise.
    speech_pad_ms:
        Padding added to each side of a detected speech segment.

    Returns
    -------
    List of dicts with ``start`` and ``end`` keys (seconds, float).
    Returns empty list if VAD is unavailable.
    """
    model, utils = _load_silero_vad()
    if model is None:
        return []

    get_speech_timestamps = utils[0]

    timestamps = get_speech_timestamps(
        audio, model,
        sampling_rate=16000,
        threshold=threshold,
        min_silence_duration_ms=min_silence_duration_ms,
        min_speech_duration_ms=min_speech_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )

    return [
        {"start": ts["start"] / 16000, "end": ts["end"] / 16000}
        for ts in timestamps
    ]



def _transcribe_whisper_gpu(
    audio_paths: list[Path],
    language: str,
    model_size: str,
    segment_cb: Callable[[int, dict], None] | None,
    progress_cb: Callable[[int, int], None] | None,
    file_start_cb: Callable[[int, str], None] | None,
    file_progress_cb: Callable[[int, int], None] | None,
    cancel_check: Callable[[], bool] | None,
    diarize: bool = False,
    num_speakers: int | None = None,
    vocals_path: str | Path | None = None,
    max_duration: float | None = None,
    max_chars: int | None = None,
    status_cb: Callable[[str], None] | None = None,
) -> list[dict]:
    """Transcribe with openai-whisper GPU + Silero VAD pre-segmentation.

    VAD (Voice Activity Detection) splits the audio at silence gaps
    before transcription.  Each speech segment is transcribed
    independently, then timestamps are adjusted to the original
    timeline.  This prevents the hallucination-loop issue that Whisper
    exhibits on long Japanese audio and produces more natural sentence
    boundaries aligned with conversational pauses.

    With *diarize*, speaker turns are computed from the same VAD spans and
    fed to the sentence splitter, so no segment ever spans two speakers.

    Falls back to whole-file transcription if VAD is unavailable.
    """
    import torch
    import whisper

    from ai_movie.config import (
        ASR_VAD_MIN_SILENCE_DURATION_MS,
        ASR_VAD_MIN_SPEECH_DURATION_MS,
        ASR_VAD_SPEECH_PAD_MS,
        ASR_VAD_THRESHOLD,
    )

    device = torch.device("cuda")
    model = whisper.load_model(model_size).to(device)
    WHISPER_SR = whisper.audio.SAMPLE_RATE  # 16 000

    all_results: list[dict] = []
    for i, p in enumerate(audio_paths):
        if cancel_check and cancel_check():
            break

        if file_start_cb:
            file_start_cb(i, p.name)

        duration = _get_audio_duration(p)

        # ── load audio & run VAD ─────────────────────────────────
        audio_np = whisper.load_audio(str(p))       # float32, 16 kHz
        audio_pt = torch.from_numpy(audio_np)

        speech_segs = _vad_detect(
            audio_pt,
            threshold=ASR_VAD_THRESHOLD,
            min_silence_duration_ms=ASR_VAD_MIN_SILENCE_DURATION_MS,
            min_speech_duration_ms=ASR_VAD_MIN_SPEECH_DURATION_MS,
            speech_pad_ms=ASR_VAD_SPEECH_PAD_MS,
        )

        if not speech_segs:
            # VAD unavailable or found nothing → whole-file fallback
            speech_segs = [{"start": 0.0, "end": len(audio_np) / WHISPER_SR}]

        # ── speaker diarization (reuses the VAD spans above) ─────
        diar = None
        if diarize:
            diar = _run_diarization(
                p, speech_segs,
                num_speakers=num_speakers, vocals_path=vocals_path,
                status_cb=status_cb,
            )

        # ── transcribe each VAD segment ──────────────────────────
        all_segs: list[dict] = []
        all_words: list[dict] = []
        chunk_errors: list[str] = []

        for vad_seg in speech_segs:
            if cancel_check and cancel_check():
                break

            start_samp = int(vad_seg["start"] * WHISPER_SR)
            end_samp = int(vad_seg["end"] * WHISPER_SR)
            chunk = audio_np[start_samp:end_samp]

            if len(chunk) < WHISPER_SR * 0.1:   # skip < 100 ms
                continue

            result = _transcribe_chunk(model, chunk, language, chunk_errors)
            if result is None:
                continue

            offset = vad_seg["start"]

            for seg in result.get("segments", []):
                all_segs.append({
                    "start": round(seg["start"] + offset, 2),
                    "end": round(seg["end"] + offset, 2),
                    "text": seg["text"].strip(),
                })
                for w in (seg.get("words") or []):
                    token = w.get("word", w.get("text", ""))
                    if not token:
                        continue
                    all_words.append({
                        "w": token,
                        "s": round(float(w["start"]) + offset, 3),
                        "e": round(float(w["end"]) + offset, 3),
                        "p": round(float(w.get("probability", 0.0)), 3),
                    })

            # per-VAD-segment progress (0..99 % within file)
            if file_progress_cb and duration > 0:
                pct = min(int(vad_seg["end"] / duration * 100), 99)
                file_progress_cb(i, pct)

        all_segs.sort(key=lambda s: s["start"])
        all_words.sort(key=lambda w: w["s"])

        segs = _finalize_segments(
            all_segs, all_words, source=str(p),
            diarization=diar,
            max_duration=max_duration, max_chars=max_chars,
            segment_cb=(lambda d, _i=i: segment_cb(_i, d)) if segment_cb else None,
        )

        if file_progress_cb:
            file_progress_cb(i, 100)

        entry = {
            "source": str(p),
            "language": language,
            "segments": segs,
            "words": all_words,
        }
        if chunk_errors:
            entry["chunk_errors"] = chunk_errors[:20]
        if diar:
            entry["diarization"] = diar
        all_results.append(entry)

        if progress_cb:
            progress_cb(i + 1, len(audio_paths))

    return all_results


def _transcribe_chunk(model, chunk, language: str,
                      errors: list[str]) -> dict | None:
    """Transcribe one VAD chunk with word timestamps and hallucination guards.

    Word timestamps use a cross-attention DTW pass that can fail on ROCm;
    on any such failure we retry once without them so the chunk still gets
    transcribed (the splitter then falls back to segment granularity).
    A failed chunk is recorded rather than silently dropped.
    """
    from ai_movie.config import (
        ASR_CONDITION_ON_PREVIOUS,
        ASR_INITIAL_PROMPT,
        ASR_WORD_TIMESTAMPS,
    )

    common = dict(
        language=language,
        verbose=False,
        condition_on_previous_text=ASR_CONDITION_ON_PREVIOUS,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    prompt = ASR_INITIAL_PROMPT.get(language)
    if prompt:
        common["initial_prompt"] = prompt

    if ASR_WORD_TIMESTAMPS:
        try:
            return model.transcribe(chunk, word_timestamps=True, **common)
        except Exception as exc:                        # noqa: BLE001
            errors.append(f"word_timestamps failed: {type(exc).__name__}: {exc}")
    try:
        return model.transcribe(chunk, **common)
    except Exception as exc:                            # noqa: BLE001
        errors.append(f"{type(exc).__name__}: {exc}")
        return None


def _finalize_segments(
    raw_segments: list[dict],
    words: list[dict],
    *,
    source: str,
    diarization: dict | None = None,
    max_duration: float | None = None,
    max_chars: int | None = None,
    segment_cb: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Re-split raw Whisper output into sentence-sized, single-speaker segments.

    Replaces the old "merge everything whose gap is <= 0.05 s" rule, which
    chained Whisper's contiguous segments into 44-second multi-speaker blobs.
    """
    from ai_movie import segmenter
    from ai_movie.config import ASR_MAX_SEGMENT_CHARS, ASR_MAX_SEGMENT_DURATION

    if max_duration is None:
        max_duration = ASR_MAX_SEGMENT_DURATION
    if max_chars is None:
        max_chars = ASR_MAX_SEGMENT_CHARS

    word_stream = words or segmenter.segments_to_words(raw_segments)
    turns = (diarization or {}).get("turns")
    speakers = (diarization or {}).get("speakers") or {}

    pieces = segmenter.split_into_sentences(
        word_stream,
        speaker_turns=turns,
        max_duration=max_duration,
        max_chars=max_chars,
    )

    out: list[dict] = []
    for piece in pieces:
        spk = piece.get("speaker") or ""
        gender = (speakers.get(spk) or {}).get("gender") if spk else None
        d = {
            "start": piece["start"],
            "end": piece["end"],
            "text": piece["text"],
            "source": source,
            "asr_conf": piece.get("asr_conf", 0.0),
        }
        if spk:
            d["speaker"] = spk
            d["speaker_conf"] = _turn_conf(turns, piece["start"], piece["end"])
        if gender:
            d["gender"] = gender
            # Back-compat alias — lip_sync.py / app.py still read tts_gender.
            d["tts_gender"] = gender
        out.append(d)
        if segment_cb:
            segment_cb(d)
    return out


def _turn_conf(turns: list[dict] | None, start: float, end: float) -> float:
    """Fraction of ``[start, end]`` covered by its dominant speaker turn."""
    if not turns:
        return 0.0
    span = max(1e-6, end - start)
    best = 0.0
    for t in turns:
        ov = min(end, float(t["end"])) - max(start, float(t["start"]))
        if ov > best:
            best = ov
    return round(max(0.0, min(1.0, best / span)), 2)


def _run_diarization(
    audio_path: Path,
    speech_segs: list[dict],
    *,
    num_speakers: int | None = None,
    vocals_path: str | Path | None = None,
    status_cb: Callable[[str], None] | None = None,
) -> dict | None:
    """Diarize *audio_path*, reusing the VAD spans already computed.

    Returns ``None`` (and logs) on any failure — transcription must not be
    lost because speaker clustering had a bad day.
    """
    try:
        from ai_movie import diarize as diarize_mod
    except Exception as exc:                            # noqa: BLE001
        print(f"[ASR] diarization unavailable: {exc}", file=sys.stderr)
        return None

    try:
        if status_cb:
            status_cb("说话人分离中…")
        return diarize_mod.diarize_file(
            audio_path,
            vocals_path=vocals_path,
            num_speakers=num_speakers,
            speech_spans=speech_segs,
            progress_cb=status_cb,
        )
    except Exception as exc:                            # noqa: BLE001
        print(f"[ASR] diarization failed ({type(exc).__name__}: {exc}) — "
              f"continuing without speaker labels", file=sys.stderr)
        return None


# ── public API ─────────────────────────────────────────────────

def transcribe_all(
    audio_paths: list[Path],
    language: str = "ja",
    model_size: str | None = None,
    backend: str = "auto",
    segment_cb: Callable[[int, dict], None] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    file_start_cb: Callable[[int, str], None] | None = None,
    file_progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    diarize: bool | None = None,
    num_speakers: int | None = None,
    vocals_path: str | Path | None = None,
    max_duration: float | None = None,
    max_chars: int | None = None,
    status_cb: Callable[[str], None] | None = None,
) -> list[dict]:
    """Transcribe audio files. Auto-selects best available backend.

    Parameters
    ----------
    backend:
        ``"auto"`` — auto-select (GPU → CPU fallback).
        ``"openai-whisper"`` — force openai-whisper GPU.
        ``"faster-whisper"`` — force faster-whisper CPU.
    segment_cb:
        Called from worker thread: ``segment_cb(file_idx, segment_dict)``
    progress_cb:
        ``progress_cb(current_file, total_files)``
    file_start_cb:
        ``file_start_cb(file_idx, filename)`` — called before processing each file
    file_progress_cb:
        ``file_progress_cb(file_idx, pct)`` — 0–100 % within the current file
    cancel_check:
        Return ``True`` to abort.
    diarize:
        Run speaker diarization and label every segment.  Defaults to
        ``config.ASR_DIARIZE``.
    num_speakers:
        Force the speaker count (``None`` = estimate automatically).
    vocals_path:
        Separated vocals track, used for cleaner speaker embeddings.
        Gender is always measured on the original audio.
    max_duration, max_chars:
        Segment caps for the sentence splitter (defaults from config).
    status_cb:
        ``status_cb(message)`` — coarse stage messages (diarization, etc).

    Returns
    -------
    list[dict] with ``source``, ``language``, ``segments``, ``words`` and
    (when diarization ran) ``diarization``.
    """
    if diarize is None:
        from ai_movie.config import ASR_DIARIZE
        diarize = ASR_DIARIZE

    if model_size is None:
        from ai_movie.config import ASR_MODEL_SIZE, ASR_OPENAI_WHISPER_MODEL
        cpu_model = ASR_MODEL_SIZE
        gpu_model = ASR_OPENAI_WHISPER_MODEL
    else:
        cpu_model = model_size
        gpu_model = model_size

    extra = dict(
        diarize=diarize, num_speakers=num_speakers, vocals_path=vocals_path,
        max_duration=max_duration, max_chars=max_chars, status_cb=status_cb,
    )

    # ── Linux / macOS ──────────────────────────────────────────────
    if sys.platform != "win32":
        # Force faster-whisper
        if backend == "faster-whisper":
            return _transcribe_cpu(
                audio_paths, language, cpu_model,
                segment_cb, progress_cb, file_start_cb,
                file_progress_cb, cancel_check, **extra,
            )

        # Force openai-whisper or auto
        if backend in ("openai-whisper", "auto"):
            try:
                import torch
                if torch.cuda.is_available():
                    return _transcribe_whisper_gpu(
                        audio_paths, language, gpu_model,
                        segment_cb, progress_cb, file_start_cb,
                        file_progress_cb, cancel_check, **extra,
                    )
                elif backend == "openai-whisper":
                    raise RuntimeError("GPU not available (torch.cuda.is_available() returned False)")
            except (ImportError, Exception) as e:
                if backend == "openai-whisper":
                    raise RuntimeError(f"openai-whisper backend failed: {e}") from e
                print(f"[ASR] GPU backend unavailable, falling back to CPU: {e}",
                      file=sys.stderr)

        # Fallback: faster-whisper CPU
        return _transcribe_cpu(
            audio_paths, language, cpu_model,
            segment_cb, progress_cb, file_start_cb,
            file_progress_cb, cancel_check, **extra,
        )

    # ── Windows: WSL+ROCm → DirectML GPU → CPU ─────────────────
    # 1. WSL + ROCm
    if wsl_rocm_available():
        try:
            return _transcribe_wsl(
                audio_paths, language, gpu_model,
                segment_cb, progress_cb, cancel_check,
            )
        except Exception as e:
            print(f"[ASR] WSL+ROCm backend failed, falling back: {e}",
                  file=sys.stderr)

    # 2. DirectML GPU
    if gpu_available():
        try:
            return _transcribe_gpu(
                audio_paths, language, gpu_model,
                segment_cb, progress_cb, cancel_check,
            )
        except Exception as e:
            print(f"[ASR] DirectML GPU backend failed, falling back: {e}",
                  file=sys.stderr)

    # 3. CPU (faster-whisper)
    return _transcribe_cpu(
        audio_paths, language, cpu_model,
        segment_cb, progress_cb, file_start_cb,
        file_progress_cb, cancel_check,
    )
