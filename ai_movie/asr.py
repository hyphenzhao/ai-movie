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

        # Prefer the copy already in the hub cache: the default path asks GitHub on every load, and a
        # dropped connection silently turned VAD off (→ whole-file transcription) for that run.
        local = Path(torch.hub.get_dir()) / "snakers4_silero-vad_master"
        if (local / "hubconf.py").exists():
            model, utils = torch.hub.load(repo_or_dir=str(local), model="silero_vad", source="local")
        else:
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
    sweep: bool = False,
    alt_audio: str | Path | None = None,
    sweep_alt: str = "whisper",
    sweep_alt_source: str = "auto",
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

    *sweep_alt* / *sweep_alt_source* pick the sweep's second decoder and the
    audio it hears (config ASR_SWEEP_ALT_DECODER / _SOURCE); *alt_audio* is
    the "other" source's file.  anime-whisper is loaded lazily before the
    first file that sweeps and freed on the way out.

    Falls back to whole-file transcription if VAD is unavailable.
    """
    import torch
    import whisper

    from ai_movie.config import (
        ASR_ANIME_BATCH,
        ASR_VAD_MIN_SILENCE_DURATION_MS,
        ASR_VAD_MIN_SPEECH_DURATION_MS,
        ASR_VAD_SPEECH_PAD_MS,
        ASR_VAD_THRESHOLD,
    )

    device = torch.device("cuda")
    model = whisper.load_model(model_size).to(device)
    WHISPER_SR = whisper.audio.SAMPLE_RATE  # 16 000

    alt_mode = _resolve_alt_source(sweep_alt, sweep_alt_source)
    alt_decoder = None
    alt_load_errors: list[str] = []
    alt_tried = False

    all_results: list[dict] = []
    try:
        for i, p in enumerate(audio_paths):
            if cancel_check and cancel_check():
                break

            if file_start_cb:
                file_start_cb(i, p.name)

            duration = _get_audio_duration(p)

            # ── load audio & run VAD ─────────────────────────────────
            audio_np = whisper.load_audio(str(p))       # float32, 16 kHz
            audio_pt = torch.from_numpy(audio_np)
            alt_np = floor_audio = None
            if sweep:
                # the other source for cross-decoding; the separated vocals set the energy floor
                other = alt_audio or vocals_path
                need_other = alt_mode == "other" or bool(vocals_path and other and str(other) == str(vocals_path))
                other_np = whisper.load_audio(str(other)) if (need_other and other and Path(other).exists()) else None
                alt_np = _pick_alt_audio(audio_np, other_np, alt_mode)
                if vocals_path and Path(vocals_path).exists():
                    floor_audio = other_np if (other_np is not None and str(other) == str(vocals_path)) \
                        else whisper.load_audio(str(vocals_path))
                if sweep_alt != "whisper" and not alt_tried:
                    alt_tried = True                    # one load attempt per call, not per file
                    if status_cb:
                        status_cb(f"loading sweep alt decoder ({sweep_alt})…")
                    alt_decoder = _load_alt_decoder(sweep_alt, alt_load_errors)

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
            chunk_errors: list[str] = list(alt_load_errors)

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

                _collect(result, offset, "vad", all_segs, all_words)

                # per-VAD-segment progress (0..99 % within file)
                if file_progress_cb and duration > 0:
                    pct = min(int(vad_seg["end"] / duration * 100), 99)
                    file_progress_cb(i, pct)

            sweep_windows = None
            if sweep:
                sweep_errors: list[str] = []
                n_sw, n_txt, sweep_windows = _sweep_pass(
                    model, audio_np, speech_segs, language, all_segs, all_words,
                    sweep_errors, alt_audio=alt_np, floor_audio=floor_audio,
                    cancel_check=cancel_check, alt_decoder=alt_decoder)
                chunk_errors += sweep_errors
                if status_cb:
                    status_cb(f"sweep: {n_sw} windows, {n_txt} with text")

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
            if sweep:
                entry["sweep_windows"] = sweep_windows
                # for the record (not fingerprinted): which second decoder the file really got;
                # "fallback" = it gave up mid-file and large-v3 decoded the rest (see chunk_errors)
                used = alt_decoder.name if alt_decoder is not None else "whisper"
                entry["sweep_alt"] = {"name": used, "requested": sweep_alt, "source": alt_mode,
                                      "revision": getattr(alt_decoder, "revision", None),
                                      "dir": str(getattr(alt_decoder, "dir", "")) or None,
                                      "batch": ASR_ANIME_BATCH if used == "anime" else None,
                                      "fallback": bool(getattr(alt_decoder, "disabled", False)),
                                      "alt_audio": alt_np is not None}
            all_results.append(entry)

            if progress_cb:
                progress_cb(i + 1, len(audio_paths))
    finally:
        if alt_decoder is not None:
            alt_decoder.close()                             # one resident model at a time, next stages need the memory

    return all_results


_SR16 = 16000          # whisper.audio.SAMPLE_RATE; the model only ever sees 16 kHz mono


def _collect(result: dict, offset: float, which: str, all_segs: list[dict],
             all_words: list[dict], alt_text: str | None = None,
             alt_by: str | None = None) -> None:
    """Append one Whisper result to the raw segment / word streams.

    Whisper's per-segment scores (``no_speech_prob``, ``avg_logprob``,
    ``compression_ratio``) are kept on the segment *and* copied onto its
    words, because ``_finalize_segments`` rebuilds sentences from the word
    stream; ``segmenter._flush`` aggregates them back per sentence.

    Sweep evidence rides the same way: ``alt_text`` is the second decode of
    the window (``""`` = it decoded nothing, which is evidence too) and
    ``alt_by`` names the decoder ("whisper" | "anime").  ``alt_by`` without
    ``alt_text`` means that decoder failed on the window.  The content
    classifier reads ``seg["alt_by"]`` (phase 2 keys its rules on it).
    """
    from ai_movie.content import raw_hallucination
    segs_in = result.get("segments", [])
    # A Whisper loop repeats one sentence over consecutive segments ("お腹が空いたら…" ×5): every copy goes.
    folded = [raw_hallucination(sg.get("text", ""), fold_only=True) for sg in segs_in]
    looped = {t for t in set(folded) if t and folded.count(t) >= 3}
    for seg, ft in zip(segs_in, folded):
        words_ = seg.get("words") or []
        mean_p = (sum(float(w.get("probability", 0.0)) for w in words_) / len(words_)) if words_ else 0.0
        if ft in looped or raw_hallucination(seg.get("text", ""), pass_=which, mean_prob=mean_p):
            continue                        # stock phrase / loop: never enters the word stream
        extra = {"nsp": round(float(seg.get("no_speech_prob", 0.0)), 3),
                 "alp": round(float(seg.get("avg_logprob", 0.0)), 3),
                 "cr": round(float(seg.get("compression_ratio", 0.0)), 3),
                 "pass": which}
        if alt_text is not None:
            extra["alt"] = alt_text
        if alt_by is not None:
            extra["alt_by"] = alt_by
        all_segs.append({"start": round(seg["start"] + offset, 2),
                         "end": round(seg["end"] + offset, 2),
                         "text": seg["text"].strip(),
                         "no_speech_prob": extra["nsp"], "avg_logprob": extra["alp"],
                         "compression_ratio": extra["cr"], "pass": which,
                         **({"alt_text": alt_text} if alt_text is not None else {}),
                         **({"alt_by": alt_by} if alt_by is not None else {})})
        for w in (seg.get("words") or []):
            token = w.get("word", w.get("text", ""))
            if not token:
                continue
            all_words.append({"w": token,
                              "s": round(float(w["start"]) + offset, 3),
                              "e": round(float(w["end"]) + offset, 3),
                              "p": round(float(w.get("probability", 0.0)), 3),
                              **extra})


def _frame_db(y: "np.ndarray", sr: int = _SR16, frame_ms: int = 20) -> "np.ndarray":
    """RMS level per *frame_ms* frame in dBFS."""
    import numpy as np
    n = sr * frame_ms // 1000
    m = len(y) // n
    if m == 0:
        return np.full(1, -120.0, dtype=np.float32)
    rms = np.sqrt(np.mean(y[:m * n].reshape(m, n) ** 2, axis=1))
    return (20 * np.log10(rms + 1e-9)).astype(np.float32)


def _sweep_windows(speech_segs: list[dict], n_samples: int, energy_db: "np.ndarray", *,
                   min_gap: float = 1.0, max_win: float = 20.0, floor_db: float = -50.0,
                   pad: float = 0.3, sr: int = _SR16, frame_ms: int = 20) -> list[dict]:
    """Windows for the second pass: what the VAD spans left uncovered.

    Gaps ≥ *min_gap* s between spans (padded *pad* s into the neighbours),
    cut into ≤ *max_win* s pieces at the quietest 100 ms of the middle 40 %
    of each piece; a piece whose 95th-percentile level never reaches
    *floor_db* holds nothing to hear and is skipped.  Pure numpy.
    """
    import numpy as np
    total = n_samples / sr
    fps = 1000 / frame_ms
    gaps, prev = [], 0.0
    for sp in sorted(speech_segs, key=lambda d: d["start"]):
        a, b = float(sp["start"]), float(sp["end"])
        if a - prev >= min_gap:
            gaps.append((max(0.0, prev - pad), min(total, a + pad)))
        prev = max(prev, b)
    if total - prev >= min_gap:
        gaps.append((max(0.0, prev - pad), total))
    out = []
    for a, b in gaps:
        pieces = [(a, b)]
        while pieces:
            x, y = pieces.pop(0)
            if y - x > max_win:
                lo, hi = int((x + 0.3 * (y - x)) * fps), int((x + 0.7 * (y - x)) * fps)
                win = max(1, int(0.1 * fps))
                seg = energy_db[lo:hi]
                if len(seg) > win:
                    k = int(np.argmin(np.convolve(seg, np.ones(win) / win, mode="valid")))
                    cut = (lo + k + win / 2) / fps
                else:
                    cut = (x + y) / 2
                pieces = [(x, cut), (cut, y)] + pieces
                continue
            lv = energy_db[int(x * fps):max(int(x * fps) + 1, int(y * fps))]
            p95 = float(np.percentile(lv, 95)) if lv.size else -120.0
            if p95 < floor_db:
                continue
            out.append({"start": round(x, 3), "end": round(y, 3), "p95_db": round(p95, 1)})
    return out


def _transcribe_sweep(model, chunk, language: str, errors: list[str]) -> dict | None:
    """Decode a sweep window: low temperatures only, no confidence gating.

    Whisper's own ``no_speech`` / ``logprob`` gates are off because the
    decision is made by content afterwards (ai_movie.content); temperatures
    above 0.4 are where the stock phrases come from.
    """
    from ai_movie.config import ASR_SWEEP_TEMPERATURES
    common = dict(language=language, verbose=False, condition_on_previous_text=False,
                  temperature=tuple(ASR_SWEEP_TEMPERATURES), beam_size=5,
                  compression_ratio_threshold=2.4, logprob_threshold=None,
                  no_speech_threshold=None)
    try:
        return model.transcribe(chunk, word_timestamps=True, **common)
    except Exception as exc:                            # noqa: BLE001
        errors.append(f"sweep word_timestamps failed: {type(exc).__name__}: {exc}")
    try:
        return model.transcribe(chunk, **common)
    except Exception as exc:                            # noqa: BLE001
        errors.append(f"sweep: {type(exc).__name__}: {exc}")
        return None


# ── the sweep's second decoder ─────────────────────────────────
#
# v3.3 decoded every text window twice with large-v3 (mix and vocals) and let
# ai_movie.content compare the two readings.  litagin/anime-whisper is an
# alternative *second model*: same large-v3 encoder, a 2-layer decoder trained
# on visual-novel speech, so it transcribes moans/laughs/breaths as text
# instead of large-v3's stock closings and hallucinates less over silence
# (model card).  Its text is the only thing used — the checkpoint carries
# large-v3's alignment heads, which index decoder layers that do not exist,
# so word timestamps are impossible (``supports_word_timestamps``) and the
# timing stays with large-v3's DTW words.

_ALT_DECODERS = ("whisper", "anime")


def _resolve_alt_source(decoder: str, mode: str) -> str:
    """Which audio the second decoder hears: "same" (the primary) or "other" (mix ↔ vocals).

    "auto" keeps v3.3's rule for large-v3 — the other source *was* the second
    signal — and gives anime-whisper the same audio, since it is the second
    model.  Pure; fingerprinted through the effective value.
    """
    if mode in ("same", "other"):
        return mode
    return "same" if decoder == "anime" else "other"


def _pick_alt_audio(audio_np, other_np, mode: str):
    """The 16 kHz array the second decoder hears, or ``None`` (no second decode).

    "other" is accepted only when it lines up with the primary: at most 1 s
    shorter or longer and at least as long, then truncated to the primary's
    length (the v3.3 rule — a shorter track within tolerance still yields
    ``None``, so window slices never run past its end).
    """
    if mode == "same":
        return audio_np
    if other_np is None:
        return None
    if abs(len(other_np) - len(audio_np)) < _SR16 and len(other_np) >= len(audio_np):
        return other_np[:len(audio_np)]
    return None


class AnimeWhisper:
    """litagin/anime-whisper as the sweep's second decoder — text only, greedy, no prompt.

    Loads lazily (3 GB fp32 on disk → fp16 on the GPU, ≈ 1.5 GB) and must be
    ``close()``d by its owner; the pipeline keeps it resident only for the
    duration of one ``_transcribe_whisper_gpu`` call, next to large-v3.
    """
    name = "anime"

    def __init__(self, model_dir: str | Path | None = None, device: str = "cuda",
                 dtype: str | None = None, attn: str | None = None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        from ai_movie.config import ASR_ANIME_ATTN, ASR_ANIME_DTYPE, ASR_ANIME_WHISPER_DIR
        self.dir = Path(model_dir or ASR_ANIME_WHISPER_DIR)
        if not (self.dir / "model.safetensors").exists():
            raise FileNotFoundError(f"anime-whisper weights missing: {self.dir / 'model.safetensors'}")
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype or ASR_ANIME_DTYPE)
        self.proc = WhisperProcessor.from_pretrained(str(self.dir), local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(self.dir), dtype=self.dtype, local_files_only=True,
            attn_implementation=attn or ASR_ANIME_ATTN).to(self.device).eval()
        # the checkpoint ships large-v3's forced_decoder_ids; transformers ignores them once
        # language/task are passed, but dropping them keeps the deprecation path out entirely
        self.model.generation_config.forced_decoder_ids = None
        self.gen = self.generate_kwargs()
        self.revision = self.read_revision(self.dir)
        self.disabled = False

    @staticmethod
    def generate_kwargs(language: str = "ja") -> dict:
        """Greedy, text-only decode.  NEVER ``prompt_ids`` / ``initial_prompt`` (model card:
        prompts make this checkpoint hallucinate) and never timestamps (no usable heads)."""
        from ai_movie.config import ASR_ANIME_MAX_NEW_TOKENS, ASR_ANIME_NO_REPEAT_NGRAM
        return dict(language=language, task="transcribe", num_beams=1, do_sample=False,
                    no_repeat_ngram_size=ASR_ANIME_NO_REPEAT_NGRAM,
                    max_new_tokens=ASR_ANIME_MAX_NEW_TOKENS, return_timestamps=False)

    @staticmethod
    def supports_word_timestamps(gen_cfg: dict, decoder_layers: int) -> bool:
        """True only when every alignment head lives in a decoder layer that exists.

        anime-whisper's generation_config lists large-v3's heads ([7,0] … [25,6])
        over a 2-layer decoder: ``return_token_timestamps`` would index
        cross-attentions that are not there.  Guards anyone from turning it on.
        """
        heads = gen_cfg.get("alignment_heads") or []
        return bool(heads) and all(int(h[0]) < decoder_layers for h in heads)

    @staticmethod
    def read_revision(model_dir: Path) -> str:
        """The HF commit the weights came from, for the run record (hf download leaves it in
        ``.cache/huggingface/download/<file>.metadata``; a ``REVISION`` file also counts)."""
        for name in ("model.safetensors.metadata", "config.json.metadata"):
            p = model_dir / ".cache" / "huggingface" / "download" / name
            if p.exists():
                first = p.read_text(encoding="utf-8").splitlines()[:1]
                if first and first[0].strip():
                    return first[0].strip()
        p = model_dir / "REVISION"
        if p.exists():
            return p.read_text(encoding="utf-8").strip() or "local"
        return "local"

    def transcribe_batch(self, chunks: list) -> list[str]:
        """Text for each ≤ 30 s float32 16 kHz chunk (short-form: one generate call, no
        chunking pipeline — its 30 s stride machinery and timestamp paths are irrelevant)."""
        import torch
        feats = self.proc(list(chunks), sampling_rate=_SR16, return_tensors="pt").input_features
        feats = feats.to(self.device, self.dtype)
        with torch.inference_mode():
            ids = self.model.generate(feats, **self.gen)
        texts = self.proc.batch_decode(ids, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)   # WordPiece-only step, BPE here
        return [t.strip() for t in texts]

    def close(self) -> None:
        """Free the weights; MuseTalk/CosyVoice share the same unified memory later on."""
        model, self.model = getattr(self, "model", None), None
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                                   # noqa: BLE001
            pass


def _load_alt_decoder(name: str, errors: list[str]) -> "AnimeWhisper | None":
    """The second decoder for *name*, or ``None`` (= large-v3 on the other source).

    A load failure (weights not there yet, out of memory, transformers API
    drift) is recorded in *errors* and the file falls back to the v3.3 alt,
    so a chunk is never lost to the experiment.
    """
    if name == "anime":
        try:
            return AnimeWhisper()
        except Exception as exc:                            # noqa: BLE001
            errors.append(f"sweep alt (anime) unavailable, large-v3 alt used: {type(exc).__name__}: {exc}")
            return None
    if name not in _ALT_DECODERS:
        errors.append(f"sweep alt {name!r} unknown, large-v3 alt used")
    return None


def _sweep_alt_texts(model, alt_audio, wins: list[dict], language: str, errors: list[str], *,
                     alt_decoder=None, batch: int | None = None,
                     cancel_check=None) -> list[tuple[str | None, str | None]]:
    """Second decode of each window in *wins*: ``(alt_text, alt_by)`` per window.

    With *alt_decoder* the windows go through it in batches of *batch*
    (config ASR_ANIME_BATCH); a failing batch yields ``(None, name)`` — the
    piece then shows *who* failed — and after three failures large-v3 decodes
    the rest of the file (the decoder is marked ``disabled`` for its owner).
    Without one, large-v3 decodes the window on *alt_audio* as in v3.3
    (a failed decode reads as ``""``, as it always did).
    """
    n = len(wins)
    if alt_audio is None or n == 0:
        return [(None, None)] * n
    out: list[tuple[str | None, str | None]] = [(None, None)] * n

    def chunk(w):
        return alt_audio[int(w["start"] * _SR16):int(w["end"] * _SR16)]

    i = 0
    if alt_decoder is not None and not getattr(alt_decoder, "disabled", False):
        if batch is None:
            from ai_movie.config import ASR_ANIME_BATCH
            batch = ASR_ANIME_BATCH
        batch = max(1, int(batch))
        failures = 0
        while i < n:
            if cancel_check and cancel_check():
                return out
            idx = list(range(i, min(n, i + batch)))
            try:
                texts = alt_decoder.transcribe_batch([chunk(wins[j]) for j in idx])
                for j, t in zip(idx, texts):
                    out[j] = (t if t is not None else "", alt_decoder.name)
            except Exception as exc:                        # noqa: BLE001
                failures += 1
                errors.append(f"sweep alt ({alt_decoder.name}) windows {wins[idx[0]]['start']}–"
                              f"{wins[idx[-1]]['end']}: {type(exc).__name__}: {exc}")
                for j in idx:
                    out[j] = (None, alt_decoder.name)
            i = idx[-1] + 1
            if failures >= 3:
                errors.append(f"sweep alt ({alt_decoder.name}): {failures} failures, "
                              f"large-v3 decodes the remaining {n - i} window(s)")
                alt_decoder.disabled = True
                break
    for j in range(i, n):
        if cancel_check and cancel_check():
            break
        alt = _transcribe_sweep(model, chunk(wins[j]), language, errors)
        out[j] = ("".join(sg.get("text", "") for sg in (alt or {}).get("segments", [])).strip(), "whisper")
    return out


def _sweep_pass(model, audio_np, speech_segs, language, all_segs, all_words, errors, *,
                alt_audio=None, floor_audio=None, cancel_check=None,
                alt_decoder=None) -> tuple[int, int, list[dict]]:
    """Second pass over the VAD gaps; returns (windows decoded, windows with text, window records).

    *floor_audio* (the separated vocals, 16 kHz) drives the energy floor —
    the mix's music would pass every window; *alt_audio* is what the second
    decoder hears (``_pick_alt_audio``), decoded again for windows that
    produced text so the classifier can compare two readings.  The second
    decode is deferred to after the primary loop so *alt_decoder* can batch
    it (same result, one GPU call per ASR_ANIME_BATCH windows).
    Every window is recorded — ``{start, end, p95_db, text, alt_text,
    alt_by}``, ``text == ""`` when large-v3 heard nothing — and goes to
    ``state["asr"]["sweep_windows"]`` so scripts/ab_sweep_alt.py can replay
    the exact audio (and a later "rescue window" measurement has the silent
    ones too).
    """
    from ai_movie.config import (ASR_SWEEP_FLOOR_DBFS, ASR_SWEEP_MAX_WINDOW_S,
                                 ASR_SWEEP_MIN_GAP_S)
    energy = _frame_db(floor_audio if floor_audio is not None else audio_np)
    wins = _sweep_windows(speech_segs, len(audio_np), energy, min_gap=ASR_SWEEP_MIN_GAP_S,
                          max_win=ASR_SWEEP_MAX_WINDOW_S, floor_db=ASR_SWEEP_FLOOR_DBFS)
    decoded: list[tuple[dict, dict | None]] = []          # (window, primary result | None)
    for w in wins:
        if cancel_check and cancel_check():
            break
        a, b = int(w["start"] * _SR16), int(w["end"] * _SR16)
        res = _transcribe_sweep(model, audio_np[a:b], language, errors)
        if not res or not any(sg.get("text", "").strip() for sg in res.get("segments", [])):
            res = None
        decoded.append((w, res))
    heard_idx = [k for k, (_, res) in enumerate(decoded) if res is not None]
    alts = dict(zip(heard_idx, _sweep_alt_texts(model, alt_audio, [decoded[k][0] for k in heard_idx],
                                                language, errors, alt_decoder=alt_decoder,
                                                cancel_check=cancel_check)))
    spans = [(float(v["start"]), float(v["end"])) for v in speech_segs]

    def inside(t):
        return any(a <= t <= b for a, b in spans)
    records = []
    for k, (w, res) in enumerate(decoded):
        rec = {"start": w["start"], "end": w["end"], "p95_db": w.get("p95_db"),
               "text": "", "alt_text": None, "alt_by": None}
        if res is not None:
            alt_text, alt_by = alts[k]
            tmp_s, tmp_w = [], []
            _collect(res, w["start"], "sweep", tmp_s, tmp_w, alt_text=alt_text, alt_by=alt_by)
            all_words += [x for x in tmp_w if not inside((x["s"] + x["e"]) / 2)]  # the VAD pass owns those
            all_segs += [x for x in tmp_s if not inside((x["start"] + x["end"]) / 2)]
            rec.update(text="".join(sg.get("text", "") for sg in res.get("segments", [])).strip(),
                       alt_text=alt_text, alt_by=alt_by)
        records.append(rec)
    return len(wins), len(heard_idx), records


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
    overlap_regions = (diarization or {}).get("overlap_regions") or []

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
        if overlap_regions:
            from ai_movie.osd import overlap_ratio
            d["overlap"] = round(overlap_ratio(overlap_regions, piece["start"],
                                               piece["end"]), 3)
        for k in ("no_speech_prob", "avg_logprob", "compression_ratio", "pass", "alt_text", "alt_by"):
            if piece.get(k) is not None:
                d[k] = piece[k]                          # Whisper scores + sweep evidence for content.classify
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
    sweep: bool | None = None,
    alt_audio: str | Path | None = None,
    sweep_alt: str | None = None,
    sweep_alt_source: str | None = None,
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
    sweep, alt_audio, sweep_alt, sweep_alt_source:
        Second ASR pass over the VAD gaps (GPU path only): the "other"
        source's file, which decoder re-reads each text window ("whisper" |
        "anime") and what it hears ("auto" | "same" | "other").  Defaults
        from config ASR_SWEEP_*.

    Returns
    -------
    list[dict] with ``source``, ``language``, ``segments``, ``words`` and
    (when diarization ran) ``diarization``; with the sweep also
    ``sweep_windows`` and ``sweep_alt``.
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
    if sweep is None:
        from ai_movie.config import ASR_SWEEP_ENABLED
        sweep = ASR_SWEEP_ENABLED
    if sweep_alt is None or sweep_alt_source is None:
        from ai_movie.config import ASR_SWEEP_ALT_DECODER, ASR_SWEEP_ALT_SOURCE
        sweep_alt = sweep_alt or ASR_SWEEP_ALT_DECODER
        sweep_alt_source = sweep_alt_source or ASR_SWEEP_ALT_SOURCE
    gpu_extra = dict(extra, sweep=bool(sweep), alt_audio=alt_audio,          # the CPU path has no sweep
                     sweep_alt=sweep_alt, sweep_alt_source=sweep_alt_source)

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
                        file_progress_cb, cancel_check, **gpu_extra,
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
