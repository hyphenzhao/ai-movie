"""Speech-to-text with automatic GPU / CPU backend selection.

Architecture (priority order)
-----------------------------
**Linux / macOS:**
  1. openai-whisper + PyTorch GPU (ROCm / CUDA) — in-process
  2. faster-whisper (CTranslate2) CPU fallback, int8 quantized

**Windows:**
  1. WSL + ROCm: launches ``wsl.exe python3 asr_wsl.py`` subprocess
  2. DirectML GPU: launches subprocess with torch-directml + openai-whisper
  3. CPU fallback: faster-whisper (CTranslate2, int8 quantized)

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
) -> list[dict]:
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
            # Per-file progress from segment end timestamp
            if duration > 0 and file_progress_cb:
                pct = min(int(seg.end / duration * 100), 99)
                file_progress_cb(i, pct)

        if file_progress_cb:
            file_progress_cb(i, 100)

        all_results.append({
            "source": str(p),
            "language": info.language,
            "segments": segs,
        })

        if progress_cb:
            progress_cb(i + 1, len(audio_paths))

    return all_results


# ── openai-whisper GPU backend (Linux ROCm / CUDA) ───────────────

import re
import threading as _threading

# Regex to parse whisper verbose output timestamps:
#   [00:29.980 --> 00:30.000] 音楽
# Group 1: end time seconds (float)
_TS_RE = re.compile(r"\[[\d:.]+ -->\s*(\d+):(\d+\.\d+)]")


def _parse_end_seconds(line: str) -> float | None:
    """Return the end timestamp in seconds from a whisper verbose line."""
    m = _TS_RE.search(line)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    return None


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


def _transcribe_whisper_gpu(
    audio_paths: list[Path],
    language: str,
    model_size: str,
    segment_cb: Callable[[int, dict], None] | None,
    progress_cb: Callable[[int, int], None] | None,
    file_start_cb: Callable[[int, str], None] | None,
    file_progress_cb: Callable[[int, int], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> list[dict]:
    """Transcribe using openai-whisper on GPU (PyTorch ROCm/CUDA).

    Captures verbose stdout to parse timestamps and drive
    *file_progress_cb* with 0–100 % based on audio position.
    """
    import torch
    import whisper

    device = torch.device("cuda")
    model = whisper.load_model(model_size).to(device)

    all_results: list[dict] = []
    for i, p in enumerate(audio_paths):
        if cancel_check and cancel_check():
            break

        if file_start_cb:
            file_start_cb(i, p.name)

        duration = _get_audio_duration(p)

        # ── capture stdout for per-file progress (whisper verbose output) ──
        stdout_pipe_r, stdout_pipe_w = os.pipe()
        old_stdout = os.dup(1)
        os.dup2(stdout_pipe_w, 1)
        os.close(stdout_pipe_w)

        last_pct = [-1]

        def _reader():
            try:
                with os.fdopen(stdout_pipe_r, "r", encoding="utf-8", errors="replace") as rf:
                    for line in rf:
                        if duration > 0:
                            end_sec = _parse_end_seconds(line)
                            if end_sec is not None:
                                pct = min(int(end_sec / duration * 100), 99)
                                if pct > last_pct[0]:
                                    last_pct[0] = pct
                                    if file_progress_cb:
                                        try:
                                            file_progress_cb(i, pct)
                                        except Exception:
                                            pass
            except Exception:
                pass

        reader_thread = _threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        try:
            result = model.transcribe(str(p), language=language, verbose=True)
        except Exception as exc:
            all_results.append({"source": str(p), "error": str(exc)})
            if progress_cb:
                progress_cb(i + 1, len(audio_paths))
            os.dup2(old_stdout, 1)
            os.close(old_stdout)
            reader_thread.join(timeout=1)
            continue
        finally:
            os.dup2(old_stdout, 1)
            os.close(old_stdout)
            reader_thread.join(timeout=1)

        if file_progress_cb:
            file_progress_cb(i, 100)

        segs: list[dict] = []
        for seg in result["segments"]:
            d = {"start": round(seg["start"], 2),
                  "end": round(seg["end"], 2),
                  "text": seg["text"].strip(),
                  "source": str(p)}
            segs.append(d)
            if segment_cb:
                segment_cb(i, d)

        all_results.append({
            "source": str(p),
            "language": result.get("language", language),
            "segments": segs,
        })

        if progress_cb:
            progress_cb(i + 1, len(audio_paths))

    return all_results


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

    Returns
    -------
    list[dict] with ``source``, ``language``, ``segments``.
    """
    if model_size is None:
        from ai_movie.config import ASR_MODEL_SIZE, ASR_OPENAI_WHISPER_MODEL
        cpu_model = ASR_MODEL_SIZE
        gpu_model = ASR_OPENAI_WHISPER_MODEL
    else:
        cpu_model = model_size
        gpu_model = model_size

    # ── Linux / macOS ──────────────────────────────────────────────
    if sys.platform != "win32":
        # Force faster-whisper
        if backend == "faster-whisper":
            return _transcribe_cpu(
                audio_paths, language, cpu_model,
                segment_cb, progress_cb, file_start_cb,
                file_progress_cb, cancel_check,
            )

        # Force openai-whisper or auto
        if backend in ("openai-whisper", "auto"):
            try:
                import torch
                if torch.cuda.is_available():
                    return _transcribe_whisper_gpu(
                        audio_paths, language, gpu_model,
                        segment_cb, progress_cb, file_start_cb,
                        file_progress_cb, cancel_check,
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
            file_progress_cb, cancel_check,
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
