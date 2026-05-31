"""Speech-to-text with automatic GPU / CPU backend selection.

Architecture (priority order)
-----------------------------
1. WSL + ROCm: launches ``wsl.exe python3 asr_wsl.py`` subprocess.
   Uses PyTorch ROCm + openai-whisper for AMD GPUs.

2. DirectML GPU: launches ``venv312/Scripts/python.exe asr_gpu.py`` subprocess.
   Uses torch-directml + openai-whisper.

3. CPU fallback: uses ``faster-whisper`` (CTranslate2, int8 quantized).

All backends communicate via stdin/stdout JSON-lines, streaming segments
in real time.  Call ``transcribe_all()`` — it auto-selects the best
available backend.
"""

import functools
import json
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
        Path(__file__).parent.parent / "venv312" / "Scripts" / "python.exe",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def gpu_available() -> bool:
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
    cancel_check: Callable[[], bool] | None,
) -> list[dict]:
    model = _load_cpu_model(model_size)
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


# ── public API ─────────────────────────────────────────────────

def transcribe_all(
    audio_paths: list[Path],
    language: str = "ja",
    model_size: str | None = None,
    segment_cb: Callable[[int, dict], None] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[dict]:
    """Transcribe audio files. Auto-selects best available backend.

    Priority: WSL+ROCm → DirectML GPU → CPU (faster-whisper)

    Parameters
    ----------
    segment_cb:
        Called from worker thread: ``segment_cb(file_idx, segment_dict)``
    progress_cb:
        ``progress_cb(current_file, total_files)``
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
        segment_cb, progress_cb, cancel_check,
    )
