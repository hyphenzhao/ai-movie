"""GPU-accelerated ASR bridge (runs in Python 3.12 venv with torch-directml).

Protocol (stdin/stdout JSON-lines)
-----------------------------------
Input:  {"audio_paths": [...], "language": "ja", "model_size": "large-v3"}
Output (streaming, one JSON object per line):
  {"type": "segment",  "source": "...", "start": 0.0, "end": 2.5, "text": "..."}
  {"type": "file_done","source": "...", "language": "ja"}
  {"type": "all_done"}
  {"type": "error",   "message": "..."}
"""

import json
import os
import sys
import traceback

_MODEL_HINT = """
Whisper 模型未下载。请先下载到本地缓存。

方式 1 — 命令行（需要能访问 HuggingFace）:
  python -c "import whisper; whisper.load_model('large-v3')"

方式 2 — 用 HF 镜像:
  export HF_ENDPOINT=https://hf-mirror.com
  python -c "import whisper; whisper.load_model('large-v3')"

方式 3 — 手动下载后指定路径:
  修改 asr_gpu.py 中 MODEL_PATH 变量指向本地模型目录。
"""


def emit(obj: dict):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    # Encourage HF mirror for China accessibility
    if "HF_ENDPOINT" not in os.environ:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    try:
        job = json.loads(sys.stdin.readline())
    except Exception:
        emit({"type": "error", "message": "Invalid input JSON"})
        return

    audio_paths = job.get("audio_paths", [])
    language = job.get("language", "ja")
    model_size = job.get("model_size", "large-v3")

    # Load model (first run downloads ~3 GB)
    try:
        import sys
        if sys.platform == "win32":
            import torch_directml
            import whisper
            device = torch_directml.device()
        else:
            # Linux/macOS: use native PyTorch CUDA/ROCm or CPU
            import torch
            import whisper
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load to CPU first to avoid device typing issues
        model = whisper.load_model(model_size, device="cpu")
        model = model.to(device)
    except Exception as e:
        msg = str(e)
        if "does not appear to have a file" in msg or "No such file" in msg:
            msg += _MODEL_HINT
        emit({"type": "error", "message": msg})
        return

    all_results = []
    for i, audio_path in enumerate(audio_paths):
        try:
            result = model.transcribe(audio_path, language=language,
                                      verbose=False)
        except Exception as exc:
            emit({"type": "error", "message": str(exc),
                  "source": audio_path})
            all_results.append({"source": audio_path, "error": str(exc)})
            continue

        for seg in result["segments"]:
            emit({
                "type": "segment",
                "source": audio_path,
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip(),
            })

        emit({"type": "file_done",
              "source": audio_path,
              "language": result.get("language", language)})

        all_results.append({
            "source": audio_path,
            "language": result.get("language", language),
            "segments": [
                {"start": round(s["start"], 2),
                 "end": round(s["end"], 2),
                 "text": s["text"].strip()}
                for s in result["segments"]
            ],
        })

    emit({"type": "all_done", "results": all_results})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        emit({"type": "error", "message": traceback.format_exc()})
