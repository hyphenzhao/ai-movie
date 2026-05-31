"""WSL+ROCm GPU-accelerated ASR bridge (runs inside WSL with PyTorch ROCm).

Protocol (stdin/stdout JSON-lines)
-----------------------------------
Same as asr_gpu.py.
Input:  {"audio_paths": [...], "language": "ja", "model_size": "large-v3"}
Output (streaming, one JSON object per line):
  {"type": "segment",    "source": "...", "start": 0.0, "end": 2.5, "text": "..."}
  {"type": "file_done",  "source": "...", "language": "ja"}
  {"type": "all_done",   "results": [...]}
  {"type": "error",      "message": "..."}
  {"type": "probe_result", "available": true/false, "device_name": "...", ...}
"""

import argparse
import json
import os
import sys
import traceback


def emit(obj: dict):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def probe_gpu() -> dict:
    """Check if ROCm GPU is available for PyTorch."""
    try:
        import torch
        available = torch.cuda.is_available()
        if not available:
            return {
                "available": False,
                "reason": "torch.cuda.is_available() returned False",
            }
        return {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "pytorch_version": torch.__version__,
        }
    except ImportError as e:
        return {"available": False, "reason": f"ImportError: {e}"}
    except Exception as e:
        return {"available": False, "reason": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true",
                       help="Check GPU availability and exit")
    args, _ = parser.parse_known_args()

    if os.environ.get("HSA_OVERRIDE_GFX_VERSION", "") == "":
        os.environ.setdefault(
            "HSA_OVERRIDE_GFX_VERSION", "11.0.0"
        )

    if args.probe:
        emit({"type": "probe_result", **probe_gpu()})
        return

    # Standard transcription mode
    try:
        job = json.loads(sys.stdin.readline())
    except Exception:
        emit({"type": "error", "message": "Invalid input JSON"})
        return

    audio_paths = job.get("audio_paths", [])
    language = job.get("language", "ja")
    model_size = job.get("model_size", "large-v3")

    try:
        import torch
        import whisper

        device = torch.device("cuda")
        model = whisper.load_model(model_size, device="cpu")
        model = model.to(device)
    except Exception as e:
        emit({"type": "error", "message": f"Model load failed: {e}"})
        return

    all_results = []
    for i, audio_path in enumerate(audio_paths):
        try:
            result = model.transcribe(
                audio_path, language=language, verbose=False
            )
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

        emit({
            "type": "file_done",
            "source": audio_path,
            "language": result.get("language", language),
        })

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
