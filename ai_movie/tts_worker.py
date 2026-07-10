"""Isolated CosyVoice synthesis worker (subprocess entry point).

Why a subprocess?  CosyVoice2 / CosyVoice3 use a Qwen2-based LLM that only
produces correct speech tokens under ``transformers==4.51.3``.  The main
application, however, must run a *newer* transformers (5.x) so it can load
the ``hy_v3`` Hunyuan-MT2 translation model.  The two requirements are
mutually exclusive inside one Python process, so all Qwen-based CosyVoice
synthesis is delegated here, where a pinned copy of transformers 4.51.3
(``vendor/tts_transformers``) is prepended to ``sys.path`` *before*
transformers is imported.

Running in a dedicated process also means synthesis executes on this
process's own main thread, satisfying CosyVoice's main-thread requirement
without any GUI-thread gymnastics on the caller side.

Protocol
--------
argv[1] : path to a JSON job file with keys::

    {
      "model_dir":  "<CosyVoice2/3 model dir>",
      "ref_audio":  "<reference wav path or speaker id>",
      "ref_text":   "<prompt/instruct text or null>",
      "method":     "instruct2" | "zero_shot" | "cross_lingual",
      "output_dir": "<dir for seg_XXXX.wav files>",
      "fp16":       true,
      "segments":   [{"index": 0, "text": "..."}, ...]
    }

Emits one JSON object per line on stdout::

    {"ev": "progress", "done": <int>, "total": <int>}
    {"ev": "result", "items": {"<index>": {"audio": "<path>"} | {"error": "<msg>"}}}
    {"ev": "fatal", "error": "<msg>"}
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PINNED_TF = _ROOT / "vendor" / "tts_transformers"


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    job_path = sys.argv[1]
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)

    # Pin transformers 4.51.3 (+ matching tokenizers / huggingface-hub) ahead
    # of the app venv's newer transformers.  MUST happen before any import
    # that pulls in transformers (i.e. before cosyvoice).  Missing pins would
    # silently fall back to the app's transformers 5.x, which decodes the Qwen
    # LLM into garbled audio — so fail loudly with rebuild instructions.
    if not (_PINNED_TF / "transformers").exists():
        _emit({"ev": "fatal", "error":
               f"pinned transformers not found at {_PINNED_TF}. "
               f"Rebuild it: bash scripts/setup_tts_transformers.sh"})
        return 1
    sys.path.insert(0, str(_PINNED_TF))

    # Make the app package importable so we can reuse call_tts().
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    try:
        import numpy as np
        import soundfile as sf
        import transformers  # noqa: F401  (import to surface version in logs)
        sys.stderr.write(f"[tts_worker] transformers={transformers.__version__}\n")

        _cv = _ROOT / "models" / "CosyVoice"
        _matcha = _cv / "third_party" / "Matcha-TTS"
        if str(_cv) not in sys.path:
            sys.path.insert(0, str(_cv))
        if str(_matcha) not in sys.path:
            sys.path.insert(0, str(_matcha))
        from cosyvoice.cli.cosyvoice import AutoModel

        from ai_movie.tts import call_tts

        model = AutoModel(model_dir=job["model_dir"], fp16=bool(job.get("fp16", True)))
    except Exception as exc:  # model / import failure is fatal for the whole job
        _emit({"ev": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        return 1

    ref_audio = job["ref_audio"]
    ref_text = job.get("ref_text")
    method = job["method"]
    out_dir = Path(job["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    segments = job["segments"]
    total = len(segments)

    def _n_hanzi(s: str) -> int:
        return sum(1 for ch in s if "一" <= ch <= "鿿")

    def _synth_once(text: str, r_audio, r_text, r_method) -> np.ndarray:
        return call_tts(model, text, r_audio, r_text, r_method)

    def _reseed(seed: int) -> None:
        import random
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def _synth_clean(text: str, base_seed: int,
                     r_audio, r_text, r_method) -> np.ndarray:
        """CosyVoice2/3 zero_shot occasionally collapses to a near-empty
        (~0.04 s) output for some RNG states.  Detect that by duration and
        retry with *different* seeds (the yaml fixes a seed at load, so
        same-seed retries are identical), keeping the LONGEST take.  Stops
        early once a take reaches a plausible length."""
        n = _n_hanzi(text)
        min_dur = n * 0.10 + 0.3   # shorter than this ⇒ a failed/empty take
        best = None
        for k in range(4):
            _reseed(base_seed + k * 7919)
            a = _synth_once(text, r_audio, r_text, r_method)
            if best is None or len(a) > len(best):
                best = a
            if len(a) / model.sample_rate >= min_dur:
                break
        return best               # longest (most complete) take

    items: dict[str, dict] = {}
    for done, seg in enumerate(segments, start=1):
        idx = seg["index"]
        text = (seg.get("text") or "").strip()
        if not text:
            items[str(idx)] = {"audio": None}
            _emit({"ev": "progress", "done": done, "total": total})
            continue
        # Optional per-segment reference override (e.g. gender routing:
        # female → soft/Taiwanese ref, male → Mandarin male ref). Falls back
        # to the job-global reference when the segment carries no override.
        if "ref_audio" in seg:
            s_audio, s_text, s_method = seg.get("ref_audio"), seg.get("ref_text"), seg.get("method")
        else:
            s_audio, s_text, s_method = ref_audio, ref_text, method
        try:
            audio_np = _synth_clean(text, 1986 + idx * 131, s_audio, s_text, s_method)
            out_path = str(out_dir / f"seg_{idx + 1:04d}.wav")
            sf.write(out_path, audio_np, model.sample_rate)
            items[str(idx)] = {"audio": out_path}
        except Exception as exc:
            items[str(idx)] = {"audio": None, "error": f"{type(exc).__name__}: {exc}"}
        _emit({"ev": "progress", "done": done, "total": total})

    _emit({"ev": "result", "items": items})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
