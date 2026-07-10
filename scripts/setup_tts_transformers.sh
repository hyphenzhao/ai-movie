#!/usr/bin/env bash
# Rebuild the pinned transformers 4.51.3 used by the isolated CosyVoice2/3 TTS
# worker (ai_movie/tts_worker.py).
#
# Why: CosyVoice2/CosyVoice3 use a Qwen2 LLM that only decodes correctly under
# transformers==4.51.3.  The main app must run a newer transformers (5.x) to
# load the Hy-MT2 (hy_v3) translation model.  The TTS worker prepends this
# pinned copy to sys.path so the two versions never collide in one process.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
TARGET="$ROOT/vendor/tts_transformers"
mkdir -p "$TARGET"
"$PY" -m pip install --no-deps --target "$TARGET" \
    "transformers==4.51.3" "tokenizers>=0.21,<0.22" "huggingface-hub>=0.30.0,<1.0"
echo "Installed pinned transformers into $TARGET"
