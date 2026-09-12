#!/usr/bin/env bash
# Build the isolated CPU venv for overlapped-speech detection (pyannote).
#
# pyannote.audio pulls a dependency tree (pyannote.*, lightning extras,
# torch-audiomentations, opentelemetry, …) that must not touch the ROCm
# application venv, so it lives in vendor/osd_venv with a CPU-only torch.
# Usage:  bash scripts/setup_osd_env.sh [python]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/vendor/osd_venv"
PY="${1:-python3}"

if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip wheel >/dev/null

# CPU torch first, so pyannote's resolver does not pull a CUDA/ROCm build.
"$VENV/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch torchaudio

ok=0
for ver in "4.0.7" "3.3.2"; do
  echo "== trying pyannote.audio==$ver"
  if "$VENV/bin/pip" install --extra-index-url https://download.pytorch.org/whl/cpu "pyannote.audio==$ver"; then
    if "$VENV/bin/python" -c "import pyannote.audio, torch; print('pyannote.audio', pyannote.audio.__version__, 'torch', torch.__version__)"; then
      ok=1; break
    fi
  fi
done
if [ "$ok" != "1" ]; then
  echo "pyannote.audio could not be installed in $VENV" >&2
  exit 1
fi
echo "OSD venv ready: $VENV"
