#!/usr/bin/env bash
# Re-render the picture side of a finished project after a lip-sync change:
# face plan (yaw gate) → MuseTalk → CodeFormer enhance → compose, then the
# v2 (cloned-voice) build, verification, demo clips and deliverables.
#
#   bash scripts/rerender_lipsync.sh test_2 <ref_female|-> <ref_male|->
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
NAME="$1"; REF_F="${2:--}"; REF_M="${3:--}"
stage() { echo "### [$(date +%H:%M:%S)] rerender($NAME) :: $*"; }

stage "faces + lipsync + enhance + compose"
$PY -u scripts/run_pipeline.py "inputs/$NAME.mp4" --name "$NAME" \
    --steps faces,lipsync,enhance,compose --force || exit 1

stage "v2 + verify + demos + assemble"
bash scripts/finish_video.sh "$NAME" "$REF_F" "$REF_M" || exit 1
stage "DONE"
