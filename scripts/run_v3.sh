#!/usr/bin/env bash
# v3 end-to-end for one film, fully automatic (no human speaker review):
#   demux → separate → osd → asr → glossary → translate → tts → compact → fit
#   → mix → faces → lipsync → enhance → compose → qc            (v1, built-in voices)
#   → auto_select_refs (F0 gate + overlap filter) → run_vc_version (v2, cloned)
#   → qc (incl. v2) → verify → eval → deliver (v2 final + demo folder)
#
#   bash scripts/run_v3.sh <name> [--upload]
# Expects inputs/<name>.mp4.  Logs to workspace/<name>/run_v3.log.
set -uo pipefail
NAME="${1:?usage: run_v3.sh <name> [--upload]}"
UPLOAD="${2:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
VIDEO="$ROOT/inputs/$NAME.mp4"
WORK="$ROOT/workspace/$NAME"
STATE="$WORK/state.json"
mkdir -p "$WORK"
LOG="$WORK/run_v3.log"
exec > >(tee -a "$LOG") 2>&1

stage() { echo; echo "===== [$(date +%H:%M:%S)] $NAME · $* ====="; }
die() { echo "FAILED: $*" >&2; exit 1; }
T0=$(date +%s)

stage "A: v1 pipeline (all stages)"
$PY -u scripts/run_pipeline.py "$VIDEO" --name "$NAME" --voice-mode sft || die "pipeline"

stage "B: auto-select VC references (F0 gate)"
$PY -u scripts/auto_select_refs.py "$STATE" || die "ref selection"

stage "C: v2 (voice conversion)"
$PY -u scripts/run_vc_version.py "$STATE" --refs-json "$WORK/refs_auto/refs.json" || die "vc version"

stage "D: QC (v1 + v2), verification, acceptance"
$PY -u scripts/run_pipeline.py "$VIDEO" --name "$NAME" --steps qc --force || die "qc"
$PY -u scripts/verify_dub.py "$STATE" --key fit -n 20 --out "$WORK/verify_v1.json" || echo "(verify v1 failed — kept going)"
$PY -u scripts/verify_dub.py "$STATE" --key vc --audio-field audio_fit -n 20 --out "$WORK/verify_v2.json" || echo "(verify v2 failed — kept going)"
$PY -u scripts/eval_pipeline.py "$STATE" || echo "(eval reported failures — kept going)"

stage "E: deliverable (v2 final + demo folder)"
if [ "$UPLOAD" = "--upload" ]; then
  $PY -u scripts/deliver.py "$STATE" --out "$ROOT/deliver" --version vc --upload || die "deliver/upload"
else
  $PY -u scripts/deliver.py "$STATE" --out "$ROOT/deliver" --version vc || die "deliver"
fi
$PY -u scripts/export_project.py "$STATE" >/dev/null || true

stage "DONE in $(( ($(date +%s) - T0) / 60 )) min"
