#!/bin/bash
# Freeze what a release produced before the next version overwrites the shared workspace:
#
#   bash scripts/archive_release.sh v3.3 [film ...]      # default: the three shorts + every SONE-846 chunk
#
# Writes workspace/_archive_<ver>/<film>/{state.json,face_plan.json,verify_v*.json,refs_auto/,deliverables/,
# output/*.mp4 (finals only), baseline/{eval.json,ACCEPTANCE.md}} — the layout scripts/accept_release.py
# reads as --baseline — plus, for a long film, <ver>/<film>_eval_long.json, the concatenated deliverable
# and the split plan/status.  baseline/eval.json is recomputed here (CPU, ffprobe + loudness) so a
# release that was never run through accept_release still has comparable rows.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
VER="${1:?version tag, e.g. v3.3}"; shift || true
BASE="$ROOT/workspace/_archive_$VER"
if [ $# -gt 0 ]; then FILMS=("$@"); else
  FILMS=(output_test test_1 test_2)
  for d in "$ROOT"/workspace/SONE-846_p[0-9][0-9]; do [ -s "$d/state.json" ] && FILMS+=("$(basename "$d")"); done
fi
mkdir -p "$BASE"
for film in "${FILMS[@]}"; do
  W="$ROOT/workspace/$film"
  [ -s "$W/state.json" ] || { echo "$film: no state.json — skipped"; continue; }
  D="$BASE/$film"; mkdir -p "$D/baseline" "$D/output"
  cp -p "$W/state.json" "$D/"
  for f in face_plan.json verify_v1.json verify_v2.json run_v3.log; do [ -e "$W/$f" ] && cp -p "$W/$f" "$D/"; done
  for sub in refs_auto deliverables runs; do [ -d "$W/$sub" ] && rsync -a --exclude '*.wav' "$W/$sub/" "$D/$sub/"; done
  find "$W/output" -maxdepth 1 -name '*.mp4' -exec cp -p {} "$D/output/" \; 2>/dev/null || true
  $PY scripts/eval_pipeline.py "$W/state.json" --out "$D/baseline/ACCEPTANCE.md" --json "$D/baseline/eval.json" \
      > "$D/baseline/eval.log" 2>&1 || echo "$film: eval_pipeline reported failures (rows still written)"
  echo "$film: archived ($(du -sh "$D" | cut -f1))"
done
# long film: film-level truth for accept_release --long and the user's deliverable
for film in SONE-846; do
  [ -d "$ROOT/workspace/$film/_split" ] || continue
  mkdir -p "$BASE/${film}_split" "$BASE/$film"
  cp -p "$ROOT/workspace/$film/_split/"{plan.json,status.txt,events.jsonl} "$BASE/${film}_split/" 2>/dev/null || true
  [ -s "$ROOT/workspace/$film/profiles.json" ] && rsync -a "$ROOT/workspace/$film/profiles.json" "$ROOT/workspace/$film/profiles" "$BASE/$film/"
  [ -s "$ROOT/deliver/${film}_full/ACCEPTANCE_FULL.json" ] && cp -p "$ROOT/deliver/${film}_full/ACCEPTANCE_FULL.json" "$BASE/${film}_eval_long.json"
  [ -d "$ROOT/deliver/${film}_full" ] && rsync -a "$ROOT/deliver/${film}_full/" "$BASE/${film}_full/"
  [ -s "$ROOT/deliver/${film}_dubbed_full.mp4" ] && cp -p "$ROOT/deliver/${film}_dubbed_full.mp4" "$BASE/"
  echo "$film: film-level records archived"
done
echo "archive $VER: $(du -sh "$BASE" | cut -f1) in $BASE"
