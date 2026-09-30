#!/usr/bin/env bash
# Long film, unattended: split at dialogue pauses → run_v3.sh per chunk → lossless concat.
#
#   bash scripts/run_long.sh inputs/FILM.mp4 NAME
#
# - scripts/smart_split.py plans the cuts (VAD gaps snapped to keyframes) and stream-copies the chunks.
# - A chunk with no speech (or whose pipeline run fails) is passed through: original picture re-encoded
#   once, original sound shifted by the dubbed chunks' median loudness offset so the joins do not jump.
# - Every chunk is resumable: a finished one is skipped, run_pipeline's own stage cache covers the rest.
# - Final: workspace/NAME/_split/ → deliver/NAME_dubbed_full.mp4 (+ deliver/NAME_full/ tables).
# Nothing is uploaded; nothing is downloaded (HF_HUB_OFFLINE).
set -uo pipefail
VIDEO="${1:?usage: run_long.sh <video> <name>}"
NAME="${2:?usage: run_long.sh <video> <name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
SPLIT="$ROOT/workspace/$NAME/_split"
mkdir -p "$SPLIT" "$ROOT/deliver/${NAME}_full"
LOG="$SPLIT/run_long.log"
exec > >(tee -a "$LOG") 2>&1
export HF_HUB_OFFLINE=1
STALL_MIN="${STALL_MIN:-40}"                        # minutes without log output = hung
MIN_SPEECH_MIN="${MIN_SPEECH_MIN:-0.03}"          # < ~2 s of detected speech → nothing to dub
RERUN="${RERUN:-}"                                # tag (e.g. v3.4): push every dub chunk through run_v3.sh again after a code
                                                  # change; the stage cache decides what really re-runs, and a chunk already
                                                  # re-run under this tag (status.txt "rerun=<tag>") is skipped on a restart

say() { echo; echo "##### [$(date '+%m-%d %H:%M:%S')] $NAME · $* #####"; }
# Film-level event log (JSON lines, with a host snapshot): what started, what ended, with which result.
ev() { $PY -m ai_movie.runlog "$SPLIT/events.jsonl" "$@" 2>/dev/null || true; }
ev film_start video="$VIDEO" pid=$$
T0=$(date +%s)

say "1: plan + cut"
$PY scripts/smart_split.py "$VIDEO" --name "$NAME" --cut || { echo "LONG_FAILED: split"; exit 1; }

say "2a: enrolment — dialogue-dense chunks first, then film-wide speaker profiles"
ENROL=$($PY - "$SPLIT/plan.json" <<'PYEOF'
import json, sys
sys.path.insert(0, ".")
from ai_movie.config import PROFILE_ENROL_DENSITY
doc = json.load(open(sys.argv[1]))
dense = [c["index"] for c in doc["chunks"] if c["minutes"] and c["speech_minutes"] / c["minutes"] >= PROFILE_ENROL_DENSITY]
top = [c["index"] for c in sorted(doc["chunks"], key=lambda c: -c["speech_minutes"])[:2]]
print(" ".join(f"{i:02d}" for i in sorted(set(dense + top))))
PYEOF
)
echo "enrolment chunks: $ENROL"
for IDX in $ENROL; do
  CN="${NAME}_p${IDX}"
  ln -sfn "$SPLIT/chunks/$CN.mp4" "$ROOT/inputs/$CN.mp4"
  # through tts so build_profiles can try each candidate reference on this person's real lines;
  # PIPE_ARGS as in run_v3.sh, or an option such as --sweep-alt gives these chunks a different asr
  # fingerprint from their later pass and they transcribe twice
  # shellcheck disable=SC2086
  $PY -u scripts/run_pipeline.py "$ROOT/inputs/$CN.mp4" --name "$CN" --steps demux,separate,osd,asr,glossary,translate,tts ${PIPE_ARGS:-} < /dev/null \
    || { echo "enrol: $CN asr failed"; echo "$IDX enrol asr_failed" >> "$SPLIT/status.txt"; }
done
if [ ! -s "$ROOT/workspace/$NAME/profiles.json" ] || [ "${REBUILD_PROFILES:-0}" = 1 ]; then
  $PY -u scripts/build_profiles.py "$NAME" < /dev/null && echo "$NAME enrol ok" >> "$SPLIT/status.txt" \
    || { echo "build_profiles failed — chunks fall back to per-chunk references"; echo "$NAME enrol failed" >> "$SPLIT/status.txt"; }
fi
[ -s "$ROOT/workspace/$NAME/profiles.json" ] && export PROFILES="$ROOT/workspace/$NAME/profiles.json"

say "2b: chunks"
$PY - "$SPLIT/plan.json" "$MIN_SPEECH_MIN" "$ENROL" > "$SPLIT/chunks.tsv" <<'PYEOF'
import json, sys
doc = json.load(open(sys.argv[1])); lim = float(sys.argv[2])
enrol = set(int(x) for x in (sys.argv[3].split() if len(sys.argv) > 3 else []))
for c in sorted(doc["chunks"], key=lambda c: (c["index"] not in enrol, c["index"])):
    print(f"{c['index']:02d}\t{c['file']}\t{'dub' if c['speech_minutes'] >= lim else 'pass'}\t{c['speech_minutes']}")
PYEOF
while IFS=$'\t' read -r IDX FILE MODE SPEECH; do
  CN="${NAME}_p${IDX}"
  ln -sfn "$FILE" "$ROOT/inputs/$CN.mp4"
  FINAL="$ROOT/workspace/$CN/output/v2_cloned_dubbed.mp4"
  if [ "$MODE" = "pass" ]; then echo "p$IDX: no speech ($SPEECH min) → passthrough"; echo "$IDX pass nospeech" >> "$SPLIT/status.txt"; continue; fi
  if [ -n "$RERUN" ]; then
    if grep -q "^$IDX dub .*rerun=$RERUN$" "$SPLIT/status.txt" 2>/dev/null; then echo "p$IDX: already re-run ($RERUN)"; continue; fi
  elif { [ -s "$FINAL" ] || grep -q "^$IDX dub v1_only" "$SPLIT/status.txt" 2>/dev/null; } && grep -q "DONE in" "$ROOT/workspace/$CN/run_v3.log" 2>/dev/null; then echo "p$IDX: already done"; continue; fi
  say "chunk p$IDX (speech $SPEECH min)"
  C0=$(date +%s)
  ev chunk_start chunk="$CN" speech_min="$SPEECH"
  # Watchdog: on 09-19 chunk 3 sat silently at "ASR: transcribing" for 33 h (no kernel/GPU error, machine
  # alive).  A chunk whose log stops growing for STALL_MIN minutes is killed and retried; the stage cache
  # makes the retry start where it stopped.
  RC=1
  for TRY in 1 2 3; do
    setsid bash scripts/run_v3.sh "$CN" < /dev/null &
    PID=$!
    while kill -0 "$PID" 2>/dev/null; do
      sleep 60
      AGE=$(( $(date +%s) - $(stat -c %Y "$ROOT/workspace/$CN/run_v3.log" 2>/dev/null || date +%s) ))
      if [ "$AGE" -gt $(( STALL_MIN * 60 )) ]; then
        echo "p$IDX: no log output for $((AGE/60)) min → killing try $TRY"
        kill -TERM -- "-$PID" 2>/dev/null; sleep 10; kill -KILL -- "-$PID" 2>/dev/null
        echo "$IDX stall try$TRY $(date '+%m-%d %H:%M')" >> "$SPLIT/status.txt"
        ev chunk_stall chunk="$CN" try="$TRY" silent_min="$((AGE/60))"
        break
      fi
    done
    wait "$PID"; RC=$?
    [ $RC -eq 0 ] && [ -s "$FINAL" ] && break
    if [ $RC -eq 0 ] && grep -q "DONE in" "$ROOT/workspace/$CN/run_v3.log" 2>/dev/null; then
      break                     # ran clean; no cloned version = no usable reference (built-in voice stays)
    fi
    [ "$TRY" -lt 3 ] && echo "p$IDX: try $TRY ended rc=$RC, retrying"
  done
  ev chunk_end chunk="$CN" rc="$RC" minutes="$(( ($(date +%s)-C0)/60 ))" final="$([ -s "$FINAL" ] && echo yes || echo no)" run="$(readlink "$ROOT/workspace/$CN/runs/latest" 2>/dev/null)"
  if [ $RC -eq 0 ] && [ -s "$FINAL" ]; then echo "$IDX dub ok $(( ($(date +%s)-C0)/60 ))min${RERUN:+ rerun=$RERUN}" >> "$SPLIT/status.txt"
  elif [ $RC -eq 0 ] && [ -s "$ROOT/workspace/$CN/output/${CN}_dubbed.mp4" ]; then
    echo "p$IDX: no cloned version → the built-in-voice dub is used for this chunk"; echo "$IDX dub v1_only${RERUN:+ rerun=$RERUN}" >> "$SPLIT/status.txt"
  else echo "p$IDX: pipeline rc=$RC → passthrough for this chunk"; echo "$IDX pass failed_rc$RC" >> "$SPLIT/status.txt"; fi
done < "$SPLIT/chunks.tsv"

say "3: passthrough chunks + concat"
# scripts/concat_long.py holds what used to be a heredoc here, so a VC-only rerun (scripts/run_vc_only.sh)
# can re-assemble the film without re-entering the chunk loop.
$PY scripts/concat_long.py "$NAME" "$SPLIT" || { echo "LONG_FAILED: concat"; exit 1; }
cp -f "$SPLIT/status.txt" "$ROOT/deliver/${NAME}_full/chunk_status.txt" 2>/dev/null
# one voice per profile across chunks (leave-one-chunk-out, CPU) → deliver/<film>_full/VOICE_CONSISTENCY.* for eval_long L3b–f
$PY -u scripts/voice_consistency.py --film "$NAME" ${PROFILES:+--profiles "$PROFILES"} || echo "(voice consistency failed — kept going)"
ev film_end minutes="$(( ($(date +%s) - T0) / 60 ))"
say "LONG_DONE in $(( ($(date +%s) - T0) / 60 )) min"
