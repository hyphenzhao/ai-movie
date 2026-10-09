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
# Outputs use deliver/. HF_HUB_OFFLINE can be set explicitly when needed.
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
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
STALL_MIN="${STALL_MIN:-40}"                        # minutes without log output = hung
MIN_SPEECH_MIN="${MIN_SPEECH_MIN:-0.03}"          # < ~2 s of detected speech → nothing to dub
RERUN="${RERUN:-}"                                # tag (e.g. v3.4): push every dub chunk through run_v3.sh again after a code
                                                  # change; the stage cache decides what really re-runs, and a chunk already
                                                  # re-run under this tag (status.txt "rerun=<tag>") is skipped on a restart
STRICT_RUN="${STRICT_RUN:-0}"                    # release runs must not concat a failed chunk's old output
V34_RUN_DIR="${V34_RUN_DIR:-}"                    # evidence-backed resume; a text status tag is insufficient
PROFILE_PATH="$ROOT/workspace/$NAME/profiles.json"

say() { echo; echo "##### [$(date '+%m-%d %H:%M:%S')] $NAME · $* #####"; }
die() { echo "LONG_FAILED: $*" >&2; exit 1; }
current_output() { "$PY" scripts/run_evidence.py output "$1"; }
# Return 3 only when there is no checkpoint yet.  A mismatching checkpoint is
# an error, never permission to rebuild voices beneath already completed chunks.
profile_checkpoint() {
  "$PY" - "$1" "$V34_RUN_DIR" "$NAME" "$PROFILE_PATH" "${V34_CODE_HASH:-}" <<'PYEOF'
import hashlib, json, os, sys, tempfile
from pathlib import Path
action, run_dir, name, profile_name, code_hash = sys.argv[1:]
run = Path(run_dir)
if not run.is_dir() or not code_hash:
    raise SystemExit("profile checkpoint needs an existing V34_RUN_DIR and V34_CODE_HASH")
marker = run / "profile_checkpoints" / (name + ".json")
if action == "check" and not marker.exists():
    raise SystemExit(3)
try:
    profile = Path(profile_name)
    raw = profile.read_bytes()
    if not raw or not isinstance(json.loads(raw), dict):
        raise ValueError("profiles.json is empty or invalid")
    expected = {"version": 1, "film": name, "code_hash": code_hash,
                "profiles": str(profile.absolute()), "sha256": hashlib.sha256(raw).hexdigest()}
    if action == "check":
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError("profile checkpoint does not match current code/profiles; start a new run")
    elif action == "record":
        if marker.exists():
            raise ValueError("refusing to overwrite an existing profile checkpoint")
        marker.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=marker.name + ".", suffix=".tmp", dir=marker.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(expected, fh, ensure_ascii=False, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(temporary, marker)
        finally:
            Path(temporary).unlink(missing_ok=True)
    else:
        raise ValueError("unknown checkpoint action")
except (OSError, ValueError, TypeError) as exc:
    raise SystemExit(str(exc))
PYEOF
}
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
) || die "enrolment plan"
echo "enrolment chunks: $ENROL"
PROFILE_RESUMED=0
if [ -n "$V34_RUN_DIR" ]; then
  if profile_checkpoint check; then
    PROFILE_RESUMED=1
    echo "profiles: verified checkpoint, reusing enrolment and references"
  else
    MARKER_RC=$?
    [ "$MARKER_RC" -eq 3 ] || die "profile checkpoint mismatch"
  fi
fi
PROFILE_READY=1
if [ "$PROFILE_RESUMED" = 0 ]; then
  for IDX in $ENROL; do
    CN="${NAME}_p${IDX}"
    ln -sfn "$SPLIT/chunks/$CN.mp4" "$ROOT/inputs/$CN.mp4"
    # Through tts so build_profiles can test references on this person's real lines.
    # PIPE_ARGS must match the later chunk pass to avoid transcribing twice.
    # shellcheck disable=SC2086
    if ! $PY -u scripts/run_pipeline.py "$ROOT/inputs/$CN.mp4" --name "$CN" --steps demux,separate,osd,asr,glossary,translate,tts ${PIPE_ARGS:-} < /dev/null; then
      echo "enrol: $CN failed"; echo "$IDX enrol failed" >> "$SPLIT/status.txt"
      [ "$STRICT_RUN" = 1 ] && die "enrolment $CN failed (STRICT_RUN=1)"
    fi
  done
  if [ ! -s "$PROFILE_PATH" ] || [ "${REBUILD_PROFILES:-0}" = 1 ]; then
    if $PY -u scripts/build_profiles.py "$NAME" < /dev/null; then
      echo "$NAME enrol ok" >> "$SPLIT/status.txt"
    else
      echo "$NAME enrol failed" >> "$SPLIT/status.txt"
      [ "$STRICT_RUN" = 1 ] && die "build_profiles failed (STRICT_RUN=1)"
      echo "build_profiles failed — chunks fall back to per-chunk references"
      PROFILE_READY=0
    fi
  fi
  if [ -n "$V34_RUN_DIR" ]; then
    [ "$PROFILE_READY" = 1 ] || die "cannot checkpoint failed profiles"
    profile_checkpoint record || die "cannot record profile checkpoint"
  fi
fi
export PROFILES=""
if [ "$PROFILE_READY" = 1 ] && [ -s "$PROFILE_PATH" ]; then export PROFILES="$PROFILE_PATH"; fi
if [ "$STRICT_RUN" = 1 ] && [ -z "$PROFILES" ]; then die "profiles missing after enrolment (STRICT_RUN=1)"; fi

say "2b: chunks"
$PY - "$SPLIT/plan.json" "$MIN_SPEECH_MIN" "$ENROL" > "$SPLIT/chunks.tsv" <<'PYEOF'
import json, sys
doc = json.load(open(sys.argv[1])); lim = float(sys.argv[2])
enrol = set(int(x) for x in (sys.argv[3].split() if len(sys.argv) > 3 else []))
for c in sorted(doc["chunks"], key=lambda c: (c["index"] not in enrol, c["index"])):
    print(f"{c['index']:02d}\t{c['file']}\t{'dub' if c['speech_minutes'] >= lim else 'pass'}\t{c['speech_minutes']}")
PYEOF
[ "$?" -eq 0 ] || die "chunk plan"
while IFS=$'\t' read -r IDX FILE MODE SPEECH; do
  CN="${NAME}_p${IDX}"
  ln -sfn "$FILE" "$ROOT/inputs/$CN.mp4"
  FINAL=""
  if [ "$MODE" = "pass" ]; then echo "p$IDX: no speech ($SPEECH min) → passthrough"; echo "$IDX pass nospeech" >> "$SPLIT/status.txt"; continue; fi
  if [ -n "$V34_RUN_DIR" ]; then
    profile_checkpoint check || die "profiles changed during this run"
    if "$PY" scripts/run_evidence.py check "$CN"; then echo "p$IDX: verified completed chunk"; continue; fi
  elif [ -n "$RERUN" ]; then
    if grep -q "^$IDX dub .*rerun=$RERUN$" "$SPLIT/status.txt" 2>/dev/null && FINAL=$(current_output "$CN" 2>/dev/null); then echo "p$IDX: already re-run ($RERUN)"; continue; fi
  elif FINAL=$(current_output "$CN" 2>/dev/null) && grep -q "DONE in" "$ROOT/workspace/$CN/run_v3.log" 2>/dev/null; then echo "p$IDX: already done"; continue; fi
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
    if [ "$RC" -eq 0 ]; then
      if FINAL=$(current_output "$CN"); then
        if [ -z "$V34_RUN_DIR" ] || "$PY" scripts/run_evidence.py check "$CN"; then break; fi
        echo "p$IDX: no valid completion receipt after successful command"
      else
        echo "p$IDX: command succeeded but the current state has no valid output"
      fi
      RC=1
    fi
    [ "$TRY" -lt 3 ] && echo "p$IDX: try $TRY ended rc=$RC, retrying"
  done
  ev chunk_end chunk="$CN" rc="$RC" minutes="$(( ($(date +%s)-C0)/60 ))" final="$([ -s "$FINAL" ] && echo yes || echo no)" run="$(readlink "$ROOT/workspace/$CN/runs/latest" 2>/dev/null)"
  if [ $RC -eq 0 ] && [ -s "$FINAL" ]; then
    echo "p$IDX: current output $FINAL"
    echo "$IDX dub ok $(( ($(date +%s)-C0)/60 ))min${RERUN:+ rerun=$RERUN}" >> "$SPLIT/status.txt"
  else
    echo "$IDX pass failed_rc$RC" >> "$SPLIT/status.txt"
    if [ "$STRICT_RUN" = 1 ]; then
      echo "LONG_FAILED: chunk p$IDX rc=$RC after retries (STRICT_RUN=1); concat skipped"
      exit 1
    fi
    echo "p$IDX: pipeline rc=$RC → passthrough for this chunk"
  fi
done < "$SPLIT/chunks.tsv"

say "3: passthrough chunks + concat"
[ -z "$V34_RUN_DIR" ] || profile_checkpoint check || die "profiles changed before concat"
# scripts/concat_long.py holds what used to be a heredoc here, so a VC-only rerun (scripts/run_vc_only.sh)
# can re-assemble the film without re-entering the chunk loop.
$PY scripts/concat_long.py "$NAME" "$SPLIT" || { echo "LONG_FAILED: concat"; exit 1; }
cp -f "$SPLIT/status.txt" "$ROOT/deliver/${NAME}_full/chunk_status.txt" 2>/dev/null
# one voice per profile across chunks (leave-one-chunk-out, CPU) → deliver/<film>_full/VOICE_CONSISTENCY.* for eval_long L3b–f
rm -f "$ROOT/deliver/${NAME}_full/VOICE_CONSISTENCY.json" "$ROOT/deliver/${NAME}_full/VOICE_CONSISTENCY.md"
if ! $PY -u scripts/voice_consistency.py --film "$NAME" ${PROFILES:+--profiles "$PROFILES"}; then
  [ "$STRICT_RUN" = 1 ] && die "film voice consistency execution failed (STRICT_RUN=1)"
  echo "(voice consistency failed — kept going)"
fi
ev film_end minutes="$(( ($(date +%s) - T0) / 60 ))"
say "LONG_DONE in $(( ($(date +%s) - T0) / 60 )) min"
