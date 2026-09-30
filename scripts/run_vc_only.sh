#!/usr/bin/env bash
# Re-run ONLY the voice-conversion version (v2) of an already finished film, plus its checks:
#   run_vc_version → qc (--force) → verify_dub v2 (n=40) → eval_pipeline
#
#   bash scripts/run_vc_only.sh <name> [extra run_vc_version args…]
#   bash scripts/run_vc_only.sh SONE-846_p07 --vc-source audio_fit --out-name v2_fitsrc --out-dir synthesized_vc_fitsrc
#
# Why not run_v3.sh: its stage A calls run_pipeline.py without --steps, which consults the stage cache —
# from a worktree whose config/code hashes differ from the stamped ones (v3.4-dev: asr and translate are
# STALE on every film) that rebuilds the whole film from asr.  This script never touches a fingerprinted
# stage: run_vc_version.py is not a cached stage, `--steps qc --force` never consults has(), and qc's
# own hash is unchanged so the main tree still sees qc VALID.  Never run --fp-adopt from a worktree.
#
# References: PROFILES=<profiles.json> or, for a long-film chunk NAME_pNN, workspace/NAME/profiles.json
# when it exists; otherwise workspace/<name>/refs_auto/refs.json (auto_select_refs must have run).
# The previous verify_v2.json is kept as verify_v2.pre_v34.json (first run only) for the before/after.
# For a long film, re-assemble afterwards with:  python scripts/concat_long.py NAME
set -uo pipefail
NAME="${1:?usage: run_vc_only.sh <name> [run_vc_version args…]}"
shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
VIDEO="$ROOT/inputs/$NAME.mp4"
WORK="$ROOT/workspace/$NAME"
STATE="$WORK/state.json"
[ -s "$STATE" ] || { echo "no state: $STATE"; exit 1; }
LOG="$WORK/run_vc_only.log"
exec > >(tee -a "$LOG") 2>&1
export HF_HUB_OFFLINE=1
stage() { echo; echo "===== [$(date +%H:%M:%S)] $NAME · $* ====="; }
die() { echo "FAILED: $*" >&2; exit 1; }
T0=$(date +%s)

FILM="${NAME%_p[0-9][0-9]}"
PROFILES="${PROFILES:-}"
if [ -z "$PROFILES" ] && [ "$FILM" != "$NAME" ] && [ -s "$ROOT/workspace/$FILM/profiles.json" ]; then
  PROFILES="$ROOT/workspace/$FILM/profiles.json"
fi
if [ -n "$PROFILES" ] && [ -s "$PROFILES" ]; then
  REF_ARGS=(--profiles "$PROFILES")
elif [ -s "$WORK/refs_auto/refs.json" ]; then
  REF_ARGS=(--refs-json "$WORK/refs_auto/refs.json")
else
  die "no profiles.json and no refs_auto/refs.json for $NAME"
fi
[ -s "$WORK/verify_v2.json" ] && [ ! -s "$WORK/verify_v2.pre_v34.json" ] && cp "$WORK/verify_v2.json" "$WORK/verify_v2.pre_v34.json"

stage "C: v2 (voice conversion) ${REF_ARGS[*]} $*"
$PY -u scripts/run_vc_version.py "$STATE" "${REF_ARGS[@]}" "$@" || die "vc version"
# one voice per speaker on the delivered v2 lines (ECAPA on CPU) → state.vc.consistency, as run_v3.sh stage C
$PY -u scripts/voice_consistency.py "$STATE" ${PROFILES:+--profiles "$PROFILES"} || echo "(voice consistency failed — kept going)"

stage "D: QC (v1 + v2), verification, acceptance"
$PY -u scripts/run_pipeline.py "$VIDEO" --name "$NAME" --steps qc --force || die "qc"
$PY -u scripts/verify_dub.py "$STATE" --key vc --audio-field audio_fit -n 40 --out "$WORK/verify_v2.json" || echo "(verify v2 failed — kept going)"
$PY -u scripts/eval_pipeline.py "$STATE" || echo "(eval reported failures — kept going)"

stage "VC_ONLY_DONE in $(( ($(date +%s) - T0) / 60 )) min"
