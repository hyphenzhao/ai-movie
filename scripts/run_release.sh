#!/usr/bin/env bash
# Unattended release: model check → pilot → three films → gate → publish.
#
#   setsid nohup bash scripts/run_release.sh v3.1.0 > /dev/null 2>&1 &
#   tail -f workspace/_release/v3.1.0/run.log         # ends in DONE / GATE_FAILED / FAILED
#
# 1. polish model: wait for the download, smoke-test it, fall back automatically
# 2. pilot on output_test (asr → glossary → translate) scored against the
#    burned-in subtitles; median below 0.90 stops the release early
# 3. per film: new ASR first, gender guard against the baseline (regression →
#    re-run that film with --gender-source vocals), then the full run_v3.sh
# 4. accept_release.py; any blocking gate failing stops here without uploading
# 5. VPS preview site + Google Drive (<version>/ subfolder)
set -uo pipefail
VER="${1:?usage: run_release.sh <version>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
REL="$ROOT/workspace/_release/$VER"
BASE="$ROOT/workspace/_archive_v3.0.0"
FILMS="${FILMS:-output_test test_1 test_2}"
MODEL="${POLISH_MODEL:-ttempvnn/HauhauCS-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4-K-M:latest}"
mkdir -p "$REL"
exec > >(tee -a "$REL/run.log") 2>&1

stage() { echo; echo "##### [$(date '+%m-%d %H:%M:%S')] $VER · $* #####"; }
die() { echo "FAILED: $*"; exit 1; }
T0=$(date +%s)
ASR_ARGS=""

stage "1: polish model"
if ! ollama list | awk 'NR>1{print $1}' | grep -qx "$MODEL"; then
  for _ in $(seq 1 360); do                      # up to 6 h (flaky VPN), then fall back
    ollama list | awk 'NR>1{print $1}' | grep -qx "$MODEL" && break
    if ! pgrep -f "pull_retry.sh|ollama pull" >/dev/null; then
      echo "download not running — starting it"
      setsid nohup bash "$REL/pull_retry.sh" >> "$REL/ollama_pull.log" 2>&1 < /dev/null &
    fi
    sleep 60
  done
fi
ollama list | awk 'NR>1{print $1}' | grep -qx "$MODEL" \
  || echo "WARNING: $MODEL still not downloaded after the wait — smoke test falls back"
if CHOSEN=$($PY scripts/smoke_polish_model.py --model "$MODEL" --json "$REL/smoke.json"); then
  export AI_MOVIE_POLISH_MODEL="$CHOSEN"
  echo "polish model: $CHOSEN"
else
  die "no polish model answers (see $REL/smoke.json)"
fi

stage "2: pilot — output_test asr/glossary/translate"
$PY -u scripts/run_pipeline.py inputs/output_test.mp4 --name output_test \
    --steps asr,glossary,translate || die "pilot pipeline"
$PY scripts/eval_against_subs.py output_test --ref-srt inputs/subs/output_test.zh.srt \
    --summary-json "$REL/pilot_subs.json" || die "pilot scoring"
MEDIAN=$($PY -c "import json;print(json.load(open('$REL/pilot_subs.json')).get('asr_median',0))")
echo "pilot ASR median: $MEDIAN"
if ! $PY -c "import sys; sys.exit(0 if float('$MEDIAN') >= 0.90 else 1)"; then
  die "pilot ASR median $MEDIAN < 0.90 — stopping before six hours of GPU work"
fi

gender_regressed() {   # $1 film → exit 0 when A3/A5 passed in the baseline but fail now
  $PY - "$1" "$BASE" <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
import eval_pipeline as ev
film, base = sys.argv[1], Path(sys.argv[2])
state = json.loads(Path(f"workspace/{film}/state.json").read_text(encoding="utf-8"))
rep = ev.Report(); ev.eval_asr(state, rep)
now = {r["key"]: r["ok"] for r in rep.rows}
was = {r["key"]: r["ok"] for r in json.loads((base / film / "baseline" / "eval.json").read_text())}
bad = [k for k in ("A3", "A5") if was.get(k) is True and now.get(k) is False]
print(f"gender guard {film}: baseline {[(k, was.get(k)) for k in ('A3','A5')]} "
      f"now {[(k, now.get(k)) for k in ('A3','A5')]} regressed {bad}")
sys.exit(0 if bad else 1)
PYEOF
}

for FILM in $FILMS; do
  stage "3: $FILM — ASR + gender guard"
  ARGS="$ASR_ARGS"
  # shellcheck disable=SC2086
  $PY -u scripts/run_pipeline.py "inputs/$FILM.mp4" --name "$FILM" $ARGS \
      --steps demux,separate,osd,asr || die "$FILM asr"
  if gender_regressed "$FILM"; then
    ARGS="$ARGS --gender-source vocals"
    echo "$FILM: gender regressed on the mix — re-running ASR with --gender-source vocals"
    # shellcheck disable=SC2086
    $PY -u scripts/run_pipeline.py "inputs/$FILM.mp4" --name "$FILM" $ARGS \
        --steps asr || die "$FILM asr (vocals gender)"
  fi
  echo "$FILM run args: ${ARGS:-(defaults)}" | tee -a "$REL/film_args.txt"

  stage "3: $FILM — full run_v3"
  PIPE_ARGS="$ARGS" bash scripts/run_v3.sh "$FILM" || die "$FILM run_v3"
done

stage "4: acceptance gate"
if ! $PY scripts/accept_release.py "$VER" --baseline "$BASE" --films "${FILMS// /,}"; then
  echo "GATE_FAILED: see $REL/ACCEPTANCE.md — nothing uploaded"
  exit 1
fi

stage "5: publish — VPS preview site"
$PY scripts/build_preview_site.py --upload || die "preview site upload"

stage "5: publish — Google Drive $VER/"
for FILM in $FILMS; do
  $PY -u scripts/deliver.py "workspace/$FILM/state.json" --out deliver --version vc \
      --upload --drive-subdir "$VER" \
      --extra "deliver/$FILM/09_subs_eval.md" --extra "$REL/ACCEPTANCE.md" \
      || die "$FILM drive upload"
done

stage "DONE in $(( ($(date +%s) - T0) / 60 )) min"
echo "DONE"
