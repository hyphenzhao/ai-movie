#!/usr/bin/env bash
# Fully-automatic two-version demo build for one video.
#
#   bash scripts/auto_demo.sh inputs/test_1.mp4 test_1
#
# Produces workspace/<name>/deliverables/{v1_standard,v2_cloned}/ with full
# films, demo clips, per-stage artifacts, plus a GUI project file.  No human
# input anywhere: speaker labels are corrected only where the automatic
# signals are decisive, and the VC reference is picked by the output-pitch
# criterion (see scripts/auto_select_refs.py) — a gender with no qualifying
# reference keeps the built-in voice instead of shipping a bad clone.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
VIDEO="$1"; NAME="$2"
WORK="workspace/$NAME"; STATE="$WORK/state.json"; D="$WORK/deliverables"

stage() { echo "### [$(date +%H:%M:%S)] $NAME :: $*"; }
die()   { echo "### [$(date +%H:%M:%S)] $NAME :: FAILED at: $*"; exit 1; }

stage "A: demux → separate → asr → glossary → translate"
$PY -u scripts/run_pipeline.py "$VIDEO" --name "$NAME" \
    --steps demux,separate,asr,glossary,translate || die "stage A"

stage "B: speaker label review (automatic signals only)"
$PY -u scripts/review_speakers.py "$STATE" --apply || die "review"
$PY - "$STATE" <<'EOF' || die "speaker csv refresh"
import json, sys
sys.path.insert(0, '.')
from pathlib import Path
from ai_movie import artifacts
state_path = Path(sys.argv[1])
s = json.loads(state_path.read_text(encoding='utf-8'))
d = state_path.parent / 'deliverables'
artifacts.export_speaker_csv(s['asr']['segments'], d / '01_speakers.csv')
artifacts.export_srt(s['asr']['segments'], d / '01_asr.ja.srt', speaker_prefix=True)
EOF

stage "C: v1 (built-in voices) tts → fit → mix → faces → lipsync → compose"
$PY -u scripts/run_pipeline.py "$VIDEO" --name "$NAME" \
    --steps tts,fit,mix,faces,lipsync,compose --voice-mode sft || die "stage C"

stage "D: auto-select VC references"
$PY -u scripts/auto_select_refs.py "$STATE" || die "ref selection"

stage "E: v2 (voice conversion)"
$PY - "$STATE" <<'EOF' > "$WORK/_refs_args" || die "refs read"
import json, sys
from pathlib import Path
picked = json.loads((Path(sys.argv[1]).parent / 'refs_auto' / 'refs.json')
                    .read_text(encoding='utf-8'))['picked']
f = picked.get('female') or '/nonexistent'
m = picked.get('male') or '/nonexistent'
print(f, m)
EOF
read -r REF_F REF_M < "$WORK/_refs_args"
if [ "$REF_F" = "/nonexistent" ] && [ "$REF_M" = "/nonexistent" ]; then
  stage "E: no qualifying reference for any gender — skipping v2 entirely"
else
  $PY -u scripts/run_vc_version.py "$STATE" \
      --ref-female "$REF_F" --ref-male "$REF_M" || die "vc version"
fi

stage "F: verification"
$PY -u scripts/verify_dub.py "$STATE" --key fit -n 20 \
    --out "$WORK/verify_v1.json" || die "verify v1"
if [ -d "$D/v2_cloned" ]; then
  $PY -u scripts/verify_dub.py "$STATE" --key vc --audio-field audio_fit -n 20 \
      --out "$WORK/verify_v2.json" || die "verify v2"
fi
$PY -u scripts/eval_pipeline.py "$STATE" || echo "(eval reported failures — kept going)"

stage "G: demo clips"
V1VIDEO=$($PY -c "import json;print(json.load(open('$STATE'))['compose']['video'])")
$PY -u scripts/make_demo_clips.py "$STATE" --video "$V1VIDEO" \
    --out "$D/v1_standard/demo" --label "标准音色" -n 3 || die "v1 clips"
if [ -d "$D/v2_cloned" ]; then
  V2VIDEO=$($PY -c "import json;print(json.load(open('$STATE'))['vc']['video'])")
  $PY -u scripts/make_demo_clips.py "$STATE" --video "$V2VIDEO" \
      --out "$D/v2_cloned/demo" --label "原声音色" -n 3 || die "v2 clips"
fi

stage "H: assemble deliverable folders + project file"
mkdir -p "$D/v1_standard" "$D/00_shared"
cp "$D/05_final_dubbed.mp4"  "$D/v1_standard/05_final_dubbed.mp4"
cp "$D/03_final_audio.wav"   "$D/v1_standard/" 2>/dev/null
cp "$D/03_tts_report.csv"    "$D/v1_standard/" 2>/dev/null
cp "$D/01_speakers.csv" "$D/01_asr.ja.srt" "$D/02_zh_sakura.srt" "$D/v1_standard/" 2>/dev/null
cp "$D/00_speaker_review.csv" "$D/ACCEPTANCE.md" "$D/v1_standard/" 2>/dev/null
cp "$D/04_face_tracks.csv" "$D/04_face_plan_summary.json" "$D/v1_standard/" 2>/dev/null
cp "$WORK/verify_v1.json" "$D/v1_standard/06_verify_audio.json" 2>/dev/null
if [ -d "$D/v2_cloned" ]; then
  cp "$D/01_speakers.csv" "$D/01_asr.ja.srt" "$D/02_zh_sakura.srt" "$D/v2_cloned/" 2>/dev/null
  cp "$WORK/verify_v2.json" "$D/v2_cloned/06_verify_audio.json" 2>/dev/null
  [ "$REF_F" != "/nonexistent" ] && cp "$REF_F" "$D/v2_cloned/07_ref_female.wav"
  [ "$REF_M" != "/nonexistent" ] && cp "$REF_M" "$D/v2_cloned/07_ref_male.wav"
fi
mv "$D/02_glossary.json" "$D/02_translation_compare.md" "$D"/track_*.jpg "$D/00_shared/" 2>/dev/null
$PY -u scripts/export_project.py "$STATE" || die "project export"

stage "DONE"
