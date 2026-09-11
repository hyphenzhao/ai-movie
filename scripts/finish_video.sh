#!/usr/bin/env bash
# Post-fix finish for one video: rebuild v2 on the current lipsync render,
# re-verify, re-clip, re-assemble.  Assumes v1 lipsync+compose are current.
#
#   bash scripts/finish_video.sh test_1 <ref_female|-> <ref_male|->
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
NAME="$1"; REF_F="${2:--}"; REF_M="${3:--}"
WORK="workspace/$NAME"; STATE="$WORK/state.json"; D="$WORK/deliverables"
[ "$REF_F" = "-" ] && REF_F=/nonexistent
[ "$REF_M" = "-" ] && REF_M=/nonexistent

stage() { echo "### [$(date +%H:%M:%S)] finish($NAME) :: $*"; }

stage "v2 rebuild"
$PY -u scripts/run_vc_version.py "$STATE" \
    --ref-female "$REF_F" --ref-male "$REF_M" || exit 1

stage "verify + eval"
$PY -u scripts/verify_dub.py "$STATE" --key fit -n 20 \
    --out "$WORK/verify_v1.json" || exit 1
$PY -u scripts/verify_dub.py "$STATE" --key vc --audio-field audio_fit -n 20 \
    --out "$WORK/verify_v2.json" || exit 1
$PY -u scripts/eval_pipeline.py "$STATE" || echo "(eval failures noted)"

stage "demo clips"
V1VIDEO=$($PY -c "import json;print(json.load(open('$STATE'))['compose']['video'])")
V2VIDEO=$($PY -c "import json;print(json.load(open('$STATE'))['vc']['video'])")
rm -rf "$D/v1_standard/demo" "$D/v2_cloned/demo"
$PY -u scripts/make_demo_clips.py "$STATE" --video "$V1VIDEO" \
    --out "$D/v1_standard/demo" --label "标准音色" -n 3 || exit 1
$PY -u scripts/make_demo_clips.py "$STATE" --video "$V2VIDEO" \
    --out "$D/v2_cloned/demo" --label "原声音色" -n 3 || exit 1

stage "assemble"
mkdir -p "$D/v1_standard" "$D/v2_cloned" "$D/00_shared"
$PY - "$STATE" <<'EOF'
import json, sys
sys.path.insert(0, '.')
from pathlib import Path
from ai_movie import artifacts
sp = Path(sys.argv[1]); s = json.loads(sp.read_text(encoding='utf-8'))
d = sp.parent / 'deliverables'
artifacts.export_speaker_csv(s['asr']['segments'], d / '01_speakers.csv')
artifacts.export_srt(s['asr']['segments'], d / '01_asr.ja.srt', speaker_prefix=True)
EOF
cp "$D/05_final_dubbed.mp4" "$D/v1_standard/05_final_dubbed.mp4"
for f in 01_speakers.csv 01_asr.ja.srt 02_zh_sakura.srt 03_tts_report.csv; do
  cp "$D/$f" "$D/v1_standard/" 2>/dev/null
  cp "$D/$f" "$D/v2_cloned/" 2>/dev/null
done
cp "$D/03_final_audio.wav" "$D/00_speaker_review.csv" "$D/ACCEPTANCE.md" \
   "$D/04_face_tracks.csv" "$D/04_face_plan_summary.json" "$D/v1_standard/" 2>/dev/null
cp "$WORK/verify_v1.json" "$D/v1_standard/06_verify_audio.json" 2>/dev/null
cp "$WORK/verify_v2.json" "$D/v2_cloned/06_verify_audio.json" 2>/dev/null
[ -f "$REF_F" ] && cp "$REF_F" "$D/v2_cloned/07_ref_female.wav"
[ -f "$REF_M" ] && cp "$REF_M" "$D/v2_cloned/07_ref_male.wav"
mv "$D/02_glossary.json" "$D/02_translation_compare.md" "$D"/track_*.jpg "$D/00_shared/" 2>/dev/null
$PY -u scripts/export_project.py "$STATE" || exit 1
stage "DONE"
