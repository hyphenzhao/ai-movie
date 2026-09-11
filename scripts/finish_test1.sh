#!/usr/bin/env bash
# test_1 corrective finish: rebuild v2 with the repaired speaker map (the
# first v2 ran before the orphan-gender repair, so 3 male segments were
# converted onto the FEMALE reference), then refresh everything downstream
# of the changed videos.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
WORK=workspace/test_1; STATE=$WORK/state.json; D=$WORK/deliverables
REF_F=$WORK/refs_auto/cand_female_seg0039.wav

stage() { echo "### [$(date +%H:%M:%S)] finish_test1 :: $*"; }

stage "v2 rebuild with corrected speakers"
$PY -u scripts/run_vc_version.py "$STATE" \
    --ref-female "$REF_F" --ref-male /nonexistent || exit 1

stage "verify v2 + eval"
$PY -u scripts/verify_dub.py "$STATE" --key vc --audio-field audio_fit -n 20 \
    --out "$WORK/verify_v2.json" || exit 1
$PY -u scripts/eval_pipeline.py "$STATE" || echo "(eval failures noted)"

stage "demo clips (both versions, videos changed)"
V1VIDEO=$($PY -c "import json;print(json.load(open('$STATE'))['compose']['video'])")
V2VIDEO=$($PY -c "import json;print(json.load(open('$STATE'))['vc']['video'])")
rm -rf "$D/v1_standard/demo" "$D/v2_cloned/demo"
$PY -u scripts/make_demo_clips.py "$STATE" --video "$V1VIDEO" \
    --out "$D/v1_standard/demo" --label "标准音色" -n 3 || exit 1
$PY -u scripts/make_demo_clips.py "$STATE" --video "$V2VIDEO" \
    --out "$D/v2_cloned/demo" --label "原声音色" -n 3 || exit 1

stage "reassemble"
$PY - <<'EOF'
import json, sys
sys.path.insert(0, '.')
from pathlib import Path
from ai_movie import artifacts
s = json.loads(Path('workspace/test_1/state.json').read_text(encoding='utf-8'))
d = Path('workspace/test_1/deliverables')
artifacts.export_speaker_csv(s['asr']['segments'], d / '01_speakers.csv')
artifacts.export_srt(s['asr']['segments'], d / '01_asr.ja.srt', speaker_prefix=True)
EOF
cp "$D/05_final_dubbed.mp4" "$D/v1_standard/05_final_dubbed.mp4"
cp "$D/01_speakers.csv" "$D/01_asr.ja.srt" "$D/v1_standard/" 2>/dev/null
cp "$D/01_speakers.csv" "$D/01_asr.ja.srt" "$D/v2_cloned/" 2>/dev/null
cp "$D/00_speaker_review.csv" "$D/ACCEPTANCE.md" "$D/v1_standard/" 2>/dev/null
cp "$WORK/verify_v2.json" "$D/v2_cloned/06_verify_audio.json"
$PY -u scripts/export_project.py "$STATE" || exit 1
stage "DONE"
