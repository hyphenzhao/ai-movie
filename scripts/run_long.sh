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
MIN_SPEECH_MIN="${MIN_SPEECH_MIN:-0.03}"          # < ~2 s of detected speech → nothing to dub

say() { echo; echo "##### [$(date '+%m-%d %H:%M:%S')] $NAME · $* #####"; }
T0=$(date +%s)

say "1: plan + cut"
$PY scripts/smart_split.py "$VIDEO" --name "$NAME" --cut || { echo "LONG_FAILED: split"; exit 1; }

say "2: chunks"
$PY - "$SPLIT/plan.json" "$MIN_SPEECH_MIN" > "$SPLIT/chunks.tsv" <<'PYEOF'
import json, sys
doc = json.load(open(sys.argv[1])); lim = float(sys.argv[2])
for c in doc["chunks"]:
    print(f"{c['index']:02d}\t{c['file']}\t{'dub' if c['speech_minutes'] >= lim else 'pass'}\t{c['speech_minutes']}")
PYEOF
while IFS=$'\t' read -r IDX FILE MODE SPEECH; do
  CN="${NAME}_p${IDX}"
  ln -sfn "$FILE" "$ROOT/inputs/$CN.mp4"
  FINAL="$ROOT/workspace/$CN/output/v2_cloned_dubbed.mp4"
  if [ "$MODE" = "pass" ]; then echo "p$IDX: no speech ($SPEECH min) → passthrough"; echo "$IDX pass nospeech" >> "$SPLIT/status.txt"; continue; fi
  if [ -s "$FINAL" ] && grep -q "DONE in" "$ROOT/workspace/$CN/run_v3.log" 2>/dev/null; then echo "p$IDX: already done"; continue; fi
  say "chunk p$IDX (speech $SPEECH min)"
  C0=$(date +%s)
  bash scripts/run_v3.sh "$CN" < /dev/null
  RC=$?
  if [ $RC -eq 0 ] && [ -s "$FINAL" ]; then echo "$IDX dub ok $(( ($(date +%s)-C0)/60 ))min" >> "$SPLIT/status.txt"
  else echo "p$IDX: pipeline rc=$RC → passthrough for this chunk"; echo "$IDX pass failed_rc$RC" >> "$SPLIT/status.txt"; fi
done < "$SPLIT/chunks.tsv"

say "3: passthrough chunks + concat"
$PY - "$NAME" "$SPLIT" <<'PYEOF' || { echo "LONG_FAILED: concat"; exit 1; }
import csv, json, statistics, subprocess, sys
from pathlib import Path
name, split = sys.argv[1], Path(sys.argv[2])
root = split.parents[2]
plan = json.loads((split / "plan.json").read_text())
def run(c): subprocess.run(c, check=True)
def probe(p, entries, stream):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries", entries,
                           "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()
finals, offsets = {}, []
for c in plan["chunks"]:
    w = root / "workspace" / f"{name}_p{c['index']:02d}"
    f = w / "output" / "v2_cloned_dubbed.mp4"
    if f.exists() and f.stat().st_size and (w / "state.json").exists():
        finals[c["index"]] = f
        st = json.loads((w / "state.json").read_text())
        off = ((st.get("vc") or {}).get("mix") or {}).get("global_offset_db", (st.get("mix") or {}).get("global_offset_db"))
        if isinstance(off, (int, float)): offsets.append(off)
gain = statistics.median(offsets) if offsets else 0.0
ref = next(iter(finals.values()), None)
fps = probe(ref, "stream=r_frame_rate", "v:0") if ref else probe(plan["video"], "stream=r_frame_rate", "v:0")
wh = probe(ref, "stream=width,height", "v:0") if ref else ""
print(f"dubbed chunks {len(finals)}/{len(plan['chunks'])}; passthrough gain {gain:+.2f} dB; fps {fps} {wh}")
parts = []
for c in plan["chunks"]:
    i = c["index"]; src = finals.get(i)
    if src is None:
        src = split / "pass" / f"{name}_p{i:02d}.mp4"; src.parent.mkdir(exist_ok=True)
        if not src.exists():
            vf = ["-vf", f"scale={wh.replace(',', ':')}"] if wh else []
            run(["ffmpeg", "-y", "-v", "error", "-i", c["file"], "-map", "0:v:0", "-map", "0:a:0", *vf,
                 "-r", fps, "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                 "-af", f"volume={gain}dB,alimiter=limit=0.89", "-ar", "44100", "-ac", "2", "-c:a", "aac", "-b:a", "192k", str(src)])
    ts = split / "ts" / f"p{i:02d}.ts"; ts.parent.mkdir(exist_ok=True)
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "mpegts", str(ts)])
    parts.append(ts)
out = root / "deliver" / f"{name}_dubbed_full.mp4"
run(["ffmpeg", "-y", "-v", "error", "-i", "concat:" + "|".join(map(str, parts)), "-c", "copy",
     "-bsf:a", "aac_adtstoasc", "-movflags", "+faststart", str(out)])
d_out = float(probe(out, "format=duration", "v:0") or subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout)
print(f"full film: {out}  {d_out/60:.2f} min (source {plan['duration']/60:.2f} min, Δ {d_out - plan['duration']:+.2f} s)")
# one table for the whole film: film-time segments with JA / ZH, for text review
rows = []
for c in plan["chunks"]:
    sp = root / "workspace" / f"{name}_p{c['index']:02d}" / "state.json"
    if not sp.exists(): continue
    st = json.loads(sp.read_text())
    segs = ((st.get("vc") or {}).get("segments") or (st.get("fit") or {}).get("segments")
            or (st.get("translate") or {}).get("segments") or (st.get("asr") or {}).get("segments") or [])
    for n, s in enumerate(segs):
        rows.append([c["index"], n, round(c["start"] + s["start"], 2), round(c["start"] + s["end"], 2),
                     s.get("speaker", ""), s.get("gender", ""), s.get("text", ""), s.get("translation", "")])
full = root / "deliver" / f"{name}_full"
with open(full / "segments_full.csv", "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh); w.writerow(["chunk", "seg", "start", "end", "speaker", "gender", "ja", "zh"]); w.writerows(rows)
(full / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{len(rows)} segments → {full / 'segments_full.csv'}")
PYEOF
cp -f "$SPLIT/status.txt" "$ROOT/deliver/${NAME}_full/chunk_status.txt" 2>/dev/null
say "LONG_DONE in $(( ($(date +%s) - T0) / 60 )) min"
