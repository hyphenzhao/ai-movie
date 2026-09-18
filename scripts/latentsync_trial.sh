#!/usr/bin/env bash
# LatentSync 1.6 A/B on the profile / small-face stretches MuseTalk skips.
#
#   bash scripts/latentsync_trial.sh            # → workspace/_ls_trial/REPORT.md
#
# Only runs when the environment is actually usable (weights complete, the
# venv's torch sees the GPU); otherwise it writes why and exits 0 so the
# nightly chain that calls it is never blocked.  It cuts one ~12 s clip per
# film around the segment the face gate skipped most, re-lip-syncs it with
# the v2 dubbed audio, and times the run — that measured frames/second is
# what the 30/60/90-minute projections are built from.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LS="$ROOT/vendor/LatentSync"
# The pipeline's own interpreter: its torch (ROCm 7.2) is the only build on this box with gfx1151 kernels —
# the rocm6.3 wheels segfault on the first GPU op.  vendor/latentsync_shims supplies what that venv lacks
# (decord stand-in, kornia_rs stub, links to the pure-Python packages).
PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT/vendor/latentsync_shims${PYTHONPATH:+:$PYTHONPATH}"
export MIOPEN_FIND_MODE=FAST      # default exhaustive conv search sat >15 min on the first step
OUT="$ROOT/workspace/_ls_trial"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/trial.log") 2>&1
echo "##### LatentSync trial $(date '+%m-%d %H:%M:%S')"

# ── readiness ──────────────────────────────────────────────────────
UNET="$LS/checkpoints/latentsync_unet.pt"
WANT=5072222488
HAVE=$(stat -c %s "$UNET" 2>/dev/null || echo 0)
if [ "$HAVE" -lt "$WANT" ]; then
  echo "NOT_READY: unet $HAVE/$WANT bytes" | tee "$OUT/REPORT.md"; exit 0
fi
if [ ! -x "$PY" ]; then echo "NOT_READY: venv missing" | tee "$OUT/REPORT.md"; exit 0; fi
DEV=$("$PY" -c "import torch;print('hip' if getattr(torch.version,'hip',None) else 'cuda' if torch.version.cuda else 'cpu', torch.cuda.is_available())" 2>&1 | tail -1)
echo "venv torch: $DEV"
case "$DEV" in *True*) ;; *) echo "NOT_READY: torch in venv cannot see the GPU ($DEV)" | tee "$OUT/REPORT.md"; exit 0;; esac
"$PY" -c "import diffusers, decord, omegaconf, kornia, DeepCache, insightface.app" 2>/dev/null || { echo "NOT_READY: deps incomplete" | tee "$OUT/REPORT.md"; exit 0; }

# ── clips: the stretch each film's face gate skipped the most ───────
"$ROOT/.venv/bin/python" - "$OUT" <<'PYEOF'
import json, subprocess, sys
from pathlib import Path
out = Path(sys.argv[1])
for film in ("test_1", "test_2"):
    st = json.loads(Path(f"workspace/{film}/state.json").read_text())
    plan = json.loads(Path(st["faces"]["plan_path"]).read_text())
    gated = plan.get("segment_gated") or {}
    segs = (st.get("fit") or st["asr"])["segments"]
    if not gated:
        print(film, "no gated segments"); continue
    i = int(max(gated, key=lambda k: gated[k]))
    s = segs[i]
    a = max(0.0, float(s["start"]) - 3.0); dur = 12.0
    src = st["vc"]["video"] if (st.get("vc") or {}).get("video") else st["compose"]["video"]
    orig = Path(st.get("input") or f"inputs/{film}.mp4")
    clip_v = out / f"{film}_orig.mp4"; clip_a = out / f"{film}_dub.wav"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{a:.2f}", "-t", f"{dur}", "-i", str(orig),
                    "-an", "-c:v", "libx264", "-crf", "16", "-r", "25", str(clip_v)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{a:.2f}", "-t", f"{dur}", "-i", src,
                    "-vn", "-ac", "1", "-ar", "16000", str(clip_a)], check=True)
    (out / f"{film}_clip.json").write_text(json.dumps(
        {"film": film, "segment": i, "gated_frames": gated[str(i)], "start": a, "dur": dur,
         "yaw": plan.get("frame_yaw", {}).get(str(int(a * plan["fps"])))}))
    print(film, f"clip from {a:.1f}s, segment {i} had {gated[str(i)]} gated frames")
PYEOF

export HF_HUB_OFFLINE=1          # everything it needs is on disk; never pull models over the metered link
# ── run ────────────────────────────────────────────────────────────
cd "$LS"
for film in test_1 test_2; do
  [ -f "$OUT/${film}_orig.mp4" ] || continue
  T0=$(date +%s)
  "$PY" -m scripts.inference \
      --unet_config_path configs/unet/stage2_512.yaml \
      --inference_ckpt_path checkpoints/latentsync_unet.pt \
      --inference_steps 20 --guidance_scale 1.5 --enable_deepcache \
      --video_path "$OUT/${film}_orig.mp4" --audio_path "$OUT/${film}_dub.wav" \
      --video_out_path "$OUT/${film}_latentsync.mp4" --temp_dir "$OUT/tmp_$film" \
      > "$OUT/${film}_infer.log" 2>&1
  RC=$?; T1=$(date +%s)
  echo "$film: rc=$RC $((T1-T0))s" | tee -a "$OUT/timing.txt"
done
cd "$ROOT"

# ── report ─────────────────────────────────────────────────────────
"$ROOT/.venv/bin/python" - "$OUT" <<'PYEOF'
import json, subprocess, sys
from pathlib import Path
out = Path(sys.argv[1]); lines = ["# LatentSync 1.6 试验", ""]
for film in ("test_1", "test_2"):
    v = out / f"{film}_latentsync.mp4"
    meta = json.loads((out / f"{film}_clip.json").read_text()) if (out / f"{film}_clip.json").exists() else {}
    t = [l for l in (out / "timing.txt").read_text().splitlines() if l.startswith(film)] if (out / "timing.txt").exists() else []
    secs = int(t[-1].split()[-1].rstrip("s")) if t else None
    if v.exists() and secs:
        frames = int(float(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                    "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(v)],
                    capture_output=True, text=True).stdout.strip() or 0))
        fps = frames / secs if secs else 0
        lines += [f"## {film}", f"- 片段 {meta.get('start', 0):.1f}s 起 {meta.get('dur', 12)} s，原门控跳过 {meta.get('gated_frames')} 帧",
                  f"- 输出 {frames} 帧，耗时 {secs} s → **{fps:.2f} 帧/秒**（{secs/frames if frames else 0:.2f} s/帧）"]
        for mins in (30, 60, 90):
            lines.append(f"  - {mins} 分钟影片若全部口型帧都处理（25 fps）：约 {mins*60*25/max(fps,1e-6)/3600:.1f} 小时；"
                         f"只处理有台词的 60%：约 {mins*60*25*0.6/max(fps,1e-6)/3600:.1f} 小时")
        lines.append(f"- 文件：`{v}`（对照原片 `{out / (film + '_orig.mp4')}`）")
    else:
        log = (out / f"{film}_infer.log").read_text(errors="replace")[-600:] if (out / f"{film}_infer.log").exists() else ""
        lines += [f"## {film}", "- **失败**", "```", log.strip(), "```"]
    lines.append("")
(out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
PYEOF
echo "LS_TRIAL_DONE"
