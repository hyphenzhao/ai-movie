#!/usr/bin/env python
"""Pick and cut demo clips from a finished dub, by measurement not by feel.

A 30-second window is only worth showing if the thing being demonstrated is
actually visible in it.  This slides a window across the film and scores each
position on what the demo is meant to prove:

  lip coverage   fraction of the window's frames the face plan drives.  A
                 window where the speaker is off-camera shows nothing.
  speech density speech seconds per window second — dead air is a bad demo.
  truncation     seconds cut because a line did not fit its slot (penalty).
  speed          mean time-stretch factor; far from 1.0 sounds unnatural.
  both speakers  windows containing only one speaker cannot show the
                 speaker-routing behaviour at all.

Windows whose lines are sexually explicit are flagged, not silently dropped:
the best-scoring window in this film is also the most explicit one, and that
is the operator's call to make, not the script's.

    python scripts/make_demo_clips.py workspace/output_test/state.json \
        --video-key lipsync --out workspace/output_test/deliverables/v1_standard/demo
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WINDOW = 30.0
STEP = 2.5
# Terms that make a clip unsuitable for an unattended customer screening.
EXPLICIT = re.compile(
    r"(おちんちん|チンチン|ちんちん|エッチ|セックス|挿|勃|射|イっ|感じちゃ"
    r"|阴茎|鸡巴|插进|插入|做爱|射精|勃起|高潮|下面|硬了)")


def ff(args: list[str]) -> None:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr[-400:]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("state")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", default=None,
                    help="dubbed video (default: state's compose/vc output)")
    ap.add_argument("--label", default="配音")
    ap.add_argument("-n", type=int, default=3, help="how many single clips")
    args = ap.parse_args()

    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    work = Path(args.state).parent
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dubbed = args.video or ((state.get("vc") or {}).get("video")
                            or (state.get("compose") or {}).get("video"))
    if not dubbed or not Path(dubbed).exists():
        print(f"no dubbed video ({dubbed})")
        return 1
    # _video is only populated on some code paths; the demux record always
    # knows what it demuxed.
    source = (state.get("_video")
              or (state.get("demux") or {}).get("video"))

    segs = ((state.get("vc") or {}).get("segments")
            or (state.get("fit") or {}).get("segments") or [])
    plan_path = (state.get("faces") or {}).get("plan_path")
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8")) if plan_path else {}
    frames = {int(k) for k in (plan.get("frames") or {})}
    fps = float(plan.get("fps") or 29.97)
    duration = float((state.get("demux") or {}).get("duration") or 0)
    if not duration:
        duration = max(float(s["end"]) for s in segs) + 5

    best = []
    t = 0.0
    while t + WINDOW <= duration:
        lo, hi = t, t + WINDOW
        inside = [s for s in segs
                  if float(s["end"]) > lo and float(s["start"]) < hi]
        if not inside:
            t += STEP
            continue
        speech = sum(min(float(s["end"]), hi) - max(float(s["start"]), lo)
                     for s in inside)
        f0, f1 = int(lo * fps), int(hi * fps)
        cover = sum(1 for f in range(f0, f1) if f in frames) / max(f1 - f0, 1)
        cut = sum(float(s.get("overrun") or 0) for s in inside)
        ratios = [float(s.get("fit_ratio") or 1.0) for s in inside]
        speed = sum(ratios) / len(ratios)
        genders = {s.get("gender") for s in inside}
        text = "".join(s.get("text_translated") or "" for s in inside)
        explicit = bool(EXPLICIT.search(text))

        score = (2.0 * cover + 1.5 * min(speech / WINDOW, 1.0)
                 - 1.0 * min(cut, 3.0) - 0.8 * abs(speed - 1.0)
                 + (0.5 if len(genders) > 1 else 0.0)
                 - (1.5 if explicit else 0.0))
        best.append({"start": round(lo, 1), "score": round(score, 3),
                     "coverage": round(cover, 3),
                     "density": round(speech / WINDOW, 3),
                     "cut": round(cut, 2), "speed": round(speed, 3),
                     "speakers": len(genders), "explicit": explicit,
                     "lines": len(inside)})
        t += STEP

    best.sort(key=lambda w: -w["score"])
    # Keep the picks apart so three clips do not show the same 30 seconds.
    picks: list[dict] = []
    for w in best:
        if all(abs(w["start"] - p["start"]) >= WINDOW for p in picks):
            picks.append(w)
        if len(picks) >= args.n:
            break

    print(f"{len(best)} candidate windows; picked {len(picks)}")
    for w in picks:
        print(f"  {w['start']:>6.1f}s score={w['score']:.2f} cover={w['coverage']:.2f} "
              f"density={w['density']:.2f} cut={w['cut']:.1f}s speed={w['speed']:.2f} "
              f"speakers={w['speakers']}{'  [EXPLICIT]' if w['explicit'] else ''}")

    for k, w in enumerate(picks, 1):
        dst = out / f"demo{k}_{int(w['start'])}-{int(w['start'] + WINDOW)}s.mp4"
        ff(["-ss", str(w["start"]), "-i", dubbed, "-t", str(WINDOW),
            "-c:v", "libx264", "-crf", "20", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k", str(dst)])
        print(f"  wrote {dst.name}")

    # Side-by-side against the untouched source: the clearest way to show what
    # changed, since both mouths are on screen at the same instant.
    if source and Path(ROOT / source).exists() and picks:
        w = picks[0]
        dst = out / f"demo_对比_原片vs{args.label}_{int(w['start'])}-{int(w['start'] + WINDOW)}s.mp4"
        ff(["-ss", str(w["start"]), "-i", str(ROOT / source),
            "-ss", str(w["start"]), "-i", dubbed, "-t", str(WINDOW),
            "-filter_complex",
            "[0:v]scale=960:-2,pad=960:1080:(ow-iw)/2:(oh-ih)/2[l];"
            "[1:v]scale=960:-2,pad=960:1080:(ow-iw)/2:(oh-ih)/2[r];"
            "[l][r]hstack=inputs=2[v]",
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-crf", "20", "-preset", "medium",
            "-c:a", "aac", "-b:a", "192k", str(dst)])
        print(f"  wrote {dst.name}")

    (out / "windows.json").write_text(
        json.dumps({"picked": picks, "all": best[:40]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
