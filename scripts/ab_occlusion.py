#!/usr/bin/env python
"""A/B the occlusion fallback: whole-frame revert (v2) vs region patch (v3).

Runs ``face_restore.occlusion_gate_video`` in both modes over the SAME
original / lip-synced pair (a finished workspace's ``lipsync.mp4`` against
the source), reports how many frames each mode touched, and dumps a few
side-by-side crops (original | frame-mode | region-mode) from the frames the
region mode patched, so the decision is made on pictures, not counts.

    python scripts/ab_occlusion.py workspace/test_1/state.json [--start S --dur D]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2                                          # noqa: E402
import numpy as np                                  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("state")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=None, help="default: whole film")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tiles", type=int, default=6)
    args = ap.parse_args()

    from ai_movie import face_restore as fr
    from ai_movie import lip_sync as ls

    state_path = Path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    video = Path(state.get("_video") or "")
    if not video.exists():
        video = ROOT / "inputs" / f"{state_path.parent.name}.mp4"
    lips = Path((state.get("lipsync") or {}).get("video") or "")
    if not lips.exists():
        print("no lipsync video in state")
        return 1
    plan = json.loads(Path(state["faces"]["plan_path"]).read_text(encoding="utf-8"))
    cuts = [int(c) for c in plan.get("cuts") or []]
    out = Path(args.out) if args.out else state_path.parent / "ab_occlusion"
    out.mkdir(parents=True, exist_ok=True)

    fps = ls._probe_fps(video)
    exact = ls._probe_fps_exact(video)
    dur = args.dur or (float(state["demux"]["duration"]) - args.start)
    o_clip, l_clip = out / "orig.mp4", out / "lips.mp4"
    ls._cut_video_clip(video, args.start, dur, o_clip, reencode=True, fps=exact)
    ls._cut_video_clip(lips, args.start, dur, l_clip, reencode=True, fps=exact)
    base = int(round(args.start * fps))
    n = int(round(dur * fps)) + 2
    from ai_movie.shots import local_cuts
    lc = local_cuts(cuts, base, n)

    res = {}
    for mode in ("frame", "region"):
        st: dict = {}
        fr.occlusion_gate_video(o_clip, l_clip, out / f"gated_{mode}.mp4",
                                occlusion_mode=mode, cuts=lc, stats=st, log_cb=print)
        res[mode] = st
        print(mode, st)

    # Contact sheet from frames the region mode patched.
    capO, capF, capR, capL = (cv2.VideoCapture(str(p)) for p in
                              (o_clip, out / "gated_frame.mp4", out / "gated_region.mp4", l_clip))
    tiles = []
    i = 0
    frames_bb = plan.get("frames") or {}
    while True:
        ok = [c.read() for c in (capO, capF, capR, capL)]
        if not all(o for o, _ in ok):
            break
        fo, ff, frg, fl = (f for _, f in ok)
        # a frame where region output differs from the raw lipsync but the
        # frame mode did not revert it (or did) — show all four
        d_region = float(np.abs(frg.astype(np.int16) - fl.astype(np.int16)).mean())
        d_frame = float(np.abs(ff.astype(np.int16) - fl.astype(np.int16)).mean())
        if d_region > 0.5 or d_frame > 0.5:
            box = frames_bb.get(str(base + i))
            if box:
                x1, y1, x2, y2 = [int(v) for v in box]
                pad = int(0.4 * (x2 - x1))
                sl = (slice(max(0, y1 - pad), y2 + pad), slice(max(0, x1 - pad), x2 + pad))
                row = np.hstack([cv2.resize(f[sl], (256, 256)) for f in (fo, fl, ff, frg)])
                cv2.putText(row, f"f{base + i} orig|lips|frame|region", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                tiles.append(row)
        i += 1
        if len(tiles) >= args.tiles * 40:
            break
    for c in (capO, capF, capR, capL):
        c.release()
    if tiles:
        step = max(1, len(tiles) // args.tiles)
        sheet = np.vstack(tiles[::step][:args.tiles])
        cv2.imwrite(str(out / "ab_occlusion.png"), sheet)
        print(f"contact sheet: {out / 'ab_occlusion.png'} ({len(tiles)} differing frames)")
    (out / "ab_occlusion.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
