#!/usr/bin/env python
"""A/B the MuseTalk paste modes (and the VAE-only ablation) on a few clips.

Picks one clip per face category from a finished workspace (frontal, 3/4
profile, small face, big face, cut-heavy), renders every clip with each
variant in ONE MuseTalk process, and measures:

  * sharp   — mouth-region sharpness ratio (Laplacian variance of the lower
              40 % of the face box at 96×48, output / source; median over
              anchored frames).  1.0 = as sharp as the original.
  * seam    — mean gradient magnitude in a ring around the paste boundary,
              output / source.  >1 means the paste adds an edge (halo).
  * psnr    — vae_only only: PSNR of the pasted face box vs the source, i.e.
              the loss of the 256² VAE round-trip + paste alone.

    python scripts/ab_fusion.py workspace/test_2/state.json --clips 5

Writes <out>/ab_fusion.json and a contact sheet <out>/ab_fusion.png.
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

CATS = ["frontal", "three_quarter", "small", "big", "cut_heavy"]


def _segments(state):
    for k in ("fit", "compact", "tts"):
        if state.get(k) and state[k].get("segments"):
            return state[k]["segments"]
    raise SystemExit("no synthesized segments in state")


def _source_video(state, state_path: Path) -> Path:
    v = state.get("_video")
    if v and Path(v).exists():
        return Path(v)
    cand = ROOT / "inputs" / f"{state_path.parent.name}.mp4"
    if cand.exists():
        return cand
    return Path(state["demux"]["video"])


def pick_clips(state, plan, segs, n: int, clip_sec: float):
    """One (segment, category) per category, longest usable first."""
    fps = plan["fps"]
    frames = plan["frames"]
    yaw = plan.get("frame_yaw") or {}
    seg_cuts = plan.get("segment_cuts") or {}
    rows = []
    for i, s in enumerate(segs):
        a, b = int(float(s["start"]) * fps), int(float(s["end"]) * fps)
        ids = [f for f in range(a, b + 1) if str(f) in frames]
        if len(ids) < int(fps * 1.0):
            continue
        ws = [frames[str(f)][2] - frames[str(f)][0] for f in ids]
        ys = [abs(yaw[str(f)]) for f in ids if str(f) in yaw]
        rows.append({"idx": i, "start": float(s["start"]), "end": float(s["end"]),
                     "n": len(ids), "w": float(np.median(ws)),
                     "yaw": float(np.median(ys)) if ys else 0.0,
                     "cuts": int(seg_cuts.get(str(i), 0))})
    picks = {}
    for r in sorted(rows, key=lambda r: -r["n"]):
        cat = None
        if r["cuts"] >= 1 and "cut_heavy" not in picks:
            cat = "cut_heavy"
        elif r["w"] < 120 and "small" not in picks:
            cat = "small"
        elif r["w"] >= 300 and r["yaw"] < 20 and "big" not in picks:
            cat = "big"
        elif 30 <= r["yaw"] <= 50 and "three_quarter" not in picks:
            cat = "three_quarter"
        elif r["yaw"] < 15 and r["w"] >= 150 and "frontal" not in picks:
            cat = "frontal"
        if cat:
            picks[cat] = r
        if len(picks) >= n:
            break
    if len(picks) < n:
        for r in sorted(rows, key=lambda r: -r["n"]):
            if r["idx"] not in {p["idx"] for p in picks.values()}:
                picks[f"extra{len(picks)}"] = r
            if len(picks) >= n:
                break
    out = []
    for cat, r in picks.items():
        s0 = max(0.0, r["start"] - 0.3)
        s1 = min(r["end"] + 0.3, s0 + clip_sec)
        out.append({"cat": cat, "seg": r["idx"], "start": s0, "end": s1,
                    "w": round(r["w"]), "yaw": round(r["yaw"], 1), "cuts": r["cuts"]})
    return out


def mouth_crop(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h = y2 - y1
    my1 = y1 + int(h * 0.60)
    c = frame[max(0, my1):max(0, y2), max(0, x1):max(0, x2)]
    if c.size == 0:
        return None
    g = cv2.cvtColor(cv2.resize(c, (96, 48), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return g


def lap_var(g):
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def seam_ring(frame, box, frac=0.06):
    """Mean |∇| on a ring around the lower half of the face box."""
    x1, y1, x2, y2 = [int(v) for v in box]
    w = max(1, x2 - x1)
    r = max(2, int(frac * w))
    yy1 = y1 + (y2 - y1) // 2
    H, W = frame.shape[:2]
    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (max(0, x1 - r), max(0, yy1 - r)), (min(W - 1, x2 + r), min(H - 1, y2 + r)), 1, -1)
    cv2.rectangle(m, (max(0, x1 + r), max(0, yy1 + r)), (min(W - 1, x2 - r), min(H - 1, y2 - r)), 0, -1)
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    sel = m > 0
    return float(mag[sel].mean()) if sel.any() else 0.0


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse < 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("state")
    ap.add_argument("--clips", type=int, default=5)
    ap.add_argument("--clip-sec", type=float, default=4.0)
    ap.add_argument("--variants", default="alpha,laplacian,vae_only")
    ap.add_argument("--out", default=None)
    ap.add_argument("--metric-only", default=None,
                    help="skip rendering; measure an existing <out>/renders dir")
    args = ap.parse_args()

    from ai_movie import lip_sync as ls
    from ai_movie.composer import build_speech_track

    state_path = Path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    plan = json.loads(Path(state["faces"]["plan_path"]).read_text(encoding="utf-8"))
    segs = _segments(state)
    video = _source_video(state, state_path)
    out = Path(args.out) if args.out else state_path.parent / "ab_fusion"
    out.mkdir(parents=True, exist_ok=True)
    renders = out / "renders"
    renders.mkdir(exist_ok=True)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    clips = pick_clips(state, plan, segs, args.clips, args.clip_sec)
    print("clips:", json.dumps(clips, ensure_ascii=False))
    fps = ls._probe_fps(video)
    exact = ls._probe_fps_exact(video)
    ms_fps = int(round(fps))

    speech = out / "speech_track.wav"
    if not speech.exists():
        build_speech_track(segs, speech)

    tasks, meta = [], []
    for ci, c in enumerate(clips):
        dur = c["end"] - c["start"]
        vclip = renders / f"clip{ci}_orig.mp4"
        aclip = renders / f"clip{ci}_audio.wav"
        bjson = renders / f"clip{ci}_bbox.json"
        if not vclip.exists():
            ls._cut_video_clip(video, c["start"], dur, vclip, reencode=True, fps=exact)
            ls._cut_audio_clip(speech, c["start"], dur, aclip)
            ls._write_clip_bbox_json(plan, c["start"], dur, fps, bjson)
        for v in variants:
            opts = {}
            if v == "laplacian":
                opts["fusion"] = "laplacian"
            elif v == "vae_only":
                opts["vae_only"] = True
            tasks.append({"video": vclip, "audio": aclip, "bbox_json": bjson,
                          "output": renders / f"clip{ci}_{v}.mp4", "opts": opts})
            meta.append((ci, v))

    if not args.metric_only:
        res = ls.musetalk_sync_batch(tasks, fps=ms_fps, use_float16=True, batch_size=4,
                                     fusion="laplacian" if "laplacian" in variants else None)
        import shutil
        for ti, p in res.items():
            ci, v = meta[ti]
            dst = renders / f"clip{ci}_{v}.mp4"
            if Path(p).resolve() != dst.resolve():
                shutil.copy2(str(p), str(dst))

    # ── metrics ──
    results = []
    sheet_rows = []
    for ci, c in enumerate(clips):
        src = read_frames(renders / f"clip{ci}_orig.mp4")
        bb = json.loads((renders / f"clip{ci}_bbox.json").read_text())["frames"]
        row = {"clip": ci, **c}
        tiles = []
        for v in variants:
            p = renders / f"clip{ci}_{v}.mp4"
            if not p.exists():
                row[v] = None
                continue
            outf = read_frames(p)
            n = min(len(src), len(outf))
            sharp, seam, ps = [], [], []
            mid = None
            for i in range(n):
                box = bb.get(str(i))
                if box is None:
                    continue
                gs, go = mouth_crop(src[i], box), mouth_crop(outf[i], box)
                if gs is None or go is None:
                    continue
                vs = lap_var(gs)
                if vs > 1e-6:
                    sharp.append(lap_var(go) / vs)
                ss = seam_ring(src[i], box)
                if ss > 1e-6:
                    seam.append(seam_ring(outf[i], box) / ss)
                x1, y1, x2, y2 = [int(t) for t in box]
                ps.append(psnr(outf[i][y1:y2, x1:x2], src[i][y1:y2, x1:x2]))
                if mid is None and i >= n // 2:
                    mid = (i, box)
            row[v] = {"sharp": round(float(np.median(sharp)), 3) if sharp else None,
                      "seam": round(float(np.median(seam)), 3) if seam else None,
                      "psnr": round(float(np.median(ps)), 2) if ps else None,
                      "frames": len(sharp)}
            if mid is not None:
                i, box = mid
                x1, y1, x2, y2 = [int(t) for t in box]
                pad = int(0.3 * (x2 - x1))
                tile = outf[i][max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                tiles.append(cv2.resize(tile, (256, 256)))
                if v == variants[0]:
                    ts = src[i][max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
                    tiles.insert(0, cv2.resize(ts, (256, 256)))
        results.append(row)
        if tiles:
            sheet_rows.append(np.hstack(tiles))
        print(f"clip{ci} {c['cat']:14s} w={c['w']:4d} yaw={c['yaw']:5.1f}  " +
              "  ".join(f"{v}: sharp={row[v]['sharp']} seam={row[v]['seam']} psnr={row[v]['psnr']}"
                        for v in variants if row.get(v)))

    (out / "ab_fusion.json").write_text(json.dumps(
        {"variants": variants, "columns": ["source"] + variants, "clips": results},
        ensure_ascii=False, indent=1), encoding="utf-8")
    if sheet_rows:
        w = max(r.shape[1] for r in sheet_rows)
        sheet = np.vstack([cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT)
                           for r in sheet_rows])
        cv2.imwrite(str(out / "ab_fusion.png"), sheet)
    print(f"wrote {out / 'ab_fusion.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
