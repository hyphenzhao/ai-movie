#!/usr/bin/env python3
"""Find burned-in subtitles and export them as glyph-only strips (no picture content).

    python scripts/screen_subs.py inputs/FILM.mp4 --srt inputs/subs/FILM.zh.srt --out workspace/FILM/_screen
    python scripts/screen_subs.py inputs/FILM.mp4 --every 1.0 --out workspace/FILM/_screen

At each probe time (SRT cue midpoints, or a fixed stride) the bottom band is
binarised: a pixel survives only if it is bright, unsaturated *and* touches a
dark outline — the way subtitle glyphs are drawn, and the way skin, sheets
and lamps are not.  Text rows are located on that mask, cropped and stacked
into tiles with the probe id in the margin, black on white.  What leaves the
machine is therefore letter shapes only, which is what the metered link and
the content rules allow.  ``scan.json`` records, per probe, whether text was
found and the row bounds; the tiles are read (by a person or a model) into
``<out>/screen.json`` as ``{"id": ["zh line", "ja line"]}``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

BAND = (860, 1080)          # rows of a 1080p frame that can hold subtitles
XR = (300, 1620)            # subtitles are centred; the margins only add noise


def srt_cues(path: Path) -> list[dict]:
    s = path.read_text(encoding="utf-8", errors="replace")

    def t(x):
        h, m, r = x.split(":")
        sec, ms = r.replace(".", ",").split(",")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000
    return [{"id": i, "start": t(a), "end": t(b), "zh": re.sub(r"\s+", " ", txt.strip())}
            for i, (a, b, txt) in enumerate(re.findall(
                r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)\s*\n(.*?)(?:\n\s*\n|\Z)", s, re.S))]


def frame_at(video: Path, t: float) -> np.ndarray | None:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
    if len(raw) != 1080 * 1920 * 3:
        return None
    return np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3)


def glyph_mask(frame: np.ndarray) -> np.ndarray:
    band = frame[BAND[0]:BAND[1], XR[0]:XR[1]]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 2] > 200) & (hsv[:, :, 1] < 50)).astype(np.uint8)
    dark = (hsv[:, :, 2] < 80).astype(np.uint8)
    ring = cv2.dilate(white, np.ones((5, 5), np.uint8)) & dark
    return white & cv2.dilate(ring, np.ones((7, 7), np.uint8))


def text_lines(glyph: np.ndarray) -> list[tuple[int, int]]:
    rows = np.where(glyph.sum(1) > 8)[0]
    runs, out = [], []
    if len(rows):
        run = [rows[0]]
        for r in rows[1:]:
            if r - run[-1] <= 3:
                run.append(r)
            else:
                runs.append((run[0], run[-1])); run = [r]
        runs.append((run[0], run[-1]))
    for a, b in runs:
        blk = glyph[a:b + 1]
        cols = np.where(blk.sum(0) > 0)[0]
        if 16 <= b - a <= 70 and blk.sum() > 150 and len(cols) and np.ptp(cols) > 120:
            out.append((int(a), int(b)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--srt", type=Path, default=None, help="probe at these cues' midpoints")
    ap.add_argument("--every", type=float, default=None, help="or probe every N seconds")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-tile", type=int, default=12)
    args = ap.parse_args()
    video = Path(args.video)
    out = args.out; (out / "strips").mkdir(parents=True, exist_ok=True); (out / "tiles").mkdir(exist_ok=True)
    if args.srt:
        probes = srt_cues(args.srt)
        for p in probes:
            p["t"] = (p["start"] + p["end"]) / 2
    else:
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(video)], capture_output=True, text=True).stdout)
        step = args.every or 1.0
        probes = [{"id": i, "t": i * step, "start": i * step, "end": (i + 1) * step} for i in range(int(dur / step))]
    scan = []
    for p in probes:
        f = frame_at(video, p["t"])
        lines = text_lines(glyph_mask(f)) if f is not None else []
        rec = {**p, "sub": bool(lines), "lines": [(a + BAND[0], b + BAND[0]) for a, b in lines]}
        if lines:
            g = glyph_mask(f)
            strip = np.vstack([np.vstack([g[a:b + 1] * 255, np.zeros((6, g.shape[1]), np.uint8)]) for a, b in lines])
            cv2.imwrite(str(out / "strips" / f"c{p['id']:04d}.png"), strip)
        scan.append(rec)
        if p["id"] % 100 == 0:
            print(p["id"], rec["sub"], flush=True)
    (out / "scan.json").write_text(json.dumps(scan, ensure_ascii=False), encoding="utf-8")
    files = sorted((out / "strips").glob("c*.png")); index = []
    for t in range(0, len(files), args.per_tile):
        rows = []
        for fp in files[t:t + args.per_tile]:
            im = cv2.imread(str(fp), 0); cid = int(fp.stem[1:])
            lab = np.zeros((im.shape[0], 110), np.uint8)
            cv2.putText(lab, str(cid), (2, min(30, im.shape[0] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2)
            rows += [np.hstack([lab, im]), np.full((3, im.shape[1] + 110), 90, np.uint8)]
        tile = np.vstack(rows)
        tile = cv2.resize(tile, (tile.shape[1] * 3 // 4, tile.shape[0] * 3 // 4), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / "tiles" / f"tile_{t // args.per_tile:02d}.png"), 255 - tile)
        index.append([t // args.per_tile, [int(fp.stem[1:]) for fp in files[t:t + args.per_tile]]])
    (out / "tiles" / "index.json").write_text(json.dumps(index))
    print(f"{sum(1 for r in scan if r['sub'])}/{len(scan)} probes have burned-in text → {len(index)} tiles in {out / 'tiles'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
