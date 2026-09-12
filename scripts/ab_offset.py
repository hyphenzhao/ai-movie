#!/usr/bin/env python
"""Sweep the A/V offset knob on one clip and measure the rendered mouth's lag.

One 30 s clip is rendered once per offset (−160 … +160 ms) in a single
MuseTalk process, each with its driving audio shifted by that offset.  For
every render the mouth-openness series (BiSeNet mouth-interior area inside
the face box, per frame) is cross-correlated with the driving audio's
per-frame RMS envelope; the lag at the correlation peak is the rendered
mouth's lead/lag against its audio.  A correct pipeline shows lag ≈ 0 at
offset 0 and lags of ∓offset/frame at ±offset — that sanity check is what
validates the metric.

Note this measures the model's mouth against *its own driving audio*
(internal sync), not against the original actor.

    python scripts/ab_offset.py workspace/test_1/state.json --start 47 --dur 20
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
    ap.add_argument("--start", type=float, default=None,
                    help="clip start (s); default: first synced range")
    ap.add_argument("--dur", type=float, default=20.0)
    ap.add_argument("--offsets", default="-160,-80,0,80,160")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-lag", type=int, default=10)
    ap.add_argument("--metric-only", action="store_true",
                    help="skip rendering; measure existing renders in <out>")
    args = ap.parse_args()

    import soundfile as sf
    import torch
    from ai_movie import lip_sync as ls
    from ai_movie.composer import build_speech_track
    from ai_movie import face_restore as fr

    state_path = Path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    plan = json.loads(Path(state["faces"]["plan_path"]).read_text(encoding="utf-8"))
    segs = None
    for k in ("fit", "compact", "tts"):
        if state.get(k) and state[k].get("segments"):
            segs = state[k]["segments"]
            break
    video = Path(state.get("_video") or "")
    if not video.exists():
        video = ROOT / "inputs" / f"{state_path.parent.name}.mp4"
    out = Path(args.out) if args.out else state_path.parent / "ab_offset"
    out.mkdir(parents=True, exist_ok=True)

    fps = ls._probe_fps(video)
    exact = ls._probe_fps_exact(video)
    if args.start is None:
        rng = plan.get("sync_ranges") or [[0.0, args.dur]]
        args.start = float(rng[0][0])
    start, dur = args.start, args.dur
    offsets = [int(x) for x in args.offsets.split(",")]

    speech = out / "speech_track.wav"
    vclip = out / "clip_orig.mp4"
    bjson = out / "clip_bbox.json"
    if not args.metric_only:
        build_speech_track(segs, speech)
        ls._cut_video_clip(video, start, dur, vclip, reencode=True, fps=exact)
        if ls._write_clip_bbox_json(plan, start, dur, fps, bjson) is None:
            print("no anchored frames in this window — pick another --start")
            return 1
        tasks = []
        for off in offsets:
            aclip = out / f"audio_{off:+d}.wav"
            ls._cut_audio_clip(speech, start, dur, aclip, offset_ms=off)
            tasks.append({"video": vclip, "audio": aclip, "bbox_json": bjson,
                          "output": out / f"render_{off:+d}.mp4"})
        res = ls.musetalk_sync_batch(tasks, fps=int(round(fps)), use_float16=True, batch_size=4)
        import shutil
        for i, off in enumerate(offsets):
            dst = out / f"render_{off:+d}.mp4"
            if i in res and Path(res[i]).resolve() != dst.resolve():
                shutil.copy2(str(res[i]), str(dst))

    # ── mouth openness vs audio envelope ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = fr._load_face_parser(device)
    bb = json.loads(bjson.read_text())["frames"]

    def openness(path):
        cap = cv2.VideoCapture(str(path))
        vals = []
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            box = bb.get(str(i))
            if box is None:
                vals.append(np.nan)
            else:
                x1, y1, x2, y2 = [int(t) for t in box]
                S = max(x2 - x1, y2 - y1)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                sx, sy = max(0, cx - S // 2), max(0, cy - S // 2)
                crop = f[sy:sy + S, sx:sx + S]
                if crop.size == 0:
                    vals.append(np.nan)
                else:
                    par = fr._parse_crop(crop, parser, device)
                    vals.append(float((par == 11).mean()))
            i += 1
        cap.release()
        return np.array(vals, dtype=np.float64)

    def envelope(path, n):
        a, sr = sf.read(str(path), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        hop = sr / fps
        env = np.array([np.sqrt(np.mean(a[int(k * hop):int((k + 1) * hop)] ** 2) + 1e-12)
                        for k in range(n)])
        return env

    def xcorr_lag(m, e, max_lag):
        """Masked normalised cross-correlation: only frames where the driven
        face is on screen count (an off-camera speaker's audio has no mouth
        to move and would otherwise swamp the correlation)."""
        ok = ~np.isnan(m)
        k = np.ones(3) / 3
        mm = np.where(ok, m, 0.0)
        mm = np.convolve(mm, k, "same")
        ee = np.convolve(e, k, "same")
        best, corr0, best_c = 0, 0.0, -9
        table = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                a, b, msk = mm[lag:], ee[:len(ee) - lag], ok[lag:]
            else:
                a, b, msk = mm[:lag], ee[-lag:], ok[:lag]
            a, b = a[msk], b[msk]
            if len(a) < 10:
                c = 0.0
            else:
                a = (a - a.mean()) / (a.std() + 1e-9)
                b = (b - b.mean()) / (b.std() + 1e-9)
                c = float(np.mean(a * b))
            table[lag] = round(c, 3)
            if lag == 0:
                corr0 = c
            if c > best_c:
                best_c, best = c, lag
        return best, corr0, best_c, table

    rows = []
    series = {}
    for off in offsets:
        p = out / f"render_{off:+d}.mp4"
        if not p.exists():
            continue
        m = openness(p)
        series[off] = m
        e = envelope(out / f"audio_{off:+d}.wav", len(m))
        lag, c0, cmax, table = xcorr_lag(m, e, args.max_lag)
        rows.append({"offset_ms": off, "best_lag_frames": lag,
                     "best_lag_ms": round(lag * 1000 / fps, 1),
                     "corr_at_0": round(c0, 3), "corr_max": round(cmax, 3),
                     "expected_lag_frames": round(-off * fps / 1000, 1)})
        print(f"offset {off:+5d} ms → mouth lag {lag:+d} frames ({rows[-1]['best_lag_ms']:+.0f} ms), "
              f"corr@0={c0:.3f} max={cmax:.3f} (expected {rows[-1]['expected_lag_frames']:+.1f})")
    # Relative check: the mouth series of an offset render should be the
    # 0-offset series shifted by that offset.  This validates that the knob
    # really moves the mouth even when the absolute audio metric is weak.
    rel = []
    if 0 in series:
        base = series[0]
        for off, m in series.items():
            if off == 0:
                continue
            lag, c0, cmax, _ = xcorr_lag(m, np.where(np.isnan(base), 0.0, base), args.max_lag)
            rel.append({"offset_ms": off, "lag_vs_zero_frames": lag,
                        "expected_frames": round(off * fps / 1000, 1), "corr_max": round(cmax, 3)})
            print(f"mouth({off:+d}) vs mouth(0): lag {lag:+d} frames (expected "
                  f"{rel[-1]['expected_frames']:+.1f}), corr {cmax:.3f}")
    (out / "ab_offset.json").write_text(json.dumps(
        {"clip": {"start": start, "dur": dur, "fps": fps}, "rows": rows, "relative": rel},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out / 'ab_offset.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
