#!/usr/bin/env python3
"""Plan (and optionally cut) a long film into chunks at dialogue pauses.

    python scripts/smart_split.py inputs/FILM.mp4 --name FILM            # plan only
    python scripts/smart_split.py inputs/FILM.mp4 --name FILM --cut      # + write chunks

A fixed 6-minute grid cuts through sentences: the line straddling the cut is
transcribed twice as two fragments, translated as two sentences and dubbed
with a seam.  Instead the cut goes where nobody is talking:

1. Silero VAD over the whole 16 kHz mono track (CPU, a few minutes).
2. Walking forward, each cut is searched in ``[target-slack, target+slack]``
   after the previous one; the candidate gaps are the non-speech stretches in
   that window, best = longest (capped, so a 40 s gap does not beat a 6 s gap
   that is far closer to the target), ties broken by distance to the target.
3. The cut is snapped to a video keyframe *inside* the gap (kept ``margin`` s
   away from speech), so chunks are stream-copied — no re-encode, exact
   concat.  A gap without a keyframe is skipped for the next best one; if the
   window has none, the window widens once, then the least-bad keyframe
   (farthest from speech) is taken and flagged ``forced``.

Output: ``workspace/<name>/_split/plan.json`` (+ ``chunks/<name>_pNN.mp4``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GAP_CAP = 8.0          # a gap longer than this earns no extra credit
MARGIN = 0.35          # keep the cut this far from detected speech


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def duration(video: Path) -> float:
    return float(_run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                       "-of", "csv=p=0", str(video)]).strip())


def keyframes(video: Path, cache: Path) -> list[float]:
    if cache.exists():
        return json.loads(cache.read_text())
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
                "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(video)])
    ks = sorted(float(x.split(",")[0]) for x in out.split() if x.strip(", "))
    cache.write_text(json.dumps(ks))
    return ks


def speech_spans(video: Path, work: Path) -> list[tuple[float, float]]:
    cache = work / "vad.json"
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]
    wav = work / "audio16k.wav"
    if not wav.exists():
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1",
                        "-ar", "16000", str(wav)], check=True)
    import soundfile as sf
    import torch
    from ai_movie.asr import _vad_detect
    torch.set_num_threads(8)
    y, sr = sf.read(str(wav), dtype="float32")
    assert sr == 16000
    spans: list[tuple[float, float]] = []
    block = 20 * 60 * sr                       # VAD state is local; 20-min blocks keep memory flat
    for off in range(0, len(y), block):
        part = torch.from_numpy(y[off:off + block])
        for s in _vad_detect(part, speech_pad_ms=100):
            spans.append((off / sr + s["start"], off / sr + s["end"]))
    spans.sort()
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    cache.write_text(json.dumps(merged))
    return [tuple(x) for x in merged]


def gaps_of(spans, total: float) -> list[tuple[float, float]]:
    out, prev = [], 0.0
    for a, b in spans:
        if a - prev > 0:
            out.append((prev, a))
        prev = max(prev, b)
    if total - prev > 0:
        out.append((prev, total))
    return out


def pick_cut(gaps, ks, lo: float, hi: float, target: float):
    """Best (time, gap_len) keyframe cut in [lo, hi], or None."""
    import bisect
    best = None
    for a, b in gaps:
        if b < lo or a > hi:
            continue
        ga, gb = a + MARGIN, b - MARGIN
        if gb <= ga:
            continue
        i = bisect.bisect_left(ks, max(ga, lo))
        cands = []
        while i < len(ks) and ks[i] <= min(gb, hi):
            cands.append(ks[i]); i += 1
        if not cands:
            continue
        mid = (ga + gb) / 2
        k = min(cands, key=lambda t: (abs(t - mid) > (gb - ga) / 4, abs(t - target)))
        score = min(b - a, GAP_CAP) - abs(k - target) / 60.0      # 1 min off target ≈ 1 s of gap
        if best is None or score > best[0]:
            best = (score, k, b - a)
    return None if best is None else (best[1], best[2])


def plan(video: Path, work: Path, target: float, slack: float) -> dict:
    total = duration(video)
    spans = speech_spans(video, work)
    ks = keyframes(video, work / "keyframes.json")
    gaps = gaps_of(spans, total)
    cuts, notes, t0 = [], [], 0.0
    while total - t0 > target + slack:
        tgt = t0 + target
        got = pick_cut(gaps, ks, tgt - slack, tgt + slack, tgt) \
            or pick_cut(gaps, ks, tgt - 1.6 * slack, tgt + 1.6 * slack, tgt)
        forced = got is None
        if forced:                                   # no silent keyframe: the one farthest from speech
            import bisect
            cand = [k for k in ks if tgt - slack <= k <= tgt + slack] or [min(ks, key=lambda k: abs(k - tgt))]
            starts = [s for s, _ in spans]
            def clearance(k):
                j = bisect.bisect_right(starts, k) - 1
                inside = j >= 0 and spans[j][0] <= k <= spans[j][1]
                return -1.0 if inside else min([abs(k - e) for _, e in spans[max(0, j):j + 1]] +
                                               [abs(s - k) for s, _ in spans[j + 1:j + 2]] + [99.0])
            k = max(cand, key=clearance)
            got = (k, 0.0)
        cuts.append(got[0]); notes.append({"gap": round(got[1], 2), "forced": forced})
        t0 = got[0]
    edges = [0.0] + cuts + [total]
    chunks = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        sp = sum(max(0.0, min(e, b) - max(s, a)) for s, e in spans)
        chunks.append({"index": i + 1, "start": round(a, 3), "end": round(b, 3),
                       "minutes": round((b - a) / 60, 2), "speech_minutes": round(sp / 60, 2),
                       **({"cut_gap_s": notes[i]["gap"], "forced": notes[i]["forced"]} if i < len(cuts) else {})})
    return {"video": str(video), "duration": total, "target_s": target, "slack_s": slack,
            "speech_minutes": round(sum(e - s for s, e in spans) / 60, 1), "chunks": chunks}


def cut(video: Path, doc: dict, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in doc["chunks"]:
        dst = out_dir / f"{name}_p{c['index']:02d}.mp4"
        c["file"] = str(dst)
        if dst.exists() and dst.stat().st_size > 0:
            continue
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{c['start']:.3f}", "-to", f"{c['end']:.3f}",
                        "-i", str(video), "-map", "0:v:0", "-map", "0:a:0", "-c", "copy",
                        "-avoid_negative_ts", "make_zero", str(dst)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--name", default=None)
    ap.add_argument("--target-min", type=float, default=8.0)
    ap.add_argument("--slack-min", type=float, default=2.5)
    ap.add_argument("--cut", action="store_true")
    args = ap.parse_args()
    video = Path(args.video)
    name = args.name or video.stem
    work = ROOT / "workspace" / name / "_split"
    work.mkdir(parents=True, exist_ok=True)
    doc = plan(video, work, args.target_min * 60, args.slack_min * 60)
    if args.cut:
        cut(video, doc, work / "chunks", name)
    (work / "plan.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    forced = sum(1 for c in doc["chunks"] if c.get("forced"))
    gaps = [c["cut_gap_s"] for c in doc["chunks"] if "cut_gap_s" in c]
    print(f"{len(doc['chunks'])} chunks, speech {doc['speech_minutes']} min of {doc['duration']/60:.1f}; "
          f"cut gaps median {sorted(gaps)[len(gaps)//2] if gaps else 0:.1f}s min {min(gaps) if gaps else 0:.1f}s; forced {forced}")
    for c in doc["chunks"]:
        print(f"  p{c['index']:02d} {c['start']/60:7.2f}–{c['end']/60:7.2f} min  {c['minutes']:5.2f} min  "
              f"speech {c['speech_minutes']:5.2f}  gap {c.get('cut_gap_s', '-')}{'  FORCED' if c.get('forced') else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
