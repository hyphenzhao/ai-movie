#!/usr/bin/env python
"""Step 3 of the long-film run: passthrough chunks + lossless concat + the film-wide segment table.

Extracted verbatim from the ``run_long.sh`` heredoc so it can be run on its
own — a voice-conversion-only rerun of the chunks (``scripts/run_vc_only.sh``)
must re-assemble the film without re-entering the chunk loop.

    python scripts/concat_long.py NAME            # workspace/NAME/_split/ → deliver/NAME_dubbed_full.mp4

Behaviour is unchanged from the shell version: a chunk without a dubbed
output is passed through (original picture re-encoded once, original sound
shifted by the dubbed chunks' median loudness offset), a chunk file that
starts one GOP early is trimmed in whole frames, everything is stream-copied
into one mp4, and ``deliver/NAME_full/segments_full.csv`` lists every line in
film time.
"""

from __future__ import annotations

import bisect
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(c):
    subprocess.run(c, check=True)


def probe(p, entries, stream):
    return subprocess.run(["ffprobe", "-v", "error", "-select_streams", stream, "-show_entries", entries,
                           "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: concat_long.py NAME [SPLIT_DIR]", file=sys.stderr)
        return 2
    name = sys.argv[1]
    split = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "workspace" / name / "_split"
    root = split.parents[2]
    plan = json.loads((split / "plan.json").read_text())
    (root / "deliver" / f"{name}_full").mkdir(parents=True, exist_ok=True)
    # A chunk file that is longer than planned starts one GOP early (see smart_split.cut): that lead duplicates
    # the previous chunk's tail and has to come off again, in whole frames.
    ks = json.loads((split / "keyframes.json").read_text()) if (split / "keyframes.json").exists() else []
    lead = {}
    for c in plan["chunks"]:
        have = float(probe(c["file"], "format=duration", "v:0") or 0)
        if ks and have - (c["end"] - c["start"]) > 1.0:
            j = bisect.bisect_left(ks, c["start"] - 1e-3)
            if j > 0:
                lead[c["index"]] = c["start"] - ks[j - 1] if abs(ks[j] - c["start"]) < 2e-3 else 0.0
    lead = {k: v for k, v in lead.items() if v > 0.05}
    if lead:
        print("chunks that start a GOP early → trimmed:", {k: round(v, 3) for k, v in lead.items()})
    finals, offsets = {}, []
    for c in plan["chunks"]:
        w = root / "workspace" / f"{name}_p{c['index']:02d}"
        f = w / "output" / "v2_cloned_dubbed.mp4"
        if not (f.exists() and f.stat().st_size):
            f = w / "output" / f"{name}_p{c['index']:02d}_dubbed.mp4"      # built-in voices beat no dub at all
        if f.exists() and f.stat().st_size and (w / "state.json").exists():
            finals[c["index"]] = f
            st = json.loads((w / "state.json").read_text())
            off = ((st.get("vc") or {}).get("mix") or {}).get("global_offset_db", (st.get("mix") or {}).get("global_offset_db"))
            if isinstance(off, (int, float)):
                offsets.append(off)
    gain = statistics.median(offsets) if offsets else 0.0
    ref = next(iter(finals.values()), None)
    fps = probe(ref, "stream=r_frame_rate", "v:0") if ref else probe(plan["video"], "stream=r_frame_rate", "v:0")
    wh = probe(ref, "stream=width,height", "v:0") if ref else ""
    print(f"dubbed chunks {len(finals)}/{len(plan['chunks'])}; passthrough gain {gain:+.2f} dB; fps {fps} {wh}")
    parts = []
    for c in plan["chunks"]:
        i = c["index"]; src = finals.get(i)
        ss = ["-ss", f"{lead[i] - 0.005:.3f}"] if i in lead else []         # output seek: frame-accurate
        if src is None:
            src = split / "pass" / f"{name}_p{i:02d}.mp4"; src.parent.mkdir(exist_ok=True)
            if not src.exists():
                vf = ["-vf", f"scale={wh.replace(',', ':')}"] if wh else []
                tmp = src.with_suffix(".part.mp4")                          # a killed encode must not look finished
                run(["ffmpeg", "-y", "-v", "error", "-i", c["file"], *ss, "-map", "0:v:0", "-map", "0:a:0", *vf,
                     "-r", fps, "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                     "-af", f"volume={gain}dB,alimiter=limit=0.89", "-ar", "44100", "-ac", "2", "-c:a", "aac", "-b:a", "192k", str(tmp)])
                tmp.replace(src)
        elif ss:
            trimmed = split / "trim" / f"{name}_p{i:02d}.mp4"; trimmed.parent.mkdir(exist_ok=True)
            if not trimmed.exists():
                tmp = trimmed.with_suffix(".part.mp4")
                run(["ffmpeg", "-y", "-v", "error", "-i", str(src), *ss, "-map", "0:v:0", "-map", "0:a:0",
                     "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                     "-c:a", "aac", "-b:a", "192k", str(tmp)])
                tmp.replace(trimmed)
            src = trimmed
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
        if not sp.exists():
            continue
        st = json.loads(sp.read_text())
        segs = ((st.get("vc") or {}).get("segments") or (st.get("fit") or {}).get("segments")
                or (st.get("translate") or {}).get("segments") or (st.get("asr") or {}).get("segments") or [])
        off = c["start"] - lead.get(c["index"], 0.0)
        for n, s in enumerate(segs):
            if s["end"] <= lead.get(c["index"], 0.0):
                continue                                                    # inside the trimmed lead
            rows.append([c["index"], n, round(off + s["start"], 2), round(off + s["end"], 2),
                         s.get("speaker", ""), s.get("gender", ""), s.get("text", ""), s.get("text_translated") or s.get("translation", "")])
    full = root / "deliver" / f"{name}_full"
    with open(full / "segments_full.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["chunk", "seg", "start", "end", "speaker", "gender", "ja", "zh"]); w.writerows(rows)
    (full / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} segments → {full / 'segments_full.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
