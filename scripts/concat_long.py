#!/usr/bin/env python
"""Step 3 of the long-film run: passthrough chunks + lossless concat + the film-wide segment table.

Extracted verbatim from the ``run_long.sh`` heredoc so it can be run on its
own — a voice-conversion-only rerun of the chunks (``scripts/run_vc_only.sh``)
must re-assemble the film without re-entering the chunk loop.

    python scripts/concat_long.py NAME            # workspace/NAME/_split/ → deliver/NAME_dubbed_full.mp4

As in the shell version, a chunk without a dubbed
output is passed through (original picture re-encoded once, original sound
shifted by the dubbed chunks' median loudness offset), a chunk file that
starts one GOP early is trimmed in whole frames, everything is stream-copied
into one mp4, and ``deliver/NAME_full/segments_full.csv`` lists every line in
film time.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import os
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


def file_signature(path: Path) -> dict:
    """Cheap identity for large media, including metadata and both file ends."""
    st = path.stat()
    size = 1024 * 1024
    with path.open("rb") as fh:
        head = fh.read(size)
        fh.seek(max(size, st.st_size - size))
        tail = fh.read(size)
    return {"path": str(path.absolute()), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "ends_sha256": hashlib.sha256(head + tail).hexdigest()}


def encode_cached(source: Path, destination: Path, settings: dict, command: list[str]) -> None:
    """Resume an encode only when both its inputs and completed output still match.

    Old trim/pass files have no proof and are rebuilt.  A killed ffmpeg leaves
    only a .part file, never a cache hit or a replacement for the last good file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    proof = destination.with_suffix(destination.suffix + ".cache.json")
    identity = {"version": 1, "source": file_signature(source), "settings": settings,
                "command": command}
    if destination.exists() and destination.stat().st_size and proof.exists():
        try:
            old = json.loads(proof.read_text(encoding="utf-8"))
            if old.get("input") == identity and old.get("output") == file_signature(destination):
                return
        except (OSError, ValueError, TypeError):
            pass
    temporary = destination.with_name(destination.stem + ".part" + destination.suffix)
    try:
        run([*command, str(temporary)])
        if not temporary.exists() or not temporary.stat().st_size:
            raise RuntimeError(f"encode produced no output: {destination}")
        if file_signature(source) != identity["source"]:
            raise RuntimeError(f"encode source changed while running: {source}")
        temporary.replace(destination)
        record = {"input": identity, "output": file_signature(destination)}
        tmp_proof = proof.with_suffix(proof.suffix + ".part")
        tmp_proof.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp_proof.replace(proof)
    finally:
        temporary.unlink(missing_ok=True)


def dubbed_source(state: dict, work: Path, root: Path, *, require_vc: bool = False) -> Path | None:
    """Use the current registered version, including an intentional v1 fallback.

    A previous cloned output may still be on disk after VC rejects every
    reference and registers the new built-in-voice compose as state.vc.video.
    The old filename must not take precedence over that explicit decision.
    """
    for key in (("vc",) if require_vc else ("vc", "compose")):
        registered = (state.get(key) or {}).get("video")
        if registered:
            path = Path(registered)
            if not path.is_absolute():
                path = root / path
            if not path.exists() or not path.stat().st_size:
                raise FileNotFoundError(f"current {key} output missing/empty: {path}")
            return path
    if require_vc:
        raise RuntimeError(f"current state has no vc.video: {work}")
    # Older states did not always record a video path; preserve their fallback.
    for path in (work / "output" / "v2_cloned_dubbed.mp4",
                 work / "output" / f"{work.name}_dubbed.mp4"):
        if path.exists() and path.stat().st_size:
            return path
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: concat_long.py NAME [SPLIT_DIR]", file=sys.stderr)
        return 2
    name = sys.argv[1]
    from run_evidence import atomic_json, file_record, min_speech_minutes
    minimum = min_speech_minutes()
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
        if c["speech_minutes"] < minimum:
            continue                    # a no-speech chunk must not reuse an old dubbed output
        w = root / "workspace" / f"{name}_p{c['index']:02d}"
        if not (w / "state.json").exists():
            if os.environ.get("STRICT_RUN") == "1" or os.environ.get("V34_RUN_DIR"):
                raise FileNotFoundError(f"speech chunk has no current state: {w}")
            continue
        st = json.loads((w / "state.json").read_text())
        f = dubbed_source(st, w, root, require_vc=(os.environ.get("STRICT_RUN") == "1"
                                                or bool(os.environ.get("V34_RUN_DIR"))))
        if f is not None:
            finals[c["index"]] = f
            off = ((st.get("vc") or {}).get("mix") or {}).get("global_offset_db", (st.get("mix") or {}).get("global_offset_db"))
            if isinstance(off, (int, float)):
                offsets.append(off)
        elif os.environ.get("STRICT_RUN") == "1" or os.environ.get("V34_RUN_DIR"):
            raise FileNotFoundError(f"speech chunk has no current output: {w}")
    gain = statistics.median(offsets) if offsets else 0.0
    ref = next(iter(finals.values()), None)
    fps = probe(ref, "stream=r_frame_rate", "v:0") if ref else probe(plan["video"], "stream=r_frame_rate", "v:0")
    wh = probe(ref, "stream=width,height", "v:0") if ref else ""
    print(f"dubbed chunks {len(finals)}/{len(plan['chunks'])}; passthrough gain {gain:+.2f} dB; fps {fps} {wh}")
    parts, manifest_parts = [], []
    records = {}
    def media_record(path):
        key = str(Path(path).resolve())
        if key not in records:
            records[key] = file_record(Path(path))
        return records[key]
    plan_record = file_record(split / "plan.json")
    for c in plan["chunks"]:
        i = c["index"]; src = finals.get(i)
        original = src or Path(c["file"])
        mode = "dub" if src is not None else "pass"
        ss = ["-ss", f"{lead[i] - 0.005:.3f}"] if i in lead else []         # output seek: frame-accurate
        if src is None:
            src = split / "pass" / f"{name}_p{i:02d}.mp4"
            vf = ["-vf", f"scale={wh.replace(',', ':')}"] if wh else []
            encode_cached(Path(c["file"]), src, {"lead": lead.get(i, 0), "gain_db": gain, "fps": fps, "wh": wh},
                ["ffmpeg", "-y", "-v", "error", "-i", c["file"], *ss, "-map", "0:v:0", "-map", "0:a:0", *vf,
                     "-r", fps, "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                     "-af", f"volume={gain}dB,alimiter=limit=0.89", "-ar", "44100", "-ac", "2", "-c:a", "aac", "-b:a", "192k"])
        elif ss:
            trimmed = split / "trim" / f"{name}_p{i:02d}.mp4"
            encode_cached(src, trimmed, {"lead": lead[i]},
                ["ffmpeg", "-y", "-v", "error", "-i", str(src), *ss, "-map", "0:v:0", "-map", "0:a:0",
                     "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                     "-c:a", "aac", "-b:a", "192k"])
            src = trimmed
        ts = split / "ts" / f"p{i:02d}.ts"; ts.parent.mkdir(exist_ok=True)
        run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "mpegts", str(ts)])
        parts.append(ts)
        manifest_parts.append({"name": f"{name}_p{i:02d}", "mode": mode,
                               "lead_seconds": lead.get(i, 0), "source": media_record(original),
                               "processed": media_record(src), "transport": media_record(ts)})
    out = root / "deliver" / f"{name}_dubbed_full.mp4"
    temporary = out.with_suffix(".part.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-i", "concat:" + "|".join(map(str, parts)), "-c", "copy",
         "-bsf:a", "aac_adtstoasc", "-movflags", "+faststart", str(temporary)])
    d_out = float(probe(temporary, "format=duration", "v:0") or subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(temporary)], capture_output=True, text=True).stdout)
    temporary.replace(out)
    atomic_json(split / "concat_manifest.json", {"version": 1, "film": name,
                "min_speech_min": minimum, "plan": plan_record,
                "parts": manifest_parts, "video": file_record(out)})
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
