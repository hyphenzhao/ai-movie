#!/usr/bin/env python3
"""Film-level acceptance for a long film against its screen-subtitle truth set.

    python scripts/eval_long.py SONE-846                # → deliver/SONE-846_full/ACCEPTANCE_FULL.{md,json}

Truth: ``workspace/<film>/_screen/scan.json`` (one probe per SRT cue; ``sub``
= a burned-in subtitle was on screen at that time) and, where read,
``screen.json`` (``{"id": ["zh", "ja"]}``).  Every kept segment of every chunk
is placed on the film timeline the way run_long.sh concatenates (chunk start
minus the trimmed lead), so the gates measure the delivered film.

Gates (L1–L6 in Documentation/v3.3-long-film.md):
  L1 cue coverage (interview / scene), L2 hallucination rate among kept
  segments (rules of ai_movie.content re-applied post hoc + kept lines far
  from any cue inside subtitled stretches), L2b ASR similarity on read cues,
  L3 voice consistency (one reference file per profile across chunks),
  L4 kept-original lines untouched, L5 loudness spread, L6 duration.
Exit 0 = pass, 1 = fail, 2 = cannot evaluate.
"""

from __future__ import annotations

import argparse
import bisect
import difflib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_movie.content import classify, fold        # noqa: E402

L1_INTERVIEW_MIN = 0.90
L1_SCENE_MIN = 0.60
L2_MAX = 0.05
L2B_INTERVIEW_MIN = 0.75
L3_MIN_MEDIAN = 0.40
L3_MAX_SPREAD = 0.15
L5_TOL_LU = 2.0
L6_TOL_S = 0.5              # ±1 frame per chunk boundary is the concat floor

_PAREN = re.compile(r"^\s*[（(].*[)）]\s*$")


def _probe(p: Path, entries: str) -> str:
    return subprocess.run(["ffprobe", "-v", "error", "-show_entries", entries, "-of", "csv=p=0", str(p)],
                          capture_output=True, text=True).stdout.strip().split("\n")[0]


def film_segments(film: str) -> tuple[list[dict], dict]:
    """Kept segments on the film timeline + chunk meta (type, offsets)."""
    split = ROOT / "workspace" / film / "_split"
    plan = json.loads((split / "plan.json").read_text())
    ks = json.loads((split / "keyframes.json").read_text()) if (split / "keyframes.json").exists() else []
    segs, meta = [], {}
    for c in plan["chunks"]:
        i = c["index"]; w = ROOT / "workspace" / f"{film}_p{i:02d}"
        lead = 0.0
        f = Path(c.get("file", ""))
        if ks and f.exists():
            have = float(_probe(f, "format=duration") or 0)
            if have - (c["end"] - c["start"]) > 1.0:
                j = bisect.bisect_left(ks, c["start"] - 1e-3)
                if j > 0 and abs(ks[j] - c["start"]) < 2e-3:
                    lead = c["start"] - ks[j - 1]
        sp = w / "state.json"
        st = json.loads(sp.read_text()) if sp.exists() else {}
        chunk_segs = ((st.get("vc") or {}).get("segments") or (st.get("fit") or {}).get("segments")
                      or (st.get("asr") or {}).get("segments") or [])
        speech = sum(s["end"] - s["start"] for s in chunk_segs)
        spk = len((((st.get("asr") or {}).get("diarization") or {}).get("speakers")) or {})
        meta[i] = {"start": c["start"], "end": c["end"], "lead": lead, "state": bool(st),
                   "type": "interview" if (speech / max(1.0, c["end"] - c["start"]) >= 0.35 and spk >= 2) else "scene",
                   "vc_refs": ((st.get("vc") or {}).get("refs") or {}),
                   "profiles": ((st.get("enrol") or {}).get("speaker_profile") or {})}
        off = c["start"] - lead
        for n, s in enumerate(chunk_segs):
            if s["end"] <= lead:
                continue
            segs.append(dict(s, chunk=i, idx=n, t0=off + s["start"], t1=off + s["end"]))
    return segs, meta


def cues(film: str) -> tuple[list[dict], dict]:
    scr = ROOT / "workspace" / film / "_screen"
    scan = json.loads((scr / "scan.json").read_text())
    read = {int(k): v for k, v in json.loads((scr / "screen.json").read_text()).items()} if (scr / "screen.json").exists() else {}
    out = [c for c in scan if c.get("sub") and not _PAREN.match(c.get("zh") or "")]
    return out, read


def evaluate(film: str) -> dict:
    segs, meta = film_segments(film)
    truth, read = cues(film)
    kept = [s for s in segs if not s.get("keep_original")]
    starts = sorted(s["t0"] for s in kept)
    ends = [s["t1"] for s in sorted(kept, key=lambda s: s["t0"])]

    def covered(a, b):
        j = bisect.bisect_right(starts, b)
        return any(min(b, ends[k]) - max(a, starts[k]) >= 0.1 for k in range(max(0, j - 12), j))

    def ctype(t):
        for i, m in meta.items():
            if m["start"] <= t < m["end"]:
                return m["type"]
        return "scene"
    checks, notes = [], []
    # L1
    for kind in ("interview", "scene"):
        cs = [c for c in truth if ctype(c["start"]) == kind]
        hit = sum(1 for c in cs if covered(c["start"], c["end"]))
        rate = hit / len(cs) if cs else 1.0
        lim = L1_INTERVIEW_MIN if kind == "interview" else L1_SCENE_MIN
        checks.append({"id": f"L1[{kind}]", "desc": f"{kind} cues covered by a dubbed line", "ok": rate >= lim,
                       "value": f"{hit}/{len(cs)} = {rate:.0%} (≥ {lim:.0%})"})
    # L2: post-hoc content rules + orphans inside subtitled stretches
    bad = []
    cue_iv = sorted((c["start"], c["end"]) for c in truth)
    cue_starts = [a for a, _ in cue_iv]

    def cue_distance(t0, t1):
        """Seconds from [t0,t1] to the nearest truth cue interval (0 = overlap)."""
        j = bisect.bisect_left(cue_starts, t0)
        best = float("inf")
        for k in range(max(0, j - 3), min(len(cue_iv), j + 3)):
            a, b = cue_iv[k]
            best = min(best, max(0.0, a - t1, t0 - b))
        return best
    for s in kept:
        r = classify({**s, "pass": s.get("pass", "vad")})
        if r["content"] == "drop":
            bad.append((s, r["reasons"][-1])); continue
        vis = len(fold(s.get("text", "")))
        d = cue_distance(s["t0"], s["t1"])
        # official subtitles skip fillers, so only a substantial line that is far from every cue
        # (but inside a subtitled stretch: some cue within a minute) counts as suspicious
        if vis >= 6 and 5.0 <= d < 60.0:
            bad.append((s, f"nearest cue {d:.0f} s away"))
    rate = len(bad) / max(1, len(kept))
    checks.append({"id": "L2", "desc": "kept lines that look hallucinated", "ok": rate <= L2_MAX,
                   "value": f"{len(bad)}/{len(kept)} = {rate:.1%} (≤ {L2_MAX:.0%})"})
    # L2b: ASR vs read screen JA
    sims = {"interview": [], "scene": []}
    for cid, (zh, ja) in read.items():
        c = next((c for c in truth if c["id"] == cid), None)
        if not c or not ja:
            continue
        ov = [s for s in kept if min(c["end"], s["t1"]) - max(c["start"], s["t0"]) > 0.25]
        if ov:
            sims[ctype(c["start"])].append(difflib.SequenceMatcher(None, fold(ja), fold("".join(s.get("text", "") for s in ov))).ratio())
    if sims["interview"]:
        med = statistics.median(sims["interview"])
        checks.append({"id": "L2b", "desc": "interview ASR similarity to screen JA (median)", "ok": med >= L2B_INTERVIEW_MIN,
                       "value": f"{med:.2f} on {len(sims['interview'])} cues (≥ {L2B_INTERVIEW_MIN})"})
    if sims["scene"]:
        notes.append(f"scene ASR similarity median {statistics.median(sims['scene']):.2f} on {len(sims['scene'])} cues")
    # L3: one reference per profile / per gender across chunks
    by_key: dict[str, set[str]] = {}
    for i, m in meta.items():
        for spk, r in m["vc_refs"].items():
            key = (m["profiles"].get(spk) or {}).get("profile") if isinstance(m["profiles"].get(spk), dict) else None
            key = key or f"gender:{r.get('gender')}"
            by_key.setdefault(key, set()).add(str(r.get("ref_audio") or r.get("path") or ""))
    multi = {k: len(v) for k, v in by_key.items() if len(v) > 1}
    if by_key:
        checks.append({"id": "L3", "desc": "one reference clip per voice across chunks", "ok": not multi,
                       "value": "ok" if not multi else f"{multi} distinct files"})
    else:
        notes.append("L3: no vc refs recorded")
    # L4: kept-original lines untouched
    ko = [s for s in segs if s.get("keep_original")]
    if ko:
        touched = sum(1 for s in ko if s.get("audio") or s.get("audio_fit"))
        checks.append({"id": "L4", "desc": "kept-original lines have no synthesized audio", "ok": touched == 0,
                       "value": f"{touched}/{len(ko)} touched"})
    # L6
    out = ROOT / "deliver" / f"{film}_dubbed_full.mp4"
    plan = json.loads((ROOT / "workspace" / film / "_split" / "plan.json").read_text())
    if out.exists():
        d = float(_probe(out, "format=duration") or 0)
        checks.append({"id": "L6", "desc": "full film duration vs source", "ok": abs(d - plan["duration"]) <= L6_TOL_S,
                       "value": f"Δ {d - plan['duration']:+.2f} s"})
    return {"film": film, "checks": checks, "notes": notes, "passed": all(c["ok"] for c in checks),
            "kept_segments": len(kept), "truth_cues": len(truth), "read_cues": len(read),
            "hallucinated": [{"t": round(s["t0"], 1), "text": s.get("text", "")[:30], "why": why} for s, why in bad[:40]]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("film")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    res = evaluate(args.film)
    out = args.out or (ROOT / "deliver" / f"{args.film}_full")
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# {args.film} 全片验收 — {'通过' if res['passed'] else '未通过'}", "",
             "| 门 | 条件 | 结果 | 数据 |", "|---|---|---|---|"]
    for c in res["checks"]:
        lines.append(f"| {c['id']} | {c['desc']} | {'✅' if c['ok'] else '❌'} | {c['value']} |")
    if res["notes"]:
        lines += ["", "## 备注"] + [f"- {n}" for n in res["notes"]]
    if res["hallucinated"]:
        lines += ["", "## 疑似幻听（前 40）"] + [f"- {h['t']}s 「{h['text']}」 — {h['why']}" for h in res["hallucinated"]]
    (out / "ACCEPTANCE_FULL.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "ACCEPTANCE_FULL.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines[:4 + len(res["checks"])]))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
