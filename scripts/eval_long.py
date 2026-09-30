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
  L3b–L3f one voice per profile on the delivered v2 lines (ECAPA; read from
  scripts/voice_consistency.py --film's report, not judged → note),
  L4 kept-original lines untouched, L5 loudness spread, L6 duration,
  L7 pronoun agreement (你/他/她) of the translate-stage lines with the
  screen Chinese of scan.json: mismatches (extra + missing) must not exceed
  the shipped v3.3 count.
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

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
from eval_against_subs import pronoun_agreement    # noqa: E402

L1_INTERVIEW_MIN = 0.90
L1_SCENE_MIN = 0.60
L2_MAX = 0.05
L2B_INTERVIEW_MIN = 0.75
# L3b–L3f thresholds live in config (VC_CONSIST_*) and are applied by scripts/voice_consistency.py --film;
# this script only reads its report (deliver/<film>_full/VOICE_CONSISTENCY.json) so it stays model-free.
L3_VOICE_GATES = (("L3b", "V2", "median d(line, own centroid) excess over the built-in voice"),
                  ("L3c", "V4", "leave-one-chunk-out centroid distance (per-chunk reference switch)"),
                  ("L3d", "V1", "seconds left in the built-in voice"),
                  ("L3e", "V5", "converted centroid moved away from the built-in voice"),
                  ("L3f", "V6", "converted centroid moved toward the enrolled profile voice"))
L5_TOL_LU = 2.0
L6_TOL_S = 0.5              # ±1 frame per chunk boundary is the concat floor
# Shipped v3.3 SONE-846 translate stage vs scan.json: 445 scored cues, 9 extra,
# 39 missing → 48 mismatches.  The old F1-style count would reward deleting
# pronouns; this one cannot get better by deletion (missing goes up).
L7_MAX_MISMATCH = 48
L7_MIN_OVERLAP_S = 0.25     # same cue↔segment overlap L2b uses

_PAREN = re.compile(r"^\s*[（(].*[)）]\s*$")


def _probe(p: Path, entries: str) -> str:
    return subprocess.run(["ffprobe", "-v", "error", "-show_entries", entries, "-of", "csv=p=0", str(p)],
                          capture_output=True, text=True).stdout.strip().split("\n")[0]


def film_segments(film: str, stage: str | None = None,
                  segments_by_chunk: dict[int, list[dict]] | None = None) -> tuple[list[dict], dict]:
    """Kept segments on the film timeline + chunk meta (type, offsets).

    *stage* forces one state key (``asr`` to measure transcription coverage
    before the rest of the chain has been re-run).  *segments_by_chunk*
    replaces the state's segment list for the chunk indices it holds (an
    in-memory A/B arm, scripts/ab_translate_context.py); other chunks still
    come from state.json, so a partial arm scores against the shipped lines."""
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
        if segments_by_chunk is not None and i in segments_by_chunk:
            chunk_segs = segments_by_chunk[i]
        else:
            chunk_segs = ((st.get(stage) or {}).get("segments") or []) if stage else (
                (st.get("vc") or {}).get("segments") or (st.get("fit") or {}).get("segments")
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


def coverage(cues_iv: list[tuple[float, float]], segs_iv: list[tuple[float, float]],
             min_overlap: float = 0.1) -> list[bool]:
    """Per cue: does some segment overlap it by ≥ *min_overlap* s."""
    order = sorted(segs_iv)
    starts = [a for a, _ in order]
    out = []
    for a, b in cues_iv:
        j = bisect.bisect_right(starts, b)
        out.append(any(min(b, order[k][1]) - max(a, order[k][0]) >= min_overlap for k in range(max(0, j - 12), j)))
    return out


def cue_distance(cue_iv: list[tuple[float, float]], t0: float, t1: float) -> float:
    """Seconds from [t0,t1] to the nearest cue interval (0 = overlap); *cue_iv* sorted."""
    starts = [a for a, _ in cue_iv]
    j = bisect.bisect_left(starts, t0)
    best = float("inf")
    for k in range(max(0, j - 3), min(len(cue_iv), j + 3)):
        a, b = cue_iv[k]
        best = min(best, max(0.0, a - t1, t0 - b))
    return best


def kept_original_stats(segs: list[dict]) -> dict:
    """L4 numbers over the kept-original segments: how many, how many seconds, how many carry
    synthesized audio (``audio`` / ``audio_fit``) although they must not.  Pure — ``evaluate``
    feeds it the placed segments (``t0``/``t1``); plain ``start``/``end`` segments work too."""
    ko = [s for s in segs if s.get("keep_original")]
    secs = sum(float(s.get("t1", s.get("end", 0))) - float(s.get("t0", s.get("start", 0))) for s in ko)
    return {"lines": len(ko), "seconds": round(secs, 1),
            "touched": sum(1 for s in ko if s.get("audio") or s.get("audio_fit"))}


def pronoun_check(truth: list[dict], segs: list[dict], key: str = "text_translated") -> dict:
    """L7 numbers: 你/他/她 agreement of our lines with the screen Chinese.

    Every cue with a ``zh`` is paired with the segments overlapping it by more
    than L7_MIN_OVERLAP_S (their *key* texts joined); cues without any line
    are not scored, so coverage (L1) cannot move this number.  The scoring
    itself is eval_against_subs.pronoun_agreement, the same function the
    output_test release gate uses."""
    groups = []
    for c in truth:
        if not c.get("zh"):
            continue
        ov = [s for s in segs if min(c["end"], s["t1"]) - max(c["start"], s["t0"]) > L7_MIN_OVERLAP_S]
        ours = " ".join((s.get(key) or "").strip() for s in ov).strip()
        groups.append({"cue": c["id"], "start": c["start"], "ref_zh": c["zh"], "ours_zh": ours})
    return pronoun_agreement(groups)


def voice_consistency_checks(film: str, checks: list[dict], notes: list[str], *,
                             report: dict | None = None, current_sigs: dict[str, str] | None = None) -> None:
    """L3b–L3f from ``deliver/<film>_full/VOICE_CONSISTENCY.json`` (scripts/voice_consistency.py --film).

    One check per gate id, over every voice the report judged; ``ok`` is appended
    ONLY as a bool.  ``evaluate()`` takes ``passed = all(c["ok"] …)`` and
    ``accept_release --long`` gates ``bool(c["ok"])``, so a "not judged" gate must
    be a note, never a check with ``ok=None``.  A report whose per-chunk ``vc_sig``
    no longer matches the chunk's state (one chunk re-cloned since) is stale and
    yields notes only — stale numbers must not gate a release.
    """
    if report is None:
        p = ROOT / "deliver" / f"{film}_full" / "VOICE_CONSISTENCY.json"
        if not p.exists():
            notes.append("L3b–f: not measured (run scripts/voice_consistency.py --film)")
            return
        try:
            report = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as exc:
            notes.append(f"L3b–f: report unreadable ({exc})")
            return
    if current_sigs is None:
        from ai_movie.voice_consistency import vc_signature
        current_sigs = {}
        for c in (report.get("chunks") or {}):
            sp = ROOT / "workspace" / f"{film}_p{int(c):02d}" / "state.json"
            if sp.exists():
                try:
                    current_sigs[str(c)] = vc_signature(json.loads(sp.read_text(encoding="utf-8")))
                except ValueError:
                    current_sigs[str(c)] = "unreadable"
    stale = sorted(c for c, m in (report.get("chunks") or {}).items()
                   if current_sigs.get(str(c)) not in (None, m.get("vc_sig")))
    if stale:
        notes.append(f"L3b–f: report stale — chunks {stale} re-cloned since; rerun scripts/voice_consistency.py --film {film}")
        return
    rows = report.get("gates") or []
    for lid, gid, desc in L3_VOICE_GATES:
        judged = [r for r in rows if r.get("gate") == gid and r.get("ok") is not None]
        unjudged = [r for r in rows if r.get("gate") == gid and r.get("ok") is None]
        if judged:
            checks.append({"id": lid, "desc": desc, "ok": all(bool(r["ok"]) for r in judged),
                           "value": "; ".join(f"{r['key']}: {r['value']}" for r in judged)})
        elif unjudged:
            notes.append(f"{lid}: not judged — " + "; ".join(f"{r['key']}: {r.get('note') or r['value']}" for r in unjudged))
    for n in report.get("notes") or []:
        notes.append(f"voice consistency: {n}")


def evaluate(film: str, stage: str | None = None) -> dict:
    segs, meta = film_segments(film, stage)
    truth, read = cues(film)
    kept = [s for s in segs if not s.get("keep_original")]
    kept_iv = [(s["t0"], s["t1"]) for s in kept]

    def ctype(t):
        for i, m in meta.items():
            if m["start"] <= t < m["end"]:
                return m["type"]
        return "scene"
    checks, notes = [], []
    # L1
    for kind in ("interview", "scene"):
        cs = [c for c in truth if ctype(c["start"]) == kind]
        hit = sum(coverage([(c["start"], c["end"]) for c in cs], kept_iv))
        rate = hit / len(cs) if cs else 1.0
        lim = L1_INTERVIEW_MIN if kind == "interview" else L1_SCENE_MIN
        checks.append({"id": f"L1[{kind}]", "desc": f"{kind} cues covered by a dubbed line", "ok": rate >= lim,
                       "value": f"{hit}/{len(cs)} = {rate:.0%} (≥ {lim:.0%})"})
    # L2: post-hoc content rules + orphans inside subtitled stretches
    bad = []
    cue_iv = sorted((c["start"], c["end"]) for c in truth)
    for s in kept:
        r = classify({**s, "pass": s.get("pass", "vad")})
        if r["content"] == "drop":
            bad.append((s, r["reasons"][-1])); continue
        vis = len(fold(s.get("text", "")))
        d = cue_distance(cue_iv, s["t0"], s["t1"])
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
    # L3b–L3f: is each delivered voice one voice (ECAPA, scripts/voice_consistency.py --film)
    voice_consistency_checks(film, checks, notes)
    # L4: kept-original lines untouched.  The line / second counts are not a gate: truth cues never
    # cover moans (0 moan-like cues among 514), so they are the only truth-independent signal of a
    # content-rule change (Documentation/v3.4: kept-original ≈ 1101 s → ≥ 1290 s on SONE-846).
    kept_original = kept_original_stats(segs)
    if kept_original["lines"]:
        checks.append({"id": "L4", "desc": "kept-original lines have no synthesized audio", "ok": kept_original["touched"] == 0,
                       "value": f"{kept_original['touched']}/{kept_original['lines']} touched"})
        notes.append(f"kept-original: {kept_original['lines']} lines, {kept_original['seconds']} s")
    # L7: pronoun agreement of the translate-stage lines with the screen Chinese
    tr_segs, _ = film_segments(film, "translate")
    pa = pronoun_check(truth, tr_segs)
    if pa["scored"]:
        checks.append({"id": "L7", "desc": "pronoun (你/他/她) mismatches vs screen Chinese", "ok": pa["mismatch"] <= L7_MAX_MISMATCH,
                       "value": f"{pa['mismatch']} on {pa['scored']} cues (extra {pa['extra']}, missing {pa['missing']}; ≤ {L7_MAX_MISMATCH})"})
    else:
        notes.append("L7: no translate-stage lines overlap a cue with Chinese")
    # L6
    out = ROOT / "deliver" / f"{film}_dubbed_full.mp4"
    plan = json.loads((ROOT / "workspace" / film / "_split" / "plan.json").read_text())
    if out.exists():
        d = float(_probe(out, "format=duration") or 0)
        checks.append({"id": "L6", "desc": "full film duration vs source", "ok": abs(d - plan["duration"]) <= L6_TOL_S,
                       "value": f"Δ {d - plan['duration']:+.2f} s"})
    return {"film": film, "checks": checks, "notes": notes, "passed": all(c["ok"] for c in checks),
            "kept_segments": len(kept), "truth_cues": len(truth), "read_cues": len(read),
            "kept_original": {"lines": kept_original["lines"], "seconds": kept_original["seconds"]},
            "hallucinated": [{"t": round(s["t0"], 1), "text": s.get("text", "")[:30], "why": why} for s, why in bad[:40]]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("film")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--stage", default=None, help="measure this state key's segments (e.g. asr) instead of the delivered ones")
    ap.add_argument("--per-chunk", action="store_true", help="also print L1 per chunk")
    args = ap.parse_args()
    res = evaluate(args.film, args.stage)
    if args.per_chunk:
        segs, meta = film_segments(args.film, args.stage)
        truth, _ = cues(args.film)
        kept = [(s["t0"], s["t1"]) for s in segs if not s.get("keep_original")]
        for i, m in sorted(meta.items()):
            cs = [(c["start"], c["end"]) for c in truth if m["start"] <= c["start"] < m["end"]]
            if cs:
                print(f"  p{i:02d} {m['type']:9s} cues {sum(coverage(cs, kept))}/{len(cs)}")
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
