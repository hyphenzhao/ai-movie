#!/usr/bin/env python3
"""Replay ``ai_movie.content.classify`` over the persisted ASR rows — no GPU, seconds.

    .venv/bin/python scripts/replay_content.py SONE-846 output_test test_1 test_2
    .venv/bin/python scripts/replay_content.py                      # every project with a 01_content.csv
    .venv/bin/python scripts/replay_content.py SONE-846 --energy    # + original-vocals p95 of each flipped row
    .venv/bin/python scripts/replay_content.py --json out.json      # the flip table, machine-readable

``step_asr`` writes every pre-classify Whisper row (scores, ``alt_text`` and
the decision it took) to ``workspace/<project>/deliverables/01_content.csv``.
Re-running the *current* rules over those rows shows exactly which lines a
rule change flips (old decision → new decision, segments and seconds), how
many seconds of kept-original voice the film gains or loses, and — where the
film has a ``_screen`` truth set — how far every flipped row is from a
subtitle cue.  That is how a content-rule change is measured in minutes
instead of a 24 h ASR re-run.

Fidelity: ``classify`` is pure, so the replay is exact for it.  The two rules
outside it are replayed too: the loop rule (``content.looped_indices``) is
text + time only; the energy rule runs first and never reads the text, so a
row it dropped stays dropped and no other row can newly trip it — those rows
(reason ``energy floor or repeated line``) are held fixed.  ``--energy`` adds
the original-vocals p95 of each flipped range as a column (is there anything
audible to restore?); it never changes a decision.

Validity: a window whose compression ratio stayed above 2.4 at every fallback
temperature (``asr.py`` l.841, up to T=1.0) is a T=1.0 *sample*, and those
are exactly the moan windows.  Their strings (「ああ×14兄ちゃん」…) differ
between ASR runs, so the flip table is only valid for the states on disk
now; after an ASR re-run, replay again before comparing.  A long film is
given by its plan name (``SONE-846`` → ``SONE-846_p01`` … from
``_split/plan.json``); a short film by its workspace directory.

Nothing is written unless ``--json`` is given; ``workspace/`` is read only.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from ai_movie.content import classify, looped_indices        # noqa: E402

WORKSPACE = ROOT / "workspace"
_FIXED_REASON = "energy floor or repeated line"          # step_asr's label for a drop outside classify()


def _num(v) -> float | None:
    try:
        return float(v) if v not in ("", None) else None
    except (TypeError, ValueError):
        return None


def load_rows(csv_path: Path) -> list[dict]:
    """The pre-classify rows of one project as ``classify`` inputs, with the delivered decision.

    The CSV is UTF-8 with BOM (Excel); ``alt_text`` '' must map back to None — the classifier
    treats "no second decode" (None) and "second decode heard nothing" ('') differently."""
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"text": r.get("text") or "", "start": _num(r.get("start")) or 0.0,
                         "end": _num(r.get("end")) or 0.0, "pass": r.get("pass") or "vad",
                         "asr_conf": _num(r.get("asr_conf")), "no_speech_prob": _num(r.get("no_speech_prob")),
                         "avg_logprob": _num(r.get("avg_logprob")),
                         "compression_ratio": _num(r.get("compression_ratio")),
                         "alt_text": r.get("alt_text") or None, "speaker": r.get("speaker"),
                         "old": r.get("decision") or "", "old_reason": r.get("reason") or ""})
    return rows


def replay_rows(rows: list[dict]) -> list[dict]:
    """Attach ``new`` / ``new_reason`` to every row: classify + loop rule; energy drops held.

    A row step_asr dropped as "only one decode heard text" had ``alt_text`` '' (the second decode
    heard nothing) — the reason string is the only record of that, so '' is put back for the replay."""
    looped = looped_indices(rows)
    for i, r in enumerate(rows):
        if r["old"] == "drop" and r["old_reason"].startswith(("energy", _FIXED_REASON)) and i not in looped:
            r["new"], r["new_reason"] = "drop", r["old_reason"]
            continue
        if r["alt_text"] is None and r["old_reason"].startswith("only one decode heard text"):
            r["alt_text"] = ""
        res = classify(r)
        kind, why = res["content"], res["reasons"][-1]
        if i in looped and kind != "nonlexical":
            kind, why = "drop", "repeated line (loop)"
        r["new"], r["new_reason"] = kind, why
    return rows


def projects_for(name: str) -> list[tuple[str, Path]]:
    """(project, workspace dir) pairs: a long film's chunks from its plan, or one directory."""
    plan = WORKSPACE / name / "_split" / "plan.json"
    if plan.exists():
        chunks = json.loads(plan.read_text(encoding="utf-8"))["chunks"]
        return [(f"{name}_p{c['index']:02d}", WORKSPACE / f"{name}_p{c['index']:02d}") for c in chunks]
    return [(name, WORKSPACE / name)]


def all_films() -> list[str]:
    seen, out = set(), []
    for p in sorted(WORKSPACE.glob("*/deliverables/01_content.csv")):
        proj = p.parent.parent.name
        film = proj.rsplit("_p", 1)[0] if "_p" in proj and (WORKSPACE / proj.rsplit("_p", 1)[0] / "_split" / "plan.json").exists() else proj
        if film not in seen:
            seen.add(film); out.append(film)
    return out


def _timeline(film: str):
    """Chunk → (offset, type) on the film timeline and the sorted truth cues, or (None, None)."""
    if not (WORKSPACE / film / "_screen" / "scan.json").exists() or not (WORKSPACE / film / "_split" / "plan.json").exists():
        return None, None
    import eval_long
    _, meta = eval_long.film_segments(film)
    truth, _ = eval_long.cues(film)
    offs = {f"{film}_p{i:02d}": (m["start"] - m["lead"], m["type"]) for i, m in meta.items()}
    return offs, sorted((c["start"], c["end"]) for c in truth)


def _p95_fn(work: Path):
    """Original-vocals p95 (dBFS) over a range, exactly as classify_segments computes it."""
    sp = work / "state.json"
    voc = ((json.loads(sp.read_text(encoding="utf-8")).get("separate") or {}).get("vocals")) if sp.exists() else None
    if not voc or not Path(voc).exists():
        return None
    import numpy as np
    from ai_movie.diarize import _load_mono16k
    y = _load_mono16k(voc)
    frame = 320
    n = len(y) // frame
    lv = 20 * np.log10(np.sqrt(np.mean(y[:n * frame].reshape(n, frame) ** 2, axis=1)) + 1e-9)

    def p95(a: float, b: float) -> float | None:
        i, j = int(a * 50), max(int(a * 50) + 1, int(b * 50))
        seg = lv[i:j]
        return float(np.percentile(seg, 95)) if seg.size else None
    return p95


def replay_film(film: str, *, energy: bool = False) -> dict:
    offs, cue_iv = _timeline(film)
    flips: dict[tuple[str, str], list[dict]] = defaultdict(list)
    n_rows = 0
    ko = {"old": [0, 0.0], "new": [0, 0.0]}
    remaining_cr: list[dict] = []
    for proj, work in projects_for(film):
        p = work / "deliverables" / "01_content.csv"
        if not p.exists():
            continue
        rows = replay_rows(load_rows(p))
        n_rows += len(rows)
        p95 = _p95_fn(work) if energy else None
        for r in rows:
            dur = r["end"] - r["start"]
            for key in ("old", "new"):
                if r[key] == "nonlexical":
                    ko[key][0] += 1; ko[key][1] += dur
            if r["new"] == "drop" and r["new_reason"].startswith("compression"):
                remaining_cr.append({"proj": proj, "t": r["start"], "dur": dur, "text": r["text"]})
            if r["new"] == r["old"]:
                continue
            e = {"proj": proj, "t": r["start"], "dur": dur, "cr": r["compression_ratio"], "text": r["text"],
                 "old_reason": r["old_reason"], "new_reason": r["new_reason"]}
            if offs and proj in offs:
                off, ctype = offs[proj]
                import eval_long
                e["chunk_type"] = ctype
                e["cue_distance"] = eval_long.cue_distance(cue_iv, off + r["start"], off + r["end"])
            if p95:
                e["vocals_p95_db"] = p95(r["start"], r["end"])
            flips[(r["old"], r["new"])].append(e)
    table = {f"{a}->{b}": {"segments": len(v), "seconds": round(sum(e["dur"] for e in v), 1)} for (a, b), v in flips.items()}
    return {"film": film, "rows": n_rows,
            "kept_original": {"old": {"lines": ko["old"][0], "seconds": round(ko["old"][1], 1)},
                              "new": {"lines": ko["new"][0], "seconds": round(ko["new"][1], 1)}},
            "flips": table, "examples": {f"{a}->{b}": v for (a, b), v in flips.items()},
            "remaining_compression_drops": remaining_cr, "truth": offs is not None}


def _fmt_example(e: dict) -> str:
    s = f"{e['proj']} {e['t']:7.1f}s {e['dur']:4.1f}s cr={e['cr'] if e['cr'] is None else round(e['cr'], 2)} {e['text'][:28]!r}"
    if "cue_distance" in e:
        d = e["cue_distance"]
        s += f"  [{e['chunk_type']}; {'on cue' if d == 0 else ('no cue' if d == float('inf') else f'cue {d:.1f} s away')}]"
    if "vocals_p95_db" in e:
        s += f"  vocals p95 {e['vocals_p95_db']:.1f} dBFS" if e["vocals_p95_db"] is not None else "  vocals p95 n/a"
    return s


def print_film(res: dict, *, max_examples: int) -> None:
    ko = res["kept_original"]
    print(f"== {res['film']}: {res['rows']} rows; kept-original {ko['old']['lines']} lines / {ko['old']['seconds']:.1f} s"
          f" -> {ko['new']['lines']} lines / {ko['new']['seconds']:.1f} s"
          + ("" if res["truth"] else "  (no _screen truth: no cue distances)"))
    if not res["flips"]:
        print("   0 flips")
    for key, t in sorted(res["flips"].items()):
        print(f"   {key:<22} {t['segments']:3d} segs {t['seconds']:6.1f} s")
        ex = res["examples"][key]
        for e in ex[:max_examples]:
            print("      ", _fmt_example(e))
        if len(ex) > max_examples:
            print(f"       … {len(ex) - max_examples} more")
        on_cue = [e for e in ex if e.get("cue_distance") == 0]
        if res["truth"] and on_cue:
            print(f"       {len(on_cue)} of these overlap a subtitle cue")
        if any("vocals_p95_db" in e and e["vocals_p95_db"] is not None for e in ex):
            vals = [e["vocals_p95_db"] for e in ex if e.get("vocals_p95_db") is not None]
            print(f"       vocals p95: min {min(vals):.1f} / median {statistics.median(vals):.1f} dBFS")
    rc = res["remaining_compression_drops"]
    if rc:
        print(f"   remaining compression drops: {len(rc)} segs {sum(e['dur'] for e in rc):.1f} s"
              + (": " + "; ".join(f"{e['proj']} {e['t']:.1f}s {e['text'][:12]!r}" for e in rc[:12]) if len(rc) <= 12 else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("films", nargs="*", help="film / project names (default: every project with a 01_content.csv)")
    ap.add_argument("--energy", action="store_true", help="add the original-vocals p95 of each flipped range (reads separated/vocals.wav)")
    ap.add_argument("--max-examples", type=int, default=40)
    ap.add_argument("--json", type=Path, default=None, help="write the per-film tables here")
    args = ap.parse_args()
    films = args.films or all_films()
    if not films:
        print("no 01_content.csv under", WORKSPACE); return 2
    results = []
    for film in films:
        res = replay_film(film, energy=args.energy)
        if not res["rows"]:
            print(f"== {film}: no 01_content.csv rows"); continue
        results.append(res)
        print_film(res, max_examples=args.max_examples)
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
