#!/usr/bin/env python
"""Dry run of the VC content-drift judge on finished states — no re-conversion (Phase 1 of V7).

Every converted line recorded in ``state["vc"]["segments"]`` (``vc`` true) is
re-read by Whisper next to the v1 wav it was converted from
(``state["fit"]["segments"][i]`` — ``audio_fit`` for v3.3 states,
``state["vc"]["source_key"]`` afterwards) and judged with
``vc_guard.judge_content`` under the production thresholds.  Only the per-line
level is probed: v3.3 states carry no chunk membership and ``synthesized_vc/``
accumulates chunk files across runs, so chunk units cannot be rebuilt from
disk.  Conversions that are newer on disk than the state are flagged.

Outputs per state, under ``--out`` (default ``workspace/<name>/drift_probe/``):

* ``lines.csv`` — one row per converted line: verdict, score, baseline, both
  readings, the existing pitch-guard verdict, and (when the film has
  verify_v1/verify_v2 rows) verify_dub's own reading of the same line with
  the "known bad" flag (max folded sim of verify's v2 reading to v1 reading /
  intended text < 0.5) so the judge's recall on verify's rows is visible;
* ``summary.json`` — counts, reject share, reasons, decode seconds, recall /
  precision proxies against verify_dub's known-bad rows;
* ``review/`` — ``<idx>_v1.wav`` + ``<idx>_conv.wav`` for every rejection and
  for ``--band-sample`` random passes with 0.5 ≤ score < 0.7, plus
  ``review.csv``: this is what the user listens to.

Transcripts go to ``workspace/<name>/vc_drift_cache.json`` — the same
content-keyed cache run_vc_version uses, so Phase 2 re-decodes only the new
conversions.  GPU (Whisper large-v3); ≈ 1 s per clip on gfx1151.

    python scripts/vc_drift_probe.py workspace/test_1/state.json workspace/SONE-846_p01/state.json
    python scripts/vc_drift_probe.py --all            # test_1, test_2, output_test, SONE-846_p01..p22
    python scripts/vc_drift_probe.py --all --report   # totals over every summary.json written so far (no GPU)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGRESSION_SET = ["test_1", "test_2", "output_test"] + [f"SONE-846_p{k:02d}" for k in range(1, 23)]


def _abs(p: str | None) -> Path | None:
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def _verify_rows(work: Path) -> dict[int, dict]:
    """{0-based idx: {"v1": heard, "v2": heard, "want", "bad"}} from the film's verify JSONs."""
    from ai_movie.vc_guard import text_sim
    try:
        v1 = {r["index"]: r for r in json.loads((work / "verify_v1.json").read_text(encoding="utf-8"))}
        src = work / "verify_v2.pre_v34.json"
        if not src.exists():
            src = work / "verify_v2.json"
        v2 = json.loads(src.read_text(encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return {}
    out = {}
    for r in v2:
        a = v1.get(r["index"])
        if not a:
            continue
        score = max(text_sim(r["heard"], a["heard"]), text_sim(r["heard"], r["want"]))
        out[r["index"] - 1] = {"v1": a["heard"], "v2": r["heard"], "want": r["want"],
                               "score": round(score, 3), "bad": score < 0.5}
    return out


def probe_state(state_path: Path, out_dir: Path, *, judge, band_sample: int, seed: int = 0) -> dict:
    from ai_movie.vc_guard import judge_content
    state = json.loads(state_path.read_text(encoding="utf-8"))
    work = state_path.parent
    vc = state.get("vc") or {}
    vsegs = vc.get("segments") or []
    fsegs = (state.get("fit") or {}).get("segments") or []
    source_key = vc.get("source_key") or "audio_fit"
    if not vsegs or not vc.get("converted"):
        return {"name": work.name, "skipped": "no converted lines"}
    st_mtime = state_path.stat().st_mtime
    units = []
    for i, s in enumerate(vsegs):
        if not s.get("vc"):
            continue
        conv = _abs(s.get("audio"))
        f = fsegs[i] if i < len(fsegs) else {}
        v1 = _abs(f.get(source_key) or f.get("audio_fit") or f.get("audio"))
        if not conv or not v1 or not conv.exists() or not v1.exists():
            continue
        units.append({"idx": i, "conv": str(conv), "v1": str(v1), "want": s.get("text_translated") or "",
                      "start": s.get("start"), "speaker": s.get("speaker"), "gender": s.get("gender"),
                      "guard": s.get("vc_guard") or "", "chunk": s.get("vc_chunk"),
                      "newer_than_state": conv.stat().st_mtime > st_mtime + 1})
    if not units:
        return {"name": work.name, "skipped": "no readable converted lines"}
    t0 = time.time()
    d0 = judge.decoded
    texts = judge.transcribe([p for u in units for p in (u["conv"], u["v1"])])
    seconds = round(time.time() - t0, 1)
    verify = _verify_rows(work)
    rows, reasons = [], {}
    judged = rejected = 0
    for u in units:
        d = judge_content(texts.get(u["conv"]), texts.get(u["v1"]), u["want"])
        rc, rv = texts.get(u["conv"]) or {}, texts.get(u["v1"]) or {}
        vr = verify.get(u["idx"])
        row = {"idx": u["idx"], "start": u["start"], "speaker": u["speaker"], "gender": u["gender"],
               "chunk": ("-".join(map(str, u["chunk"])) if u.get("chunk") else ""),
               "judged": int(d["judged"]), "ok": int(d["ok"]), "reason": d["reason"], "gate": d.get("gate") or "",
               "score": d["score"], "baseline": d["baseline"],
               "lang_conv": (rc.get("auto") or {}).get("language"), "lang_v1": (rv.get("auto") or {}).get("language"),
               "heard_v1": d["heard_v1"], "heard_conv": d["heard_conv"],
               "heard_conv_zh": ((rc.get("zh") or {}).get("text") if rc.get("zh") else ""),
               "want": u["want"], "pitch_guard": u["guard"], "newer_than_state": int(u["newer_than_state"]),
               "verify_sampled": int(vr is not None), "verify_bad": (int(vr["bad"]) if vr else ""),
               "verify_score": (vr["score"] if vr else ""), "verify_heard_v2": (vr["v2"] if vr else ""),
               "conv": u["conv"], "v1": u["v1"]}
        rows.append(row)
        if d["judged"]:
            judged += 1
            if not d["ok"]:
                rejected += 1
                k = d["reason"].split("_")[0]
                reasons[k] = reasons.get(k, 0) + 1
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "lines.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    # review folder: every rejection + a random sample of the 0.5–0.7 band
    review = out_dir / "review"
    if review.exists():
        shutil.rmtree(review)
    review.mkdir()
    picked = [r for r in rows if r["judged"] and not r["ok"]]
    band = [r for r in rows if r["judged"] and r["ok"] and r["score"] is not None and 0.5 <= r["score"] < 0.7]
    rnd = random.Random(seed)
    picked += rnd.sample(band, min(band_sample, len(band)))
    rv_rows = []
    for r in picked:
        tag = f"{r['idx']:04d}"
        shutil.copy2(r["v1"], review / f"{tag}_v1.wav")
        shutil.copy2(r["conv"], review / f"{tag}_conv.wav")
        rv_rows.append({k: r[k] for k in ("idx", "reason", "score", "baseline", "heard_v1", "heard_conv", "want", "pitch_guard")}
                       | {"kind": "reject" if not r["ok"] else "band_pass"})
    with open(review / "review.csv", "w", newline="", encoding="utf-8-sig") as fh:
        if rv_rows:
            w = csv.DictWriter(fh, fieldnames=list(rv_rows[0]))
            w.writeheader(); w.writerows(rv_rows)
    # recall / precision proxies against verify_dub's known-bad rows (sampled converted lines only)
    sampled = [r for r in rows if r["verify_sampled"]]
    known_bad = [r for r in sampled if r["verify_bad"] == 1]
    caught = [r for r in known_bad if r["judged"] and not r["ok"]]
    rej_sampled = [r for r in sampled if r["judged"] and not r["ok"]]
    rej_sampled_bad = [r for r in rej_sampled if r["verify_bad"] == 1]
    summary = {"name": work.name, "converted": len(units), "judged": judged, "rejected": rejected,
               "reject_share_judged": round(rejected / judged, 3) if judged else None,
               "reject_share_converted": round(rejected / len(units), 3),
               "reasons": reasons, "seconds": seconds, "decoded": judge.decoded - d0,
               "newer_than_state": sum(r["newer_than_state"] for r in rows),
               "review_rejects": len([r for r in picked if not r["ok"]]), "review_band": len(picked) - len([r for r in picked if not r["ok"]]),
               "verify": {"sampled_converted": len(sampled), "known_bad": len(known_bad), "caught": len(caught),
                          "judge_rejected_in_sample": len(rej_sampled), "of_which_known_bad": len(rej_sampled_bad),
                          "missed_unjudgeable": len([r for r in known_bad if not r["judged"]]),
                          "missed_judged_ok": len([r for r in known_bad if r["judged"] and r["ok"]])}}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def report(names: list[str]) -> int:
    tot = {"converted": 0, "judged": 0, "rejected": 0, "seconds": 0.0, "decoded": 0}
    ver = {"sampled_converted": 0, "known_bad": 0, "caught": 0, "judge_rejected_in_sample": 0,
           "of_which_known_bad": 0, "missed_unjudgeable": 0, "missed_judged_ok": 0}
    reasons: dict[str, int] = {}
    print(f"{'film':14} {'conv':>5} {'judged':>6} {'rej':>4} {'share':>6} {'sec':>6}  verify: bad caught | rej rej_bad")
    for n in names:
        p = ROOT / "workspace" / n / "drift_probe" / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        if s.get("skipped"):
            print(f"{n:14} skipped: {s['skipped']}")
            continue
        for k in tot:
            tot[k] += s.get(k) or 0
        for k in ver:
            ver[k] += (s.get("verify") or {}).get(k) or 0
        for k, v in (s.get("reasons") or {}).items():
            reasons[k] = reasons.get(k, 0) + v
        v = s.get("verify") or {}
        print(f"{n:14} {s['converted']:5d} {s['judged']:6d} {s['rejected']:4d} {s['reject_share_judged'] or 0:6.1%} {s['seconds']:6.0f}"
              f"  {v.get('known_bad', 0):3d} {v.get('caught', 0):6d} | {v.get('judge_rejected_in_sample', 0):3d} {v.get('of_which_known_bad', 0):7d}")
    if tot["judged"]:
        print(f"\nTOTAL converted {tot['converted']}, judged {tot['judged']} ({tot['judged'] / tot['converted']:.0%}), "
              f"rejected {tot['rejected']} ({tot['rejected'] / tot['judged']:.1%} of judged, "
              f"{tot['rejected'] / tot['converted']:.1%} of converted); reasons {reasons}; "
              f"{tot['decoded']} clips decoded in {tot['seconds']:.0f} s ({tot['seconds'] / max(1, tot['decoded']):.2f} s/clip)")
        print(f"verify_dub rows: {ver['known_bad']} known-bad converted lines of {ver['sampled_converted']} sampled; "
              f"judge caught {ver['caught']} (recall {ver['caught'] / max(1, ver['known_bad']):.0%}; "
              f"{ver['missed_unjudgeable']} un-judgeable, {ver['missed_judged_ok']} judged ok); "
              f"judge rejected {ver['judge_rejected_in_sample']} sampled lines of which {ver['of_which_known_bad']} known-bad "
              f"(precision proxy {ver['of_which_known_bad'] / max(1, ver['judge_rejected_in_sample']):.0%})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="*", help="state.json paths (or workspace names)")
    ap.add_argument("--all", action="store_true", help="the 25-workspace regression set")
    ap.add_argument("--out", default=None, help="output dir (single state only; default workspace/<name>/drift_probe)")
    ap.add_argument("--band-sample", type=int, default=30, help="random 0.5–0.7 passes copied for listening")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report", action="store_true", help="only print totals over existing summaries")
    args = ap.parse_args()
    names = list(args.states)
    if args.all:
        names += REGRESSION_SET
    if not names:
        ap.error("give state.json paths or --all")
    paths = []
    for n in names:
        p = Path(n)
        paths.append(p if p.suffix == ".json" else ROOT / "workspace" / n / "state.json")
    if args.report:
        return report([p.parent.name for p in paths])
    from ai_movie.asr import WhisperClips
    judge = None
    try:
        for p in paths:
            if not p.exists():
                print(f"{p.parent.name}: no state.json", flush=True)
                continue
            work = p.parent
            if judge is None or judge.cache_path.parent != work:
                if judge is not None:
                    judge._save_cache()
                    model = judge._model
                    judge = WhisperClips(cache=work / "vc_drift_cache.json", device=args.device)
                    judge._model = model                # one load per process, cache per workspace
                else:
                    judge = WhisperClips(cache=work / "vc_drift_cache.json", device=args.device)
            out_dir = Path(args.out) if (args.out and len(paths) == 1) else work / "drift_probe"
            t0 = time.time()
            s = probe_state(p, out_dir, judge=judge, band_sample=args.band_sample)
            if s.get("skipped"):
                print(f"{work.name}: {s['skipped']}", flush=True)
                continue
            v = s["verify"]
            print(f"{work.name}: {s['rejected']}/{s['judged']} judged rejected of {s['converted']} converted "
                  f"{s['reasons']} — {s['decoded']} clips in {s['seconds']} s ({time.time() - t0:.0f} s total); "
                  f"verify known-bad {v['known_bad']} → caught {v['caught']}"
                  + (f"; {s['newer_than_state']} conversions NEWER than state (stale mix?)" if s["newer_than_state"] else ""),
                  flush=True)
    finally:
        if judge is not None:
            judge.close()
    return report([p.parent.name for p in paths])


if __name__ == "__main__":
    raise SystemExit(main())
