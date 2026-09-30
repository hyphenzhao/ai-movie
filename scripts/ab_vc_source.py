#!/usr/bin/env python
"""A/B of the voice-conversion source (V6): natural line vs slot-fitted line, on one finished state.

Both variants come from ``run_vc_version.py`` runs of the same state.json with
their own ``--out-name`` / ``--out-dir`` (each writes
``deliverables/<out-name>/vc_state.json``), e.g.

    bash scripts/run_vc_only.sh output_test --vc-source audio_fit --out-name v2_fitsrc --out-dir synthesized_vc_fitsrc
    bash scripts/run_vc_only.sh output_test                      # B = config default (audio), v2_cloned
    python scripts/ab_vc_source.py workspace/output_test/state.json --a v2_fitsrc --b v2_cloned

Only lines whose v1 ``fit_ratio`` > 1 can differ (every other line is
byte-identical converter input either way), so the comparison is per line on
that subset: guard / content-judge verdicts, ``vc_pin_ratio``,
``vc_len_ratio``, ECAPA similarity of the pinned wav to the speaker's
reference (CPU; a noisy metric on 0.3–2 s lines — read the median, not the
per-line delta), and paired wavs ``ab_v6/<idx>_A.wav|_B.wav`` for
listening.  Film-level guards that must not move are printed too.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _abs(p):
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def _load_variant(work: Path, name: str) -> dict:
    p = work / "deliverables" / name / "vc_state.json"
    if not p.exists():
        raise SystemExit(f"{p} missing — run run_vc_version.py --out-name {name} first")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state")
    ap.add_argument("--a", default="v2_fitsrc", help="deliverable name of the audio_fit-source variant")
    ap.add_argument("--b", default="v2_cloned", help="deliverable name of the natural-source variant")
    ap.add_argument("--all", action="store_true", help="every converted line, not only v1 fit_ratio > 1")
    ap.add_argument("--no-ecapa", action="store_true")
    args = ap.parse_args()
    state_path = Path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    work = state_path.parent
    v1 = (state.get("fit") or {}).get("segments") or []
    A, B = _load_variant(work, args.a), _load_variant(work, args.b)
    sa, sb = A["segments"], B["segments"]
    out = work / "deliverables" / args.b / "ab_v6"
    out.mkdir(parents=True, exist_ok=True)

    sim = None
    if not args.no_ecapa:
        try:
            from ai_movie.diarize import similarity
            sim = lambda a, b: similarity(a, b, device="cpu")   # noqa: E731
        except Exception as exc:                        # noqa: BLE001
            print(f"ECAPA unavailable: {exc}")

    rows = []
    for i, f in enumerate(v1):
        if i >= len(sa) or i >= len(sb):
            break
        fr = float(f.get("fit_ratio") or 1.0)
        if not args.all and fr <= 1.0:
            continue
        a, b = sa[i], sb[i]
        if not (a.get("vc") or b.get("vc")):
            continue
        spk = f.get("speaker")
        ref = ((B.get("refs") or {}).get(spk) or (A.get("refs") or {}).get(spk) or {}).get("ref_audio")
        row = {"idx": i, "speaker": spk, "gender": f.get("gender"), "v1_fit_ratio": fr,
               "dur": round(float(f.get("end", 0)) - float(f.get("start", 0)), 2),
               "A_vc": int(bool(a.get("vc"))), "B_vc": int(bool(b.get("vc"))),
               "A_guard": a.get("vc_guard") or "", "B_guard": b.get("vc_guard") or "",
               "A_drift": a.get("vc_drift"), "B_drift": b.get("vc_drift"),
               "A_pin": a.get("vc_pin_ratio"), "B_pin": b.get("vc_pin_ratio"),
               "A_len": a.get("vc_len_ratio"), "B_len": b.get("vc_len_ratio"),
               "A_sim": None, "B_sim": None, "text": (f.get("text_translated") or "")[:30]}
        pa, pb = _abs(a.get("audio_fit")), _abs(b.get("audio_fit"))
        for tag, p in (("A", pa), ("B", pb)):
            if p and p.exists():
                shutil.copy2(p, out / f"{i:04d}_{tag}.wav")
                if sim and ref and _abs(ref).exists():
                    try:
                        row[f"{tag}_sim"] = round(sim(str(p), str(_abs(ref))), 3)
                    except Exception:                   # noqa: BLE001
                        pass
        rows.append(row)
    if not rows:
        print("no differing lines (no v1 fit_ratio > 1 converted lines)")
        return 0
    with open(out / "ab_v6.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    def med(key):
        xs = [r[key] for r in rows if isinstance(r[key], (int, float))]
        return (round(statistics.median(xs), 3), len(xs)) if xs else (None, 0)

    print(f"{work.name}: {len(rows)} lines with v1 fit_ratio > 1 (A={args.a}, B={args.b})")
    print(f"  converted   A {sum(r['A_vc'] for r in rows)}  B {sum(r['B_vc'] for r in rows)}")
    print(f"  guard drops A {sum(1 for r in rows if r['A_guard'])}  B {sum(1 for r in rows if r['B_guard'])}"
          f"   (A: {[r['A_guard'] for r in rows if r['A_guard']]}; B: {[r['B_guard'] for r in rows if r['B_guard']]})")
    print(f"  drift score median A {med('A_drift')}  B {med('B_drift')}")
    print(f"  pin ratio median    A {med('A_pin')}  B {med('B_pin')}   len ratio median A {med('A_len')}  B {med('B_len')}")
    if sim:
        print(f"  ECAPA sim median    A {med('A_sim')}  B {med('B_sim')}")
    for tag, V in (("A", A), ("B", B)):
        g = V.get("guard") or {}
        print(f"  film {tag}: converted {V.get('converted')}, guard {g.get('rejected')}/{g.get('checked')} "
              f"(drift {(g.get('drift') or {}).get('rejected')}/{(g.get('drift') or {}).get('judged')}), "
              f"reused_lipsync {V.get('reused_lipsync')}, max_drift {V.get('max_drift_ms')} ms, "
              f"len_ratio {V.get('len_ratio')}")
    print(f"  pairs → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
