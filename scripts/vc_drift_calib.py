#!/usr/bin/env python
"""Replay the VC content-drift judge over verify_dub's paired rows (offline, CPU, JSON only).

The judge's thresholds (config VC_DRIFT_*) were set on the 417 paired
verify_v1 / verify_v2 rows of the v3.3 release (299 converted lines). This
replays ``vc_guard.judge_content`` over those same ASR readings; it is an
in-sample consistency check, not independent precision/recall or evidence
of zero human false rejections. Legacy rows lost the auto transcript when
forced Chinese was accepted. The live guard also uses a different decoder
and raw rather than fitted audio, so its decisions need not reproduce this
replay. No GPU is used:

    python scripts/vc_drift_calib.py            # the 25-workspace regression set (v3.3 rows)
    python scripts/vc_drift_calib.py test_1 SONE-846_p03
    python scripts/vc_drift_calib.py --after    # Phase-2 rows (verify_v2.json as rewritten by run_vc_only.sh)

"proxy flagged" = folded max(sim(v2 reading, v1 reading),
sim(v2 reading, intended)) < 0.5 on verify's settled reading.  Prints:
judged share, rejections by reason, how many known-bad lines the judge
catches, the residual floor, and every rejection / miss for inspection.
Before rows AND conversion membership come from the frozen _archive_v3.3
state and verify files when available. Pairing historical rows with a
current overwritten state would silently change converted/built-in counts.
``--after`` uses current verify rows and state, requiring matching intended
text and time as well as index. Proxy improvements do not prove an audible
improvement, and built-in fallback is not automatically human-correct.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_movie.vc_guard import fold_zh, judge_content, text_sim     # noqa: E402

REGRESSION_SET = ["test_1", "test_2", "output_test"] + [f"SONE-846_p{k:02d}" for k in range(1, 23)]


def reading(r: dict) -> dict:
    """WhisperClips-shaped reading; legacy zh* rows cannot recover raw auto text."""
    if "heard_auto" in r:
        return {"auto": {"text": r["heard_auto"], "language": r.get("lang_auto")},
                "zh": ({"text": r["forced_zh"]} if r.get("forced_zh") is not None else None)}
    if r["lang"] in ("zh", "zh*"):        # zh*: forced decode accepted; the raw auto text was not kept
        return {"auto": {"text": r["heard"], "language": "zh"}, "zh": None}
    return {"auto": {"text": r["heard"], "language": r.get("lang_auto")},
            "zh": ({"text": r["forced_zh"]} if r.get("forced_zh") else None)}


def same_line(a: dict, b: dict) -> bool:
    """Index alone is insufficient after segmentation/translation has changed."""
    return (fold_zh(a.get("want")) == fold_zh(b.get("want"))
            and a.get("start") is not None and b.get("start") is not None
            and abs(float(a["start"]) - float(b["start"])) <= 0.001)


def load_rows(work: Path, *, after: bool = False) -> list[dict]:
    frozen = work.parent / "_archive_v3.3" / work.name
    source = work if after or not frozen.is_dir() else frozen
    # A historical transcript without its historical state cannot establish which lines were VC.
    if not after and source == work and (work / "verify_v2.pre_v34.json").exists():
        raise ValueError("historical verify rows need _archive_v3.3/state.json; current state may be overwritten")
    v1 = {r["index"]: r for r in json.loads((source / "verify_v1.json").read_text(encoding="utf-8"))}
    v2 = json.loads((source / "verify_v2.json").read_text(encoding="utf-8"))
    st = json.loads((source / "state.json").read_text(encoding="utf-8"))
    vsegs = (st.get("vc") or {}).get("segments") or []
    rows = []
    for r in v2:
        a = v1.get(r["index"])
        if a is None:
            continue
        i = r["index"] - 1
        s = vsegs[i] if 0 <= i < len(vsegs) else {}
        state_row = {"want": s.get("text_translated"), "start": s.get("start")}
        if not same_line(a, r) or not same_line(r, state_row):
            raise ValueError(f"line {r['index']}: verify/state text or time mismatch; cannot pair by index")
        rows.append(dict(ws=work.name, idx=r["index"], vc=bool(s.get("vc")), want=r["want"], v1=a, v2=r,
                         dur=float(s.get("end", 0)) - float(s.get("start", 0)), source=str(source)))
    return rows


def main() -> int:
    after = "--after" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or REGRESSION_SET
    rows = []
    invalid = 0
    for n in names:
        w = ROOT / "workspace" / n
        try:
            rows.extend(load_rows(w, after=after))
        except Exception as exc:                        # noqa: BLE001
            print(f"{n}: excluded ({type(exc).__name__}: {exc})")
            invalid += 1
            continue
    if not rows:
        print("no paired verify rows")
        return 1
    for r in rows:
        r["bad"] = max(text_sim(r["v2"]["heard"], r["v1"]["heard"]), text_sim(r["v2"]["heard"], r["want"])) < 0.5
        r["d"] = judge_content(reading(r["v2"]), reading(r["v1"]), r["want"])
    conv = [r for r in rows if r["vc"]]
    bad = [r for r in conv if r["bad"]]
    judged = [r for r in conv if r["d"]["judged"]]
    rej = [r for r in judged if not r["d"]["ok"]]
    caught = [r for r in bad if r["d"]["judged"] and not r["d"]["ok"]]
    fp = [r for r in rej if not r["bad"]]
    bad_all = [r for r in rows if r["bad"]]
    legacy = sum(r[k].get("lang") == "zh*" and "heard_auto" not in r[k] for r in rows for k in ("v1", "v2"))
    print("ASR proxy replay only: labels reuse the scored transcripts; no human precision/recall is measured.")
    print(f"sources: {'current state/verify' if after else 'frozen v3.3 where available'}; "
          f"{legacy} legacy forced-zh rows have no original auto transcript; excluded workspaces: {invalid}")
    print(f"{'AFTER' if after else 'BEFORE'}: paired {len(rows)}, proxy flagged over all sampled lines {len(bad_all)} "
          f"({len(bad_all) / len(rows):.1%}); converted {len(conv)}, proxy-flagged converted {len(bad)} "
          f"({len(bad) / max(1, len(conv)):.1%}); built-in proxy flags {len(bad_all) - len(bad)}")
    print(f"judged {len(judged)}/{len(conv)} ({len(judged) / max(1, len(conv)):.0%}); rejected {len(rej)} "
          f"({len(rej) / max(1, len(judged)):.1%} of judged, {len(rej) / max(1, len(conv)):.1%} of converted); "
          f"reasons {dict(Counter(r['d']['reason'].split('_')[0] for r in rej))}; "
          f"gates {dict(Counter(r['d']['gate'] or 'applied' for r in conv))}")
    print(f"proxy flags rejected {len(caught)}/{len(bad)}; unrejected proxy flags "
          f"{len(bad) - len(caught)}/{len(conv)} = {(len(bad) - len(caught)) / max(1, len(conv)):.1%}; "
          f"rejected-without-proxy-flag {len(fp)} (not a human false-positive count)")
    bi = [r for r in rows if not r["vc"]]
    print(f"built-in lines {len(bi)}: judge rejects {sum(1 for r in bi if r['d']['judged'] and not r['d']['ok'])}")
    per: dict[str, list[int]] = {}
    for r in conv:
        p = per.setdefault(r["ws"], [0, 0])
        if r["d"]["judged"]:
            p[0] += 1
            p[1] += int(not r["d"]["ok"])
    print("per film rejected/judged:", {k: f"{v[1]}/{v[0]}" for k, v in per.items()})
    print("\nrejections:")
    for r in rej:
        print(f"  {r['ws']:13} #{r['idx']:<4} {r['d']['reason']:12} dur={r['dur']:.2f} bad={int(r['bad'])} "
              f"want={r['want'][:12]!r} v1={r['v1']['heard'][:12]!r} v2={r['v2']['heard'][:16]!r}")
    print("\nunrejected proxy flags:")
    for r in bad:
        if not (r["d"]["judged"] and not r["d"]["ok"]):
            print(f"  {r['ws']:13} #{r['idx']:<4} gate={r['d']['gate']} dur={r['dur']:.2f} "
                  f"want={r['want'][:12]!r} v1={r['v1']['heard'][:12]!r} v2={r['v2']['heard'][:16]!r}")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
