#!/usr/bin/env python
"""Replay the VC content-drift judge over verify_dub's paired rows (offline, CPU, JSON only).

The judge's thresholds (config VC_DRIFT_*) were set on the 417 paired
verify_v1 / verify_v2 rows of the v3.3 release (299 converted lines).  This
replays ``vc_guard.judge_content`` over those rows using verify_dub's own
readings (auto language + forced-zh re-decode — the regime
``asr.WhisperClips`` mirrors), so the numbers behind the pre-registered
Phase-2 target can be reproduced without a GPU:

    python scripts/vc_drift_calib.py            # the 25-workspace regression set (v3.3 rows)
    python scripts/vc_drift_calib.py test_1 SONE-846_p03
    python scripts/vc_drift_calib.py --after    # Phase-2 rows (verify_v2.json as rewritten by run_vc_only.sh)

"known bad" = the design's flag: folded max(sim(v2 reading, v1 reading),
sim(v2 reading, intended)) < 0.5 on verify's settled reading.  Prints:
judged share, rejections by reason, how many known-bad lines the judge
catches, the residual floor, and every rejection / miss for inspection.
verify_v2.pre_v34.json (kept by run_vc_only.sh) is preferred so the replay
stays on the v3.3 rows after a Phase-2 rerun overwrites verify_v2.json;
``--after`` reads verify_v2.json instead and pairs it with the SAME
verify_v1.json (the v1 rows are index-keyed and unchanged), which is how the
pre-registered Phase-2 numbers are read: the bad share over ALL sampled
lines (v3.3 baseline 43/417 = 10.3 %; built-in lines re-read at 1.00, so a
line the judge sent back to the built-in voice counts as fixed) and over the
converted ones (43/299 = 14.4 %).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_movie.vc_guard import judge_content, text_sim     # noqa: E402

REGRESSION_SET = ["test_1", "test_2", "output_test"] + [f"SONE-846_p{k:02d}" for k in range(1, 23)]


def reading(r: dict) -> dict:
    """WhisperClips-shaped reading from a verify_dub row."""
    if r["lang"] in ("zh", "zh*"):        # zh*: forced decode accepted; the raw auto text was not kept
        return {"auto": {"text": r["heard"], "language": "zh"}, "zh": None}
    return {"auto": {"text": r["heard"], "language": r.get("lang_auto")},
            "zh": ({"text": r["forced_zh"]} if r.get("forced_zh") else None)}


def main() -> int:
    after = "--after" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or REGRESSION_SET
    rows = []
    for n in names:
        w = ROOT / "workspace" / n
        try:
            v1 = {r["index"]: r for r in json.loads((w / "verify_v1.json").read_text(encoding="utf-8"))}
            src = w / "verify_v2.pre_v34.json"
            if after or not src.exists():
                src = w / "verify_v2.json"
            v2 = json.loads(src.read_text(encoding="utf-8"))
            st = json.loads((w / "state.json").read_text(encoding="utf-8"))
        except Exception as exc:                        # noqa: BLE001
            print(f"{n}: skipped ({type(exc).__name__})")
            continue
        vsegs = (st.get("vc") or {}).get("segments") or []
        for r in v2:
            a = v1.get(r["index"])
            if not a:
                continue
            s = vsegs[r["index"] - 1] if r["index"] - 1 < len(vsegs) else {}
            rows.append(dict(ws=n, idx=r["index"], vc=bool(s.get("vc")), want=r["want"], v1=a, v2=r,
                             dur=float(s.get("end", 0)) - float(s.get("start", 0))))
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
    print(f"{'AFTER' if after else 'BEFORE'}: paired {len(rows)}, bad over all sampled lines {len(bad_all)} "
          f"({len(bad_all) / len(rows):.1%}); converted {len(conv)}, known-bad converted {len(bad)} "
          f"({len(bad) / max(1, len(conv)):.1%}); built-in bad {len(bad_all) - len(bad)}")
    print(f"judged {len(judged)}/{len(conv)} ({len(judged) / max(1, len(conv)):.0%}); rejected {len(rej)} "
          f"({len(rej) / max(1, len(judged)):.1%} of judged, {len(rej) / max(1, len(conv)):.1%} of converted); "
          f"reasons {dict(Counter(r['d']['reason'].split('_')[0] for r in rej))}; "
          f"gates {dict(Counter(r['d']['gate'] or 'applied' for r in conv))}")
    print(f"known-bad caught {len(caught)}/{len(bad)}; residual floor on the verify metric "
          f"{len(bad) - len(caught)}/{len(conv)} = {(len(bad) - len(caught)) / max(1, len(conv)):.1%}; "
          f"rejected-but-not-known-bad {len(fp)}")
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
    print("\nknown-bad missed:")
    for r in bad:
        if not (r["d"]["judged"] and not r["d"]["ok"]):
            print(f"  {r['ws']:13} #{r['idx']:<4} gate={r['d']['gate']} dur={r['dur']:.2f} "
                  f"want={r['want'][:12]!r} v1={r['v1']['heard'][:12]!r} v2={r['v2']['heard'][:16]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
