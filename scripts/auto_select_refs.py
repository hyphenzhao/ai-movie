#!/usr/bin/env python
"""Pick voice-conversion reference clips for a video, fully automatically.

The criterion is the one that actually predicts VC quality on this pipeline:
**does converting known-good audio onto the candidate reproduce the expected
pitch?**  On the first film, the shipped reference halved every female line's
F0 (232 Hz → 117 Hz, reads as male) while scoring a *higher* ECAPA similarity
than some working clips — ECAPA is largely pitch-invariant, so it cannot see
octave collapse.  See Documentation/vc-gate-result.md.

Procedure per gender found in the diarization:

  1. cut the top candidate windows from the separated vocals — longest
     segments with the most voiced frames (voiced frames matter more than
     duration: an off-mic speaker buried under the other's laughter can have
     a 7-second window with 4 usable frames)
  2. voice-convert 2 known-clean SFT probe lines onto each candidate
     (the probe wavs are generic Chinese and reusable across videos)
  3. keep candidates whose *outputs* land in the gender's pitch range;
     among those prefer the output/reference F0 ratio closest to 1.0
  4. if nothing qualifies, emit null — the caller keeps the built-in voice
     for that gender rather than shipping an octave-collapsed clone

Runs the actual conversion via scripts/vc_ref_probe.py in a subprocess (it
needs the pinned transformers), measures in this process (needs librosa +
the app env).

    python scripts/auto_select_refs.py workspace/test_1/state.json
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

from ai_movie.config import OSD_REF_MAX_OVERLAP, TTS_GENDER_HZ as GENDER_HZ  # noqa: E402
from ai_movie.pitch import f0_median as _f0_median                            # noqa: E402

# Generic Chinese probe lines (built-in SFT voices) reused across videos.
SOURCES = {
    "female": ["asset/vc_probe/f1.wav", "asset/vc_probe/f2.wav"],
    "male": ["asset/vc_probe/m2.wav", "asset/vc_probe/m0.wav"],
}
N_CANDIDATES = 3
MIN_SEG_SECONDS = 2.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("state")
    ap.add_argument("--out", default=None,
                    help="dir for candidate/selected wavs (default <work>/refs_auto)")
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    from ai_movie.diarize import _load_mono16k

    state_path = Path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    work = state_path.parent
    out = Path(args.out) if args.out else work / "refs_auto"
    out.mkdir(parents=True, exist_ok=True)

    vocals_path = (state.get("separate") or {}).get("vocals")
    if not vocals_path or not Path(vocals_path).exists():
        print("no separated vocals")
        return 1
    vocals = _load_mono16k(vocals_path)
    segs = state["asr"]["segments"]

    def f0_profile(a: "np.ndarray"):
        return _f0_median(a)

    def f0_of(path: str):
        return _f0_median(path)[0]

    # Overlapped-speech regions (osd stage): a window where two people talk
    # is useless as a timbre reference, whatever its pitch says.
    overlap = ((state.get("osd") or {}).get("regions")) or []

    def overlap_ratio(a: float, b: float) -> float:
        if b <= a or not overlap:
            return 0.0
        cov = sum(max(0.0, min(b, float(e)) - max(a, float(s_)))
                  for s_, e in overlap)
        return cov / (b - a)

    genders = sorted({s.get("gender") for s in segs if s.get("gender")})
    print(f"genders present: {genders}")

    # ── stage 1: cut candidates ────────────────────────────────────────
    candidates: dict[str, list[dict]] = {}
    for g in genders:
        scored = []
        for i, s in enumerate(segs):
            if s.get("gender") != g:
                continue
            d = float(s["end"]) - float(s["start"])
            if d < MIN_SEG_SECONDS:
                continue
            if overlap_ratio(float(s["start"]), float(s["end"])) > OSD_REF_MAX_OVERLAP:
                continue
            a = vocals[int(s["start"] * 16000):int(s["end"] * 16000)]
            med, voiced = f0_profile(a)
            # A measurable F0 in the wrong range means the *other* speaker
            # dominates this window (the overlap problem) — not a candidate.
            lo, hi = GENDER_HZ[g]
            if med is not None and not (lo * 0.9 <= med <= hi * 1.1):
                continue
            scored.append((voiced, d, i, med))
        scored.sort(reverse=True)
        cands = []
        for voiced, d, i, med in scored[:N_CANDIDATES]:
            s = segs[i]
            a = vocals[int(s["start"] * 16000):int(s["end"] * 16000)]
            p = out / f"cand_{g}_seg{i + 1:04d}.wav"
            sf.write(str(p), a, 16000)
            cands.append({"path": str(p), "seg": i + 1, "dur": round(d, 2),
                          "voiced": voiced, "f0": med})
            print(f"  {g} candidate #{i + 1}: {d:.1f}s voiced={voiced} "
                  f"f0={med and round(med, 1)}")
        candidates[g] = cands
        if not cands:
            print(f"  {g}: NO candidate windows at all")

    # ── stage 2: convert probes onto every candidate ───────────────────
    all_refs = [c["path"] for cs in candidates.values() for c in cs]
    if not all_refs:
        (out / "refs.json").write_text(json.dumps({g: None for g in genders}),
                                       encoding="utf-8")
        print("no candidates anywhere — refs.json is all null")
        return 0

    probe_out = out / "probe"
    srcs = sorted({str(ROOT / s) for g in genders for s in SOURCES.get(g, [])})
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "vc_ref_probe.py"),
           "--refs", *all_refs, "--sources", *srcs, "--out", str(probe_out)]
    print("running VC probe subprocess…")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
    if r.returncode != 0:
        print(f"probe failed:\n{r.stderr[-800:]}")
        return 1
    manifest = json.loads((probe_out / "manifest.json").read_text(encoding="utf-8"))

    # ── stage 3: score by output pitch ─────────────────────────────────
    by_ref: dict[str, list[dict]] = {}
    for rec in manifest:
        by_ref.setdefault(rec["ref"], []).append(rec)

    picked: dict[str, str | None] = {}
    report: dict[str, list] = {}
    for g, cands in candidates.items():
        lo, hi = GENDER_HZ[g]
        rows = []
        for c in cands:
            outs = [f0_of(rec["vc"]) for rec in by_ref.get(c["path"], [])]
            measurable = [o for o in outs if o is not None]
            in_range = [o for o in measurable if lo <= o <= hi]
            ratio = None
            if c["f0"] and measurable:
                med_out = sorted(measurable)[len(measurable) // 2]
                ratio = med_out / c["f0"]
            rows.append({**c, "out_f0": [o and round(o, 1) for o in outs],
                         "n_in_range": len(in_range),
                         "n_measurable": len(measurable),
                         "ratio": ratio and round(ratio, 3)})
            print(f"  {g} seg{c['seg']}: outputs {rows[-1]['out_f0']} "
                  f"in-range {len(in_range)}/{len(outs)} ratio={rows[-1]['ratio']}")
        report[g] = rows
        # Qualify: every measurable output in range, and at least one
        # measurable.  Prefer ratio nearest 1.0, then more voiced frames.
        ok = [r_ for r_ in rows
              if r_["n_measurable"] > 0 and r_["n_in_range"] == r_["n_measurable"]]
        ok.sort(key=lambda r_: (abs((r_["ratio"] or 1.0) - 1.0), -r_["voiced"]))
        picked[g] = ok[0]["path"] if ok else None
        print(f"  {g} → {picked[g] and Path(picked[g]).name}"
              f"{'' if picked[g] else ' (none qualified — keep built-in voice)'}")

    (out / "refs.json").write_text(
        json.dumps({"picked": picked, "candidates": report},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out / 'refs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
