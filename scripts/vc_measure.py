#!/usr/bin/env python
"""Score the VC probe from :mod:`scripts.vc_feasibility` against the gate.

Runs in the *app* environment (needs faster-whisper + the diarization
encoder), which is why it is a second process: the probe itself must run
under the pinned transformers 4.51.3.

Three measurements per case:

  language   Whisper with ``language=None`` (auto-detect).  The zero-shot
             version failed here — 10 of 20 sampled segments came back as
             Japanese/Korean gibberish.  Also counts residual kana, since
             Whisper sometimes labels kana output as ``zh``.
  timbre     ECAPA cosine similarity against the speaker's reference clip.
             Wrong-timbre control measured 0.10 previously.
  duration   VC output length vs the source it converted, from the probe's
             manifest.

    python scripts/vc_measure.py workspace/output_test/vc_probe
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KANA = re.compile(r"[぀-ゟ゠-ヿ]")
HANZI = re.compile(r"[一-鿿]")

SIM_GATE = 0.45
RATIO_LO, RATIO_HI = 0.98, 1.02


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("probe_dir")
    # Default to the project's own local checkout: passing "large-v3" makes
    # faster-whisper resolve it against HuggingFace, which stalls here.
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    probe = Path(args.probe_dir)
    manifest = json.loads((probe / "manifest.json").read_text(encoding="utf-8"))

    from ai_movie.asr import _load_cpu_model
    from ai_movie.config import ASR_MODEL_SIZE
    from ai_movie.diarize import similarity

    # Reuse the pipeline's loader: ctranslate2 has no ROCm build on this box,
    # so it must fall back to CPU int8 rather than dying on the CUDA probe.
    model = args.model or ASR_MODEL_SIZE
    print(f"loading whisper: {model}", flush=True)
    asr = _load_cpu_model(model)

    def read_back(path: str) -> tuple[str, float, str]:
        segs, info = asr.transcribe(path, language=None, beam_size=5,
                                    vad_filter=False)
        text = "".join(s.text for s in segs).strip()
        return info.language, float(info.language_probability), text

    def overlap_of(want_text: str, heard: str) -> float:
        want = set(HANZI.findall(want_text))
        return len(want & set(HANZI.findall(heard))) / max(len(want), 1)

    # The other speaker's reference — the floor for "similarity means nothing".
    other_ref = {rec["gender"]: rec["ref"] for rec in manifest}
    flip = {"female": "male", "male": "female"}

    rows = []
    for rec in manifest:
        if not rec.get("vc"):
            rows.append({**rec, "verdict": "ERROR"})
            continue
        # Control: read back the SFT *source* through the identical path, so a
        # short clip that Whisper already mangles is not blamed on VC.
        s_lang, _, s_text = read_back(rec["sft"])
        lang, prob, text = read_back(rec["vc"])
        sim = similarity(rec["vc"], rec["ref"])
        sim_wrong = similarity(rec["vc"], other_ref[flip[rec["gender"]]])
        rows.append({
            "key": rec["key"], "gender": rec["gender"], "text": rec["text"],
            "heard": text, "lang": lang, "lang_p": round(prob, 3),
            "kana": len(KANA.findall(text)),
            "overlap": round(overlap_of(rec["text"], text), 3),
            "src_heard": s_text, "src_lang": s_lang,
            "src_kana": len(KANA.findall(s_text)),
            "src_overlap": round(overlap_of(rec["text"], s_text), 3),
            "sim": round(sim, 3), "sim_wrong": round(sim_wrong, 3),
            "ratio": rec.get("dur_ratio"),
            "sft_dur": round(rec.get("sft_dur", 0), 2),
        })

    (probe / "measure.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 96)
    print("VC output vs the SFT source it was converted from (source = control)")
    print(f"{'key':4} {'dur':>5} | {'lang':5} {'chars':>6} {'sim':>6} {'wrong':>6} {'ratio':>6} "
          f"| {'src':5} {'chars':>6}")
    print("-" * 96)
    for r in rows:
        print(f"{r['key']:4} {r.get('sft_dur',0):5.2f} | {r.get('lang','-'):5} "
              f"{r.get('overlap',0):6.2f} {r.get('sim',0):6.3f} "
              f"{r.get('sim_wrong',0):6.3f} {(r.get('ratio') or 0):6.3f} "
              f"| {r.get('src_lang','-'):5} {r.get('src_overlap',0):6.2f}")
    print()
    for r in rows:
        print(f"  {r['key']}  want: {r['text']}")
        print(f"      src : {r.get('src_heard','')}")
        print(f"      vc  : {r.get('heard','')}")

    ok_lang = sum(1 for r in rows if r.get("lang") == "zh" and not r.get("kana"))
    src_lang_ok = sum(1 for r in rows if r.get("src_lang") == "zh" and not r.get("src_kana"))
    print(f"\ncontrol: the SFT sources themselves score {src_lang_ok}/{len(rows)} "
          f"on the same language test")
    wrongs = [r["sim_wrong"] for r in rows if "sim_wrong" in r]
    if wrongs:
        print(f"control: similarity against the *other* speaker's reference "
              f"ranges {min(wrongs):.3f}–{max(wrongs):.3f}")
    sims = [r["sim"] for r in rows if "sim" in r]
    ratios = [r["ratio"] for r in rows if r.get("ratio")]
    n = len(rows)

    print("=" * 78)
    print(f"GATE 1 language : {ok_lang}/{n} zh & kana-free "
          f"{'PASS' if ok_lang == n else 'FAIL'}")
    if sims:
        worst = min(sims)
        print(f"GATE 2 timbre   : min={worst:.3f} median={sorted(sims)[len(sims)//2]:.3f} "
              f"(gate {SIM_GATE}) {'PASS' if worst >= SIM_GATE else 'FAIL'}")
    if ratios:
        inside = sum(1 for x in ratios if RATIO_LO <= x <= RATIO_HI)
        print(f"GATE 3 duration : {inside}/{len(ratios)} in [{RATIO_LO},{RATIO_HI}], "
              f"range {min(ratios):.3f}–{max(ratios):.3f}")
        print("                  (a post-VC re-fit snaps every segment back to "
              "its exact slot, so\n                   deviations this small are "
              "absorbed as <4% tempo change)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
