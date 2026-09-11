#!/usr/bin/env python
"""Acceptance check on the synthesized speech itself, segment by segment.

The previous release passed every pipeline-level check and was still worse to
listen to, because nothing ever inspected the *audio that came out*.  Two
failures hid there: half the sampled segments were not Chinese at all (the
zero-shot LLM was conditioned on a Japanese reference transcript), and 27% of
"female" segments came out in the male pitch range.

So both are measured directly:

  language  Whisper with auto-detect on each synthesized wav.  Counts kana
            separately, because Whisper sometimes labels kana output ``zh``.
  content   hanzi overlap between what was intended and what comes back.
  pitch     median F0 against the range for the segment's assigned gender.
            Not a similarity score — just "is this voice the right sex".

    python scripts/verify_dub.py workspace/output_test/state.json --key fit -n 20
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
# Deliberately wide, and overlapping in the middle: the question is whether a
# voice is unambiguously the wrong sex, not where the boundary sits.
FEMALE_HZ = (165.0, 320.0)
MALE_HZ = (70.0, 175.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("state")
    ap.add_argument("--key", default="fit",
                    choices=["fit", "tts", "vc"],
                    help="which stage's segments to check")
    ap.add_argument("--audio-field", default="audio_fit")
    ap.add_argument("-n", type=int, default=20, help="how many to sample (0=all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    from ai_movie.asr import _load_cpu_model
    from ai_movie.config import ASR_MODEL_SIZE
    from ai_movie.diarize import _load_mono16k, pitch_track

    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    segs = (state.get(args.key) or {}).get("segments") or []
    pool = [(i, s) for i, s in enumerate(segs)
            if (s.get(args.audio_field) or s.get("audio"))
            and (s.get("text_translated") or "").strip()]
    if not pool:
        print(f"no audio under state['{args.key}'] field '{args.audio_field}'")
        return 1
    if args.n and args.n < len(pool):
        # Even spread across the film rather than the first N, so a problem
        # confined to one stretch cannot hide.
        step = len(pool) / args.n
        pool = [pool[int(k * step)] for k in range(args.n)]

    print(f"loading whisper: {ASR_MODEL_SIZE}", flush=True)
    asr = _load_cpu_model(ASR_MODEL_SIZE)

    rows = []
    for i, s in pool:
        path = s.get(args.audio_field) or s["audio"]
        want = (s.get("text_translated") or "").strip()
        gender = s.get("gender") or "female"
        out, info = asr.transcribe(path, language=None, beam_size=5,
                                   vad_filter=False)
        heard = "".join(x.text for x in out).strip()
        wset = set(HANZI.findall(want))
        overlap = len(wset & set(HANZI.findall(heard))) / max(len(wset), 1)

        # Whisper's language ID is unreliable on sub-second clips — it labelled
        # two correct Chinese lines "ko" while transcribing them accurately.
        # Settle those by forcing a Chinese decode: if that returns the text we
        # asked for, the audio is Chinese and the auto-detect was the error.
        forced = None
        if info.language != "zh":
            f_out, _ = asr.transcribe(path, language="zh", beam_size=5,
                                      vad_filter=False)
            forced = "".join(x.text for x in f_out).strip()
            f_overlap = (len(wset & set(HANZI.findall(forced)))
                         / max(len(wset), 1))
            if f_overlap >= 0.6 and not KANA.search(forced):
                heard, overlap = forced, f_overlap
                info_language = "zh*"          # * = confirmed by forced decode
            else:
                info_language = info.language
        else:
            info_language = "zh"

        a = _load_mono16k(path)
        f0, ok, _ = pitch_track(a, cache_key=None)
        med = float(np.median(f0[ok])) if int(ok.sum()) >= 6 else None
        lo, hi = FEMALE_HZ if gender == "female" else MALE_HZ
        in_range = None if med is None else (lo <= med <= hi)

        rows.append({"index": i + 1, "start": s.get("start"), "gender": gender,
                     "lang": info_language, "lang_auto": info.language,
                     "forced_zh": forced,
                     "lang_p": round(float(info.language_probability), 3),
                     "kana": len(KANA.findall(heard)),
                     "overlap": round(overlap, 3),
                     "f0": round(med, 1) if med else None,
                     "f0_ok": in_range, "want": want, "heard": heard})
        print(f"  #{i+1:>3} {gender:6} {info_language:3} kana={rows[-1]['kana']:>2} "
              f"chars={overlap:.2f} f0={rows[-1]['f0'] or '-':>6} "
              f"{'ok' if in_range else ('OUT' if in_range is False else '-')}  "
              f"{heard[:26]}", flush=True)

    n = len(rows)
    zh = sum(1 for r in rows
             if r["lang"].startswith("zh") and not r["kana"])
    forced_n = sum(1 for r in rows if r["lang"] == "zh*")
    kana = sum(r["kana"] for r in rows)
    measured = [r for r in rows if r["f0_ok"] is not None]
    outr = [r for r in measured if not r["f0_ok"]]
    ov = sorted(r["overlap"] for r in rows)

    print("\n" + "=" * 72)
    print(f"language : {zh}/{n} Chinese & kana-free   (total kana chars: {kana})")
    if forced_n:
        print(f"           {forced_n} of those needed a forced Chinese decode to "
              f"confirm —\n           Whisper's auto language-ID mislabels "
              f"sub-second clips, the audio itself was fine")
    print(f"content  : median hanzi overlap {ov[len(ov)//2]:.2f}, "
          f"worst {ov[0]:.2f}")
    if measured:
        print(f"pitch    : {len(outr)}/{len(measured)} outside the range for their "
              f"assigned gender ({100*len(outr)/len(measured):.0f}%)")
        for r in outr:
            print(f"           #{r['index']} {r['gender']} f0={r['f0']} Hz")
    print("=" * 72)

    out = Path(args.out) if args.out else Path(args.state).parent / f"verify_{args.key}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
