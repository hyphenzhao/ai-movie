#!/usr/bin/env python
"""Which reference clip does voice conversion actually reproduce?

The feasibility gate measured language, ECAPA similarity and duration — and
missed the defect that matters most on this material: converting a 232 Hz
female line against the shipped reference produced a 117 Hz voice, an octave
down, which reads as male.  ECAPA similarity did not catch it because that
embedding is largely pitch-invariant.

So this probe measures the one thing that was missing: the **F0 of the
converted output** against the F0 of the reference it was converted onto.  A
usable reference reproduces its own pitch range; one that halves it is
unusable no matter how good its similarity score looks.

    python scripts/vc_ref_probe.py --refs workspace/output_test/refs_v2/*.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINNED_TF = ROOT / "vendor" / "tts_transformers"
sys.path.insert(0, str(PINNED_TF))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "CosyVoice"))
sys.path.insert(0, str(ROOT / "models" / "CosyVoice" / "third_party" / "Matcha-TTS"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--sources", nargs="+", default=None,
                    help="source wavs to convert (default: the gate probe's)")
    ap.add_argument("--out", default="workspace/output_test/vc_ref_probe")
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    from cosyvoice.cli.cosyvoice import AutoModel

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sources = args.sources or sorted(
        str(p) for p in (ROOT / "workspace/output_test/vc_probe/sft").glob("f*.wav"))
    print(f"sources: {[Path(s).name for s in sources]}")

    print("loading CosyVoice3-0.5B …", flush=True)
    cv3 = AutoModel(model_dir=str(ROOT / "models" / "CosyVoice3-0.5B"), fp16=True)

    rows = []
    for ref in args.refs:
        for src in sources:
            key = f"{Path(ref).stem}__{Path(src).stem}"
            try:
                chunks = [g["tts_speech"].squeeze(0).cpu().numpy()
                          for g in cv3.inference_vc(src, ref, stream=False)]
                audio = np.concatenate(chunks)
                p = out / f"{key}.wav"
                sf.write(str(p), audio, cv3.sample_rate)
                rows.append({"ref": ref, "src": src, "vc": str(p)})
                print(f"  {key}: {len(audio)/cv3.sample_rate:.2f}s", flush=True)
            except Exception as exc:
                print(f"  {key}: FAILED {type(exc).__name__}: {exc}", flush=True)

    (out / "manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
