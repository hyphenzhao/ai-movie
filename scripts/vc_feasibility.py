#!/usr/bin/env python
"""Step 0 gate: can ``inference_vc`` replace zero-shot cloning?

Zero-shot cloning failed because CosyVoice's LLM path is driven by
``prompt_text`` — we feed it *Japanese*, so Japanese leaks into the Chinese
output and some references deterministically push the model into a
degenerate mode (see Documentation/v2-quality-upgrade.md).

``inference_vc`` takes a different path entirely: ``frontend_vc`` builds no
``text`` and no ``llm_embedding``, and ``vc_job`` feeds the source speech
tokens straight into the token queue, bypassing the LLM.  So the *content*
comes from a source wav we already trust and only the *timbre* comes from
the reference.

This script produces the evidence:

  stage A  CosyVoice-300M-SFT synthesizes 6 Chinese sentences with built-in
           voices (中文女 / 中文男) — these stand in for v1's audio.
  stage B  CosyVoice3-0.5B voice-converts each onto the real speaker's
           reference clip.

Measurement lives in ``scripts/vc_measure.py`` (needs the app's own
transformers, so it cannot share this process).

    python scripts/vc_feasibility.py --out workspace/output_test/vc_probe
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINNED_TF = ROOT / "vendor" / "tts_transformers"

# Must precede any transformers import: CosyVoice3's Qwen LLM decodes to
# garbage under the app's transformers 5.x (project_cosyvoice3_garble).
if not (PINNED_TF / "transformers").exists():
    raise SystemExit(f"pinned transformers missing at {PINNED_TF}; "
                     f"run bash scripts/setup_tts_transformers.sh")
sys.path.insert(0, str(PINNED_TF))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "CosyVoice"))
sys.path.insert(0, str(ROOT / "models" / "CosyVoice" / "third_party" / "Matcha-TTS"))

# Real lines from the film, so the probe exercises the same text the pipeline
# will. Three per speaker, mid-length.
HANZI_RE = re.compile(r"[\u4e00-\u9fff]")

CASES = [
    ("f0", "female", "中文女", "我参加了电视剧集。"),
    ("f1", "female", "中文女", "毕竟我们姑且还是老师和学生的关系。"),
    ("f2", "female", "中文女", "能穿上梦寐以求的校服，我非常开心。"),
    ("m0", "male", "中文男", "虽然开始了，不过大家还好吗？"),
    ("m1", "male", "中文男", "这次是第几部电影？"),
    ("m2", "male", "中文男", "因为感觉像在当老师，所以还挺开心的吧。"),
]


def cases_from_state(state_path: Path, n: int) -> list[tuple[str, str, str, str]]:
    """Sample real pipeline lines, stratified by length.

    The first probe showed every problem clustering in the shortest clips, so
    a length-stratified sample is the only way to tell a general VC weakness
    from a short-utterance one.
    """
    state = json.loads(state_path.read_text(encoding="utf-8"))
    segs = (state.get("fit") or state.get("translate") or {}).get("segments") or []
    pool = [s for s in segs if (s.get("text_translated") or "").strip()]
    out: list[tuple[str, str, str, str]] = []
    for gender, spk in (("female", "中文女"), ("male", "中文男")):
        mine = sorted((s for s in pool if (s.get("gender") or "female") == gender),
                      key=lambda s: len(HANZI_RE.findall(s["text_translated"])))
        if not mine:
            continue
        half = max(1, n // 2)
        # Even spread across the length range, shortest to longest.
        picks = [mine[round(k * (len(mine) - 1) / max(half - 1, 1))]
                 for k in range(half)]
        seen: set[str] = set()
        for k, s in enumerate(picks):
            text = s["text_translated"].strip()
            if text in seen:
                continue
            seen.add(text)
            out.append((f"{gender[0]}{k}", gender, spk, text))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="workspace/output_test/vc_probe")
    ap.add_argument("--ref-female",
                    default="workspace/output_test/synthesized/ref_S0_alt1.wav")
    ap.add_argument("--ref-male",
                    default="workspace/output_test/synthesized/ref_S1.wav")
    ap.add_argument("--from-state", default=None,
                    help="sample real lines from a run's state.json instead of "
                         "the built-in six")
    ap.add_argument("-n", type=int, default=16,
                    help="how many lines to sample with --from-state")
    args = ap.parse_args()

    cases = CASES
    if args.from_state:
        cases = cases_from_state(Path(args.from_state), args.n)
        print(f"sampled {len(cases)} lines from {args.from_state}", flush=True)

    out = Path(args.out)
    (out / "sft").mkdir(parents=True, exist_ok=True)
    (out / "vc").mkdir(parents=True, exist_ok=True)

    import soundfile as sf
    import torch
    from cosyvoice.cli.cosyvoice import AutoModel

    refs = {"female": str(ROOT / args.ref_female),
            "male": str(ROOT / args.ref_male)}
    manifest: list[dict] = []

    # ── stage A: built-in-voice sources (stand-in for v1) ──────────────
    print("[A] loading CosyVoice-300M-SFT …", flush=True)
    sft = AutoModel(model_dir=str(ROOT / "models" / "CosyVoice-300M-SFT"),
                    fp16=True)
    print(f"[A] built-in speakers: {sft.list_available_spks()}", flush=True)
    for key, gender, spk, text in cases:
        chunks = [g["tts_speech"].squeeze(0).cpu().numpy()
                  for g in sft.inference_sft(text, spk, stream=False)]
        import numpy as np
        audio = np.concatenate(chunks)
        path = out / "sft" / f"{key}.wav"
        sf.write(str(path), audio, sft.sample_rate)
        dur = len(audio) / sft.sample_rate
        print(f"[A] {key} {spk} {dur:5.2f}s  {text}", flush=True)
        manifest.append({"key": key, "gender": gender, "spk": spk,
                         "text": text, "sft": str(path), "sft_dur": dur,
                         "ref": refs[gender]})

    del sft
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── stage B: voice conversion onto the real speakers ───────────────
    print("[B] loading CosyVoice3-0.5B …", flush=True)
    cv3 = AutoModel(model_dir=str(ROOT / "models" / "CosyVoice3-0.5B"),
                    fp16=True)
    for rec in manifest:
        try:
            # This fork's frontend loads its own audio: _extract_speech_token /
            # _extract_spk_embedding / _extract_speech_feat all call
            # load_wav(arg, sr) internally, so BOTH arguments are *paths*,
            # despite the parameter being named ``source_speech_16k``.
            import numpy as np
            chunks = [g["tts_speech"].squeeze(0).cpu().numpy()
                      for g in cv3.inference_vc(rec["sft"], rec["ref"],
                                                stream=False)]
            audio = np.concatenate(chunks)
            path = out / "vc" / f"{rec['key']}.wav"
            sf.write(str(path), audio, cv3.sample_rate)
            rec["vc"] = str(path)
            rec["vc_dur"] = len(audio) / cv3.sample_rate
            rec["dur_ratio"] = round(rec["vc_dur"] / rec["sft_dur"], 4)
            print(f"[B] {rec['key']} {rec['vc_dur']:5.2f}s  "
                  f"ratio={rec['dur_ratio']:.3f}", flush=True)
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[B] {rec['key']} FAILED: {rec['error']}", flush=True)

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
