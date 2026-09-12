#!/usr/bin/env python
"""Build the original-timbre version on top of a finished built-in-voice run.

v1 speaks every line with a CosyVoice built-in speaker (中文女 / 中文男).
That is the only configuration where the Japanese source cannot leak into the
Chinese output, because ``frontend_sft`` conditions the model on text plus a
speaker embedding and never on a reference transcript.  It also means the dub
sounds like two stock voices rather than the people on screen.

This script converts v1's audio to the real speakers' timbre with
``inference_vc``, which takes its *content* from v1's wavs and only its
*timbre* from a reference clip — so there is still no text conditioning that
could reintroduce Japanese.  Evidence in ``Documentation/vc-gate-result.md``.

The point of converting v1's *fitted* audio (rather than re-synthesizing) is
that voice conversion preserves duration to within one 25 Hz token frame
(~40 ms).  Each converted segment is then stretched to exactly the duration of
the v1 wav it came from, so the timeline is identical by construction and the
converted track can reuse v1's lip-sync render instead of paying for a second
one.  (Re-running the ordinary slot fit does *not* work here — it recompresses
audio v1 already compressed; measured drift up to 120 ms.)

    python scripts/run_vc_version.py workspace/output_test/state.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Chosen by measuring whether each candidate *reproduces its own pitch*
# (scripts/vc_ref_probe.py), not by heuristic score or ECAPA similarity.
# The reference decides the outcome: converting onto the clip the previous
# release shipped drove a 232 Hz line down to 117 Hz — an octave, which reads
# as male — while this one holds the ratio at 0.97-1.01. ECAPA barely
# separated the two (0.44 vs 0.70), because that embedding is largely
# pitch-invariant; the F0 ratio is what exposes it.
DEFAULT_REFS = {
    "female": "workspace/output_test/refs_v2/ref_seg0027_female.wav",
    "male": "workspace/output_test/synthesized/ref_S1.wav",
}
# Any segment whose timing moves more than one frame breaks the premise that
# v1's lip-sync video can be reused.
FRAME_TOLERANCE = 1.0 / 29.97


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("state", help="workspace/<name>/state.json from the v1 run")
    ap.add_argument("--out-name", default="v2_cloned")
    ap.add_argument("--ref-female", default=DEFAULT_REFS["female"])
    ap.add_argument("--ref-male", default=DEFAULT_REFS["male"])
    args = ap.parse_args()

    from ai_movie import artifacts, tts as tts_mod
    from ai_movie.composer import (build_speech_track, compose_video,
                                   mix_audio, stretch_audio)

    state_path = Path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    work = state_path.parent
    deliver = work / "deliverables" / args.out_name
    deliver.mkdir(parents=True, exist_ok=True)

    v1_segs = (state.get("fit") or {}).get("segments") or []
    if not v1_segs:
        log("state has no fitted segments — run v1 first")
        return 1
    v1_ends = {i: s.get("fit_end") for i, s in enumerate(v1_segs)}

    refs = {}
    for spk, meta in ((state["asr"]["diarization"].get("speakers")) or {}).items():
        g = meta.get("gender")
        p = ROOT / (args.ref_female if g == "female" else args.ref_male)
        if p.exists():
            refs[spk] = {"ref_audio": str(p), "gender": g}
    log(f"references: { {k: Path(v['ref_audio']).name for k, v in refs.items()} }")
    if not refs:
        log("no usable reference clips")
        return 1

    # ── convert ────────────────────────────────────────────────────────
    segs = [dict(s) for s in v1_segs]
    out_dir = work / "synthesized_vc"
    log(f"voice-converting {len(segs)} segments…")
    items = tts_mod.run_vc_conversion(
        segs, refs, out_dir,
        progress_cb=lambda d, t: log(f"  VC {d}/{t}") if d % 10 == 0 else None)

    converted = 0
    for i, s in enumerate(segs):
        it = items.get(i, {})
        if it.get("audio"):
            s["audio"] = it["audio"]
            s["vc"] = bool(it.get("vc"))
            converted += int(bool(it.get("vc")))
        else:
            s["audio"] = None
        s.pop("audio_fit", None)
        s.pop("fit_ratio", None)
        s.pop("fit_end", None)
    log(f"converted {converted}/{len(segs)} segments "
        f"({len(segs) - converted} kept the built-in voice)")

    # ── pin onto v1's timeline ─────────────────────────────────────────
    #
    # Not fit_segments_to_timeline: that recomputes the slot fit from scratch,
    # and its input here is v1's *already* fitted audio.  Measured, it drifted
    # up to 120 ms — segments v1 had compressed to 1.60× got compressed a
    # second time, moving their ends *earlier*, while untouched segments moved
    # later by the ~30-50 ms voice conversion adds.
    #
    # Matching each converted wav to the exact duration of the v1 wav it came
    # from makes the timeline identical by construction rather than
    # approximately.  The stretch needed is only the conversion overhead
    # (1.02-1.05×), which is inaudible.
    log("pinning each segment to its v1 duration…")
    fitted_dir = out_dir / "fitted"
    fitted_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    for i, s in enumerate(segs):
        v1 = v1_segs[i]
        src, ref_wav = s.get("audio"), v1.get("audio_fit")
        if not src or not ref_wav or not Path(ref_wav).exists():
            continue
        target = sf.info(ref_wav).duration
        have = sf.info(src).duration
        dst = fitted_dir / (Path(src).stem + ".fit.wav")
        stretch_audio(Path(src), dst, have / target if target > 0 else 1.0)
        # Trim or pad the residue so the sample count matches exactly; atempo
        # lands within a millisecond or two but "within" is not "equal".
        a, sr = sf.read(str(dst), dtype="float32")
        want = int(round(target * sr))
        if len(a) > want:
            a = a[:want]
        elif len(a) < want:
            import numpy as np
            a = np.concatenate([a, np.zeros(want - len(a), dtype="float32")])
        sf.write(str(dst), a, sr)
        s["audio_fit"] = str(dst)
        s["fit_ratio"] = round(have / target, 4) if target > 0 else 1.0
        s["fit_end"] = v1.get("fit_end")
        s["overrun"] = v1.get("overrun", 0)

    drifts = []
    for i, s in enumerate(segs):
        a, b = v1_ends.get(i), s.get("fit_end")
        if a is not None and b is not None:
            drifts.append((abs(b - a), i))
    drifts.sort(reverse=True)
    worst = drifts[0] if drifts else (0.0, -1)
    over = [d for d, _ in drifts if d > FRAME_TOLERANCE]
    log(f"timeline vs v1: max drift {worst[0] * 1000:.1f} ms (segment {worst[1] + 1}); "
        f"{len(over)}/{len(drifts)} segments beyond one frame")
    reuse_lipsync = not over
    if not reuse_lipsync:
        log("  → drift too large to reuse v1's lip-sync; v2 needs its own render")

    # ── mix and mux ────────────────────────────────────────────────────
    from ai_movie.composer import mix_for_state
    audio_out = out_dir / "final_audio.wav"
    mix_stats: dict = {}
    mix_for_state(state, segs, audio_out, stats=mix_stats)   # stereo bed if present
    shutil.copy2(str(audio_out), str(deliver / "03_final_audio.wav"))

    # Prefer the CodeFormer-enhanced render (same frames, sharper mouth).
    video = ((state.get("enhance") or {}).get("video")
             or (state.get("lipsync") or {}).get("video"))
    if not video or not Path(video).exists():
        log("no lip-sync video in state")
        return 1
    final = work / "output" / f"{args.out_name}_dubbed.mp4"
    final.parent.mkdir(parents=True, exist_ok=True)
    log(f"muxing onto {'v1 lip-sync' if reuse_lipsync else 'v1 lip-sync (DRIFTED)'}…")
    compose_video(Path(video), audio_out, final)
    shutil.copy2(str(final), str(deliver / "05_final_dubbed.mp4"))

    rows = [{
        "idx": i, "start": s.get("start"), "end": s.get("end"),
        "speaker": s.get("speaker"), "gender": s.get("gender"),
        "voice": "克隆(VC)" if s.get("vc") else "内置",
        "fit_ratio": s.get("fit_ratio"), "fit_end": s.get("fit_end"),
        "v1_fit_end": v1_ends.get(i),
        "drift_ms": (round((s["fit_end"] - v1_ends[i]) * 1000, 1)
                     if s.get("fit_end") is not None and v1_ends.get(i) is not None
                     else ""),
        "overrun_cut": s.get("overrun", 0),
        "text": (s.get("text_translated") or "")[:40],
    } for i, s in enumerate(segs)]
    artifacts.export_csv(rows, deliver / "03_tts_report.csv")

    state["vc"] = {"segments": segs, "refs": refs, "video": str(final),
                   "converted": converted, "reused_lipsync": reuse_lipsync,
                   "max_drift_ms": round(worst[0] * 1000, 1),
                   "mix": mix_stats}
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    log(f"done → {deliver / '05_final_dubbed.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
