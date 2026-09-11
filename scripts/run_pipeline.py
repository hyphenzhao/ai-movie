#!/usr/bin/env python
"""Headless pipeline runner — the whole dub without the GUI.

All orchestration used to live inside ``ai_movie/gui/app.py`` (4 700 lines of
Tk), with ``ai_movie/pipeline.py`` an empty stub.  That made every end-to-end
change impossible to verify unattended.  This script drives the same library
functions the GUI does, keeps its state in one JSON file so stages can be
re-run individually, and writes reviewable artifacts for every stage.

Usage
-----
    python scripts/run_pipeline.py VIDEO [--steps demux,separate,asr,...]
    python scripts/run_pipeline.py VIDEO --steps asr --force

Stages: demux, separate, asr, glossary, translate, tts, fit, mix, faces,
        lipsync, compose
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_movie import artifacts                      # noqa: E402
from ai_movie.config import WORKSPACE_DIR           # noqa: E402
from ai_movie.utils import ensure_dir               # noqa: E402

ALL_STEPS = ["demux", "separate", "asr", "glossary", "translate",
             "tts", "fit", "mix", "faces", "lipsync", "enhance", "compose"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Ctx:
    """Workspace paths plus the resumable state document."""

    def __init__(self, video: Path, name: str | None = None):
        self.video = Path(video).resolve()
        self.name = name or self.video.stem
        self.work = ensure_dir(WORKSPACE_DIR / self.name)
        self.deliver = ensure_dir(self.work / "deliverables")
        self.state_path = self.work / "state.json"
        self.state: dict = {}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.state_path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=1),
            encoding="utf-8")

    def has(self, step: str) -> bool:
        return bool(self.state.get(step))

    def put(self, step: str, data: dict) -> None:
        self.state[step] = data
        self.save()
        self.export_project()
        log(f"✓ {step}")

    def export_project(self) -> None:
        """Mirror the run into a GUI-loadable ``.aimovie.json``.

        state.json is this script's own format; the GUI reads ProjectLog.
        Writing both after every stage keeps a headless run visible in the
        GUI instead of leaving whatever the last GUI session saved.
        """
        try:
            from export_project import build          # same directory
            from ai_movie.config import PROJECTS_DIR
            st = dict(self.state)
            st.setdefault("_video", str(self.video))
            log_obj = build(st, self.name, str(self.video))
            log_obj.save(Path(PROJECTS_DIR) / f"{self.name}.aimovie.json")
        except Exception as exc:                        # noqa: BLE001
            print(f"[warn] could not update project file: "
                  f"{type(exc).__name__}: {exc}", flush=True)


# ── stages ─────────────────────────────────────────────────────────

def step_demux(ctx: Ctx) -> None:
    from ai_movie.demuxer import demux_video
    out = demux_video(ctx.video, ctx.work / "demuxed" / "original")
    ctx.put("demux", out)


def step_separate(ctx: Ctx) -> None:
    from ai_movie.composer import separate_vocals
    from ai_movie.config import VOCAL_SEPARATION_BACKEND
    res = separate_vocals(Path(ctx.state["demux"]["audio"]),
                          ctx.work / "separated",
                          backend=VOCAL_SEPARATION_BACKEND)
    ctx.put("separate", {k: str(v) for k, v in res.items()})


def step_asr(ctx: Ctx, args) -> None:
    from ai_movie import asr as asr_mod
    from ai_movie import diarize as diarize_mod

    audio = Path(ctx.state["demux"]["audio"])
    vocals = ctx.state.get("separate", {}).get("vocals")

    # Pass 1: transcribe and split on punctuation / pauses only.
    log("ASR: transcribing…")
    res = asr_mod.transcribe_all(
        [audio], language=args.language, backend=args.asr_backend,
        diarize=False,
        file_progress_cb=lambda i, p: log(f"  ASR {p}%") if p % 25 == 0 else None,
    )
    segments = res[0].get("segments", [])
    words = res[0].get("words", [])
    log(f"ASR: {len(segments)} segments, {len(words)} words")

    diar = None
    if not args.no_diarize:
        # Pass 2: diarize using those segments as units (they already end at
        # pauses, so a short question is its own unit).
        log("Diarization…")
        diar = diarize_mod.diarize_file(
            audio, vocals_path=vocals, segments=segments,
            num_speakers=args.num_speakers, progress_cb=log)

        if args.dialogue_refine:
            diar = diarize_mod.refine_speakers_with_dialogue(
                segments, diar, model=args.dialogue_model, progress_cb=log)

        # Pass 3: re-split, now cutting at speaker changes too.  Falls back to
        # the pass-1 segments as pseudo-words when word timestamps were not
        # available (an empty word list would otherwise erase the transcript).
        segments = asr_mod._finalize_segments(
            segments if not words else [], words,
            source=str(audio), diarization=diar)
        log(f"After speaker-aware re-split: {len(segments)} segments")

    (ctx.work / "asr_words.json").write_text(
        json.dumps(words, ensure_ascii=False), encoding="utf-8")

    artifacts.export_srt(segments, ctx.deliver / "01_asr.ja.srt",
                         speaker_prefix=bool(diar))
    artifacts.export_speaker_csv(segments, ctx.deliver / "01_speakers.csv")
    if diar:
        artifacts.export_json(diar, ctx.deliver / "01_diarization.json")
        artifacts.export_speaker_demo_wavs(
            segments, audio, ctx.deliver, prefix="01_spk")

    ctx.put("asr", {"segments": segments, "diarization": diar,
                    "language": args.language})


def step_glossary(ctx: Ctx, args) -> None:
    from ai_movie import glossary as gl
    segs = ctx.state["asr"]["segments"]
    terms = gl.build_glossary(segs, model=args.translate_helper,
                              progress_cb=log)
    artifacts.export_json(terms, ctx.deliver / "02_glossary.json")
    ctx.put("glossary", terms)


def step_translate(ctx: Ctx, args) -> None:
    from ai_movie import translator

    segs = [dict(s) for s in ctx.state["asr"]["segments"]]
    gloss = ctx.state.get("glossary") or {}
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]

    variants: dict[str, list[str]] = {}
    for eng in engines:
        log(f"Translating with engine '{eng}' ({len(segs)} segments)…")
        t0 = time.time()
        try:
            out = translator.translate_segments(
                segs, engine=eng, glossary=gloss,
                progress_cb=lambda d, t, e=eng: log(f"  {e}: {d}/{t}")
                if d % 20 == 0 else None,
            )
        except Exception as exc:                        # noqa: BLE001
            log(f"  engine {eng} FAILED: {type(exc).__name__}: {exc}")
            continue
        variants[eng] = out
        log(f"  {eng} done in {time.time() - t0:.0f}s")
        tagged = [{**s, "text_translated": t} for s, t in zip(segs, out)]
        artifacts.export_srt(tagged, ctx.deliver / f"02_zh_{eng}.srt",
                             text_key="text_translated", speaker_prefix=True)

    if not variants:
        raise RuntimeError("all translation engines failed")

    artifacts.export_translation_compare(
        segs, variants, ctx.deliver / "02_translation_compare.md")

    chosen = args.chosen_engine if args.chosen_engine in variants \
        else list(variants)[0]
    log(f"Using '{chosen}' for downstream stages")
    for s, t in zip(segs, variants[chosen]):
        s["text_translated"] = t
    ctx.put("translate", {"segments": segs, "variants": variants,
                          "chosen": chosen})


def step_tts(ctx: Ctx, args) -> None:
    from ai_movie import diarize as diarize_mod
    from ai_movie import tts as tts_mod

    segs = [dict(s) for s in ctx.state["translate"]["segments"]]
    diar = ctx.state["asr"].get("diarization") or {}
    audio = Path(ctx.state["demux"]["audio"])
    vocals = ctx.state.get("separate", {}).get("vocals")
    out_dir = ensure_dir(ctx.work / "synthesized")

    refs: dict[str, dict] = {}
    if args.voice_mode == "clone" and diar:
        log("Extracting per-speaker reference clips…")
        refs = diarize_mod.extract_speaker_references(
            diar, segs, audio, vocals_path=vocals, out_dir=out_dir)
        for spk, r in refs.items():
            log(f"  {spk} ({r.get('gender')}): {r['duration']}s "
                f"@{r['start']}s  voiced={r['voiced_ratio']}  “{r['text'][:24]}”")

        # Pick by measured clone quality rather than by heuristic score.
        if not args.no_ref_probe:
            refs = tts_mod.select_best_reference(
                refs, out_dir=out_dir, progress_cb=log)
            for spk, r in refs.items():
                log(f"  {spk} final ref: {r['duration']}s @{r['start']}s "
                    f"probe_sim={r.get('probe_similarity')}")
        artifacts.export_json(refs, ctx.deliver / "03_speaker_refs.json")

    seg_texts = [(i, (s.get("text_translated") or "").strip())
                 for i, s in enumerate(segs)]
    seg_refs, modes = tts_mod.build_seg_refs(
        segs, refs, force_sft=(args.voice_mode == "sft"))
    log(f"Voice modes: {modes}")

    items = tts_mod.run_cloned_synthesis(
        seg_texts, seg_refs, out_dir,
        progress_cb=lambda d, t: log(f"  TTS {d}/{t}") if d % 10 == 0 else None,
    )
    ok = 0
    for i, s in enumerate(segs):
        it = items.get(i, {})
        s["audio"] = it.get("audio")
        if it.get("tts_error"):
            s["tts_error"] = it["tts_error"]
        if s["audio"]:
            ok += 1
    log(f"TTS: {ok}/{len(segs)} segments synthesized")

    quality = {}
    if refs:
        log("Measuring clone similarity…")
        quality = tts_mod.verify_clone_quality(segs, refs)
        for spk, q in quality.items():
            log(f"  {spk}: similarity {q['similarity']} (n={q['n']}) "
                f"{'OK' if q['ok'] else 'BELOW THRESHOLD'}")

        # A clone that does not sound like the speaker is worse than a clean
        # built-in voice — re-synthesize those speakers with 中文男/中文女.
        bad = [spk for spk, q in quality.items() if not q["ok"]]
        if bad:
            log(f"Clone below threshold for {bad} — falling back to built-in voices")
            redo = [(i, (s.get("text_translated") or "").strip())
                    for i, s in enumerate(segs)
                    if s.get("speaker") in bad and (s.get("text_translated") or "").strip()]
            fb_refs = {}
            for i, _ in redo:
                g = segs[i].get("gender") or segs[i].get("tts_gender") or "female"
                fb_refs[i] = ((tts_mod._SFT_MALE_SPK if g == "male"
                               else tts_mod._SFT_FEMALE_SPK), None, "sft")
            items2 = tts_mod.run_cloned_synthesis(redo, fb_refs, out_dir)
            for i, _ in redo:
                it = items2.get(i, {})
                if it.get("audio"):
                    segs[i]["audio"] = it["audio"]
                    segs[i]["tts_fallback"] = True
            for spk in bad:
                refs.pop(spk, None)
            log(f"  re-synthesized {len(redo)} segments with built-in voices")

        artifacts.export_json(quality, ctx.deliver / "03_clone_quality.json")
        for spk, r in refs.items():
            same = [s for s in segs if s.get("speaker") == spk and s.get("audio")]
            if same:
                try:
                    artifacts.export_ab_wav(
                        r["ref_audio"], same[len(same) // 2]["audio"],
                        ctx.deliver / f"03_ab_{spk}.wav")
                except Exception:                       # noqa: BLE001
                    pass

    ctx.put("tts", {"segments": segs, "refs": refs, "quality": quality,
                    "ok": ok})


def step_fit(ctx: Ctx) -> None:
    from ai_movie.composer import fit_segments_to_timeline

    segs = [dict(s) for s in ctx.state["tts"]["segments"]]
    fit_segments_to_timeline(segs, out_dir=ctx.work / "synthesized" / "fitted")
    ratios = [s.get("fit_ratio", 1.0) for s in segs if s.get("audio_fit")]
    sped = [s for s in segs if s.get("fit_ratio", 1.0) > 1.0]
    cut = [s for s in segs if s.get("overrun")]
    log(f"Fitted {len(ratios)} segments; {len(sped)} sped up; "
        f"max total factor {max(ratios) if ratios else 1.0:.2f}; "
        f"{len(cut)} truncated at the next segment "
        f"(max {max((s['overrun'] for s in cut), default=0):.2f}s cut)")
    rows = [{
        "idx": i, "start": s.get("start"), "end": s.get("end"),
        "speaker": s.get("speaker"), "gender": s.get("gender"),
        "voice": "内置" if s.get("tts_fallback") else "克隆",
        "rate_factor": s.get("rate_factor"),
        "fit_ratio": s.get("fit_ratio"), "fit_end": s.get("fit_end"),
        "overrun_cut": s.get("overrun", 0),
        "text": (s.get("text_translated") or "")[:40],
    } for i, s in enumerate(segs)]
    artifacts.export_csv(rows, ctx.deliver / "03_tts_report.csv")
    ctx.put("fit", {"segments": segs})


def step_mix(ctx: Ctx) -> None:
    from ai_movie.composer import build_speech_track, mix_audio

    segs = ctx.state.get("fit", ctx.state["tts"])["segments"]
    bg = ctx.state.get("separate", {}).get("background")
    out = ctx.work / "synthesized" / "final_audio.wav"
    if bg and Path(bg).exists():
        mix_audio(segs, Path(bg), out)
    else:
        build_speech_track(segs, out)
    import shutil
    shutil.copy2(str(out), str(ctx.deliver / "03_final_audio.wav"))
    ctx.put("mix", {"audio": str(out)})


def step_faces(ctx: Ctx, args) -> None:
    from ai_movie import faces as faces_mod
    from ai_movie.translator import free_gpu_for_local_work

    free_gpu_for_local_work(log_cb=log)

    segs = ctx.state.get("fit", ctx.state.get("tts", ctx.state["asr"]))["segments"]
    plan = faces_mod.build_face_plan(
        ctx.video, segs,
        out_json=ctx.work / "face_plan.json",
        tracks_cache=ctx.work / "face_tracks_cache.json",
        progress_cb=log,
    )
    faces_mod.save_track_thumbnails(ctx.video, plan, ctx.deliver)
    rows = [{"id": t["id"], "gender": t.get("gender"), "conf": t.get("conf"),
             "votes": t.get("votes"), "first": t.get("first"),
             "last": t.get("last"), "keyframes": t.get("n"),
             "mean_area": round(t.get("mean_area", 0))}
            for t in plan["tracks"]]
    artifacts.export_csv(rows, ctx.deliver / "04_face_tracks.csv")
    artifacts.export_json(
        {"speaker_track": plan["speaker_track"], "tracks": rows,
         "anchored_frames": len(plan["frames"]), "n_frames": plan["n_frames"],
         "gate": plan.get("gate"), "segment_gated": plan.get("segment_gated")},
        ctx.deliver / "04_face_plan_summary.json")
    ctx.put("faces", {"plan_path": str(ctx.work / "face_plan.json"),
                      "speaker_track": plan["speaker_track"],
                      "tracks": rows})


def step_lipsync(ctx: Ctx, args) -> None:
    from ai_movie.lip_sync import segment_based_lip_sync
    from ai_movie.translator import free_gpu_for_local_work

    free_gpu_for_local_work(log_cb=log)

    segs = ctx.state.get("fit", ctx.state["tts"])["segments"]
    plan_path = ctx.state.get("faces", {}).get("plan_path")
    out = ctx.work / "lipsync.mp4"
    res = segment_based_lip_sync(
        ctx.video, segs, out,
        backend=args.lipsync_backend,
        occlusion_gate=True,
        face_plan=plan_path,
        progress_cb=lambda d, t: log(f"  lipsync {d}/{t}"),
    )
    if res is None:
        raise RuntimeError("lip-sync returned None (cancelled)")

    faces_state = ctx.state.get("faces") or {}
    bindings = faces_state.get("speaker_track") or {}
    seg_bind = {}
    seg_gated = {}
    try:
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        seg_bind = plan.get("segment_track") or {}
        seg_gated = plan.get("segment_gated") or {}
    except Exception:                                   # noqa: BLE001
        pass

    def _tid(i, s):
        return seg_bind.get(str(i), bindings.get(s.get("speaker") or ""))

    rows = [{
        "idx": i,
        "start": s.get("start"), "end": s.get("end"),
        "speaker": s.get("speaker"), "gender": s.get("gender"),
        "bound_track": _tid(i, s),
        "lipsynced": "是" if _tid(i, s) is not None
                     else "否（画面无此人，直通原视频）",
        "gated_frames": seg_gated.get(str(i), 0),
        "text": (s.get("text_translated") or "")[:36],
    } for i, s in enumerate(segs)]
    artifacts.export_csv(rows, ctx.deliver / "04_lipsync_report.csv")
    import shutil
    shutil.copy2(str(out), str(ctx.deliver / "04_lipsync.mp4"))
    ctx.put("lipsync", {"video": str(out)})


def step_enhance(ctx: Ctx, args) -> None:
    """CodeFormer pass over the lip-synced video (only the regenerated lower
    face is touched; frames with no confident face pass through).

    Measured on test_2: MuseTalk's mouth keeps ~50% of the source's sharpness
    even on frontal faces; CodeFormer with the lips *unprotected* brings it
    back to ~parity (44→82, 64→80) without changing the generated shape.
    """
    from ai_movie.config import FACE_ENHANCE_FIDELITY, FACE_ENHANCE_PROTECT_LIPS
    from ai_movie.face_restore import (codeformer_available, frame_filter_from_plan,
                                       restore_video)
    from ai_movie.translator import free_gpu_for_local_work
    import shutil

    src = ctx.state.get("lipsync", {}).get("video")
    plan_path = ctx.state.get("faces", {}).get("plan_path")
    if not src or not Path(src).exists():
        raise RuntimeError("enhance needs the lipsync step's video")
    if not codeformer_available():
        log("CodeFormer weights missing — enhance skipped")
        ctx.put("enhance", {"video": None, "skipped": True})
        return
    free_gpu_for_local_work(log_cb=log)
    out = ctx.work / "lipsync_enhanced.mp4"
    res = restore_video(
        Path(src), out,
        fidelity_weight=args.enhance_fidelity if args.enhance_fidelity is not None
        else FACE_ENHANCE_FIDELITY,
        protect_lips=FACE_ENHANCE_PROTECT_LIPS if args.enhance_protect_lips is None
        else bool(args.enhance_protect_lips),
        frame_filter=frame_filter_from_plan(plan_path),
        progress_cb=lambda d, t: log(f"  enhance {d}/{t}") if d % 300 == 0 or d == t else None,
        log_cb=log,
    )
    if res is None:
        raise RuntimeError("face enhance returned None (cancelled)")
    shutil.copy2(str(out), str(ctx.deliver / "04_lipsync_enhanced.mp4"))
    ctx.put("enhance", {"video": str(out)})


def step_compose(ctx: Ctx) -> None:
    from ai_movie.composer import compose_video
    import shutil

    video = (ctx.state.get("enhance", {}).get("video")
             or ctx.state.get("lipsync", {}).get("video") or str(ctx.video))
    audio = ctx.state["mix"]["audio"]
    out = ensure_dir(ctx.work / "output") / f"{ctx.name}_dubbed.mp4"
    compose_video(Path(video), Path(audio), out)
    shutil.copy2(str(out), str(ctx.deliver / "05_final_dubbed.mp4"))
    ctx.put("compose", {"video": str(out)})


# ── driver ─────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--name", default=None)
    ap.add_argument("--steps", default=",".join(ALL_STEPS))
    ap.add_argument("--force", action="store_true",
                    help="re-run steps even if state already has them")
    ap.add_argument("--language", default="ja")
    ap.add_argument("--asr-backend", default="openai-whisper")
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--no-diarize", action="store_true")
    # OFF by default: the only local model strong enough for this task
    # (gpt-oss-120b) loads onto the GPU but its server never becomes
    # available on this ROCm build, so enabling it hangs the run.
    ap.add_argument("--dialogue-refine", action="store_true",
                    help="refine speaker attribution with an LLM over the transcript")
    ap.add_argument("--dialogue-model", default=None)
    ap.add_argument("--translate-helper", default=None)
    # Default to sakura only: the gpt-oss polish stage cannot complete on
    # this ROCm build (see Documentation/v2-quality-upgrade.md).  Pass a
    # comma-separated list to compare engines in one run.
    ap.add_argument("--engines", default="sakura")
    ap.add_argument("--chosen-engine", default=None)
    ap.add_argument("--no-ref-probe", action="store_true",
                    help="skip probing candidate reference clips (faster, "
                         "picks by heuristic score only)")
    ap.add_argument("--voice-mode", default="clone",
                    choices=["clone", "sft", "gender", "female", "male"],
                    help="'sft' forces every segment onto a built-in speaker "
                         "(中文女/中文男) — no reference audio anywhere, so the "
                         "Japanese source cannot leak into the output")
    ap.add_argument("--lipsync-backend", default="musetalk")
    ap.add_argument("--enhance-fidelity", type=float, default=None,
                    help="CodeFormer fidelity w (default: config FACE_ENHANCE_FIDELITY)")
    ap.add_argument("--enhance-protect-lips", type=int, default=None, choices=[0, 1],
                    help="1 = leave the generated lips untouched (default: config)")
    args = ap.parse_args()

    ctx = Ctx(Path(args.video), args.name)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    log(f"Workspace: {ctx.work}")

    dispatch = {
        "demux": lambda: step_demux(ctx),
        "separate": lambda: step_separate(ctx),
        "asr": lambda: step_asr(ctx, args),
        "glossary": lambda: step_glossary(ctx, args),
        "translate": lambda: step_translate(ctx, args),
        "tts": lambda: step_tts(ctx, args),
        "fit": lambda: step_fit(ctx),
        "mix": lambda: step_mix(ctx),
        "faces": lambda: step_faces(ctx, args),
        "lipsync": lambda: step_lipsync(ctx, args),
        "enhance": lambda: step_enhance(ctx, args),
        "compose": lambda: step_compose(ctx),
    }

    for s in steps:
        if s not in dispatch:
            log(f"unknown step: {s}")
            return 2
        if ctx.has(s) and not args.force:
            log(f"· {s} (cached)")
            continue
        log(f"▶ {s}")
        t0 = time.time()
        dispatch[s]()
        log(f"  {s} took {time.time() - t0:.0f}s")

    log(f"Deliverables: {ctx.deliver}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
