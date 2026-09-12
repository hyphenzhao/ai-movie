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

Stages: demux, separate, osd, asr, glossary, translate, tts, compact, fit,
        mix, faces, lipsync, enhance, compose, qc
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

ALL_STEPS = ["demux", "separate", "osd", "asr", "glossary", "translate",
             "tts", "compact", "fit", "mix", "faces", "lipsync", "enhance",
             "compose", "qc"]


def _tts_segments(ctx: "Ctx") -> list[dict]:
    """Synthesized segments: the compact stage's if it ran, else tts's."""
    st = ctx.state
    if st.get("compact") and st["compact"].get("segments"):
        return st["compact"]["segments"]
    return st["tts"]["segments"]


def _timeline_segments(ctx: "Ctx") -> list[dict]:
    """Segments with placement info: fit → compact → tts → asr."""
    st = ctx.state
    for key in ("fit", "compact", "tts", "asr"):
        if st.get(key) and st[key].get("segments"):
            return st[key]["segments"]
    raise KeyError("no segments in state")

# ── stage fingerprints ──────────────────────────────────────────────
#
# ``state.json`` used to cache a stage by its *name* only: once "fit" had
# run, changing TTS_FIT_MAX_SPEEDUP, editing fit_segments_to_timeline or
# re-running "tts" left a stale "fit" that the runner happily reused.  Each
# stage now records a fingerprint of everything its output depends on —
# the input video, the config constants it reads, the source of the
# functions it calls and the fingerprints of the stages it consumes — and a
# stage whose fingerprint no longer matches is re-run automatically.
#
# States written before this existed carry no ``_fp`` entry and are treated
# as valid ("legacy"); ``--fp-adopt`` stamps them so later changes are
# detected.

STEP_DEPS: dict[str, list[str]] = {
    "demux": [],
    "separate": ["demux"],
    "osd": ["separate"],
    "asr": ["demux", "separate", "osd"],
    "glossary": ["asr"],
    "translate": ["asr", "glossary"],
    "tts": ["translate"],
    "compact": ["tts"],
    "fit": ["compact", "tts"],
    "mix": ["fit", "separate"],
    "faces": ["fit"],
    "lipsync": ["fit", "faces"],
    "enhance": ["lipsync", "faces"],
    "compose": ["enhance", "mix"],
    "qc": ["compose"],
}

# Config constants each stage reads (missing names are simply skipped, so a
# stage may list constants that a later version of config.py introduces).
STEP_CONFIG: dict[str, list[str]] = {
    "demux": [],
    "separate": ["VOCAL_SEPARATION_BACKEND", "UVR_MODEL_NAME", "DEMUCS_MODEL",
                 "SEPARATE_FULL_RATE_BED", "SEPARATE_ANALYSIS_FROM_FULL"],
    "osd": ["OSD_ENABLED", "OSD_MODEL"],
    "asr": ["ASR_MODEL_SIZE", "ASR_OPENAI_WHISPER_MODEL", "ASR_WORD_TIMESTAMPS",
            "ASR_VAD_THRESHOLD", "ASR_MAX_SEGMENT_DURATION", "ASR_MAX_SEGMENT_CHARS",
            "ASR_PAUSE_SPLIT_SEC", "ASR_MIN_SEGMENT_DURATION", "DIARIZE_GENDER_HZ",
            "DIARIZE_AHC_THRESHOLD", "OSD_SEED_EXCLUDE"],
    "glossary": ["GLOSSARY_AUTO_EXTRACT", "GLOSSARY_MAX_TERMS", "GLOSSARY_MIN_COUNT"],
    "translate": ["TRANSLATION_CTX_BEFORE", "TRANSLATION_CTX_AFTER",
                  "OLLAMA_SAKURA_MODEL", "GLOSSARY_ENFORCE_MODEL"],
    "tts": ["TTS_PREFERRED_MODEL", "TTS_REF_MIN_DURATION", "TTS_REF_MAX_DURATION",
            "TTS_CLONE_MIN_SIMILARITY", "TTS_F0_GATE", "TTS_F0_RATIO_RANGE",
            "OSD_REF_MAX_OVERLAP"],
    "compact": ["COMPACT_ENABLED", "COMPACT_TRIGGER_RATIO", "COMPACT_TARGET_RATIO",
                "COMPACT_MAX_ROUNDS", "COMPACT_MODEL", "COMPACT_MIN_CHARS",
                "TTS_COMPACT_SEC_PER_CHAR_DEFAULT"],
    "fit": ["TTS_FIT_MAX_SPEEDUP", "TTS_FIT_MIN_SPEEDUP", "TTS_FIT_MAX_SPEEDUP_HARD",
            "TTS_FIT_MAX_TRUNCATE", "TTS_FIT_TAIL_TOLERANCE", "TTS_FIT_MAX_TAIL",
            "TTS_FIT_MIN_GAP", "TTS_FIT_BACKEND", "TTS_NATURAL_SEC_PER_CHAR",
            "TTS_RATE_MIN_CORRECTION", "TTS_RATE_MAX_CORRECTION"],
    "mix": ["MIX_DUCK_DB", "MIX_DUCK_ATTACK_MS", "MIX_DUCK_RELEASE_MS",
            "MIX_MATCH_LOUDNESS", "MIX_MATCH_CLAMP_DB", "MIX_TARGET_LUFS",
            "MIX_TRUE_PEAK_DB"],
    "faces": ["FACE_DET_EVERY", "FACE_DET_MAX_WIDTH", "FACE_DET_CONF", "FACE_TRACK_IOU",
              "FACE_TRACK_MIN_FRAMES", "FACE_TRACK_MAX_GAP", "FACE_GENDER_SAMPLES",
              "FACE_GENDER_MIN_CONF", "FACE_BIND_MIN_SCORE", "FACE_YAW_MAX",
              "FACE_MIN_WIDTH", "FACE_MIN_WIDTH_SR", "FACE_GATE_SMOOTH",
              "SHOT_DETECT", "SHOT_SCDET_THRESHOLD"],
    "lipsync": ["MUSETALK_BOX_SMOOTH", "MUSETALK_SHARPEN", "MUSETALK_FUSION",
                "LIPSYNC_AUDIO_OFFSET_MS", "LIPSYNC_SMALL_FACE_UPSCALE",
                "LIPSYNC_SR_MIN_FRAC", "OCCLUSION_MODE", "OCCLUSION_FULL_LIP_THRESH"],
    "enhance": ["FACE_ENHANCE_FIDELITY", "FACE_ENHANCE_PROTECT_LIPS"],
    "compose": [],
    "qc": ["QC_ASR_CONF_WARN", "QC_SPEAKER_CONF_WARN", "QC_FIT_WARN", "QC_FIT_FAIL",
           "QC_OVERRUN_WARN", "QC_OVERRUN_FAIL", "QC_GATED_FRAC_WARN",
           "QC_OVERLAP_WARN", "QC_OVERLAP_FAIL"],
}

# Functions whose source text each stage's output depends on.  Hashing at
# function level (not module level) means editing mix_audio does not
# invalidate "fit".  Names that do not resolve hash as "missing" so a stage
# may list functions that appear in a later version.
STEP_CODE: dict[str, list[str]] = {
    "demux": ["ai_movie.demuxer.demux_video"],
    "separate": ["ai_movie.composer.separate_vocals", "ai_movie.composer._separate_uvr",
                 "ai_movie.composer._separate_demucs",
                 "ai_movie.composer.separate_for_pipeline"],
    "osd": ["ai_movie.osd.run_osd"],
    "asr": ["ai_movie.asr.transcribe_all", "ai_movie.asr._finalize_segments",
            "ai_movie.segmenter.split_into_sentences", "ai_movie.diarize.diarize_file",
            "ai_movie.diarize._refine_with_channel", "ai_movie.diarize.split_by_pitch"],
    "glossary": ["ai_movie.glossary.build_glossary"],
    "translate": ["ai_movie.translator.translate_segments",
                  "ai_movie.translator.enforce_glossary",
                  "ai_movie.translator._sakura_translate"],
    "tts": ["ai_movie.tts.run_cloned_synthesis", "ai_movie.tts.build_seg_refs",
            "ai_movie.diarize.extract_speaker_references",
            "ai_movie.tts.verify_clone_quality"],
    "compact": ["ai_movie.translator.compact_translation",
                "ai_movie.composer.segment_slots",
                "ai_movie.composer.speaker_sec_per_char"],
    "fit": ["ai_movie.composer.fit_segments_to_timeline",
            "ai_movie.composer.fit_audio_to_slot",
            "ai_movie.composer.estimate_rate_correction",
            "ai_movie.composer.segment_slots"],
    "mix": ["ai_movie.composer.mix_audio", "ai_movie.composer.build_speech_track",
            "ai_movie.composer.loudnorm_two_pass"],
    "faces": ["ai_movie.faces.build_face_plan", "ai_movie.faces.detect_face_tracks",
              "ai_movie.faces.gate_frames", "ai_movie.faces.bind_speakers_to_tracks",
              "ai_movie.faces.bind_segments_to_tracks", "ai_movie.faces.interpolate_track",
              "ai_movie.shots.detect_cuts"],
    "lipsync": ["ai_movie.lip_sync.segment_based_lip_sync",
                "ai_movie.lip_sync.musetalk_sync_batch",
                "ai_movie.lip_sync._cut_audio_clip",
                "ai_movie.face_restore.occlusion_gate_video"],
    "enhance": ["ai_movie.face_restore.restore_video"],
    "compose": ["ai_movie.composer.compose_video"],
    "qc": ["ai_movie.qc.build_qc"],
}

# Files (relative to ROOT) whose bytes a stage depends on.
STEP_FILES: dict[str, list[str]] = {
    "lipsync": ["patches/musetalk_rotation_align.patch",
                "patches/musetalk_target_face.patch",
                "patches/musetalk_quality.patch",
                "patches/musetalk_fusion.patch"],
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _sha1(data: bytes) -> str:
    import hashlib
    return hashlib.sha1(data).hexdigest()


def input_fingerprint(video: Path) -> str:
    """Cheap identity for a large input file: size, mtime, first/last 4 MiB."""
    p = Path(video)
    try:
        st = p.stat()
    except OSError:
        return "missing"
    chunk = 4 * 1024 * 1024
    with open(p, "rb") as fh:
        head = fh.read(chunk)
        if st.st_size > chunk:
            fh.seek(max(0, st.st_size - chunk))
            tail = fh.read(chunk)
        else:
            tail = b""
    return _sha1(f"{st.st_size}|{st.st_mtime_ns}|".encode()
                 + _sha1(head).encode() + _sha1(tail).encode())


def _source_hash(dotted: str) -> str:
    """sha1 of a function's source text, or "missing" if it cannot be found."""
    import importlib
    import inspect
    mod_name, _, attr = dotted.rpartition(".")
    try:
        mod = importlib.import_module(mod_name)
        obj = getattr(mod, attr)
        return _sha1(inspect.getsource(obj).encode("utf-8"))
    except Exception:                                   # noqa: BLE001
        return "missing"


def _args_extra(step: str, args) -> dict:
    """CLI choices that change a stage's output."""
    if args is None:
        return {}
    pick = {
        "asr": ["language", "asr_backend", "num_speakers", "no_diarize",
                "dialogue_refine", "dialogue_model"],
        "glossary": ["translate_helper"],
        "translate": ["engines", "chosen_engine"],
        "tts": ["voice_mode", "no_ref_probe"],
        "compact": ["no_compact", "voice_mode"],
        "lipsync": ["lipsync_backend", "lipsync_audio_offset_ms", "occlusion_mode"],
        "enhance": ["enhance_fidelity", "enhance_protect_lips"],
    }
    return {k: getattr(args, k, None) for k in pick.get(step, [])}


def step_fingerprint(ctx: "Ctx", step: str, args=None) -> dict:
    """Everything ``step``'s output depends on, plus a combined hash."""
    from ai_movie import config as _cfg

    cfg = {}
    for name in STEP_CONFIG.get(step, []):
        if hasattr(_cfg, name):
            cfg[name] = repr(getattr(_cfg, name))
    code = {d: _source_hash(d) for d in STEP_CODE.get(step, [])}
    files = {}
    for rel in STEP_FILES.get(step, []):
        p = ROOT / rel
        files[rel] = _sha1(p.read_bytes()) if p.exists() else "missing"
    fps = ctx.state.get("_fp") or {}
    up = {}
    for dep in STEP_DEPS.get(step, []):
        if dep not in ctx.state:
            continue                    # optional upstream absent
        up[dep] = (fps.get(dep) or {}).get("hash", "legacy")
    extra = _args_extra(step, args)
    body = {"v": 1, "input": ctx.input_fp, "cfg": cfg, "code": code,
            "files": files, "up": up, "extra": extra}
    body["hash"] = _sha1(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                    default=str).encode("utf-8"))
    return body


def _fp_diff(old: dict, new: dict) -> list[str]:
    """Human-readable list of what changed between two fingerprints."""
    out = []
    if old.get("input") != new.get("input"):
        out.append("input video")
    for part in ("cfg", "code", "files", "up", "extra"):
        a, b = old.get(part) or {}, new.get(part) or {}
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                out.append(f"{part}:{k}")
    return out


class Ctx:
    """Workspace paths plus the resumable state document."""

    def __init__(self, video: Path, name: str | None = None, *,
                 args=None, no_fp: bool = False):
        self.video = Path(video).resolve()
        self.name = name or self.video.stem
        self.work = ensure_dir(WORKSPACE_DIR / self.name)
        self.deliver = ensure_dir(self.work / "deliverables")
        self.state_path = self.work / "state.json"
        self.args = args
        self.no_fp = no_fp
        self.state: dict = {}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        # Always record the real source so downstream scripts (demo clips,
        # deliver) never chase a path that was moved after the run.
        if self.video.exists():
            self.state["_video"] = str(self.video)
        self.input_fp = input_fingerprint(self.video)

    def save(self) -> None:
        """Atomic write: a crash mid-write must never leave a truncated state."""
        import os
        tmp = self.state_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(self.state, ensure_ascii=False, indent=1))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.state_path)

    def fingerprint(self, step: str) -> dict:
        return step_fingerprint(self, step, self.args)

    def has(self, step: str) -> bool:
        """Cached *and* still valid for the current inputs, config and code."""
        if not self.state.get(step):
            return False
        if self.no_fp:
            return True
        old = (self.state.get("_fp") or {}).get(step)
        if old is None:
            return True                 # legacy state: trust it
        new = self.fingerprint(step)
        if old.get("hash") == new["hash"]:
            return True
        why = ", ".join(_fp_diff(old, new)[:6]) or "hash"
        log(f"· {step} STALE ({why}) → re-running")
        return False

    def stamp(self, step: str) -> None:
        self.state.setdefault("_fp", {})[step] = self.fingerprint(step)

    def put(self, step: str, data: dict) -> None:
        self.state[step] = data
        self.stamp(step)
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
    from ai_movie.composer import separate_for_pipeline
    from ai_movie.config import SEPARATE_FULL_RATE_BED, VOCAL_SEPARATION_BACKEND
    from ai_movie.demuxer import extract_full_audio

    dm = ctx.state["demux"]
    full = dm.get("audio_full")
    if SEPARATE_FULL_RATE_BED and (not full or not Path(full).exists()):
        # Workspace demuxed before v3: add the full-rate track in place.
        dst = Path(dm["audio"]).parent / "audio_full.wav"
        dm.update(extract_full_audio(ctx.video, dst))
        full = dm["audio_full"]
        log(f"  extracted full-rate audio: {dm.get('sample_rate')} Hz "
            f"× {dm.get('channels')} ch")
    res = separate_for_pipeline(
        Path(dm["audio"]), Path(full) if (SEPARATE_FULL_RATE_BED and full) else None,
        ctx.work / "separated", backend=VOCAL_SEPARATION_BACKEND)
    ctx.put("separate", {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in res.items()})


def step_osd(ctx: Ctx, args) -> None:
    """Overlapped-speech regions on the separated vocals (CPU, isolated venv)."""
    from ai_movie import osd as osd_mod
    from ai_movie.config import OSD_ENABLED

    if not OSD_ENABLED or getattr(args, "no_osd", False):
        ctx.put("osd", {"available": False, "regions": [], "reason": "disabled"})
        return
    src = ctx.state.get("separate", {}).get("vocals") or ctx.state["demux"]["audio"]
    doc = osd_mod.run_osd(src, ctx.work / "osd.json", log_cb=log)
    if doc.get("available"):
        log(f"  OSD: {len(doc['regions'])} overlap regions, "
            f"{doc['total_overlap_s']:.1f}s total")
        artifacts.export_json(doc, ctx.deliver / "01_overlap.json")
    else:
        log(f"  OSD unavailable: {doc.get('reason')}")
    ctx.put("osd", doc)


def step_asr(ctx: Ctx, args) -> None:
    from ai_movie import asr as asr_mod
    from ai_movie import diarize as diarize_mod

    audio = Path(ctx.state["demux"]["audio"])
    vocals = ctx.state.get("separate", {}).get("vocals")
    overlap = (ctx.state.get("osd") or {}).get("regions") or []

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
            num_speakers=args.num_speakers, progress_cb=log,
            overlap_regions=overlap)

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


def _synthesize(ctx: Ctx, args, segs: list[dict], idxs: list[int],
                refs: dict[str, dict], out_dir: Path) -> int:
    """Synthesize ``segs[i]`` for every i in *idxs* with the run's voice routing.

    Shared by the tts stage (all lines) and the compact stage (only the
    rewritten lines).  Speakers whose clone fell back to a built-in voice
    (``tts_fallback``) stay on the built-in voice.  Sets ``audio`` /
    ``tts_error`` on the segments; returns the number synthesized.
    """
    from ai_movie import tts as tts_mod

    seg_texts = [(i, (segs[i].get("text_translated") or "").strip())
                 for i in idxs]
    seg_refs, modes = tts_mod.build_seg_refs(
        segs, refs, force_sft=(args.voice_mode == "sft"))
    for i in idxs:
        s = segs[i]
        if s.get("tts_fallback") or (s.get("speaker") and s["speaker"] not in refs
                                     and args.voice_mode == "clone"):
            g = s.get("gender") or s.get("tts_gender") or "female"
            seg_refs[i] = ((tts_mod._SFT_MALE_SPK if g == "male"
                            else tts_mod._SFT_FEMALE_SPK), None, "sft")
    seg_refs = {i: r for i, r in seg_refs.items() if i in set(idxs)}
    if len(idxs) == len(segs):
        log(f"Voice modes: {modes}")

    items = tts_mod.run_cloned_synthesis(
        seg_texts, seg_refs, out_dir,
        progress_cb=lambda d, t: log(f"  TTS {d}/{t}") if d % 10 == 0 else None,
    )
    ok = 0
    for i in idxs:
        s = segs[i]
        it = items.get(i, {})
        s["audio"] = it.get("audio")
        s.pop("tts_error", None)
        if it.get("tts_error"):
            s["tts_error"] = it["tts_error"]
        if s["audio"]:
            _trim_silence_inplace(Path(s["audio"]))
            ok += 1
    return ok


def _trim_silence_inplace(path: Path, thr_db: float = -45.0,
                          pad_ms: float = 40.0) -> float:
    """Cut leading/trailing silence off a synthesized line (keeps *pad_ms*).

    Measured on output_test: some built-in-voice lines carried 0.3 s of
    silence at each end inside a 1.4 s file — a third of a slot spent on
    nothing, then "fixed" by speeding the words up.  Returns seconds removed.
    """
    import numpy as np
    import soundfile as sf
    try:
        x, sr = sf.read(str(path), dtype="float32")
    except Exception:                                   # noqa: BLE001
        return 0.0
    mono = x if x.ndim == 1 else x.mean(axis=1)
    thr = 10 ** (thr_db / 20)
    idx = np.where(np.abs(mono) > thr)[0]
    if len(idx) == 0:
        return 0.0
    pad = int(sr * pad_ms / 1000)
    a = max(0, int(idx[0]) - pad)
    b = min(len(mono), int(idx[-1]) + pad)
    removed = (len(mono) - (b - a)) / sr
    if removed < 0.02:
        return 0.0
    sf.write(str(path), x[a:b], sr)
    return removed


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
            diar, segs, audio, vocals_path=vocals, out_dir=out_dir,
            overlap_regions=(ctx.state.get("osd") or {}).get("regions") or [])
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

    ok = _synthesize(ctx, args, segs, list(range(len(segs))), refs, out_dir)
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


def step_compact(ctx: Ctx, args) -> None:
    """Rewrite lines whose synthesized duration cannot fit their slot.

    Measured after TTS (not predicted): natural duration / slot above
    COMPACT_TRIGGER_RATIO → ask the instruct model for a shorter line with a
    character budget derived from the speaker's measured s/char, re-synth
    just those lines, re-measure; at most COMPACT_MAX_ROUNDS.  Rejected
    rewrites keep the full line — fit's speed-up and truncation remain the
    backstop.  ``text_translated_full`` always keeps the original wording.
    """
    import soundfile as sf
    from ai_movie import translator
    from ai_movie.composer import _visible_chars, segment_slots, speaker_sec_per_char
    from ai_movie.config import (
        COMPACT_ENABLED, COMPACT_MAX_ROUNDS, COMPACT_MIN_CHARS, COMPACT_MODEL,
        COMPACT_TARGET_RATIO, COMPACT_TRIGGER_RATIO,
    )

    segs = [dict(s) for s in ctx.state["tts"]["segments"]]
    refs = ctx.state["tts"].get("refs") or {}
    gloss = ctx.state.get("glossary") or {}
    if not COMPACT_ENABLED or getattr(args, "no_compact", False):
        log("compact: disabled")
        ctx.put("compact", {"segments": segs, "skipped": True,
                            "attempted": 0, "rewritten": 0, "rounds": 0})
        return

    slots = segment_slots(segs)
    out_dir = ensure_dir(ctx.work / "synthesized" / "compact")

    def _dur(s):
        p = s.get("audio")
        try:
            return float(sf.info(str(p)).duration) if p and Path(p).exists() else None
        except Exception:                               # noqa: BLE001
            return None

    report: dict[int, dict] = {}
    attempted = rewritten = 0
    rounds = 0
    for rnd in range(1, COMPACT_MAX_ROUNDS + 1):
        spc = speaker_sec_per_char(segs)
        todo = []
        for i, s in enumerate(segs):
            d = _dur(s)
            if d is None or slots[i] <= 0:
                continue
            ratio = d / slots[i]
            if ratio > COMPACT_TRIGGER_RATIO:
                todo.append((i, d, ratio))
        if not todo:
            break
        rounds = rnd
        log(f"compact round {rnd}: {len(todo)} lines exceed "
            f"{COMPACT_TRIGGER_RATIO}× their slot")
        changed: list[int] = []
        with translator.exclusive_engine("ollama", ollama_model=COMPACT_MODEL,
                                         log_cb=log):
            for i, d, ratio in todo:
                s = segs[i]
                zh = s.get("text_translated") or ""
                s.setdefault("text_translated_full", zh)
                rate = spc.get(s.get("speaker") or "", 0.24)
                # Shrink proportionally: budget so the rewrite lands near
                # COMPACT_TARGET_RATIO × slot at this speaker's measured rate.
                budget = int(slots[i] * COMPACT_TARGET_RATIO / max(rate, 0.05))
                if rnd > 1:
                    budget = int(budget * 0.8)
                budget = max(COMPACT_MIN_CHARS, budget)
                n_before = _visible_chars(zh)
                rec = report.setdefault(i, {
                    "idx": i, "speaker": s.get("speaker"), "slot": round(slots[i], 2),
                    "ratio_before": round(ratio, 2), "chars_before": n_before,
                    "text_full": s.get("text_translated_full"), "status": "kept",
                    "notes": []})
                rec.update({"budget": budget, "round": rnd})
                if n_before <= budget:
                    rec["notes"].append(f"r{rnd}:within_budget")
                    continue
                attempted += 1
                ctx_lines = [(segs[j].get("text_translated") or "")
                             for j in range(max(0, i - 2), i)]
                cand = translator.compact_translation(
                    s.get("text") or "", zh, budget, glossary=gloss,
                    context=ctx_lines)
                if cand:
                    rec[f"_prev_{rnd}"] = (zh, s.get("audio"), d)
                    s["text_translated"] = cand
                    changed.append(i)
                    log(f"  #{i} {n_before}→{_visible_chars(cand)} 字 "
                        f"(ratio {ratio:.2f}, budget {budget}): {cand}")
                else:
                    rec["notes"].append(f"r{rnd}:rewrite_rejected")
        if not changed:
            break
        log(f"  re-synthesizing {len(changed)} lines…")
        # Re-synthesize into a per-round directory so a reverted line's
        # previous audio is still on disk.
        _synthesize(ctx, args, segs, changed, refs, ensure_dir(out_dir / f"r{rnd}"))
        for i in changed:
            rec = report[i]
            prev_zh, prev_audio, prev_d = rec.pop(f"_prev_{rnd}")
            d2 = _dur(segs[i])
            # The built-in voice's tempo varies ±40 % between takes, so a
            # shorter text is not always shorter audio.  Keep whichever
            # (text, audio) pair is actually shorter; never ship the longer.
            if d2 is None or (prev_d is not None and d2 >= prev_d * 0.97):
                segs[i]["text_translated"] = prev_zh
                segs[i]["audio"] = prev_audio
                rec["notes"].append(
                    f"r{rnd}:no_gain_reverted({prev_d:.2f}s→{d2 if d2 is None else round(d2, 2)}s)")
                continue
            rec.update({"text_compact": segs[i]["text_translated"],
                        "chars_after": _visible_chars(segs[i]["text_translated"]),
                        "status": "rewritten",
                        "ratio_after": round(d2 / slots[i], 2)})
            rec["notes"].append(f"r{rnd}:accepted({prev_d:.2f}s→{d2:.2f}s)")

    rewritten = sum(1 for r in report.values() if r["status"] == "rewritten")
    for i, rec in report.items():
        rec.setdefault("ratio_after", rec["ratio_before"])
        rec["start"] = segs[i].get("start")
        rec["notes"] = ";".join(rec["notes"])
        if rec["status"] != "rewritten":
            segs[i]["text_translated"] = rec["text_full"]
    rows = sorted(report.values(), key=lambda r: r["idx"])
    artifacts.export_csv(rows, ctx.deliver / "03_compact_report.csv")
    artifacts.export_srt(segs, ctx.deliver / "03_zh_compact.srt",
                         text_key="text_translated", speaker_prefix=True)
    still = sum(1 for r in rows if r["ratio_after"] > COMPACT_TRIGGER_RATIO)
    log(f"compact: {rewritten}/{attempted} rewrites accepted in {rounds} round(s); "
        f"{still} line(s) still above {COMPACT_TRIGGER_RATIO}× (fit will speed/cut)")
    ctx.put("compact", {"segments": segs, "attempted": attempted,
                        "rewritten": rewritten, "rounds": rounds,
                        "still_over": still, "report": rows})


def step_fit(ctx: Ctx) -> None:
    from ai_movie.composer import fit_segments_to_timeline

    segs = [dict(s) for s in _tts_segments(ctx)]
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
        "text_full": (s.get("text_translated_full") or "")[:40],
    } for i, s in enumerate(segs)]
    artifacts.export_csv(rows, ctx.deliver / "03_tts_report.csv")
    ctx.put("fit", {"segments": segs})


def step_mix(ctx: Ctx) -> None:
    from ai_movie.composer import mix_for_state

    segs = _timeline_segments(ctx)
    out = ctx.work / "synthesized" / "final_audio.wav"
    stats: dict = {}
    mix_for_state(ctx.state, segs, out, stats=stats)
    import shutil
    shutil.copy2(str(out), str(ctx.deliver / "03_final_audio.wav"))
    if stats:
        g = [abs(v) for v in stats.get("gain_db", {}).values()]
        log(f"  mix: {stats.get('sr')} Hz × {stats.get('channels')} ch, "
            f"loudness match={'on' if stats.get('match_loudness') else 'off'}, "
            f"|gain| median {float(__import__('numpy').median(g)) if g else 0:.1f} dB")
    ctx.put("mix", {"audio": str(out), **stats})


def step_faces(ctx: Ctx, args) -> None:
    from ai_movie import faces as faces_mod
    from ai_movie.translator import free_gpu_for_local_work

    free_gpu_for_local_work(log_cb=log)

    segs = _timeline_segments(ctx)
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

    segs = _timeline_segments(ctx)
    plan_path = ctx.state.get("faces", {}).get("plan_path")
    out = ctx.work / "lipsync.mp4"
    stats: dict = {}
    res = segment_based_lip_sync(
        ctx.video, segs, out,
        backend=args.lipsync_backend,
        occlusion_gate=True,
        face_plan=plan_path,
        audio_offset_ms=args.lipsync_audio_offset_ms,
        fusion=args.fusion,
        occlusion_mode=args.occlusion_mode,
        stats=stats,
        progress_cb=lambda d, t: log(f"  lipsync {d}/{t}"),
    )
    if res is None:
        raise RuntimeError("lip-sync returned None (cancelled)")

    faces_state = ctx.state.get("faces") or {}
    bindings = faces_state.get("speaker_track") or {}
    seg_bind = {}
    seg_gated = {}
    seg_sr = {}
    seg_cuts = {}
    try:
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        seg_bind = plan.get("segment_track") or {}
        seg_gated = plan.get("segment_gated") or {}
        seg_sr = plan.get("segment_sr") or {}
        seg_cuts = plan.get("segment_cuts") or {}
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
        "sr_frames": seg_sr.get(str(i), 0),
        "cuts_inside": seg_cuts.get(str(i), 0),
        "text": (s.get("text_translated") or "")[:36],
    } for i, s in enumerate(segs)]
    artifacts.export_csv(rows, ctx.deliver / "04_lipsync_report.csv")
    import shutil
    shutil.copy2(str(out), str(ctx.deliver / "04_lipsync.mp4"))
    if stats:
        log(f"  lipsync: {stats.get('clips')} clips, {stats.get('sr_clips')} at 2×, "
            f"occlusion={stats.get('occlusion_mode')} "
            f"(reverted {stats.get('reverted_frames')} frames, "
            f"region-patched {stats.get('region_frames')}), fusion={stats.get('fusion')}, "
            f"audio offset {stats.get('audio_offset_ms')} ms")
    ctx.put("lipsync", {"video": str(out), **{k: v for k, v in stats.items()
                                              if k != "per_clip"},
                        "per_clip": stats.get("per_clip")})


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
    cuts = None
    try:
        cuts = json.loads(Path(plan_path).read_text(encoding="utf-8")).get("cuts")
    except Exception:                                   # noqa: BLE001
        pass
    res = restore_video(
        Path(src), out,
        fidelity_weight=args.enhance_fidelity if args.enhance_fidelity is not None
        else FACE_ENHANCE_FIDELITY,
        protect_lips=FACE_ENHANCE_PROTECT_LIPS if args.enhance_protect_lips is None
        else bool(args.enhance_protect_lips),
        frame_filter=frame_filter_from_plan(plan_path),
        cuts=cuts,
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


def step_qc(ctx: Ctx, args) -> None:
    """Per-segment PASS/WARN/FAIL (ai_movie/qc.py) for the current version(s)."""
    from ai_movie import qc as qc_mod

    docs = {}
    q = qc_mod.build_qc(ctx.state)
    paths = qc_mod.write_outputs(q, ctx.deliver)
    docs["fit"] = {"summary": q["summary"],
                   **{f"{k}_path": str(v) for k, v in paths.items()}}
    log(f"  QC ({q['key']}): {q['summary']['PASS']} PASS / {q['summary']['WARN']} WARN / "
        f"{q['summary']['FAIL']} FAIL of {q['summary']['n']}")
    for r, n in list(q["summary"]["reasons"].items())[:6]:
        log(f"    {n:4d}  {r}")
    if (ctx.state.get("vc") or {}).get("segments"):
        qv = qc_mod.build_qc(ctx.state, key="vc")
        pv = qc_mod.write_outputs(qv, ctx.deliver, suffix="_vc")
        docs["vc"] = {"summary": qv["summary"],
                      **{f"{k}_path": str(v) for k, v in pv.items()}}
        log(f"  QC (vc): {qv['summary']['PASS']} PASS / {qv['summary']['WARN']} WARN / "
            f"{qv['summary']['FAIL']} FAIL")
    ctx.put("qc", docs)


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
    ap.add_argument("--no-osd", action="store_true",
                    help="skip overlapped-speech detection")
    ap.add_argument("--no-compact", action="store_true",
                    help="skip the duration-constrained rewrite stage")
    ap.add_argument("--lipsync-backend", default="musetalk")
    ap.add_argument("--lipsync-audio-offset-ms", type=float, default=None,
                    help="delay the driving audio vs picture (ms, +=later); "
                         "default config LIPSYNC_AUDIO_OFFSET_MS")
    ap.add_argument("--fusion", default=None, choices=[None, "alpha", "laplacian"],
                    help="MuseTalk paste mode (default config MUSETALK_FUSION)")
    ap.add_argument("--occlusion-mode", default=None, choices=[None, "frame", "region"],
                    help="occluded-mouth fallback granularity (default config OCCLUSION_MODE)")
    ap.add_argument("--enhance-fidelity", type=float, default=None,
                    help="CodeFormer fidelity w (default: config FACE_ENHANCE_FIDELITY)")
    ap.add_argument("--enhance-protect-lips", type=int, default=None, choices=[0, 1],
                    help="1 = leave the generated lips untouched (default: config)")
    ap.add_argument("--no-fp", action="store_true",
                    help="ignore stage fingerprints (cache by stage name only)")
    ap.add_argument("--fp-adopt", action="store_true",
                    help="stamp every cached stage with its current fingerprint and exit")
    ap.add_argument("--fp-explain", default=None, metavar="STEP",
                    help="print why STEP is (or is not) stale and exit")
    args = ap.parse_args()

    ctx = Ctx(Path(args.video), args.name, args=args, no_fp=args.no_fp)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    log(f"Workspace: {ctx.work}")

    if args.fp_adopt:
        n = 0
        for s in STEP_DEPS:
            if ctx.state.get(s):
                ctx.stamp(s)
                n += 1
        ctx.save()
        log(f"fingerprints adopted for {n} cached stages")
        return 0
    if args.fp_explain:
        s = args.fp_explain
        old = (ctx.state.get("_fp") or {}).get(s)
        if not ctx.state.get(s):
            log(f"{s}: not cached")
        elif old is None:
            log(f"{s}: cached, legacy (no fingerprint) — valid")
        else:
            new = ctx.fingerprint(s)
            diff = _fp_diff(old, new)
            log(f"{s}: {'VALID' if not diff else 'STALE — ' + ', '.join(diff)}")
        return 0

    dispatch = {
        "demux": lambda: step_demux(ctx),
        "separate": lambda: step_separate(ctx),
        "osd": lambda: step_osd(ctx, args),
        "asr": lambda: step_asr(ctx, args),
        "glossary": lambda: step_glossary(ctx, args),
        "translate": lambda: step_translate(ctx, args),
        "tts": lambda: step_tts(ctx, args),
        "compact": lambda: step_compact(ctx, args),
        "fit": lambda: step_fit(ctx),
        "mix": lambda: step_mix(ctx),
        "faces": lambda: step_faces(ctx, args),
        "lipsync": lambda: step_lipsync(ctx, args),
        "enhance": lambda: step_enhance(ctx, args),
        "compose": lambda: step_compose(ctx),
        "qc": lambda: step_qc(ctx, args),
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
