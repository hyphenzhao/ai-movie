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

The converter takes v1's *natural* line (``config.VC_SOURCE_KEY = "audio"``;
``--vc-source audio_fit`` restores the v3.3 input for an A/B) and the whole
fit ratio is paid once, after conversion: each converted wav is stretched to
exactly the sample count of the v1 fitted wav it stands in for
(``pin_to_v1``), so the timeline is identical by construction and the
converted track reuses v1's lip-sync render instead of paying for a second
one.  Converting the fitted audio instead meant a sped-up line went
rubberband(rate) → rubberband(slot) → VC → rubberband again, with the
converter seeing 1.25–1.6× speech.  (Re-running the ordinary slot fit does
*not* work here — it recompresses audio v1 already compressed; measured drift
up to 120 ms.)  A line the guard or the converter left in the built-in voice
is v1's fitted wav copied bit-exact, never re-rendered.

Every converted line is judged twice against the v1 line it came from
(``ai_movie/vc_guard.py``): pitch (voicing / band / octave jumps) and, when
``VC_DRIFT_JUDGE`` is on, content — both wavs are re-read by Whisper
(``asr.WhisperClips``, loaded once per run) and a line whose words drifted
falls back to the built-in voice like a pitch failure; over
``VC_GUARD_MAX_REJECT`` of judged lines rejected → the next reference clip.

    python scripts/run_vc_version.py workspace/output_test/state.json --refs-json …
    python scripts/run_vc_version.py workspace/F_p01/state.json --profiles workspace/F/profiles.json
    # A/B on one state without overwriting the deliverable:
    python scripts/run_vc_version.py … --vc-source audio_fit --out-name v2_fitsrc --out-dir synthesized_vc_fitsrc
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

# There is no default reference clip: the reference decides the outcome (a
# bad one drove a 232 Hz line down to 117 Hz — an octave — while ECAPA
# similarity barely noticed, Documentation/vc-gate-result.md), so it must
# come from a source that measured it — --profiles (build_profiles.py) or
# --refs-json (auto_select_refs.py) — or be named explicitly with --ref-*.
# Any segment whose timing moves more than one frame breaks the premise that
# v1's lip-sync video can be reused.
FRAME_TOLERANCE = 1.0 / 29.97


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _refs_from_profiles(state: dict, path: Path) -> tuple[dict, str]:
    """speaker → {ref_audio, gender, profile} from a film-wide profiles.json.

    A chunk speaker uses the profile the enrol stage assigned it; without an
    assignment it takes the gender's default profile.  A profile with no
    reference (nothing qualified) contributes nothing, so its lines keep the
    built-in voice — the same fallback ``run_vc_conversion`` applies.
    """
    import hashlib
    raw = path.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    profiles = doc.get("profiles") or {}
    assigned = ((state.get("enrol") or {}).get("speaker_profile")) or {}
    default = {p.get("gender"): pid for pid, p in profiles.items() if p.get("default_for_gender")}
    refs = {}
    for spk, meta in ((state["asr"]["diarization"].get("speakers")) or {}).items():
        g = meta.get("gender")
        a = assigned.get(spk)
        pid = (a.get("profile") if isinstance(a, dict) else a) or default.get(g)
        prof = profiles.get(pid) if pid else None
        ref = (prof or {}).get("ref_audio")
        if not ref:
            continue
        rp = Path(ref) if Path(ref).is_absolute() else path.parent / ref
        if rp.exists():
            refs[spk] = {"ref_audio": str(rp), "gender": g, "profile": pid}
    return refs, hashlib.sha1(raw).hexdigest()


def _ref_options(refs: dict, args, profiles_doc: dict | None) -> list[dict]:
    """The reference sets to try, best first: the chosen clips, then the alternates.

    With profiles the alternates are each profile's ``ref_alternatives``; with
    refs.json they are the other clips that passed the gate (``reject`` None).
    Every option maps *every* speaker (a speaker without an alternate keeps
    its primary clip).
    """
    alts: dict[str, list[str]] = {}
    if profiles_doc:
        base = Path(args.profiles).parent
        for spk, v in refs.items():
            prof = (profiles_doc.get("profiles") or {}).get(v.get("profile") or "") or {}
            primary = Path(v["ref_audio"]).resolve()
            alts[spk] = [str((base / a).resolve() if not Path(a).is_absolute() else Path(a).resolve())
                         for a in (prof.get("ref_alternatives") or [])]
            alts[spk] = [a for a in alts[spk] if Path(a) != primary]
    elif args.refs_json:
        doc = json.loads(Path(args.refs_json).read_text(encoding="utf-8"))
        for spk, v in refs.items():
            rows = (doc.get("candidates") or {}).get(v.get("gender"), []) or []
            primary = Path(v["ref_audio"]).resolve()
            alts[spk] = [str((ROOT / r["path"]).resolve() if not Path(r["path"]).is_absolute() else Path(r["path"]).resolve())
                         for r in rows if r.get("reject") is None]
            alts[spk] = [a for a in alts[spk] if Path(a) != primary]
    n = max([len(a) for a in alts.values()] + [0])
    options = [refs]
    for k in range(n):
        opt = {}
        for spk, v in refs.items():
            a = alts.get(spk) or []
            opt[spk] = dict(v, ref_audio=a[k]) if k < len(a) and Path(a[k]).exists() else v
        options.append(opt)
    return options[:3]


def _merge_attempts(attempts: list, segs: list[dict]) -> tuple[dict, dict, dict, dict]:
    """Per speaker, take the attempt with the lowest judged-rejection rate for that speaker's lines;
    attempts that judged none of the speaker's lines (conversion failed) lose to any that did.

    The merged stats carry each chosen line's verdict (``verdicts[i]``: f0 / voiced counts and the
    content judge's scores) so they reach the report, and a ``drift`` block that aggregates the
    content judge across attempts (units / seconds per attempt, the merged judged / rejected counts)."""
    speakers = sorted({s.get("speaker") or "" for s in segs})
    by_spk = {spk: [i for i, s in enumerate(segs) if (s.get("speaker") or "") == spk] for spk in speakers}
    items, refs, chosen, verdicts = {}, {}, {}, {}
    tot_checked = tot_rejected = 0
    reasons: dict[str, int] = {}
    for spk, idxs in by_spk.items():
        best = None
        for k, it, st, ref_set in attempts:
            vs = [st["verdicts"].get(i) for i in idxs]
            judged = [v for v in vs if v and v.get("judged")]
            converted = sum(1 for i in idxs if (it.get(i) or {}).get("vc"))
            rej = sum(1 for v in judged if not v["ok"])
            key = (0 if judged else 1, rej / len(judged) if judged else 1.0, -converted, k)
            if best is None or key < best[0]:
                best = (key, k, it, st, ref_set, len(judged), rej)
        _, k, it, st, ref_set, n_j, rej = best
        chosen[spk] = k
        for i in idxs:
            if i in it:
                items[i] = it[i]
            if i in st["verdicts"]:
                verdicts[i] = st["verdicts"][i]
        if spk in ref_set:
            refs[spk] = ref_set[spk]
        tot_checked += n_j; tot_rejected += rej
        for i in idxs:
            v = st["verdicts"].get(i)
            if v and v.get("judged") and not v["ok"]:
                key2 = v["reason"].split("_")[0]
                reasons[key2] = reasons.get(key2, 0) + 1
    per_attempt = [dict(st.get("drift") or {}, attempt=k) for k, _it, st, _r in attempts]
    drift = {"enabled": any(d.get("enabled") for d in per_attempt),
             "judged": sum(1 for v in verdicts.values() if _drift_judged(v)),
             "rejected": sum(1 for v in verdicts.values() if _drift_judged(v) and _drift_failed(v)),
             "seconds": round(sum(d.get("seconds") or 0.0 for d in per_attempt), 1),
             "attempts": per_attempt}
    errs = [d["error"] for d in per_attempt if d.get("error")]
    if errs:
        drift["error"] = errs[-1]
    stats = {"checked": tot_checked, "rejected": tot_rejected, "reasons": reasons,
             "dropped": sum(1 for it in items.values() if it.get("guard")),
             "verdicts": verdicts, "drift": drift}
    return items, refs, stats, chosen


def _drift_judged(v: dict) -> bool:
    """A verdict the content judge could evaluate on either level (line or chunk)."""
    return any((v.get(k) or {}).get("judged") for k in ("drift", "drift_chunk"))


def _drift_failed(v: dict) -> bool:
    return any((v.get(k) or {}).get("judged") and not (v.get(k) or {}).get("ok") for k in ("drift", "drift_chunk"))


def pin_to_v1(segs: list[dict], v1_segs: list[dict], fitted_dir: Path, items: dict, *,
              source_key: str, stretch=None) -> list[dict]:
    """Pin every line onto v1's timeline; mutates and returns *segs*.

    Not fit_segments_to_timeline: that recomputes the slot fit from scratch,
    and its input here is v1's already fitted audio.  Measured, it drifted up
    to 120 ms — segments v1 had compressed to 1.60× got compressed a second
    time, moving their ends *earlier*, while untouched segments moved later
    by the ~30-50 ms voice conversion adds.  Matching each wav to the exact
    sample count of the v1 fitted wav it stands in for makes the timeline
    identical by construction rather than approximately.

    * ``vc=False`` items (no reference, too short, converter or guard
      fallback): v1's ``audio_fit`` is copied bit-exact — no re-render.
    * converted lines: stretched by have/target in one rubberband pass, then
      trimmed / zero-padded to ``round(target·sr)`` samples (atempo lands
      within a millisecond or two, and "within" is not "equal").

    Per segment: ``audio_fit`` (pinned), ``fit_ratio`` = v1's (what the slot
    demanded; keeps qc's fit warn/fail semantics identical to v1),
    ``vc_pin_ratio`` = have/target (the stretch actually applied to the
    converted wav — qc applies the fit thresholds to it too, so a split-back
    overshoot stays a FAIL), ``vc_len_ratio`` = have / source duration (the
    conversion's own length overhead, source-key agnostic; ≈ 1.0–1.05 for a
    healthy singleton, up to ~2 for a short piece cut out of a chunk),
    ``overrun`` copied from v1, ``fit_end`` recomputed from the pinned file
    the way composer.fit_segments_to_timeline does, so the caller's drift
    check measures instead of copying.
    """
    import numpy as np
    import soundfile as sf
    if stretch is None:
        from ai_movie.composer import stretch_audio as stretch
    fitted_dir.mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(segs)), key=lambda i: float(segs[i].get("start", 0.0)))
    nxt = {i: (segs[order[n + 1]] if n + 1 < len(order) else None) for n, i in enumerate(order)}
    for i, s in enumerate(segs):
        v1 = v1_segs[i] if i < len(v1_segs) else {}
        ref_wav = v1.get("audio_fit")
        it = items.get(i) or {}
        src = s.get("audio")
        if not ref_wav or not Path(ref_wav).exists():
            continue
        dst = fitted_dir / f"seg_{i + 1:04d}.fit.wav"
        target = float(sf.info(ref_wav).duration)
        if not it.get("vc") or not src or not Path(src).exists():
            shutil.copy2(ref_wav, dst)
            have, pin = target, 1.0
        else:
            have = float(sf.info(src).duration)
            pin = have / target if target > 0 else 1.0
            stretch(Path(src), dst, pin)
            a, sr = sf.read(str(dst), dtype="float32")
            want = int(round(target * sr))
            if len(a) > want:
                a = a[:want]
            elif len(a) < want:
                a = np.concatenate([a, np.zeros(want - len(a), dtype="float32")])
            sf.write(str(dst), a, sr)
        source = it.get("source")
        source_dur = it.get("source_dur")
        if source_dur is None and source and Path(source).exists():
            source_dur = float(sf.info(source).duration)
        s["audio_fit"] = str(dst)
        s["fit_ratio"] = v1.get("fit_ratio", 1.0)
        s["vc_pin_ratio"] = round(pin, 4)
        s["vc_len_ratio"] = round(have / source_dur, 4) if (it.get("vc") and source_dur) else None
        s["overrun"] = v1.get("overrun", 0)
        start = float(s.get("start", 0.0))
        dur = float(sf.info(str(dst)).duration)
        hard_limit = float("inf")
        if nxt.get(i) is not None:
            hard_limit = max(float(s.get("end", 0.0)), float(nxt[i].get("start", float("inf"))))
        s["fit_end"] = round(min(start + dur, hard_limit), 2)
    return segs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("state", help="workspace/<name>/state.json from the v1 run")
    ap.add_argument("--out-name", default="v2_cloned")
    ap.add_argument("--ref-female", default=None, help="explicit female reference clip (else --refs-json / --profiles)")
    ap.add_argument("--ref-male", default=None, help="explicit male reference clip")
    ap.add_argument("--refs-json", default=None,
                    help="refs_auto/refs.json from auto_select_refs.py; overrides --ref-*; "
                         "if no gender qualified, registers the v1 film as state['vc']")
    ap.add_argument("--profiles", default=None,
                    help="film-wide profiles.json (scripts/build_profiles.py): one reference clip per "
                         "speaker profile shared by every chunk; a chunk speaker maps to a profile via "
                         "state['enrol']['speaker_profile'] or the gender's default profile")
    ap.add_argument("--vc-source", default=None, choices=["audio", "audio_fit"],
                    help="which v1 wav the converter takes its content from (default config.VC_SOURCE_KEY): "
                         "audio = the natural line, pinned once afterwards; audio_fit = v3.3's slot-fitted input")
    ap.add_argument("--out-dir", default="synthesized_vc",
                    help="per-line output folder under the workspace (default synthesized_vc); give each A/B "
                         "variant its own so they do not overwrite each other")
    ap.add_argument("--no-drift-judge", action="store_true",
                    help="skip the Whisper content judge (config.VC_DRIFT_JUDGE); pitch guard only")
    args = ap.parse_args()

    from ai_movie import artifacts, tts as tts_mod
    from ai_movie.composer import (build_speech_track, compose_video,
                                   mix_audio, stretch_audio)

    state_path = Path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    work = state_path.parent
    deliver = work / "deliverables" / args.out_name
    deliver.mkdir(parents=True, exist_ok=True)

    if args.refs_json:
        picked = (json.loads(Path(args.refs_json).read_text(encoding="utf-8"))
                  .get("picked") or {})
        args.ref_female = picked.get("female") or "/nonexistent"
        args.ref_male = picked.get("male") or "/nonexistent"
        if args.ref_female == "/nonexistent" and args.ref_male == "/nonexistent":
            # Nothing qualified through the F0 gate: the built-in voices ARE
            # the deliverable.  Register v1 as the "vc" version so the
            # downstream tooling (qc, deliver) has one place to look.
            if not (state.get("fit") or {}).get("segments") or not (state.get("compose") or {}).get("video"):
                log("no qualifying reference and no v1 film to fall back to")
                return 1
            state["vc"] = {"segments": state["fit"]["segments"], "refs": {},
                           "video": state["compose"]["video"], "converted": 0,
                           "reused_lipsync": True, "max_drift_ms": 0.0,
                           "note": "no VC reference qualified — built-in voices"}
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
            log("no qualifying reference for any gender — registered v1 as the vc version")
            return 0

    v1_segs = (state.get("fit") or {}).get("segments") or []
    if not v1_segs:
        log("state has no fitted segments — run v1 first")
        return 1
    v1_ends = {i: s.get("fit_end") for i, s in enumerate(v1_segs)}

    refs = {}
    profiles_sha1 = None
    profiles_doc = None
    if args.profiles:
        refs, profiles_sha1 = _refs_from_profiles(state, Path(args.profiles))
        profiles_doc = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
        log(f"profiles {Path(args.profiles).name} ({profiles_sha1[:8]}): "
            f"{ {k: v.get('profile') for k, v in refs.items()} }")
    else:
        if not (args.ref_female or args.ref_male or args.refs_json):
            log("no reference given: pass --profiles, --refs-json or --ref-female/--ref-male "
                "(a baked-in default once cloned every film with output_test's voice)")
            return 2
        for spk, meta in ((state["asr"]["diarization"].get("speakers")) or {}).items():
            g = meta.get("gender")
            chosen = args.ref_female if g == "female" else args.ref_male
            if not chosen:
                continue
            p = ROOT / chosen
            if p.exists():
                refs[spk] = {"ref_audio": str(p), "gender": g}
    log(f"references: { {k: Path(v['ref_audio']).name for k, v in refs.items()} }")
    if not refs:
        if args.profiles and (state.get("fit") or {}).get("segments") and (state.get("compose") or {}).get("video"):
            state["vc"] = {"segments": state["fit"]["segments"], "refs": {},
                           "video": state["compose"]["video"], "converted": 0,
                           "reused_lipsync": True, "max_drift_ms": 0.0,
                           "profiles_sha1": profiles_sha1,
                           "note": "no profile has a reference for this chunk's speakers — built-in voices"}
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
            log("no profile reference applies — registered v1 as the vc version")
            return 0
        log("no usable reference clips")
        return 1

    # ── convert, guard, retry with the next clip when a reference wrecks the lines ─────────
    from ai_movie.config import (VC_DRIFT_BASELINE_MIN, VC_DRIFT_BATCH, VC_DRIFT_BEAM, VC_DRIFT_JUDGE,
                                 VC_DRIFT_KANA_LEAK, VC_DRIFT_MIN_CHARS, VC_DRIFT_MIN_SIM,
                                 VC_DRIFT_WHISPER_MODEL, VC_GUARD_MAX_REJECT, VC_SOURCE_KEY)
    from ai_movie.vc_guard import guard_lines
    source_key = args.vc_source or VC_SOURCE_KEY
    segs = [dict(s) for s in v1_segs]
    out_dir = work / args.out_dir
    options = _ref_options(refs, args, profiles_doc)
    # The content judge: one Whisper load per run, outside the CosyVoice worker (which is a separate
    # pinned subprocess, gone by the time the guard runs).  Its transcript cache is keyed by wav content,
    # so retries (try{k}/chunks/…) and A/B variants re-decode only the new converted clips.  A judge
    # that cannot load or crashes is logged and the attempt continues pitch-only: it never fails v2.
    judge = None
    transcribe = None
    use_judge = VC_DRIFT_JUDGE and not args.no_drift_judge
    if use_judge:
        try:
            from ai_movie.asr import WhisperClips
            judge = WhisperClips(cache=work / "vc_drift_cache.json")
            transcribe = judge.transcribe
            log(f"content judge: whisper {judge.model_size} (beam {judge.beam}, mel batch {judge.batch})")
        except Exception as exc:                        # noqa: BLE001
            log(f"content judge unavailable ({type(exc).__name__}: {exc}) — pitch guard only")
            judge = transcribe = None
    attempts = []          # (k, items, stats, ref_set)
    try:
        for k, ref_set in enumerate(options):
            names = {spk: Path(v["ref_audio"]).name for spk, v in ref_set.items()}
            log(f"voice-converting {len(segs)} segments ({source_key}) with {names}…")
            items = tts_mod.run_vc_conversion(
                segs, ref_set, out_dir / (f"try{k}" if k else ""), source_key=source_key,
                progress_cb=lambda d, t: log(f"  VC {d}/{t}") if d % 10 == 0 else None)
            stats = guard_lines(segs, items, v1_segs, source_key=source_key, log=log, transcribe=transcribe)
            attempts.append((k, items, stats, ref_set))
            rate = stats["rejected"] / max(1, stats["checked"])
            if k == 0 and stats["checked"] == 0:
                break                                       # nothing measurable — no basis to retry
            if stats["checked"] and rate <= VC_GUARD_MAX_REJECT:
                break
            if k + 1 < len(options):
                log(f"  {rate:.0%} of judged lines rejected — trying the next reference clip")
    finally:
        if judge is not None:
            judge.close()                                   # free the ~6 GB before mix/mux
    # Choose per speaker: each speaker keeps the attempt where *its* lines fared best — an attempt that
    # converted nothing (worker died: checked 0) never wins, and a speaker with a good primary clip is
    # not dragged onto a worse alternate by another speaker's numbers.
    items, refs, stats, chosen = _merge_attempts(attempts, segs)
    reject_rate = stats["rejected"] / max(1, stats["checked"])
    if len(attempts) > 1:
        log(f"  kept per speaker: {chosen} → {reject_rate:.0%} rejected overall")
    drift_info = dict(stats.get("drift") or {})
    if judge is not None:
        drift_info.update(model=judge.model_size, beam=judge.beam, batch=judge.batch,
                          decoded=judge.decoded, model_seconds=round(judge.seconds, 1))
    drift_info.update(min_sim=VC_DRIFT_MIN_SIM, baseline_min=VC_DRIFT_BASELINE_MIN,
                      min_chars=VC_DRIFT_MIN_CHARS, kana_leak=VC_DRIFT_KANA_LEAK,
                      config={"model": VC_DRIFT_WHISPER_MODEL, "beam": VC_DRIFT_BEAM, "batch": VC_DRIFT_BATCH})
    if use_judge:
        log(f"content judge: {drift_info.get('rejected', 0)}/{drift_info.get('judged', 0)} judged lines drifted"
            + (f", {drift_info['decoded']} clips decoded in {drift_info['model_seconds']} s" if judge else "")
            + (f" — ERROR {drift_info['error']}" if drift_info.get("error") else ""))

    converted = 0
    verdicts = stats.get("verdicts") or {}
    for i, s in enumerate(segs):
        it = items.get(i, {})
        v1 = v1_segs[i]
        s["vc"] = bool(it.get("vc"))
        # Every vc=False line is v1's own fitted wav (bit-exact copy in pin_to_v1) — never the natural
        # take the converter hands back as its fallback (run_vc_conversion docstring).
        s["audio"] = it["audio"] if (it.get("vc") and it.get("audio")) else v1.get("audio_fit")
        if it.get("guard"):
            s["vc_guard"] = it["guard"]
        if it.get("chunk"):
            s["vc_chunk"] = list(it["chunk"])
        if it.get("skipped") or it.get("tts_error"):
            s["vc_skip"] = it.get("skipped") or it.get("tts_error")
        v = verdicts.get(i) or {}
        d = v.get("drift") or {}
        if d.get("score") is not None:
            s["vc_drift"] = d["score"]
        converted += int(bool(it.get("vc")))
        s.pop("audio_fit", None)
        s.pop("fit_ratio", None)
        s.pop("fit_end", None)
    log(f"converted {converted}/{len(segs)} segments "
        f"({len(segs) - converted} kept the built-in voice)")

    # ── pin onto v1's timeline (see pin_to_v1) ─────────────────────────
    log("pinning each segment to its v1 duration…")
    fitted_dir = out_dir / "fitted"
    pin_to_v1(segs, v1_segs, fitted_dir, items, source_key=source_key, stretch=stretch_audio)
    len_ratios = sorted(s["vc_len_ratio"] for s in segs if s.get("vc_len_ratio"))
    len_ratio = ({"median": len_ratios[len(len_ratios) // 2], "p90": len_ratios[int(0.9 * (len(len_ratios) - 1))],
                  "max": len_ratios[-1], "n": len(len_ratios)} if len_ratios else {"n": 0})
    if len_ratios:
        log(f"conversion length overhead (out / source): median {len_ratio['median']:.3f}, "
            f"p90 {len_ratio['p90']:.3f}, max {len_ratio['max']:.3f} over {len(len_ratios)} converted lines")

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

    def _vd(i: int, key: str):
        return (verdicts.get(i) or {}).get(key)

    rows = [{
        "idx": i, "start": s.get("start"), "end": s.get("end"),
        "speaker": s.get("speaker"), "gender": s.get("gender"),
        "voice": "克隆(VC)" if s.get("vc") else "内置",
        "guard": s.get("vc_guard") or "", "skip": s.get("vc_skip") or "",
        "chunk": ("-".join(str(x) for x in s["vc_chunk"]) if s.get("vc_chunk") else ""),
        "fit_ratio": s.get("fit_ratio"), "vc_pin_ratio": s.get("vc_pin_ratio"),
        "vc_len_ratio": s.get("vc_len_ratio"),
        "f0_conv": _vd(i, "f0_conv"), "f0_v1": _vd(i, "f0_v1"),
        "voiced_conv": _vd(i, "voiced_conv"), "voiced_v1": _vd(i, "voiced_v1"),
        "drift_score": (_vd(i, "drift") or {}).get("score"),
        "drift_baseline": (_vd(i, "drift") or {}).get("baseline"),
        "drift_chunk_score": (_vd(i, "drift_chunk") or {}).get("score"),
        "heard_v1": ((_vd(i, "drift") or {}).get("heard_v1") or "")[:40],
        "heard_conv": ((_vd(i, "drift") or {}).get("heard_conv") or "")[:40],
        "fit_end": s.get("fit_end"),
        "v1_fit_end": v1_ends.get(i),
        "drift_ms": (round((s["fit_end"] - v1_ends[i]) * 1000, 1)
                     if s.get("fit_end") is not None and v1_ends.get(i) is not None
                     else ""),
        "overrun_cut": s.get("overrun", 0),
        "text": (s.get("text_translated") or "")[:40],
    } for i, s in enumerate(segs)]
    artifacts.export_csv(rows, deliver / "03_tts_report.csv")

    state["vc"] = {"segments": segs, "refs": refs, "video": str(final), "profiles_sha1": profiles_sha1,
                   "source_key": source_key, "out_dir": str(out_dir), "out_name": args.out_name,
                   "len_ratio": len_ratio,
                   "guard": {"checked": stats["checked"], "rejected": stats["rejected"], "reasons": stats["reasons"],
                             "dropped": stats.get("dropped"), "attempts": len(attempts), "chosen": chosen,
                             "reject_rate": round(reject_rate, 3), "drift": drift_info},
                   "converted": converted, "reused_lipsync": reuse_lipsync,
                   "max_drift_ms": round(worst[0] * 1000, 1),
                   "mix": mix_stats}
    # The same block next to the deliverable, so an A/B variant that is later overwritten in
    # state["vc"] keeps its own numbers.
    (deliver / "vc_state.json").write_text(json.dumps(state["vc"], ensure_ascii=False, indent=1), encoding="utf-8")
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    log(f"done → {deliver / '05_final_dubbed.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
