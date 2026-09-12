"""Per-segment quality control: PASS / WARN / FAIL with reasons and timecodes.

Every stage already computes a confidence of some kind — ASR word
probability, speaker-turn coverage, fit ratio and truncation, clone
similarity, face binding, yaw gating, overlap ratio — but nobody read them
together, so a reviewer had to watch the whole film.  This module folds
them into one record per segment and a review list of only the segments
that need eyes.

Thresholds live in ``config`` (``QC_*``).  Every rule is a pure function
of the state document, so the report can be regenerated at any time.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _tc(t: float | None) -> str:
    if t is None:
        return ""
    t = float(t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def _overlap_ratio(regions, a: float, b: float) -> float:
    if b <= a or not regions:
        return 0.0
    cov = 0.0
    for s, e in regions:
        cov += max(0.0, min(b, float(e)) - max(a, float(s)))
    return min(1.0, cov / (b - a))


def _thresholds() -> dict:
    from ai_movie import config as c
    return {
        "asr_conf_warn": getattr(c, "QC_ASR_CONF_WARN", 0.5),
        "speaker_conf_warn": getattr(c, "QC_SPEAKER_CONF_WARN", 0.6),
        "fit_warn": getattr(c, "QC_FIT_WARN", 1.25),
        "fit_fail": getattr(c, "QC_FIT_FAIL", 1.60),
        "overrun_warn": getattr(c, "QC_OVERRUN_WARN", 0.15),
        "overrun_fail": getattr(c, "QC_OVERRUN_FAIL", 0.30),
        "gated_frac_warn": getattr(c, "QC_GATED_FRAC_WARN", 0.5),
        "overlap_warn": getattr(c, "QC_OVERLAP_WARN", 0.3),
        "overlap_fail": getattr(c, "QC_OVERLAP_FAIL", 0.6),
        "clone_sim_warn": getattr(c, "QC_CLONE_SIM_WARN",
                                  getattr(c, "TTS_CLONE_MIN_SIMILARITY", 0.40)),
        "mix_gain_warn": getattr(c, "QC_MIX_GAIN_WARN", 2.9),
        "occlusion_frac_warn": getattr(c, "QC_OCCLUSION_FRAC_WARN", 0.25),
    }


def build_qc(state: dict, *, plan: dict | None = None, osd: dict | None = None,
             key: str | None = None) -> dict:
    """Build the QC document from a pipeline state.

    ``key`` selects the segment list (``"vc"`` for the cloned version,
    otherwise fit → compact → tts).  ``plan`` is the face plan (loaded from
    ``state["faces"]["plan_path"]`` when omitted); ``osd`` the overlap
    document (``state["osd"]`` when omitted).
    """
    th = _thresholds()
    if key is None:
        key = next((k for k in ("fit", "compact", "tts") if (state.get(k) or {}).get("segments")), None)
    segs = ((state.get(key) or {}).get("segments")) if key else None
    if not segs:
        return {"key": key, "segments": [], "summary": {"PASS": 0, "WARN": 0, "FAIL": 0,
                                                        "n": 0, "reasons": {}},
                "thresholds": th}
    if plan is None:
        pp = (state.get("faces") or {}).get("plan_path")
        if pp and Path(pp).exists():
            try:
                plan = json.loads(Path(pp).read_text(encoding="utf-8"))
            except Exception:                           # noqa: BLE001
                plan = None
    plan = plan or {}
    if osd is None:
        osd = state.get("osd") or {}
    regions = osd.get("regions") or []
    fps = float(plan.get("fps") or 25.0)
    frames = plan.get("frames") or {}
    frames_sr = set(int(f) for f in plan.get("frames_sr") or [])
    speaker_track = plan.get("speaker_track") or {}
    segment_track = plan.get("segment_track") or {}
    segment_gated = plan.get("segment_gated") or {}
    segment_cuts = plan.get("segment_cuts") or {}
    tracks = plan.get("tracks") or []
    quality = (state.get("tts") or {}).get("quality") or {}
    vc_refs = (state.get("vc") or {}).get("refs") or {}
    mix_doc = (state.get("mix") or {})
    if key == "vc" and ((state.get("vc") or {}).get("mix") or {}).get("gain_db"):
        mix_doc = (state.get("vc") or {}).get("mix") or {}
    mix_gain = mix_doc.get("gain_db") or {}
    mix_off = float(mix_doc.get("global_offset_db") or 0.0)
    lip_per_clip = ((state.get("lipsync") or {}).get("per_clip")) or []
    compact_report = {r["idx"]: r for r in ((state.get("compact") or {}).get("report") or [])}

    def _same_gender_track_present(a: int, b: int, gender: str | None) -> bool:
        if not gender:
            return False
        for t in tracks:
            if t.get("gender") == gender and not (t.get("last", -1) < a or t.get("first", 1e12) > b):
                return True
        return False

    def _occluded_in(a_s: float, b_s: float) -> tuple[int, float]:
        """Occlusion-fallback frames of the clips overlapping this segment and
        the worst clip-level fraction (a clip spans several segments, so the
        count alone over-attributes; the fraction is what the rule uses)."""
        n, worst = 0, 0.0
        for c in lip_per_clip:
            if c.get("end", 0) <= a_s or c.get("start", 1e12) >= b_s:
                continue
            o = c.get("occlusion") or {}
            k = int(o.get("reverted_frames", 0)) + int(o.get("region_frames", 0))
            n += k
            worst = max(worst, k / max(1, int(o.get("frames", 0) or 1)))
        return n, worst

    recs = []
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    reasons_count: dict[str, int] = {}
    for i, s in enumerate(segs):
        start, end = float(s.get("start", 0)), float(s.get("end", 0))
        a, b = int(start * fps), int(end * fps)
        spk = s.get("speaker") or ""
        gender = s.get("gender") or s.get("tts_gender")
        tid = segment_track.get(str(i), speaker_track.get(spk))
        anchored = sum(1 for f in range(a, b + 1) if str(f) in frames)
        gated = int(segment_gated.get(str(i), 0))
        sr = sum(1 for f in range(a, b + 1) if f in frames_sr)
        ov = _overlap_ratio(regions, start, end) if regions else None
        sim = None
        if key == "vc" and spk in vc_refs:
            sim = (vc_refs.get(spk) or {}).get("similarity")
        elif spk in quality:
            sim = quality[spk].get("similarity")
        fit_ratio = float(s.get("fit_ratio") or 1.0)
        overrun = float(s.get("overrun") or 0.0)
        gain = mix_gain.get(str(i), mix_gain.get(i))
        comp = compact_report.get(i)
        occ, occ_frac = _occluded_in(start, end)

        warn, fail = [], []
        asr_conf = s.get("asr_conf")
        if asr_conf is not None and asr_conf < th["asr_conf_warn"]:
            warn.append(f"asr_conf<{th['asr_conf_warn']}")
        spk_conf = s.get("speaker_conf")
        if spk_conf is not None and spk_conf < th["speaker_conf_warn"]:
            warn.append(f"speaker_conf<{th['speaker_conf_warn']}")
        if fit_ratio > th["fit_fail"]:
            fail.append(f"fit_ratio>{th['fit_fail']}")
        elif fit_ratio > th["fit_warn"]:
            warn.append(f"fit_ratio>{th['fit_warn']}")
        if overrun > th["overrun_fail"]:
            fail.append(f"truncated>{th['overrun_fail']}s")
        elif overrun > th["overrun_warn"]:
            warn.append(f"truncated>{th['overrun_warn']}s")
        if s.get("tts_error") or not (s.get("audio") or s.get("audio_fit")):
            fail.append("no_audio")
        if s.get("tts_fallback"):
            warn.append("clone_fallback_builtin")
        if sim is not None and sim < th["clone_sim_warn"]:
            warn.append(f"clone_similarity<{th['clone_sim_warn']}")
        if ov is not None:
            if ov > th["overlap_fail"]:
                fail.append(f"overlap>{th['overlap_fail']}")
            elif ov > th["overlap_warn"]:
                warn.append(f"overlap>{th['overlap_warn']}")
        if tid is not None and (anchored + gated) > 0 and gated / (anchored + gated) > th["gated_frac_warn"]:
            warn.append("mostly_gated(profile/small)")
        if tid is None and _same_gender_track_present(a, b, gender):
            warn.append("unbound_but_same_gender_face_onscreen")
        if occ_frac >= th["occlusion_frac_warn"]:
            warn.append(f"occlusion_fallback({occ_frac:.0%} of clip)")
        if comp and comp.get("status") == "rewritten":
            warn.append("compact_rewritten")
        # Lines under 0.7 s keep the built-in voice by design (VC minimum);
        # only a longer line that was not converted is worth a look.
        if key == "vc" and not s.get("vc") and (end - start) >= 0.7:
            warn.append("vc_kept_builtin")

        status = "FAIL" if fail else ("WARN" if warn else "PASS")
        counts[status] += 1
        for r in fail + warn:
            reasons_count[r] = reasons_count.get(r, 0) + 1
        recs.append({
            "idx": i, "status": status, "tc": _tc(start), "start": round(start, 2),
            "end": round(end, 2), "speaker": spk, "gender": gender,
            "asr_conf": asr_conf, "speaker_conf": spk_conf,
            "text_ja": s.get("text"), "text_zh": s.get("text_translated"),
            "text_zh_full": s.get("text_translated_full"),
            "fit_ratio": round(fit_ratio, 3), "rate_factor": s.get("rate_factor"),
            "overrun": round(overrun, 2), "tts_fallback": bool(s.get("tts_fallback")),
            "vc": bool(s.get("vc")), "clone_similarity": sim,
            "mix_gain_db": gain, "bound_track": tid, "anchored_frames": anchored,
            "gated_frames": gated, "sr_frames": sr, "cuts_inside": int(segment_cuts.get(str(i), 0)),
            "occlusion_frames": occ, "overlap_ratio": None if ov is None else round(ov, 3),
            "reasons": ";".join(fail + warn),
        })

    top = sorted(reasons_count.items(), key=lambda kv: -kv[1])
    return {"key": key, "segments": recs,
            "summary": {**counts, "n": len(recs),
                        "fail_frac": round(counts["FAIL"] / max(1, len(recs)), 3),
                        "reasons": dict(top)},
            "thresholds": th}


def write_outputs(qc: dict, deliver_dir: str | Path, *, suffix: str = "") -> dict[str, Path]:
    """Write ``06_qc{suffix}.json``, ``06_qc{suffix}_review.csv`` and a summary."""
    d = Path(deliver_dir)
    d.mkdir(parents=True, exist_ok=True)
    pj = d / f"06_qc{suffix}.json"
    pj.write_text(json.dumps(qc, ensure_ascii=False, indent=1), encoding="utf-8")
    pc = d / f"06_qc{suffix}_review.csv"
    cols = ["idx", "status", "tc", "start", "end", "speaker", "gender", "reasons",
            "fit_ratio", "overrun", "overlap_ratio", "clone_similarity", "bound_track",
            "gated_frames", "sr_frames", "occlusion_frames", "cuts_inside",
            "text_ja", "text_zh", "text_zh_full"]
    with open(pc, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in qc["segments"]:
            if r["status"] != "PASS":
                w.writerow(r)
    ps = d / f"06_qc{suffix}_summary.txt"
    sm = qc["summary"]
    lines = [f"segments: {sm['n']}  PASS {sm['PASS']}  WARN {sm['WARN']}  FAIL {sm['FAIL']}"
             f"  (fail {sm.get('fail_frac', 0):.1%})", "reasons:"]
    for k, v in sm["reasons"].items():
        lines.append(f"  {v:4d}  {k}")
    ps.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": pj, "csv": pc, "summary": ps}
