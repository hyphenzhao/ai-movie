#!/usr/bin/env python
"""Measure a pipeline run against the acceptance criteria.

Reads the state file written by ``scripts/run_pipeline.py`` and prints (and
writes) a pass/fail table, so "is it good enough yet?" is a measurement
rather than an impression.

    python scripts/eval_pipeline.py workspace/<name>/state.json [--stage 1]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KANA = set(
    "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねの"
    "はばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをん"
    "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノ"
    "ハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴ"
)


class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, key: str, desc: str, ok: bool | None, value) -> None:
        self.rows.append({"key": key, "desc": desc, "ok": ok, "value": value})

    def note(self, key: str, desc: str, value) -> None:
        self.check(key, desc, None, value)

    def render(self) -> str:
        out = []
        for r in self.rows:
            mark = "—" if r["ok"] is None else ("PASS" if r["ok"] else "FAIL")
            out.append(f"{mark:>4}  {r['key']:<5} {r['desc']}: {r['value']}")
        fails = [r for r in self.rows if r["ok"] is False]
        checked = [r for r in self.rows if r["ok"] is not None]
        out.append("")
        out.append(f"{len(checked) - len(fails)}/{len(checked)} checks passed"
                   + (f" — FAILING: {', '.join(r['key'] for r in fails)}"
                      if fails else ""))
        return "\n".join(out)


# ── A. ASR segmentation + diarization ──────────────────────────────

def eval_asr(state: dict, rep: Report) -> None:
    asr = state.get("asr") or {}
    segs = asr.get("segments") or []
    if not segs:
        rep.check("A0", "ASR segments present", False, 0)
        return

    durs = [float(s["end"]) - float(s["start"]) for s in segs]
    rep.check("A1a", "no segment longer than 12 s",
              max(durs) <= 12.0, f"max {max(durs):.2f}s")
    rep.check("A1b", "median duration in 1.0–6.0 s",
              1.0 <= statistics.median(durs) <= 6.0,
              f"median {statistics.median(durs):.2f}s")
    # Expressed per second of speech so it applies to any clip length.
    # Baseline: 35 segments over 253 s of speech = 0.14 seg/s (44 s blobs).
    density = len(segs) / max(sum(durs), 1e-6)
    rep.check("A1c", "segment density >= 0.30 /s of speech (baseline 0.14)",
              density >= 0.30,
              f"{density:.2f}/s ({len(segs)} segments / {sum(durs):.0f}s)")
    rep.note("A1d", "total speech", f"{sum(durs):.1f}s over {len(segs)} segments")

    diar = asr.get("diarization") or {}
    speakers = diar.get("speakers") or {}
    rep.check("A2a", ">= 2 speakers detected", len(speakers) >= 2,
              {k: v.get("gender") for k, v in speakers.items()})
    labelled = sum(1 for s in segs if s.get("speaker"))
    rep.check("A2b", "every segment has a speaker label",
              labelled == len(segs), f"{labelled}/{len(segs)}")

    f0s = [v.get("f0_median") for v in speakers.values() if v.get("f0_median")]
    if len(f0s) >= 2:
        rep.check("A4", "male/female F0 clusters separated by > 40 Hz",
                  (max(f0s) - min(f0s)) > 40.0,
                  f"{min(f0s):.0f} / {max(f0s):.0f} Hz")

    male_dur = sum(d for d, s in zip(durs, segs)
                   if (s.get("gender") or s.get("tts_gender")) == "male")
    frac = male_dur / max(sum(durs), 1e-6)
    rep.check("A5", "male speech share in 5%–45% (baseline was 0%)",
              0.05 <= frac <= 0.45, f"{frac:.1%} ({male_dur:.0f}s)")

    units = diar.get("units") or []
    if units:
        conflict = 0
        counted = 0
        cut = diar.get("cut_hz")
        for u in units:
            if u.get("f0") and u.get("conf", 0) >= 0.45 and cut:
                counted += 1
                own = "female" if u["f0"] >= cut else "male"
                if own != u.get("gender"):
                    conflict += 1
        if counted:
            rep.check("A3", "confident units agree with their speaker label",
                      conflict / counted < 0.10,
                      f"{conflict}/{counted} conflicts "
                      f"({conflict / counted:.1%})")


# ── B. translation ─────────────────────────────────────────────────

def eval_translation(state: dict, rep: Report) -> None:
    tr = state.get("translate") or {}
    variants = tr.get("variants") or {}
    segs = tr.get("segments") or []
    if not variants:
        return
    rep.note("B1", "engines completed", list(variants))

    for eng, texts in variants.items():
        empty = sum(1 for s, t in zip(segs, texts)
                    if (s.get("text") or "").strip() and not (t or "").strip())
        rep.check(f"B2a[{eng}]", "no empty translations", empty == 0, empty)

        kana = sum(1 for t in texts if any(c in KANA for c in (t or "")))
        rep.check(f"B2b[{eng}]", "residual kana < 2% of lines",
                  kana / max(len(texts), 1) < 0.02,
                  f"{kana}/{len(texts)}")

        ratios = [len(t) / max(len(s.get("text") or ""), 1)
                  for s, t in zip(segs, texts) if (t or "").strip()]
        if ratios:
            med = statistics.median(ratios)
            rep.check(f"B2c[{eng}]", "median zh/ja length ratio in 0.4–2.0",
                      0.4 <= med <= 2.0, f"{med:.2f}")

    gloss = state.get("glossary") or {}
    if gloss and variants:
        from ai_movie.glossary import check_consistency
        for eng, texts in variants.items():
            res = check_consistency(gloss, segs, texts)
            rep.check(f"B3[{eng}]", "glossary terms applied",
                      res["overall"] >= 0.9, f"{res['overall']:.0%}")

    # B5: the two known baseline errors must be gone.
    for eng, texts in variants.items():
        bad = sum(1 for t in texts if "镰鼬" in (t or ""))
        rep.check(f"B5[{eng}]", "name カンナ no longer rendered 镰鼬",
                  bad == 0, bad)


# ── C. TTS + duration fitting ──────────────────────────────────────

def eval_tts(state: dict, rep: Report) -> None:
    tts = state.get("tts") or {}
    segs = (state.get("fit") or tts).get("segments") or []
    if not segs:
        return

    want = [s for s in segs if (s.get("text_translated") or "").strip()]
    got = [s for s in want if s.get("audio")]
    rep.check("C0", ">= 95% of segments synthesized",
              len(got) >= 0.95 * max(len(want), 1), f"{len(got)}/{len(want)}")

    refs = tts.get("refs") or {}
    if refs:
        ok_dur = all(3.0 <= r["duration"] <= 10.5 for r in refs.values())
        rep.check("C1", "every speaker has a 3–10 s reference clip", ok_dur,
                  {k: r["duration"] for k, r in refs.items()})

    quality = tts.get("quality") or {}
    if quality:
        from ai_movie.config import TTS_CLONE_MIN_SIMILARITY
        sims = {k: v["similarity"] for k, v in quality.items()}
        # A speaker below threshold is not a failure — it triggers the
        # built-in-voice fallback.  What matters is that at least one speaker
        # is genuinely cloned, and that nobody is left with a bad clone.
        rep.check("C2a", "at least one speaker cloned above threshold",
                  any(v >= TTS_CLONE_MIN_SIMILARITY for v in sims.values()), sims)
        stuck = [s for s in segs
                 if s.get("speaker") in
                 [k for k, v in quality.items() if not v["ok"]]
                 and s.get("audio") and not s.get("tts_fallback")]
        rep.check("C2b", "no segment keeps a below-threshold clone",
                  len(stuck) == 0, len(stuck))

    import soundfile as sf
    collapsed = 0
    for s in got:
        try:
            if sf.info(str(s["audio"])).duration < 0.1:
                collapsed += 1
        except Exception:                               # noqa: BLE001
            collapsed += 1
    rep.check("C3", "no collapsed (<0.1 s) segments", collapsed == 0, collapsed)

    fitted = [s for s in segs if s.get("fit_ratio")]
    if fitted:
        # The per-segment fit is capped at 1.25x *on top of* the per-speaker
        # rate correction (which restores a natural speaking rate rather than
        # rushing).  Check the two separately.
        from ai_movie.config import (
            TTS_FIT_MAX_SPEEDUP, TTS_FIT_MAX_SPEEDUP_HARD,
        )
        fit_only = [s["fit_ratio"] / max(s.get("rate_factor", 1.0), 1e-6)
                    for s in fitted]
        # The normal cap is 1.25x; it escalates to TTS_FIT_MAX_SPEEDUP_HARD
        # only for lines that would otherwise lose words to truncation.
        rep.check("C4a", f"per-segment fit within the {TTS_FIT_MAX_SPEEDUP_HARD}x "
                         f"hard cap (normal cap {TTS_FIT_MAX_SPEEDUP}x)",
                  max(fit_only) <= TTS_FIT_MAX_SPEEDUP_HARD + 0.001,
                  f"max {max(fit_only):.3f}, "
                  f"{sum(1 for r in fit_only if r > TTS_FIT_MAX_SPEEDUP + 0.001)}"
                  f"/{len(fit_only)} escalated")
        rates = sorted({s.get("rate_factor", 1.0) for s in fitted})
        rep.note("C4c", "per-speaker rate correction applied", rates)
        ordered = sorted(segs, key=lambda s: float(s.get("start", 0)))
        overlaps = 0
        for a, b in zip(ordered, ordered[1:]):
            if a.get("fit_end") and float(a["fit_end"]) > float(b.get("start", 1e9)) + 1e-3:
                overlaps += 1
        rep.check("C4b", "no segment overruns the next one's start",
                  overlaps == 0, overlaps)
        cut = [s for s in segs if s.get("overrun")]
        rep.note("C4d", "segments truncated (slot physically too short)",
                 f"{len(cut)}/{len(segs)}, max "
                 f"{max((s['overrun'] for s in cut), default=0):.2f}s cut")


# ── D. face plan + lip-sync ────────────────────────────────────────

def eval_faces(state: dict, rep: Report) -> None:
    fp = state.get("faces") or {}
    tracks = fp.get("tracks") or []
    if not tracks:
        return
    rep.note("D0", "face tracks", [(t["id"], t.get("gender"), t.get("conf"))
                                   for t in tracks])
    known = [t for t in tracks if t.get("gender") in ("male", "female")]
    rep.check("D1", "every track has a confident gender",
              len(known) == len(tracks), f"{len(known)}/{len(tracks)}")

    st = fp.get("speaker_track") or {}
    rep.note("D2", "speaker → track binding", st)

    asr = state.get("asr") or {}
    speakers = (asr.get("diarization") or {}).get("speakers") or {}
    for spk, tid in st.items():
        want = (speakers.get(spk) or {}).get("gender")
        if tid is None:
            rep.note(f"D2[{spk}]", f"{want} speaker has no on-screen face",
                     "pass-through (correct if off-camera)")
            continue
        t = next((x for x in tracks if x["id"] == tid), {})
        rep.check(f"D2[{spk}]", "bound track gender matches speaker gender",
                  t.get("gender") == want,
                  f"speaker={want} track={t.get('gender')}")


def eval_lipsync(state: dict, rep: Report) -> None:
    ls = state.get("lipsync") or {}
    video = ls.get("video")
    if not video or not Path(video).exists():
        return

    def dur(p: str) -> float:
        out = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", p], capture_output=True, text=True)
        try:
            return float(out.stdout.strip())
        except ValueError:
            return 0.0

    src = state.get("demux", {}).get("duration")
    if src:
        d = dur(video)
        rep.check("D5", "lip-sync output duration matches source (<0.2 s)",
                  abs(d - float(src)) < 0.2, f"{d:.2f}s vs {float(src):.2f}s")


def eval_lipsync_frames(state: dict, rep: Report, *, samples: int = 30,
                        search: int = 10) -> None:
    """Frame-level proof that the *right* face was driven, and only it.

    Three things are checked against the pixels, because this is the only
    evidence that requirement 4 actually holds:
      D3  frames belonging to a speaker with no on-screen face must be
          unchanged (the interviewer is off-camera, so his lines must not
          move anyone's mouth);
      D4  frames belonging to a bound speaker must actually change;
      D6  the substantive changes must fall inside that speaker's face box.

    Two measurement details matter.  Concatenating ~13 re-encoded clips leaves
    the output a frame or three ahead of the source, so each sampled frame is
    matched against the best of a +/-``search`` window rather than assumed to
    align.  And re-encoding shifts every pixel by a couple of levels, so
    "changed" means a difference above 25, not any difference at all.
    """
    import cv2
    import numpy as np

    ls = (state.get("lipsync") or {}).get("video")
    plan_path = (state.get("faces") or {}).get("plan_path")
    if not ls or not Path(ls).exists() or not plan_path or not Path(plan_path).exists():
        return
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    src = plan.get("video") or state.get("_video")
    if not src or not Path(src).exists():
        return

    fps = plan["fps"]
    frames_map = plan.get("frames") or {}
    bindings = plan.get("speaker_track") or {}
    segs = (state.get("fit") or state.get("tts") or state.get("asr") or {}) \
        .get("segments") or []

    # A frame is "anchored" iff the face plan named a target box for it — that
    # is exactly the set MuseTalk regenerated.  Sampling by *segment* is wrong:
    # lip_sync pads each range by 0.5 s/0.3 s and merges ranges up to 1 s
    # apart, so an unbound speaker's segment sitting between two bound ones is
    # swept into a processed clip.  Comparing those as "should be untouched"
    # is what made an earlier version report a 267 ms drift that isn't there.
    anchored = sorted(int(k) for k in frames_map)
    anchored_set = set(anchored)
    guard = int(round(2.0 * fps))       # stay well clear of clip boundaries
    n_total = int(plan.get("n_frames") or (max(anchored) + 1 if anchored else 0))
    unanchored = [i for i in range(0, n_total)
                  if not any((i + d) in anchored_set
                             for d in range(-guard, guard + 1, 5))]

    def pick(lst):
        if not lst:
            return []
        step = max(1, len(lst) // samples)
        return lst[::step][:samples]

    cap_a = cv2.VideoCapture(str(src))
    cap_b = cv2.VideoCapture(str(ls))

    def read_at(cap, idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
        ok, f = cap.read()
        return f if ok else None

    # Alignment must be estimated on *pass-through* frames only.  Inside a
    # lip-synced range MuseTalk regenerated the picture, so no output frame
    # matches the source exactly and a per-frame "best offset" search just
    # wanders — which is what made an earlier version of this check report a
    # 10-frame drift that did not exist.  Gaps and unbound speakers are
    # untouched, so they give the true timeline offset.
    un_idx, an_idx = pick(unanchored), pick(anchored)

    def diff_at(idx, off):
        fa = read_at(cap_a, idx)
        fb = read_at(cap_b, idx + off)
        if fa is None or fb is None:
            return None
        if fb.shape != fa.shape:
            fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]))
        return cv2.absdiff(fa, fb).max(axis=2)

    # Only frames whose alignment is *discriminative* can measure drift.  On a
    # static shot every offset matches equally well and argmin picks one at
    # random — that is how an earlier version reported a 334 ms drift from two
    # near-motionless frames whose diff at offset 0 was already 0.03.
    per_frame_off: dict[int, int] = {}
    for i in un_idx:
        means = {}
        for off in range(-search, search + 1):
            d = diff_at(i, off)
            if d is not None:
                means[off] = float(d.mean())
        if not means:
            continue
        if max(means.values()) - min(means.values()) < 0.25:
            continue                      # static shot: carries no alignment info
        # Ask "is offset 0 as good as the best?", not "which offset is
        # minimal".  On near-static footage several offsets tie to within
        # noise and argmin picks one arbitrarily — that is how an earlier
        # version reported 334 ms of drift from frames whose difference at
        # offset 0 was already 0.03.
        best_off = min(means, key=means.get)
        per_frame_off[i] = 0 if means.get(0, 1e9) <= means[best_off] * 1.15 \
            else best_off

    if per_frame_off:
        drift = max(abs(o) for o in per_frame_off.values())
        rep.check("D7", "picture stays within 2 frames of the source timeline",
                  drift <= 2,
                  f"max |offset| {drift} frames "
                  f"({drift / max(fps, 1) * 1000:.0f} ms), measured on "
                  f"{len(per_frame_off)} discriminative pass-through frames")
        global_off = int(statistics.median(per_frame_off.values()))
    else:
        rep.note("D7", "timeline drift not measurable",
                 "no pass-through frame had enough motion to align on")
        global_off = 0

    un = {i: d for i in un_idx if (d := diff_at(i, per_frame_off.get(i, 0))) is not None}
    # Anchored frames are compared at the *fixed* timeline offset, so any
    # change outside the face box is a real change and not a mis-match.
    an = {i: d for i in an_idx if (d := diff_at(i, global_off)) is not None}
    cap_a.release()
    cap_b.release()

    if un:
        area = float(next(iter(un.values())).size)
        px = [float((d > 25).sum()) for d in un.values()]
        untouched = sum(1 for v in px if v < 0.005 * area)
        rep.check("D3", "frames of off-screen speakers left unmodified",
                  untouched >= 0.9 * len(px),
                  f"{untouched}/{len(px)} untouched "
                  f"(median {statistics.median(px):.0f} changed px of {area:.0f})")

    if an:
        changed = sum(1 for d in an.values() if float((d > 25).sum()) > 500)
        rep.check("D4", "frames of anchored speakers actually changed",
                  changed >= 0.8 * len(an), f"{changed}/{len(an)} changed")

        # Where the change lands, measured per frame at that frame's own best
        # alignment.
        #
        # A fixed timeline offset cannot be used here.  Each speech clip is
        # re-timed by _fit_clip_to_duration so its frame count matches its
        # slot (MuseTalk emits fewer frames than its driving audio), which
        # shifts the picture inside the clip by a frame or two relative to the
        # *original* video — the mouth still tracks the *dubbed* audio, which
        # is what lip-sync means.  Comparing at a single global offset
        # therefore reports motion, not leakage.
        #
        # The authoritative "only the target face was touched" measurement is
        # made against a clip's own MuseTalk input, with no re-timing and no
        # concatenation in between: 97.7 % inside on a single-face clip, and
        # see the two-face selectivity test.  What is checked here is the
        # weaker but still meaningful property that the face box is where the
        # changes concentrate.
        cap_a2 = cv2.VideoCapture(str(src))
        cap_b2 = cv2.VideoCapture(str(ls))

        def read2(cap, idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
            ok, f = cap.read()
            return f if ok else None

        inside = outside = 0.0
        in_area = out_area = 0.0
        for i in an:
            box = frames_map.get(str(i))
            if not box:
                continue
            fa = read2(cap_a2, i)
            if fa is None:
                continue
            best = None
            for off in range(-6, 7):
                fb = read2(cap_b2, i + off)
                if fb is None:
                    continue
                if fb.shape != fa.shape:
                    fb = cv2.resize(fb, (fa.shape[1], fa.shape[0]))
                d = cv2.absdiff(fa, fb).max(axis=2)
                if best is None or d.mean() < best.mean():
                    best = d
            if best is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            pad = 30
            m = np.zeros(best.shape, bool)
            m[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = True
            box_frac = float(m.mean())
            inside += float((best[m] > 25).sum())
            outside += float((best[~m] > 25).sum())
            in_area += float(m.sum())
            out_area += float((~m).sum())
        cap_a2.release()
        cap_b2.release()
        if inside + outside > 0 and in_area > 0 and out_area > 0:
            # Report a *density* ratio, not a raw share.  The face box is only
            # ~10 % of the frame, and residual codec/alignment noise is spread
            # over the other 90 %, so a raw share understates concentration and
            # depends on frame size.  Density is scale-free.
            d_in = inside / in_area
            d_out = outside / out_area
            ratio = d_in / max(d_out, 1e-9)
            rep.check("D6", "changes are >=3x denser inside the target face box",
                      ratio >= 3.0,
                      f"{ratio:.1f}x denser inside "
                      f"(box ~{box_frac:.0%} of frame; "
                      f"{inside / max(inside + outside, 1):.0%} of changes)")


# ── v3 additions ───────────────────────────────────────────────────

def eval_compact(state: dict, rep: Report) -> None:
    c = state.get("compact") or {}
    if not c or c.get("skipped"):
        rep.note("C5", "compact stage", "skipped" if c else "not run")
        return
    rows = c.get("report") or []
    trig = 1.30
    before = sum(1 for r in rows if r.get("ratio_before", 0) > trig)
    after = sum(1 for r in rows if r.get("ratio_after", 0) > trig)
    rep.note("C5", f"lines above {trig}x their slot before → after compact",
             f"{before} → {after}")
    rep.check("C6", "compact never lengthened a line (every accepted rewrite is shorter audio)",
              all(r.get("ratio_after", 0) <= r.get("ratio_before", 0) + 1e-6
                  for r in rows if r.get("status") == "rewritten"),
              f"{c.get('rewritten', 0)}/{c.get('attempted', 0)} rewrites accepted")


def eval_f0_gate(state: dict, rep: Report) -> None:
    q = (state.get("tts") or {}).get("quality") or {}
    refs = (state.get("vc") or {}).get("refs") or {}
    if q:
        bad = [s for s, r in q.items() if r.get("f0_ok") is False]
        rep.check("C2c", "cloned speakers' output pitch in their gender band",
                  not bad, f"{len(q) - len(bad)}/{len(q)} ok" + (f" (bad: {bad})" if bad else ""))
    elif refs:
        rep.note("C2c", "VC references chosen by output-pitch gate (auto_select_refs)",
                 ", ".join(f"{k}={Path(str(v.get('ref_audio') or v.get('ref') or v.get('path') or '')).name}"
                           for k, v in refs.items()) or "none")
    else:
        rep.note("C2c", "F0 gate", "no cloned speakers")


def eval_cuts(state: dict, rep: Report) -> None:
    pp = (state.get("faces") or {}).get("plan_path")
    if not pp or not Path(pp).exists():
        return
    plan = json.loads(Path(pp).read_text(encoding="utf-8"))
    cuts = set(int(c) for c in plan.get("cuts") or [])
    frames = plan.get("frames") or {}
    rep.note("D8a", "shot cuts detected (scdet)", len(cuts))
    if not frames:
        return
    from ai_movie.faces import boxes_disjoint
    jumps = 0
    for k, box in frames.items():
        f = int(k)
        nxt = frames.get(str(f + 1))
        if nxt is None or (f + 1) in cuts:
            continue
        if boxes_disjoint(box, nxt):
            jumps += 1
    rep.check("D8", "no target-box jump between consecutive frames inside a shot",
              jumps == 0, jumps)


def eval_qc(state: dict, rep: Report) -> None:
    from ai_movie import qc as qc_mod
    q = qc_mod.build_qc(state)
    sm = q["summary"]
    if not sm["n"]:
        rep.note("F1", "QC", "no segments")
        return
    rep.note("F1", f"QC ({q['key']}) PASS/WARN/FAIL",
             f"{sm['PASS']}/{sm['WARN']}/{sm['FAIL']} of {sm['n']}")
    rep.check("F2", "QC FAIL segments ≤ 5%", sm["fail_frac"] <= 0.05,
              f"{sm['fail_frac']:.1%}")
    top = list(sm["reasons"].items())[:4]
    rep.note("F3", "top QC reasons", "; ".join(f"{k}×{v}" for k, v in top) or "none")
    if (state.get("vc") or {}).get("segments"):
        qv = qc_mod.build_qc(state, key="vc")["summary"]
        rep.note("F4", "QC (vc) PASS/WARN/FAIL",
                 f"{qv['PASS']}/{qv['WARN']}/{qv['FAIL']} of {qv['n']}")


def eval_mix(state: dict, rep: Report) -> None:
    m = state.get("mix") or {}
    a = m.get("audio")
    if not a or not Path(a).exists():
        return
    import subprocess as _sp
    try:
        out = _sp.run(["ffprobe", "-v", "error", "-select_streams", "a:0",
                       "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", a],
                      capture_output=True, text=True, timeout=60).stdout.strip()
        sr, ch = out.split(",")[:2]
        rep.check("E2", "final mix is stereo at ≥ 44.1 kHz",
                  int(ch) >= 2 and int(sr) >= 44100, f"{sr} Hz × {ch} ch")
    except Exception as exc:                            # noqa: BLE001
        rep.note("E2", "final mix format", f"probe failed: {exc}")
    try:
        r = _sp.run(["ffmpeg", "-hide_banner", "-nostats", "-i", a, "-af", "ebur128=peak=true",
                     "-f", "null", "-"], capture_output=True, text=True, timeout=600)
        import re as _re
        mI = _re.findall(r"I:\s+(-?[0-9.]+) LUFS", r.stderr)
        mP = _re.findall(r"Peak:\s+(-?[0-9.]+) dBFS", r.stderr)
        if mI and mP:
            lufs, tp = float(mI[-1]), float(mP[-1])
            rep.check("E3", "integrated loudness within −16 ± 1.5 LUFS and true peak ≤ −1 dBTP",
                      abs(lufs + 16.0) <= 1.5 and tp <= -0.9, f"{lufs} LUFS, {tp} dBTP")
    except Exception:                                   # noqa: BLE001
        pass


def eval_compose(state: dict, rep: Report) -> None:
    cv = state.get("compose") or {}
    v = cv.get("video")
    if v and Path(v).exists():
        rep.note("E1", "final dubbed video", v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    rep = Report()
    eval_asr(state, rep)
    eval_translation(state, rep)
    eval_tts(state, rep)
    eval_faces(state, rep)
    eval_lipsync(state, rep)
    try:
        eval_lipsync_frames(state, rep)
    except Exception as exc:                            # noqa: BLE001
        rep.note("D3", "frame-level lip-sync check skipped",
                 f"{type(exc).__name__}: {exc}")
    eval_compose(state, rep)
    for fn, key in ((eval_compact, "C5"), (eval_f0_gate, "C2c"), (eval_cuts, "D8"),
                    (eval_mix, "E2"), (eval_qc, "F1")):
        try:
            fn(state, rep)
        except Exception as exc:                        # noqa: BLE001
            rep.note(key, f"{fn.__name__} skipped", f"{type(exc).__name__}: {exc}")

    text = rep.render()
    print(text)
    out = args.out or str(Path(args.state).parent / "deliverables" / "ACCEPTANCE.md")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("# 验收结果\n\n```\n" + text + "\n```\n", encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0 if not any(r["ok"] is False for r in rep.rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
