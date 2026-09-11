#!/usr/bin/env python
"""Cross-check every segment's speaker label with three independent signals.

Hand-sampling put diarization at 15/18 correct.  That was tolerable while
both speakers were cloned from their own voice, but v1 gives each *gender* a
different built-in voice, so a wrong label is now an audibly wrong voice —
and "说话人日志准确清晰" is one of v1's three demo points.

No single signal settles it, so this runs three that fail in different ways
and reports where they disagree:

  pitch    median F0 of the separated vocals over the segment, against the
           GMM cut the diarizer already derived.  Silent on segments with
           too few voiced frames — exactly the short off-mic questions.
  channel  log-mel mean/std through a classifier seeded from confident-pitch
           segments.  Keeps the close-lav vs off-mic difference that ECAPA
           deliberately discards (89% cross-validated on this recording).
  anchor   ECAPA similarity against the two hand-verified reference clips.
           Weak on channel, strong on voice identity — the complement of the
           other two.

Segments where all three agree are settled.  The rest are exported as short
wavs under ``review/`` so a human can decide by ear in a few minutes; this
script cannot listen and does not pretend to.

    python scripts/review_speakers.py workspace/output_test/state.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Below this gap the anchor signal is barely better than a coin flip; see the
# comment where it is applied.
ANCHOR_MIN_MARGIN = 0.20
# A pitch median this far (in log space) from the GMM cut is taken as settled;
# nearer than that and an octave error is plausible.
PITCH_MIN_RATIO = 1.20


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("state", help="workspace/<name>/state.json")
    ap.add_argument("--out", default=None,
                    help="deliverables dir (default: <workspace>/deliverables)")
    ap.add_argument("--clip-pad", type=float, default=0.15)
    ap.add_argument("--truth", default=None,
                    help="JSON of {labels: {index: gender}} confirmed by ear; "
                         "these override every signal and become classifier seeds")
    ap.add_argument("--apply", action="store_true",
                    help="write the accepted labels back into state.json")
    args = ap.parse_args()

    truth: dict[int, str] = {}
    if args.truth:
        blob = json.loads(Path(args.truth).read_text(encoding="utf-8"))
        truth = {int(k) - 1: v for k, v in (blob.get("labels") or {}).items()}
        print(f"ground truth supplied for {len(truth)} segments: "
              f"{sorted(v for v in set(truth.values()))}")

    from ai_movie import artifacts
    from ai_movie.diarize import (_load_mono16k, channel_features, pitch_track,
                                  similarity, unit_f0)

    state_path = Path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    work = state_path.parent
    deliver = Path(args.out) if args.out else work / "deliverables"
    deliver.mkdir(parents=True, exist_ok=True)
    review_dir = deliver / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    segs = state["asr"]["segments"]
    diar = state["asr"]["diarization"]
    cut = float(diar.get("cut_hz") or 165.0)
    refs = (state.get("tts") or {}).get("refs") or {}

    # Anchor per gender, from the reference clips the previous run verified.
    anchors: dict[str, str] = {}
    for spk, r in refs.items():
        g, path = r.get("gender"), r.get("ref_audio")
        if g and path and Path(path).exists():
            anchors[g] = path
    print(f"anchors: { {g: Path(p).name for g, p in anchors.items()} }")

    vocals_path = (state.get("separate") or {}).get("vocals")
    if not vocals_path or not Path(vocals_path).exists():
        print("no separated vocals — cannot run pitch/channel signals")
        return 1
    print(f"loading vocals: {vocals_path}")
    vocals = _load_mono16k(vocals_path)

    units = [(float(s["start"]), float(s["end"])) for s in segs]

    # ── signal 1: pitch ────────────────────────────────────────────────
    print("pitch tracking…")
    f0, ok, fps = pitch_track(vocals, cache_key=vocals_path)
    pitch_g: list[str | None] = []
    pitch_hz: list[float | None] = []
    for s, e in units:
        v = unit_f0(f0, ok, fps, s, e)
        pitch_hz.append(v)
        pitch_g.append(None if v is None else ("female" if v >= cut else "male"))
    n_pitch = sum(1 for g in pitch_g if g)
    print(f"  measurable on {n_pitch}/{len(units)} segments (cut {cut:.1f} Hz)")

    # ── signal 2: channel classifier ───────────────────────────────────
    print("channel classifier…")
    X, keep = channel_features(vocals, units)
    pos = {u: k for k, u in enumerate(keep)}
    # Seed only from segments whose pitch is clear of the boundary, so the
    # classifier is not trained on the very cases it must decide.
    seeds = [(i, g) for i, (g, hz) in enumerate(zip(pitch_g, pitch_hz))
             if g and hz and abs(np.log(hz) - np.log(cut)) > np.log(1.15)
             and i in pos and i not in truth]
    # Ear-confirmed labels are the best seeds available, and they land exactly
    # where pitch fails: the male interviewer's questions, whose rising
    # intonation pushes F0 into the female range (one confirmed case read
    # 205 Hz). They also relieve the male/female seed imbalance that made the
    # classifier lean female.
    seeds += [(i, g) for i, g in truth.items() if i in pos]
    tr_m = [pos[i] for i, g in seeds if g == "male"]
    tr_f = [pos[i] for i, g in seeds if g == "female"]
    print(f"  seeds: male={len(tr_m)} female={len(tr_f)}")

    chan_g: list[str | None] = [None] * len(units)
    chan_p: list[float | None] = [None] * len(units)
    cv = 0.0
    chan_min_margin = 1.0            # abstain unless calibration earns better
    # 3 seeds a side is enough to *try* — the per-class calibration gate
    # below decides whether the result is actually trustworthy.  On material
    # where one speaker rarely gets a clean pitch reading (an off-mic
    # interviewer), demanding 6 male seeds silently disabled the one signal
    # that could label his unmeasurable short questions.
    if len(tr_m) >= 3 and len(tr_f) >= 3:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        Xs = np.vstack([X[tr_m], X[tr_f]])
        ys = np.array([0] * len(tr_m) + [1] * len(tr_f))
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=0.3, max_iter=2000,
                                               class_weight="balanced"))
        folds = min(5, len(tr_m), len(tr_f))
        cv = float(cross_val_score(clf, Xs, ys, cv=folds).mean())
        print(f"  cross-validated accuracy {cv:.3f}")

        # Calibrate the confidence threshold on OUT-OF-FOLD predictions.
        # Scoring the classifier on the seeds it was fitted to would report
        # ~100% and set the bar far too low.  The gate is PER-CLASS: with a
        # 24-vs-3 seed imbalance, overall accuracy is fooled by a classifier
        # that just answers "female" — both classes must clear the bar, and
        # the minority class must actually appear in the band.
        oof = cross_val_predict(clf, Xs, ys, cv=folds, method="predict_proba")[:, 1]
        print("  out-of-fold per-class accuracy by confidence band:")
        for t in (0.0, 0.1, 0.2, 0.3, 0.4):
            m = np.abs(oof - 0.5) >= t
            pred = (oof >= 0.5).astype(int)
            accs, ns = [], []
            for cls in (0, 1):
                sel = m & (ys == cls)
                ns.append(int(sel.sum()))
                accs.append(float((pred[sel] == cls).mean()) if sel.any() else 0.0)
            if sum(ns) < 6:
                continue
            print(f"    |p-0.5|>={t:.1f}: male {accs[0]:.2f} (n={ns[0]})  "
                  f"female {accs[1]:.2f} (n={ns[1]})")
            if (min(accs) >= 0.9 and min(ns) >= 2
                    and chan_min_margin > t):
                chan_min_margin = t
        print(f"  → trusting the channel signal at |p-0.5| >= {chan_min_margin:.1f}")

        clf.fit(Xs, ys)
        pf = clf.predict_proba(X)[:, 1]
        for i in range(len(units)):
            if i in pos:
                p = float(pf[pos[i]])
                chan_p[i] = p
                if abs(p - 0.5) >= chan_min_margin:
                    chan_g[i] = "female" if p >= 0.5 else "male"
    else:
        print("  not enough seeds — channel signal unavailable")

    # ── signal 3: ECAPA against the anchors ────────────────────────────
    print("anchor similarity…")
    import soundfile as sf

    clip_dir = work / "review_clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    anc_g: list[str | None] = []
    anc_margin: list[float | None] = []
    clips: list[str | None] = []
    for i, (s, e) in enumerate(units):
        a = max(0, int((s - args.clip_pad) * 16000))
        b = min(len(vocals), int((e + args.clip_pad) * 16000))
        if b - a < int(0.5 * 16000) or len(anchors) < 2:
            anc_g.append(None)
            anc_margin.append(None)
            clips.append(None)
            continue
        clip = clip_dir / f"seg_{i + 1:04d}.wav"
        sf.write(str(clip), vocals[a:b], 16000)
        clips.append(str(clip))
        sims = {g: similarity(str(clip), p) for g, p in anchors.items()}
        best = max(sims, key=sims.get)
        other = min(sims, key=sims.get)
        margin = sims[best] - sims[other]
        # ECAPA is trained to be channel-invariant, which is precisely the cue
        # separating these two speakers, so it abstains unless the gap is wide.
        # Measured against confident pitch: 0.72 accuracy overall, 0.88 at
        # margin>=0.15, 1.00 at >=0.20 (but only 9/128 segments get that far).
        anc_g.append(best if margin >= ANCHOR_MIN_MARGIN else None)
        anc_margin.append(round(margin, 3))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(units)}")

    # ── combine ────────────────────────────────────────────────────────
    # Precedence, not majority: the three signals are not equally reliable, so
    # counting votes lets the weakest one outvote the strongest. Measured
    # against confident pitch on this recording, channel scores 0.94 and the
    # anchor 0.72 — a 2-1 "majority" of channel+anchor over pitch would be
    # wrong more often than pitch alone.
    def decide(i: int) -> tuple[str | None, str]:
        if i in truth:
            return truth[i], "ear"
        hz = pitch_hz[i]
        if hz and abs(np.log(hz) - np.log(cut)) >= np.log(PITCH_MIN_RATIO):
            return pitch_g[i], "pitch"
        if chan_g[i]:
            return chan_g[i], "channel"
        if pitch_g[i]:
            # A pitch reading this close to the cut is one octave error away
            # from meaning the opposite. Only act on it when the channel
            # classifier — even below its own confidence bar — is not leaning
            # the other way; otherwise the two weak signals cancel out.
            p = chan_p[i]
            if p is not None and (p >= 0.5) != (pitch_g[i] == "female"):
                return None, "contested"
            return pitch_g[i], "pitch(weak)"
        if anc_g[i]:
            return anc_g[i], "anchor"
        return None, "none"

    rows = []
    disputed = []
    for i, s in enumerate(segs):
        cur = s.get("gender") or s.get("tts_gender") or "female"
        votes = [g for g in (pitch_g[i], chan_g[i], anc_g[i]) if g]
        n_f = votes.count("female")
        n_m = votes.count("male")
        verdict, by = decide(i)
        majority = verdict or cur
        # "Settled" needs a decisive signal AND no dissent from the others.
        unanimous = bool(verdict) and n_f * n_m == 0
        rows.append({
            "decided_by": by,
            "index": i + 1,
            "start": round(float(s["start"]), 2),
            "end": round(float(s["end"]), 2),
            "speaker": s.get("speaker"),
            "current": cur,
            "pitch": pitch_g[i] or "",
            "pitch_hz": round(pitch_hz[i], 1) if pitch_hz[i] else "",
            "channel": chan_g[i] or "",
            "channel_p_female": round(chan_p[i], 3) if chan_p[i] is not None else "",
            "anchor": anc_g[i] or "",
            "anchor_margin": anc_margin[i] if anc_margin[i] is not None else "",
            "votes_f": n_f, "votes_m": n_m,
            "majority": majority,
            "unanimous": "yes" if unanimous else "no",
            "agrees_current": "yes" if majority == cur else "NO",
            "text": (s.get("text") or "")[:40],
        })
        if not unanimous or majority != cur:
            disputed.append((i, majority, cur, clips[i]))

    csv_path = deliver / "00_speaker_review.csv"
    artifacts.export_csv(rows, csv_path)

    settled = sum(1 for r in rows if r["unanimous"] == "yes"
                  and r["agrees_current"] == "yes")
    flips = [r for r in rows if r["agrees_current"] == "NO"]
    print("\n" + "=" * 70)
    print(f"segments                       : {len(rows)}")
    print(f"all three agree with the label : {settled}")
    print(f"needs a human ear              : {len(disputed)}")
    print(f"  of which the vote flips the current label: {len(flips)}")
    print(f"channel classifier cv accuracy : {cv:.3f}")
    print(f"\nwrote {csv_path}")

    # Export the disputed clips grouped by what the vote proposes, so the
    # listening pass is "do these all sound like one person?" rather than 30
    # unrelated decisions.
    # Only the flips are worth anyone's time: everything else keeps the label
    # it already had, so listening to it cannot change the output.
    manifest = []
    for i, majority, cur, clip in disputed:
        if not clip or majority == cur:
            continue
        dst = review_dir / f"seg{i + 1:04d}_{cur}_to_{majority}.wav"
        dst.write_bytes(Path(clip).read_bytes())
        manifest.append({
            "index": i + 1, "proposed": majority, "current": cur,
            "start": rows[i]["start"], "end": rows[i]["end"],
            "decided_by": rows[i]["decided_by"],
            "pitch_hz": rows[i]["pitch_hz"],
            "channel_p_female": rows[i]["channel_p_female"],
            "clip": dst.name, "text": rows[i]["text"],
            # Who is speaking either side of it — a question answered by the
            # next line is usually the interviewer's, and that context settles
            # cases no acoustic signal can.
            "prev": (f"{rows[i-1]['current']}: {rows[i-1]['text']}"
                     if i > 0 else ""),
            "next": (f"{rows[i+1]['current']}: {rows[i+1]['text']}"
                     if i + 1 < len(rows) else ""),
        })
    (review_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {len(manifest)} clips needing a decision to {review_dir}")
    for m in manifest:
        print(f"  #{m['index']:>3} {m['start']:>7}s {m['current']}→{m['proposed']} "
              f"({m['decided_by']})  {m['text'][:24]}")

    if args.apply:
        # Only ear-confirmed labels and decisive-signal flips are written; a
        # segment with no decisive signal keeps whatever it already had.
        changed = 0

        def fix_speaker(s: dict, g: str) -> None:
            """Point the segment at a speaker whose gender matches *g*."""
            spks = diar.setdefault("speakers", {})
            same_g = [spk for spk, meta in spks.items()
                      if meta.get("gender") == g]
            if len(same_g) == 1:
                s["speaker"] = same_g[0]
            elif not same_g:
                # Diarization merged this speaker away entirely (an off-mic
                # interviewer with too few voiced frames to form a pitch
                # mode).  Leaving the old speaker id would bind these
                # segments to the *other* speaker's face and drive that
                # mouth with the wrong-gender voice — so mint a speaker for
                # the orphan gender.  With no on-screen face of that gender,
                # face binding then correctly leaves the picture untouched.
                new_id = f"S{max((int(k[1:]) for k in spks
                                  if k.startswith('S') and k[1:].isdigit()),
                                 default=-1) + 1}"
                spks[new_id] = {"gender": g, "f0_median": None,
                                "total_speech": 0.0, "n_turns": 0,
                                "synthesized_by": "review_speakers orphan-gender"}
                s["speaker"] = new_id
                print(f"  minted speaker {new_id} ({g}) for orphan gender")
            # with 2+ same-gender speakers there is no basis to choose — keep

        for i, s in enumerate(segs):
            r = rows[i]
            if r["agrees_current"] != "NO" or r["decided_by"] in ("none", "contested"):
                continue
            g = r["majority"]
            s["gender"] = g
            s["tts_gender"] = g
            fix_speaker(s, g)
            changed += 1

        # Consistency repair, independent of this run's flips: a segment
        # whose gender doesn't match its speaker's recorded gender (e.g. a
        # flip applied on an earlier run, before orphan-gender minting
        # existed) binds to the wrong face downstream.
        repaired = 0
        for s in segs:
            spk_meta = (diar.get("speakers") or {}).get(s.get("speaker") or "")
            g = s.get("gender")
            if g and spk_meta and spk_meta.get("gender") != g:
                fix_speaker(s, g)
                repaired += 1
        if repaired:
            print(f"  repaired speaker id on {repaired} gender-mismatched segments")
        # Keep the translated copy in step with ASR, since TTS reads that one.
        by_time = {(round(float(s["start"]), 2), round(float(s["end"]), 2)): s
                   for s in segs}
        for key in ("translate", "tts", "fit"):
            for s in ((state.get(key) or {}).get("segments") or []):
                src = by_time.get((round(float(s["start"]), 2),
                                   round(float(s["end"]), 2)))
                if src:
                    s["gender"] = src["gender"]
                    s["tts_gender"] = src["tts_gender"]
                    s["speaker"] = src["speaker"]
        backup = state_path.with_suffix(".json.bak")
        backup.write_text(state_path.read_text(encoding="utf-8"), encoding="utf-8")
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        n_f = sum(1 for s in segs if s.get("gender") == "female")
        print(f"\napplied {changed} label changes to {state_path} "
              f"(backup: {backup.name})")
        print(f"  now {n_f} female / {len(segs) - n_f} male")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
