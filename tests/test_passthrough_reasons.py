"""faces.passthrough_reasons (v3.4 L8 step 1): every non-anchored frame inside
a speech window gets exactly one reason, and the counts add up.  Synthetic
plan only (no video, no detector); the live workspace plans are only used
for the invariant, never for pinned numbers (they change on every re-run).

    .venv/bin/python tests/test_passthrough_reasons.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie import faces  # noqa: E402

FPS = 30.0
N = 900                     # 30 s
DET = 5
PAD = int(round(0.7 * FPS))     # 21 frames, as build_face_plan


def _track(tid, first, last, box, yaw=0.0, gaps=(), yaw_runs=()):
    """Keyframes every DET frames; ``gaps`` = (a, b) spans with no keyframe;
    ``yaw_runs`` = (a, b, deg) spans with that yaw."""
    kfs, yaws = {}, {}
    for f in range(first, last + 1, DET):
        if any(a <= f <= b for a, b in gaps):
            continue
        kfs[f] = list(box)
        y = yaw
        for a, b, deg in yaw_runs:
            if a <= f <= b:
                y = deg
        yaws[f] = y
    return {"id": tid, "keyframes": kfs, "yaw": yaws, "first": first, "last": last,
            "n": len(kfs), "gender": "female", "conf": 0.9}


def _synthetic():
    # track 0: on screen frames 0–600, big frontal face; a yaw excursion 300–360;
    #          a long keyframe gap 450–520 (> DET·(MAX_GAP+1)) → track_gap
    # track 1: small face (width 30 < 40) 600–899
    # track 2: frames 700–760 then a cut at 730 between keyframes → "cut" for interpolation
    t0 = _track(0, 0, 600, [100, 100, 300, 300], gaps=[(451, 519)], yaw_runs=[(300, 360, 70.0)])
    t1 = _track(1, 600, 899, [500, 100, 530, 130])
    t2 = _track(2, 700, 760, [800, 100, 1000, 300])
    # make the two keyframes around 730 disjoint so interpolation refuses (a cut)
    t2["keyframes"] = {725: [800, 100, 1000, 300], 735: [10, 10, 210, 210],
                       **{k: v for k, v in t2["keyframes"].items() if k not in (725, 730, 735)}}
    tracks = [t0, t1, t2]
    segments = [
        {"start": 1.0, "end": 3.0, "speaker": "S0", "text": "こんにちは"},              # anchored
        {"start": 4.0, "end": 5.0, "speaker": "S1", "text": "はい"},                    # unbound
        {"start": 10.0, "end": 12.0, "speaker": "S0", "text": "そうですね"},            # yaw run inside
        {"start": 15.5, "end": 17.0, "speaker": "S0", "text": "ええと"},                # track gap
        {"start": 21.0, "end": 22.0, "speaker": "S2", "text": "うん", "content": "nonlexical"},
        {"start": 24.0, "end": 25.0, "speaker": "S3", "text": "きた"},                  # size (track 1)
        {"start": 26.0, "end": 27.0, "speaker": "S0", "text": "終わり"},                # offscreen (track 0 ended)
    ]
    segments = faces.mark_no_lipsync(segments)
    plan_tracks = {"fps": FPS, "n_frames": N, "size": (1920, 1080), "det_every": DET, "tracks": tracks}
    return plan_tracks, segments


def _build_plan(plan_tracks, segments, cuts):
    """Reproduce build_face_plan's frame selection without detection/binding."""
    tracks = plan_tracks["tracks"]
    bindings = {"S0": 0, "S1": None, "S2": None, "S3": 1}
    per_track = {t["id"]: faces.interpolate_track(t, N, det_every=DET, cuts=cuts) for t in tracks}
    per_yaw = {t["id"]: faces.interpolate_scalar(t["yaw"], N, det_every=DET, cuts=cuts) for t in tracks}
    frames = {}
    for i, seg in enumerate(segments):
        if seg.get("no_lipsync"):
            continue
        tid = bindings.get(seg["speaker"])
        if tid is None:
            continue
        a = max(0, int(seg["start"] * FPS) - PAD)
        b = min(N, int(seg["end"] * FPS) + PAD)
        cand = {f: per_track[tid][f] for f in range(a, b + 1) if f in per_track[tid]}
        if not cand:
            continue
        gated, _sr = faces.gate_frames(list(cand), cand, per_yaw[tid], yaw_max=55.0,
                                       min_width=80.0, smooth=15, min_width_sr=40.0)
        for f, box in cand.items():
            if f not in gated:
                frames[f] = box
    return {"fps": FPS, "n_frames": N, "det_every": DET, "cuts": cuts,
            "speaker_track": bindings, "segment_track": {},
            "frames": {str(k): v for k, v in sorted(frames.items())},
            "gate": {"yaw_max": 55.0, "min_width": 80.0, "min_width_sr": 40.0, "smooth": 15},
            "_tracks_full": tracks}


def test_synthetic_plan_reasons():
    plan_tracks, segments = _synthetic()
    plan = _build_plan(plan_tracks, segments, cuts=[])
    r = faces.passthrough_reasons(plan, segments)
    tot = sum(r["frames"].values())
    assert r["speech_frames"] == r["anchored_in_windows"] + tot, r
    win = lambda s: (min(N, int(s["end"] * FPS) + PAD) - max(0, int(s["start"] * FPS) - PAD) + 1)  # noqa: E731
    assert r["speech_frames"] == sum(win(s) for s in segments)
    seg = r["segments"]
    assert "0" not in seg                                            # fully anchored
    # segment 1's window [99, 171] overlaps segment 0's [9, 111]; those 13
    # frames carry segment 0's box, and anchored always wins
    assert seg["1"] == {"unbound": win(segments[1]) - 13}
    assert set(seg["2"]) == {"yaw"} and seg["2"]["yaw"] > 0           # only the yaw run inside
    assert set(seg["3"]) == {"track_gap"} and seg["3"]["track_gap"] > 0
    assert seg["4"] == {"nonlexical": win(segments[4])}
    assert seg["5"] == {"size": win(segments[5])}                    # 30 px face → gated as size
    assert seg["6"] == {"offscreen": win(segments[6])}
    assert set(r["frames"]) == {"unbound", "yaw", "track_gap", "nonlexical", "size", "offscreen"}


def test_synthetic_plan_cut_reason():
    plan_tracks, segments = _synthetic()
    cuts = [730]
    segs = segments + [{"start": 23.6, "end": 24.9, "speaker": "S4", "text": "あの"}]
    plan = _build_plan(plan_tracks, segs, cuts)
    plan["speaker_track"]["S4"] = 2
    r = faces.passthrough_reasons(plan, segs)
    tot = sum(r["frames"].values())
    assert r["speech_frames"] == r["anchored_in_windows"] + tot
    last = r["segments"][str(len(segs) - 1)]
    assert last.get("cut", 0) > 0, last                              # 726–734: not interpolated across the cut
    assert last.get("offscreen", 0) > 0, last                        # window extends past the track's span


def test_missing_track_is_unbound_and_smooth_is_named():
    plan_tracks, segments = _synthetic()
    plan = _build_plan(plan_tracks, segments, cuts=[])
    plan["speaker_track"]["S0"] = 99                                 # dangling track id
    r = faces.passthrough_reasons(plan, segments)
    assert r["speech_frames"] == r["anchored_in_windows"] + sum(r["frames"].values())
    assert r["segments"]["2"] == {"unbound": r["segments"]["2"]["unbound"]}
    # a frame the median filter pulled in (yaw fine, width fine) is "smooth":
    # two short profile runs around a 7-frame frontal gap — the 15-frame
    # median gates the gap too, and those frames must not be blamed on yaw
    t = plan_tracks["tracks"][0]
    for f in range(300, 361, DET):
        t["yaw"][f] = 70.0 if f in (320, 325, 335, 340) else 0.0
    plan2 = _build_plan(plan_tracks, segments, cuts=[])
    r2 = faces.passthrough_reasons(plan2, segments)
    assert r2["speech_frames"] == r2["anchored_in_windows"] + sum(r2["frames"].values())
    assert r2["segments"]["2"] == {"yaw": 16, "smooth": 7}, r2["segments"]["2"]


def test_load_tracks_cache_plan_roundtrip():
    plan_tracks, _ = _synthetic()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "face_tracks_cache.json"
        faces._save_tracks_cache(p, "abc", plan_tracks)
        got = faces.load_tracks_cache_plan(p)
        assert got is not None and len(got["tracks"]) == 3
        assert all(isinstance(k, int) for t in got["tracks"] for k in t["keyframes"])
        assert faces._load_tracks_cache(p, "wrong-key") is None      # the keyed loader still refuses
        assert faces.load_tracks_cache_plan(Path(d) / "missing.json") is None


def test_mark_no_lipsync_rule():
    segs = [{"text": "あっ", "content": "nonlexical"}, {"text": "はい", "content": "speech"},
            {"text": "んん…"}, {"text": "こんにちは"}, {"text": "はい", "no_lipsync": True}]
    out = faces.mark_no_lipsync(segs)
    assert [bool(s.get("no_lipsync")) for s in out] == [True, False, True, False, True]
    assert segs[2].get("no_lipsync") is None                          # input untouched


def test_eval_recomputes_from_plan_when_state_has_no_counts():
    spec = importlib.util.spec_from_file_location("eval_pipeline", ROOT / "scripts" / "eval_pipeline.py")
    ep = importlib.util.module_from_spec(spec); spec.loader.exec_module(ep)     # type: ignore[union-attr]
    plan_tracks, segments = _synthetic()
    plan = _build_plan(plan_tracks, segments, cuts=[])
    with tempfile.TemporaryDirectory() as d:
        pp = Path(d) / "face_plan.json"
        pp.write_text(json.dumps({k: v for k, v in plan.items() if k != "_tracks_full"}))
        faces._save_tracks_cache(Path(d) / "face_tracks_cache.json", "k", plan_tracks)
        state = {"faces": {"plan_path": str(pp)},
                 "fit": {"segments": [dict(s) for s in segments]},
                 "lipsync": {"gated_frames": 100, "use_orig_by_reason": {"plan": 5, "no_face": 2, "lip": 1},
                             "occluded_sporadic": 1, "switches": 2, "fade_frames": 4, "faded_frames": 12,
                             "switch_list": [{"step_hard": 50.0, "step_faded": 12.0},
                                             {"step_hard": 40.0, "step_faded": 60.0}]}}
        rep = ep.Report()
        ep.eval_passthrough(state, rep)
    keys = {r["key"]: r for r in rep.rows}
    assert "D9a" in keys and "[recomputed" in keys["D9a"]["desc"]
    assert keys["D9a"]["ok"] is None                                  # notes, never gates
    assert "unbound" in keys["D9a"]["value"]
    assert "plan 5, no_face 2, lip 1" in keys["D9b"]["value"]
    assert keys["D9c"]["value"].startswith("0.87 over 2 switches, 1 with a larger")


def test_live_plans_hold_the_invariant():
    """Only the invariant on whatever plans exist — no pinned numbers."""
    ws = ROOT / "workspace"
    n = 0
    for film in ("output_test", "test_1", "test_2"):
        pp, cp, sp = ws / film / "face_plan.json", ws / film / "face_tracks_cache.json", ws / film / "state.json"
        if not (pp.exists() and cp.exists() and sp.exists()):
            continue
        plan = json.loads(pp.read_text(encoding="utf-8"))
        cache = faces.load_tracks_cache_plan(cp)
        state = json.loads(sp.read_text(encoding="utf-8"))
        segs = next((state[k]["segments"] for k in ("fit", "compact", "tts", "asr")
                     if (state.get(k) or {}).get("segments")), None)
        if cache is None or not segs:
            continue
        r = faces.passthrough_reasons(plan, faces.mark_no_lipsync(segs), tracks=cache["tracks"])
        assert r["speech_frames"] == r["anchored_in_windows"] + sum(r["frames"].values()), film
        assert set(r["frames"]) <= set(faces.PASSTHROUGH_REASONS)
        n += 1
    if not n:
        print("  (skipped: no workspace plans)")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
