"""Switch crossfade (v3.4 L5): blend weights, chunk equivalence, frame-count
invariance, fade only on generated frames, switch bookkeeping; L8 counts in
qc.picture_counts.  Pure numpy — no video, no GPU, no torch.

    .venv/bin/python tests/test_switch_fade.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie.switch_fade import FadeWriter, SwitchLog, frame_step, switch_fade_weights  # noqa: E402

FADE = 4


def _partials(w):
    return [i for i, x in enumerate(w) if 0.0 < x < 1.0]


# ── switch_fade_weights ────────────────────────────────────────────

def test_length_is_always_preserved():
    for n in (0, 1, 2, 7, 30):
        for pat in (lambda i: True, lambda i: False, lambda i: i % 3 == 0, lambda i: i < n // 2):
            flags = [pat(i) for i in range(n)]
            for fade in (0, 1, 3, 4, 5):
                w = switch_fade_weights(flags, fade)
                assert len(w) == n, (n, fade)


def test_single_boundary_fades_only_the_generated_side():
    # 10 original frames then 20 generated: exactly 4 partial frames, all on
    # the generated side, 4/5 → 1/5 monotone; the original side stays at 1.
    flags = [True] * 10 + [False] * 20
    w = switch_fade_weights(flags, FADE, cuts={30})       # no fade-out at the clip end
    assert list(w[:10]) == [1.0] * 10
    assert _partials(w) == [10, 11, 12, 13]
    assert np.allclose(w[10:14], [4 / 5, 3 / 5, 2 / 5, 1 / 5])
    assert all(a > b for a, b in zip(w[10:14], w[11:15]))
    assert list(w[14:]) == [0.0] * 16
    # and the mirror: generated then original → ramp up at the end of the run
    flags = [False] * 20 + [True] * 10
    w = switch_fade_weights(flags, FADE, cuts={0})
    assert _partials(w) == [16, 17, 18, 19]
    assert np.allclose(w[16:20], [1 / 5, 2 / 5, 3 / 5, 4 / 5])
    assert list(w[20:]) == [1.0] * 10


def test_cut_at_the_boundary_means_no_ramp():
    flags = [True] * 10 + [False] * 20 + [True] * 5
    w = switch_fade_weights(flags, FADE, cuts={10, 30})
    assert _partials(w) == []
    w = switch_fade_weights(flags, FADE, cuts={10})
    assert _partials(w) == [26, 27, 28, 29]
    w = switch_fade_weights(flags, FADE, cuts={30})
    assert _partials(w) == [10, 11, 12, 13]


def test_short_runs_never_overlap_ramps():
    for L in (1, 2, 3, 5, 8, 9):
        flags = [True] * 5 + [False] * L + [True] * 5
        w = switch_fade_weights(flags, FADE)
        assert w.min() >= 0.0 and w.max() <= 1.0
        assert list(w[:5]) == [1.0] * 5 and list(w[5 + L:]) == [1.0] * 5
        f_in = min(FADE, L // 2)
        f_out = min(FADE, L - f_in)
        assert f_in + f_out <= L
        assert len(_partials(w)) == f_in + f_out, (L, list(w))
    # a lone generated frame is low-passed (50 %), not shown at full strength
    assert switch_fade_weights([True, False, True], FADE)[1] == 0.5


def test_fade_zero_is_identity():
    flags = [True, False, False, True, False, True, True, False]
    w = switch_fade_weights(flags, 0)
    assert list(w) == [1.0 if f else 0.0 for f in flags]


def test_all_generated_clip_fades_both_ends():
    w = switch_fade_weights([False] * 30, FADE)
    assert _partials(w) == [0, 1, 2, 3, 26, 27, 28, 29]
    w = switch_fade_weights([False] * 30, FADE, cuts={0, 30})      # neighbours continue generated
    assert _partials(w) == []


def test_original_frames_are_never_blended():
    rng = random.Random(7)
    for _ in range(200):
        n = rng.randint(1, 60)
        flags = [rng.random() < 0.4 for _ in range(n)]
        cuts = {rng.randint(0, n) for _ in range(rng.randint(0, 3))}
        w = switch_fade_weights(flags, rng.choice([1, 2, 3, 4, 5]), cuts)
        for f, x in zip(flags, w):
            if f:
                assert x == 1.0
        assert w.min() >= 0.0 and w.max() <= 1.0


# ── FadeWriter: chunked == single pass, count invariant ─────────────

def _items(flags, boxes=None, reasons=None, seed=0):
    """Synthetic 8×8 frames: original = 100, generated = 200 (uint8)."""
    out = []
    for i, f in enumerate(flags):
        orig = np.full((8, 8, 3), 100, np.uint8)
        gen = np.full((8, 8, 3), 200, np.uint8)
        meta = {"box": boxes[i] if boxes else None,
                "reason": (reasons[i] if reasons else ("plan" if f else None))}
        out.append((orig, gen, f, meta))
    return out


def _run(flags, fade, cuts=(), chunks=None):
    frames, weights = [], []
    fw = FadeWriter(fade, cuts, lambda idx, out, w, item: (frames.append(out.copy()), weights.append(w)))
    items = _items(flags)
    if chunks is None:
        fw.push(items, last=True)
    else:
        pos = 0
        for c in chunks:
            fw.push(items[pos:pos + c])
            pos += c
        fw.push(items[pos:])
        fw.flush()
    assert fw.written == len(flags)
    return frames, weights


def test_chunked_writer_matches_single_pass():
    flags = [True] * 6 + [False] * 20 + [True] * 7 + [False] * 3 + [True] * 2 + [False] * 15
    ref_frames, ref_w = _run(flags, FADE)
    # boundaries inside a generated run (12), inside a fade ramp (27 → run
    # [26, 29) is 3 long), and one chunk shorter than the hold-back (2)
    for chunks in ([12, 15, 2], [1] * 50, [27, 5], [2, 2, 2, 2], [25]):
        frames, w = _run(flags, FADE, chunks=chunks)
        assert len(frames) == len(flags) == len(ref_frames)
        assert w == ref_w, chunks
        assert all(np.array_equal(a, b) for a, b in zip(frames, ref_frames)), chunks


def test_chunked_writer_matches_single_pass_random():
    rng = random.Random(3)
    for _ in range(150):
        n = rng.randint(1, 70)
        flags = [rng.random() < 0.45 for _ in range(n)]
        fade = rng.choice([0, 1, 2, 3, 4, 5])
        cuts = {rng.randint(0, n) for _ in range(rng.randint(0, 3))}
        ref_frames, ref_w = _run(flags, fade, cuts)
        assert np.allclose(ref_w, switch_fade_weights(flags, fade, cuts))
        chunks, left = [], n
        while left > 0:
            c = rng.randint(1, max(1, min(left, 12)))
            chunks.append(c)
            left -= c
        frames, w = _run(flags, fade, cuts, chunks=chunks[:-1])
        assert w == ref_w, (flags, fade, cuts, chunks)
        assert all(np.array_equal(a, b) for a, b in zip(frames, ref_frames))


def test_blend_values_and_max_step():
    # orig 100 / gen 200 → a hard switch steps 100; the ramp steps ≤ 100/(fade+1) (+1 for rounding)
    flags = [True] * 6 + [False] * 20 + [True] * 6
    frames, w = _run(flags, FADE)
    vals = [int(f[0, 0, 0]) for f in frames]
    assert vals[:6] == [100] * 6 and vals[-6:] == [100] * 6
    assert vals[6:10] == [120, 140, 160, 180] and vals[10:22] == [200] * 12
    assert vals[22:26] == [180, 160, 140, 120]
    steps = [abs(a - b) for a, b in zip(vals, vals[1:])]
    assert max(steps) <= 100 // (FADE + 1) + 1
    # unblended frames are the input objects themselves (bit-exact, no copy)
    fw = FadeWriter(FADE, (), None)
    items = _items([True, False])
    fw.push(items, last=True)
    assert fw.weights == [1.0, 0.5]
    assert fw.faded == 1


# ── SwitchLog: reasons, kinds, step metrics ────────────────────────

def test_switch_log_records_kind_dir_and_steps():
    flags = [True] * 6 + [False] * 12 + [True] * 6 + [False] * 10 + [True] * 4
    reasons = ["plan"] * 6 + [None] * 12 + ["lip"] * 6 + [None] * 10 + ["no_face"] * 4
    log = SwitchLog(FADE)
    frames = []
    fw = FadeWriter(FADE, (), lambda idx, out, w, item: (frames.append(out), log.observe(idx, out, item)))
    fw.push(_items(flags, reasons=reasons), last=True)
    recs = log.records()
    assert [(r["frame"], r["dir"], r["kind"]) for r in recs] == [
        (6, "in", "plan"), (18, "out", "occlusion"), (24, "in", "occlusion"), (34, "out", "occlusion")]
    for r in recs:
        assert r["fade"] == FADE
        assert r["step_hard"] == 100.0             # what the hard cut would have shown
        assert r["step_faded"] <= 100 / (FADE + 1) + 1
        assert r["step_src"] == 0.0                # static synthetic source
        assert "_until" not in r
    # fade 0 → the written step IS the hard step
    log0 = SwitchLog(0)
    fw0 = FadeWriter(0, (), lambda idx, out, w, item: log0.observe(idx, out, item))
    fw0.push(_items(flags, reasons=reasons), last=True)
    assert all(r["step_faded"] == 100.0 for r in log0.records())


def test_frame_step_box():
    a = np.zeros((10, 10, 3), np.uint8)
    b = a.copy()
    b[2:4, 2:4] = 255
    assert abs(frame_step(a, b) - 255 * 4 / 100) < 1e-4
    assert frame_step(a, b, (2, 4, 2, 4)) == 255.0
    assert frame_step(a, b, (5, 9, 5, 9)) == 0.0
    assert abs(frame_step(a, b, (-3, 4, -3, 4)) - 255 * 4 / 16) < 1e-4      # clipped to the frame


# ── qc.picture_counts (L8 step 1, film level) ──────────────────────

def test_qc_picture_counts_from_state():
    from ai_movie import qc
    state = {
        "faces": {"passthrough": {"frames": {"unbound": 30, "yaw": 10}, "speech_frames": 200,
                                  "anchored_in_windows": 160, "segments": {}}},
        "lipsync": {"clips": 3, "gated_frames": 300, "anchored_frames": 250, "reverted_frames": 40,
                    "use_orig_by_reason": {"plan": 50, "no_face": 30, "lip": 10},
                    "occluded_sporadic": 4, "fade_frames": 4, "faded_frames": 32, "switches": 5,
                    "passthrough_clips": 1, "passthrough_clip_frames": 60,
                    "switch_list": [{"kind": "plan", "dir": "in"}, {"kind": "plan", "dir": "out"},
                                    {"kind": "occlusion", "dir": "in"}]},
    }
    pic = qc.picture_counts(state)
    assert pic["plan"] == {"speech_window_frames": 200, "anchored": 160, "passthrough": 40,
                           "passthrough_frac": 0.2, "by_reason": {"unbound": 30, "yaw": 10}}
    assert pic["gate"]["by_reason"] == {"plan": 50, "no_face": 30, "lip": 10}
    assert pic["gate"]["switch_kinds"] == {"plan_in": 1, "plan_out": 1, "occlusion_in": 1}
    assert pic["gate"]["switches"] == 5 and pic["gate"]["faded_frames"] == 32
    # a v3.3 state: reverted counts only, no reasons — still summarised, never raises
    old = {"lipsync": {"reverted_frames": 7, "per_clip": [
        {"occlusion": {"frames": 100, "reverted_frames": 7}}, {"occlusion": {"frames": 50}}]}}
    pic = qc.picture_counts(old)
    assert pic["gate"]["by_reason"] == {"occlusion": 7} and pic["gate"]["gated_frames"] == 150
    assert "plan" not in pic
    assert qc.picture_counts({}) == {}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
