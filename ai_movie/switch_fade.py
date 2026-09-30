"""Crossfade at the switches between generated and original frames (v3.4, L5).

Why this module exists
----------------------
The lip-sync output switches between MuseTalk's generated mouth and the
untouched original at two kinds of boundary: the face plan's per-frame gate
(a hard profile, a tiny face, an unbound line — MuseTalk simply pastes the
original frame there) and the occlusion gate (``face_restore.
occlusion_gate_video`` reverts a sustained run of frames whose mouth is
covered or whose face S3FD lost).  Both were hard cuts: an open synthetic
mouth on frame *t* and the closed real one on *t+1* — measured on the
regression set, ~25–60 such pops per short film, most of them inside a
speech window and not on a shot cut, so nothing hides them.

The rule implemented here: **the fade lives only on the generated frames.**
A frame the plan or the gate says must be original stays 100 % original —
a generated mouth is never painted onto an occluder or a profile face, not
even at 20 %.  The generated mouth instead fades *in* over the first
``fade`` generated frames of a run and *out* over the last ``fade``.  The
blend is per frame between the original and the generated frame of the
SAME index, so no frame is added or dropped: the clip's frame count — and
therefore ``lip_sync._fit_clip_to_duration``'s exact ``-vframes`` — is
untouched (see the assembly-frame-loss note in the project memory: every
cut must stay frame-count exact).

The gate reads its clips in 600-frame chunks, and a frame's weight depends
on where its run ends, so :class:`FadeWriter` holds back the last
``2·fade`` frames of every chunk until the next chunk (or the end) settles
them; the result is provably identical to a single-chunk computation (see
tests/test_switch_fade.py, which checks it against random chunkings).

Numpy only — no torch, no cv2 — so the tests import nothing heavy and
``face_restore`` (which imports torch at module level) is not needed to
reason about the blend.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable, Sequence

import numpy as np


def switch_fade_weights(use_orig: Sequence[bool], fade: int,
                        cuts: Iterable[int] = ()) -> np.ndarray:
    """Per-frame weight of the ORIGINAL frame: 1 = original, 0 = generated.

    ``use_orig[i]`` is True where the frame must show the original.  For every
    maximal run of generated frames ``[i, j)`` the weight ramps down over the
    first ``f_in`` frames and up over the last ``f_out`` frames::

        w[i + t]     = (f_in  - t) / (f_in  + 1)      t < f_in     (4/5, 3/5, 2/5, 1/5 at fade=4)
        w[j - 1 - t] = (f_out - t) / (f_out + 1)      t < f_out

    with ``f_in = min(fade, L // 2)`` and ``f_out = min(fade, L - f_in)`` so
    the two ramps never overlap (``f_in + f_out ≤ L``); a run shorter than
    ``2·fade`` is low-passed rather than flickered, which is wanted.  A
    boundary index listed in *cuts* is a shot change (or a clip edge the
    caller knows continues generated): the switch is already invisible
    there, so that side gets no ramp.  Original frames always keep weight 1.
    ``fade <= 0`` returns the plain 0/1 vector.  ``len(w) == len(use_orig)``
    always.
    """
    w = np.asarray([1.0 if bool(x) else 0.0 for x in use_orig], dtype=np.float32)
    n = len(w)
    fade = int(fade)
    if fade <= 0 or n == 0:
        return w
    cut_set = {int(c) for c in cuts}
    i = 0
    while i < n:
        if w[i] != 0.0:
            i += 1
            continue
        j = i
        while j < n and w[j] == 0.0:
            j += 1
        L = j - i
        f_in = 0 if i in cut_set else min(fade, L // 2)
        f_out = 0 if j in cut_set else min(fade, L - f_in)
        for t in range(f_in):
            w[i + t] = (f_in - t) / (f_in + 1)
        for t in range(f_out):
            w[j - 1 - t] = (f_out - t) / (f_out + 1)
        i = j
    return w


def blend_frames(orig: np.ndarray, gen: np.ndarray, w: float) -> np.ndarray:
    """``w·orig + (1−w)·gen`` for uint8 frames; the inputs are returned
    unchanged (no copy) at the endpoints so unblended frames stay bit-exact."""
    if w >= 1.0:
        return orig
    if w <= 0.0:
        return gen
    out = orig.astype(np.float32) * np.float32(w) + gen.astype(np.float32) * np.float32(1.0 - w)
    return np.rint(out).astype(np.uint8)


def frame_step(a: np.ndarray, b: np.ndarray, box=None) -> float:
    """Mean |a − b| (0–255) inside ``box = (y1, y2, x1, x2)``, or over the
    whole frame when *box* is None — the size of the visual jump between two
    consecutive frames, used to record what each switch looked like."""
    if box is not None:
        y1, y2, x1, x2 = [int(v) for v in box]
        y1, x1 = max(0, y1), max(0, x1)
        y2, x2 = min(a.shape[0], y2), min(a.shape[1], x2)
        if y2 > y1 and x2 > x1:
            a = a[y1:y2, x1:x2]
            b = b[y1:y2, x1:x2]
    if a.shape != b.shape or a.size == 0:
        return 0.0
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


class SwitchLog:
    """Records every generated↔original switch as the frames are written.

    Fed in order with ``observe(idx, out, item)`` where ``item`` is the
    ``(orig, gen, use_orig, meta)`` tuple the writer blended and ``out`` the
    frame it wrote; ``meta`` may carry ``box`` (``(y1, y2, x1, x2)`` for the
    step measurement) and ``reason`` (``"plan"`` / ``"no_face"`` / ``"lip"``
    for original frames).  A switch is recorded at the first frame of the
    new side: ``dir="in"`` when the generated mouth starts (its ramp is the
    next ``fade`` frames, so the record is finalised later) and ``"out"``
    when it ends (its ramp is the previous ``fade`` frames, read back).

    Per switch: ``step_hard`` — mean |Δ| the hard cut would have shown
    (generated frame against the previous original, or vice versa);
    ``step_faded`` — the largest step actually written across the ramp;
    ``step_src`` — the largest step of the original footage across the same
    frames, i.e. the motion floor.  Bookkeeping only: ``step_faded``
    includes source motion, so it is not comparable to ``step_hard`` on a
    moving shot (the reviewer's eyes decide; ``04_switches.csv`` tells them
    where to look).
    """

    def __init__(self, fade: int):
        self.fade = max(0, int(fade))
        self.switches: list[dict] = []
        self._open: list[dict] = []
        self._recent: deque = deque(maxlen=self.fade + 1)
        self._prev: dict = {}

    def observe(self, idx: int, out: np.ndarray, item: tuple) -> None:
        orig, gen, flag, meta = item
        flag = bool(flag)
        meta = meta or {}
        box = meta.get("box")
        prev = self._prev
        if prev:
            step_written = frame_step(out, prev["out"], box)
            step_src = frame_step(orig, prev["orig"], box)
            self._recent.append((step_written, step_src))
            if flag != prev["flag"]:
                if flag:                        # generated → original
                    ramp = list(self._recent)
                    self.switches.append({
                        "frame": idx, "dir": "out",
                        "kind": "plan" if meta.get("reason") == "plan" else "occlusion",
                        "fade": self.fade,
                        "step_hard": round(frame_step(orig, prev["gen"], box), 2),
                        "step_faded": round(max(s for s, _ in ramp), 2),
                        "step_src": round(max(s for _, s in ramp), 2)})
                else:                           # original → generated
                    rec = {"frame": idx, "dir": "in",
                           "kind": "plan" if prev.get("reason") == "plan" else "occlusion",
                           "fade": self.fade,
                           "step_hard": round(frame_step(gen, prev["orig"], box), 2),
                           "step_faded": step_written, "step_src": step_src,
                           "_until": idx + self.fade}
                    self.switches.append(rec)
                    self._open.append(rec)
            for rec in list(self._open):
                if idx > rec["frame"]:
                    rec["step_faded"] = max(rec["step_faded"], step_written)
                    rec["step_src"] = max(rec["step_src"], step_src)
                if idx >= rec["_until"]:
                    self._open.remove(rec)
        self._prev = {"out": out, "orig": orig, "gen": gen, "flag": flag,
                      "reason": meta.get("reason")}

    def records(self) -> list[dict]:
        """The switches, finalised (open ramps closed, floats rounded)."""
        out = []
        for rec in self.switches:
            r = {k: v for k, v in rec.items() if k != "_until"}
            r["step_faded"] = round(float(r["step_faded"]), 2)
            r["step_src"] = round(float(r["step_src"]), 2)
            out.append(r)
        return out


class FadeWriter:
    """Chunk-by-chunk blender that reproduces the single-pass weights exactly.

    ``push(items, last)`` takes ``(orig, gen, use_orig, meta)`` tuples for the
    next frames; ``sink(index, out_frame, weight, item)`` is called once per
    frame, in order, with the clip-local index.  A fade-out depends on the
    end of the generated run, which may lie in the next chunk, so the last
    ``hold = 2·fade`` frames of a push are kept back until the next push (or
    ``last=True``).  The weights of the frames that are written are final:
    the run start is visible through the ``prefix`` of already-written flags
    (also ``2·fade`` long), and a run that is still open at the end of the
    known frames is extended by ``hold`` frames — the "run continues"
    assumption — which cannot change any written frame's weight because a
    frame more than ``2·fade`` from the end of the known flags is never in
    a fade-out and, if it is in a fade-in, its run is already ≥ ``2·fade``
    long so ``f_in`` is saturated.  ``written`` equals the number of frames
    pushed once ``last=True`` has been seen; the count never changes.
    """

    def __init__(self, fade: int, cuts: Iterable[int] = (),
                 sink: Callable[[int, np.ndarray, float, tuple], None] | None = None):
        self.fade = max(0, int(fade))
        self.hold = 2 * self.fade
        self.cuts = {int(c) for c in cuts}
        self.sink = sink
        self._held: list[tuple] = []
        self._prefix: deque = deque(maxlen=max(1, self.hold))
        self.written = 0
        self.faded = 0
        self.weights: list[float] = []      # per written frame (kept for stats/tests)

    def push(self, items: Iterable[tuple], last: bool = False) -> int:
        """Feed the next frames; returns how many frames were written now."""
        pending = self._held + list(items)
        m = len(pending)
        if m == 0:
            self._held = []
            return 0
        flags = [bool(x[2]) for x in pending]
        prefix = list(self._prefix) if self.hold else []
        ext = [] if last else [flags[-1]] * self.hold
        offset = self.written - len(prefix)             # clip index of prefix[0]
        span = len(prefix) + m + len(ext)
        local_cuts = {c - offset for c in self.cuts if 0 <= c - offset <= span}
        w_all = switch_fade_weights(prefix + flags + ext, self.fade, local_cuts)
        w = w_all[len(prefix):len(prefix) + m]
        n_write = m if last else max(0, m - self.hold)
        for t in range(n_write):
            orig, gen, flag, _meta = pending[t]
            wt = float(w[t])
            out = blend_frames(orig, gen, wt)
            if 0.0 < wt < 1.0:
                self.faded += 1
            self.weights.append(wt)
            if self.sink is not None:
                self.sink(self.written, out, wt, pending[t])
            self.written += 1
            if self.hold:
                self._prefix.append(bool(flag))
        self._held = [] if last else pending[n_write:]
        return n_write

    def flush(self) -> int:
        """Write everything still held (end of clip)."""
        return self.push([], last=True)
