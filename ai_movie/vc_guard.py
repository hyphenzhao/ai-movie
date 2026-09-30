"""Per-line safety check on voice-converted audio.

A reference clip can pass the probe gate (two clean built-in lines convert
to the right pitch) and still wreck real lines: on the first long film the
film-wide female reference left 7 of 10 long lines with no measurable
pitch at all — husky, half-voiced output that reads as a man — and the
rest an octave down in one chunk.  No reference selection can rule that
out in advance, so every converted line is measured against the built-in
line it came from and falls back to that line when the conversion lost the
voice:

* voicing: the converted line keeps fewer than ``VC_GUARD_MIN_VOICED_KEEP``
  of the built-in line's confidently-voiced frames (per second, so a slightly
  longer output does not hide a loss);
* band: the converted pitch is outside the speaker's gender band;
* jump: the converted / built-in pitch ratio is outside ``VC_GUARD_RATIO``
  (an octave collapse is 0.5).

Pure measurement; the caller decides what to do with the verdicts.
"""

from __future__ import annotations

from pathlib import Path

from ai_movie.config import VC_GUARD_MIN_BASE_FRAMES, VC_GUARD_MIN_VOICED_KEEP, VC_GUARD_RATIO
from ai_movie.pitch import f0_median, gender_band


def _dur(path: str) -> float:
    import soundfile as sf
    try:
        return max(1e-3, float(sf.info(path).duration))
    except Exception:                                   # noqa: BLE001
        return 1e-3


def judge_line(conv_wav: str, v1_wav: str, gender: str | None) -> dict:
    """``{"ok": bool, "reason": str, "f0_conv", "f0_v1", "voiced_conv", "voiced_v1"}``."""
    f_c, n_c = f0_median(conv_wav)
    f_v, n_v = f0_median(v1_wav)
    rate_c, rate_v = n_c / _dur(conv_wav), n_v / _dur(v1_wav)
    res = {"ok": True, "reason": "", "f0_conv": f_c and round(f_c, 1), "f0_v1": f_v and round(f_v, 1),
           "voiced_conv": n_c, "voiced_v1": n_v, "judged": n_v >= VC_GUARD_MIN_BASE_FRAMES}
    # A baseline with a handful of confidently-voiced frames (short or soft built-in lines, most male
    # lines) cannot judge anything: n_c vs n_v is noise there and produced random fallbacks mid-sentence.
    if n_v < VC_GUARD_MIN_BASE_FRAMES:
        return res
    if rate_c < VC_GUARD_MIN_VOICED_KEEP * rate_v:
        res.update(ok=False, reason=f"lost_voicing_{rate_c / rate_v:.2f}")
        return res
    if f_c is None and f_v is not None:
        res.update(ok=False, reason="unvoiced_output")
        return res
    band = gender_band(gender)
    if f_c is not None and band and not (band[0] <= f_c <= band[1]):
        res.update(ok=False, reason=f"band_{gender}_{f_c:.0f}Hz")
        return res
    if f_c is not None and f_v:
        r = f_c / f_v
        if not (VC_GUARD_RATIO[0] <= r <= VC_GUARD_RATIO[1]):
            res.update(ok=False, reason=f"pitch_jump_{r:.2f}")
    return res


def guard_lines(segs: list[dict], items: dict, v1_segs: list[dict], *, log=None) -> dict:
    """Judge every converted line; returns ``{"checked", "rejected", "reasons", "verdicts": {i: …}}``.

    *items* is ``run_vc_conversion``'s result; a rejected line's entry is
    rewritten in place to the built-in audio with ``vc=False`` and
    ``guard=<reason>`` so the caller's bookkeeping needs no change.
    """
    checked = rejected = 0
    reasons: dict[str, int] = {}
    verdicts: dict[int, dict] = {}
    bad: set[int] = set()
    for i, s in enumerate(segs):
        it = items.get(i) or {}
        if not it.get("vc") or not it.get("audio"):
            continue
        v1 = v1_segs[i] if i < len(v1_segs) else {}
        v1_wav = v1.get("audio_fit") or v1.get("audio")
        if not v1_wav or not Path(v1_wav).exists():
            continue
        try:
            v = judge_line(it["audio"], v1_wav, s.get("gender") or s.get("tts_gender"))
        except Exception as exc:                        # noqa: BLE001  (unreadable wav → treat as lost)
            v = {"ok": False, "reason": f"unreadable_{type(exc).__name__}", "judged": True}
        verdicts[i] = v
        if not v.get("judged"):
            continue
        checked += 1
        if not v["ok"]:
            bad.add(i)
            key = v["reason"].split("_")[0]
            reasons[key] = reasons.get(key, 0) + 1
    # A chunk (short neighbouring lines converted as one utterance, tts._vc_chunks) is kept or dropped
    # whole: a built-in line spliced between two cloned ones inside a sentence is the "several people
    # talking" effect chunking exists to prevent.
    drop: set[int] = set()
    for i in bad:
        members = (items.get(i) or {}).get("chunk")
        drop.update(range(members[0], members[1] + 1) if members else [i])
    for i in sorted(drop):
        it = items.get(i) or {}
        if not it.get("vc"):
            continue
        v1_wav = (v1_segs[i].get("audio_fit") or v1_segs[i].get("audio")) if i < len(v1_segs) else None
        if not v1_wav:
            continue
        rejected += int(i in bad)
        reason = verdicts.get(i, {}).get("reason") or "chunk_member"
        items[i] = {"audio": v1_wav, "mode": "builtin", "vc": False, "guard": reason}
    if log:
        log(f"  VC guard: {rejected}/{checked} judged lines failed → {len(drop)} lines back to the built-in voice"
            + (f" ({', '.join(f'{k}×{n}' for k, n in sorted(reasons.items(), key=lambda kv: -kv[1]))})" if reasons else ""))
    return {"checked": checked, "rejected": rejected, "dropped": len(drop), "reasons": reasons, "verdicts": verdicts}
