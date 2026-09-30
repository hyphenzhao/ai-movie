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

Pitch cannot see a line whose *words* changed.  Paired verify_dub readings of
the v3.3 release (417 sampled lines, 299 converted) found 43 converted lines
(14 %) whose Whisper reading resembles neither the built-in line's reading
nor the intended text — language flips (「不妙啊」→ "Tchau, tchau", 「没有吗？」
→ "Mijn oma") and word drift (「把腿伸直」→「把腿先举」, 「射出来射出来」→
「秀出來」) — while built-in lines re-read by the same Whisper score 1.00
(p5 = 1.00), so the loss is in the conversion, not the metric.  The content
judge (:func:`judge_content`) therefore re-reads both wavs and scores the
conversion by folded similarity to max(v1 reading, intended text):

* hard rules on any length: ``script_flip`` (the conversion decodes to no CJK
  while the built-in line did, and a forced-Chinese re-decode does not rescue
  it) and ``kana_leak`` (≥ ``VC_DRIFT_KANA_LEAK`` kana where v1 had none);
* ``drift_<score>`` when the score is under ``VC_DRIFT_MIN_SIM``, judged only
  when the intended text has ≥ ``VC_DRIFT_MIN_CHARS`` folded characters and
  Whisper heard the built-in line itself (baseline ≥ ``VC_DRIFT_BASELINE_MIN``).

Both verdicts merge into one per-line result, so the caller's drop rule
(chunks go whole) and the retry rule (reject share over
``VC_GUARD_MAX_REJECT`` → next reference clip) need no second path.  Pure
measurement; the caller decides what to do with the verdicts.
"""

from __future__ import annotations

import difflib
import re
import sys
import unicodedata
from pathlib import Path

from ai_movie.config import (VC_DRIFT_BASELINE_MIN, VC_DRIFT_KANA_LEAK, VC_DRIFT_MIN_CHARS,
                             VC_DRIFT_MIN_SIM, VC_GUARD_MIN_BASE_FRAMES, VC_GUARD_MIN_VOICED_KEEP,
                             VC_GUARD_RATIO)
from ai_movie.pitch import f0_median, gender_band

KANA_RE = re.compile(r"[぀-ゟ゠-ヿ]")
CJK_RE = re.compile(r"[一-鿿]")
_NONWORD_RE = re.compile(r"[^\w]|_")

_T2S = None          # lazy OpenCC("t2s"); False once the import failed (identity fold, warned once)


def _t2s(text: str) -> str:
    global _T2S
    if _T2S is None:
        try:
            from opencc import OpenCC
            _T2S = OpenCC("t2s")
        except Exception as exc:                        # noqa: BLE001
            print(f"[vc_guard] opencc unavailable ({type(exc).__name__}): traditional characters "
                  "are not folded, similarity of a 「小雞雞」/「小鸡鸡」 pair drops to ~0.3", file=sys.stderr)
            _T2S = False
    return _T2S.convert(text) if _T2S else text


def fold_zh(text: str | None) -> str:
    """Fold a Whisper reading (or intended text) to comparable characters.

    1. traditional → simplified: Whisper writes traditional characters for
       converted timbres (「小雞雞」 vs 「小鸡鸡」 scored 0.29 unfolded);
    2. NFKC + lowercase: full-width digits/Latin → ASCII;
    3. katakana → hiragana (as ``content.fold``), so a kana leak compares on
       one script;
    4. every non-word character goes: punctuation, spaces, 「」…——.
    """
    t = _t2s(text or "")
    t = unicodedata.normalize("NFKC", t).lower()
    t = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in t)
    return _NONWORD_RE.sub("", t)


def text_sim(a: str | None, b: str | None) -> float:
    """SequenceMatcher ratio on folded text; 0 when either side folds to nothing.

    Same family as ``content.classify`` (CONTENT_AGREE_MIN) and
    ``eval_against_subs.ratio`` so the numbers compare across the repo."""
    fa, fb = fold_zh(a), fold_zh(b)
    if not fa or not fb:
        return 0.0
    return difflib.SequenceMatcher(None, fa, fb, autojunk=False).ratio()


def _readings(r) -> tuple[str, str | None]:
    """(auto reading, forced-zh reading or None) from a transcript entry.

    An entry is either a plain string (one reading) or ``{"auto": {"text", …},
    "zh": {"text", …} | None}`` as :class:`ai_movie.asr.WhisperClips` returns."""
    if r is None:
        return "", None
    if isinstance(r, str):
        return r, None
    auto = r.get("auto")
    zh = r.get("zh")
    a = (auto.get("text") if isinstance(auto, dict) else auto) or ""
    z = (zh.get("text") if isinstance(zh, dict) else zh) if zh is not None else None
    return a, (z if z else None)


def judge_content(conv, v1, want: str | None, *, min_chars: int = VC_DRIFT_MIN_CHARS,
                  baseline_min: float = VC_DRIFT_BASELINE_MIN, min_sim: float = VC_DRIFT_MIN_SIM,
                  kana_leak: int = VC_DRIFT_KANA_LEAK) -> dict:
    """Judge one converted unit by its Whisper reading (pure; no audio here).

    *conv* / *v1* are transcript entries (see :func:`_readings`) of the
    converted wav and of the built-in wav it was converted from; *want* is the
    intended text.  Returns ``{"judged", "ok", "reason", "score", "baseline",
    "heard_conv", "heard_v1"}``.

    Decode regime mirrors ``scripts/verify_dub.py`` (auto language, forced-zh
    re-decode when auto ≠ zh) so the thresholds calibrated on its rows
    transfer: a forced-zh decode alone would turn every Portuguese output
    into phonetic Chinese and ``script_flip`` would never fire.

    Rules, in order:

    1. not judged when no reading of the built-in line contains CJK — Whisper
       could not hear the *built-in* line (moans, 「哥哥……」 heard as "Фига!"),
       so it cannot judge its conversion (same principle as
       ``VC_GUARD_MIN_BASE_FRAMES``);
    2. ``script_flip`` (any length): the auto reading of the conversion has
       no CJK and the forced-zh reading, if any, scores under *min_sim* —
       catches "Shut up"/"Bye-bye" on 2-char lines, which are exactly the
       sub-0.7 s lines where every catastrophic failure was measured;
    3. ``kana_leak`` (any length): ≥ *kana_leak* kana in the conversion's
       auto reading, none in the built-in reading, not rescued by forced-zh;
    4. ``drift_<score>``: score = max over the conversion's readings of
       max(sim to v1 reading, sim to intended text) — anchoring on the v1
       reading calibrates Whisper's own error on TTS audio out, the intended
       text removes the false rejections where Whisper mis-heard v1 but
       heard the conversion (p02 #5); judged only when the intended text has
       ≥ *min_chars* folded characters and baseline = sim(v1 reading, want)
       ≥ *baseline_min*;
    5. otherwise ok — ``judged`` is True when any rule could be evaluated:
       the hard rules need a non-empty conversion reading (so a 2-char line
       whose conversion reads as CJK counts as judged and passed — the flip
       test was applied), the drift rule needs its two gates; a line whose
       conversion decodes to nothing and whose drift gates are not met is
       not judged.
    """
    want = want or ""
    v1_auto, v1_zh = _readings(v1)
    c_auto, c_zh = _readings(conv)
    # ``gate`` says why the drift rule was not applied: no_v1_cjk / short / baseline / None (applied).
    res = {"judged": False, "ok": True, "reason": "", "score": None, "baseline": None,
           "heard_conv": c_auto, "heard_v1": v1_auto, "gate": None}
    # The built-in line's forced-zh reading only counts when it reads the intended text (verify_dub's
    # rule); otherwise a v1 heard as "Back off" would pass rule 1 on the strength of phonetic Chinese.
    v1_texts = [t for t in (v1_auto,) if t]
    if v1_zh and text_sim(v1_zh, want) >= min_sim:
        v1_texts.append(v1_zh)
    if not any(CJK_RE.search(t) for t in v1_texts):
        res["gate"] = "no_v1_cjk"
        return res                                      # rule 1: built-in line not heard as Chinese
    baseline, heard_v1 = max(((text_sim(t, want), t) for t in v1_texts), key=lambda x: x[0])   # ties → auto reading
    res.update(baseline=round(baseline, 3), heard_v1=heard_v1)

    def _score(t: str | None) -> float:
        return max(text_sim(t, heard_v1), text_sim(t, want)) if t else 0.0

    s_auto, s_zh = _score(c_auto), (_score(c_zh) if c_zh else None)
    score = max(s_auto, s_zh or 0.0)
    res["score"] = round(score, 3)
    rescued = s_zh is not None and s_zh >= min_sim      # verify_dub's rule: forced zh reads the line → audio is Chinese
    if fold_zh(c_auto):
        res["judged"] = True
        if not CJK_RE.search(c_auto) and not rescued:
            res.update(ok=False, reason="script_flip")   # rule 2
            return res
        if (len(KANA_RE.findall(c_auto)) >= kana_leak and not any(KANA_RE.search(t) for t in v1_texts)
                and not rescued):
            res.update(ok=False, reason="kana_leak")     # rule 3
            return res
    if len(fold_zh(want)) < min_chars:
        res["gate"] = "short"
    elif baseline < baseline_min:
        res["gate"] = "baseline"
    else:
        res["judged"] = True
        if score < min_sim:
            res.update(ok=False, reason=f"drift_{score:.2f}")   # rule 4 (an empty reading scores 0.00)
    return res


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


def _v1_source(v1: dict, source_key: str) -> str | None:
    """The v1 wav the conversion took its content from — like is judged against like."""
    return v1.get(source_key) or v1.get("audio_fit") or v1.get("audio")


def _drift_units(segs: list[dict], items: dict, v1_segs: list[dict], *, source_key: str) -> list[dict]:
    """One unit per converted line (its piece vs the v1 line) plus one per chunk
    (the converted chunk vs the concatenated built-in source).

    Both levels are judged: the catastrophic flips are the sub-0.7 s lines,
    which are exactly the ones converted inside a chunk, and a chunk decode
    whose longer mate is intact would score ≥ 0.5 and hide them."""
    units: list[dict] = []
    chunks: dict[tuple[int, int], dict] = {}
    for i, s in enumerate(segs):
        it = items.get(i) or {}
        if not it.get("vc") or not it.get("audio"):
            continue
        v1 = v1_segs[i] if i < len(v1_segs) else {}
        v1_wav = _v1_source(v1, source_key)
        want = s.get("text_translated") or ""
        if v1_wav and Path(v1_wav).exists():
            units.append({"kind": "line", "idx": i, "conv": it["audio"], "v1": v1_wav, "want": want})
        ch = it.get("chunk")
        if (ch and it.get("chunk_src") and it.get("chunk_out")
                and Path(it["chunk_src"]).exists() and Path(it["chunk_out"]).exists()):
            key = (int(ch[0]), int(ch[1]))
            u = chunks.setdefault(key, {"kind": "chunk", "members": [], "conv": it["chunk_out"],
                                        "v1": it["chunk_src"], "wants": []})
            u["members"].append(i)
            u["wants"].append(want)
    for u in chunks.values():
        u["want"] = "。".join(u.pop("wants"))
        units.append(u)
    return units


def guard_lines(segs: list[dict], items: dict, v1_segs: list[dict], *, source_key: str = "audio_fit",
                log=None, transcribe=None, drift: dict | None = None) -> dict:
    """Judge every converted line; returns ``{"checked", "rejected", "dropped", "reasons",
    "verdicts": {i: …}, "drift": {…}}``.

    *items* is ``run_vc_conversion``'s result; a rejected line's entry is
    rewritten in place to v1's fitted audio with ``vc=False`` and
    ``guard=<reason>`` so the caller's bookkeeping needs no change.
    *source_key* names the v1 wav that was converted (the pitch baseline and
    the content baseline are read from it; voiced-frames-per-second is
    tempo-invariant, so the verdict logic is the same for either key); the
    fallback is always v1's ``audio_fit`` — the exact fitted line, so a
    fallback can never move the timeline.

    *transcribe* — ``callable(paths) -> {path: reading}`` (see
    :class:`ai_movie.asr.WhisperClips`) — enables the content judge; ``None``
    reproduces the pitch-only behaviour.  *drift* overrides the judge's
    thresholds (keys min_chars / baseline_min / min_sim / kana_leak).
    """
    verdicts: dict[int, dict] = {}
    for i, s in enumerate(segs):
        it = items.get(i) or {}
        if not it.get("vc") or not it.get("audio"):
            continue
        v1 = v1_segs[i] if i < len(v1_segs) else {}
        v1_wav = _v1_source(v1, source_key)
        if not v1_wav or not Path(v1_wav).exists():
            continue
        try:
            v = judge_line(it["audio"], v1_wav, s.get("gender") or s.get("tts_gender"))
        except Exception as exc:                        # noqa: BLE001  (unreadable wav → treat as lost)
            v = {"ok": False, "reason": f"unreadable_{type(exc).__name__}", "judged": True}
        verdicts[i] = v

    drift_stats: dict = {"enabled": bool(transcribe), "units": 0, "judged": 0, "rejected": 0}
    if transcribe is not None:
        drift_stats.update(_merge_drift(segs, items, v1_segs, verdicts, transcribe=transcribe,
                                        source_key=source_key, thresholds=drift or {}))

    # Accounting after the merge, so a content rejection counts exactly like a pitch one.
    checked = rejected = 0
    reasons: dict[str, int] = {}
    bad: set[int] = set()
    for i, v in verdicts.items():
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
        items[i] = {"audio": v1_wav, "mode": "builtin", "vc": False, "guard": reason,
                    "source": it.get("source"), "source_dur": it.get("source_dur"), "chunk": it.get("chunk")}
    if log:
        log(f"  VC guard: {rejected}/{checked} judged lines failed → {len(drop)} lines back to the built-in voice"
            + (f" ({', '.join(f'{k}×{n}' for k, n in sorted(reasons.items(), key=lambda kv: -kv[1]))})" if reasons else "")
            + (f"; content judge: {drift_stats['rejected']}/{drift_stats['judged']} of {drift_stats['units']} units"
               + (f" in {drift_stats['seconds']:.0f} s" if drift_stats.get("seconds") is not None else "")
               if transcribe is not None else "")
            + (f"; content judge FAILED: {drift_stats['error']}" if drift_stats.get("error") else ""))
    return {"checked": checked, "rejected": rejected, "dropped": len(drop), "reasons": reasons,
            "verdicts": verdicts, "drift": drift_stats}


def _merge_drift(segs, items, v1_segs, verdicts, *, transcribe, source_key, thresholds) -> dict:
    """Run the content judge over every unit and fold its verdicts into *verdicts* in place.

    Per line: ``judged |= drift.judged``, ``ok &= drift.ok``; the pitch reason
    wins when both fail, then the line's own drift reason, then the chunk's
    (``…_chunk``).  ``v["drift"]`` / ``v["drift_chunk"]`` keep the scores for
    the report.  Any exception from the transcriber is recorded under
    ``error`` and leaves the pitch verdicts untouched — the judge must never
    fail the v2 build.
    """
    import time
    units = _drift_units(segs, items, v1_segs, source_key=source_key)
    out: dict = {"units": len(units), "judged": 0, "rejected": 0, "seconds": None}
    if not units:
        return out
    t0 = time.time()
    try:
        paths = sorted({p for u in units for p in (u["conv"], u["v1"])})
        texts = transcribe(paths) or {}
    except Exception as exc:                            # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["seconds"] = round(time.time() - t0, 1)
        return out
    out["seconds"] = round(time.time() - t0, 1)
    for u in units:
        d = judge_content(texts.get(u["conv"]), texts.get(u["v1"]), u["want"], **thresholds)
        if u["kind"] == "line":
            targets, tag, key = [u["idx"]], "", "drift"
        else:
            targets, tag, key = u["members"], "_chunk", "drift_chunk"
            d = dict(d, reason=(d["reason"] + tag) if d["reason"] else "")
        if d["judged"]:
            out["judged"] += 1
            out["rejected"] += int(not d["ok"])
        for i in targets:
            v = verdicts.get(i)
            if v is None:
                continue
            v[key] = {"judged": d["judged"], "ok": d["ok"], "reason": d["reason"], "score": d["score"],
                      "baseline": d["baseline"], "gate": d.get("gate"),
                      "heard_conv": d["heard_conv"], "heard_v1": d["heard_v1"]}
            if d["judged"]:
                v["judged"] = True
                if not d["ok"] and v["ok"]:
                    v["ok"] = False
                    v["reason"] = d["reason"]
    return out
