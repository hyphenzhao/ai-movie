"""Decide what a transcribed segment *is* before anything downstream trusts it.

Whisper returns text for whatever it is given.  On quiet, breathy or
non-speech audio that text is often a stock phrase from its training data
(「ご清聴ありがとうございました」), Latin garbage ("inse Philaname") or an
onomatopoeia (「グーグー」) — and each of those was translated, synthesised
and lip-synced on the first long film.  Confidence does not separate them
from real lines: the user's rule is *content first* — a low-confidence real
line is dubbed, a meaningless vocalisation keeps the original voice, and a
hallucination is dropped.

``classify`` applies film-independent rules in order and returns one of

* ``"speech"``      — translate and dub as usual, whatever ``asr_conf`` says
* ``"nonlexical"``  — moans / laughs / sighs: keep the original audio, no
                      lip-sync (``keep_original`` + ``no_lipsync``)
* ``"drop"``        — not speech at all: remove the segment

Segments carry ``pass`` = ``"vad"`` (first Whisper pass over VAD spans) or
``"sweep"`` (second pass over the gaps).  Sweep segments have no VAD backing,
so the rules are stricter for them and a second decode of the same window on
the other audio source (``alt_text``) is the strongest evidence either way.
"""

from __future__ import annotations

import difflib
import re

from ai_movie.segmenter import _HALLUCINATION_PHRASES, _visible_len
from ai_movie.units import is_nonlexical

from ai_movie.config import (CONTENT_AGREE_MIN, CONTENT_CONFLICT_MAX, CONTENT_ENERGY_FLOOR_DBFS,   # noqa: E402
                             CONTENT_LOGPROB_DROP, CONTENT_MAX_CPS, CONTENT_NSP_DROP, CONTENT_REPEAT_DROP,
                             CONTENT_WEAK_CONF, CONTENT_WEAK_LOGPROB)

# Whole-line stock outputs (folded); the segmenter's substring list covers the
# YouTube boilerplate, these are the polite closings Whisper emits over silence.
_STOCK_LINES = {
    "ご清聴ありがとうございました", "ご静聴ありがとうございました",
    "ありがとうございました", "ありがとうございます", "どうもありがとうございました",
    "おやすみなさい", "お疲れ様でした", "お疲れさまでした", "失礼します", "失礼しました",
    "ご視聴ありがとうございました", "またね", "バイバイ", "ごちそうさまでした", "いただきます",
}
_PUNCT = r"[\s、。，,．.！!？?…‥・「」『』（）()～~♪♬\-—]"
_CJK = re.compile(r"[぀-ヿ一-鿿]")


def fold(text: str) -> str:
    t = re.sub(_PUNCT, "", text or "")
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in t)


def raw_hallucination(text: str, *, pass_: str = "vad", mean_prob: float = 0.0,
                      fold_only: bool = False) -> bool | str:
    """Whole-line stock output check on a *raw* Whisper segment, before the
    sentence re-split scatters 「ご視聴ありがとうございました」 across word
    boundaries and speaker turns (「ご視」｜「聴ありがとうございました」).
    With ``fold_only`` returns the folded text instead (for loop detection)."""
    ft = fold(text).rstrip("。")
    if fold_only:
        return ft
    if not ft:
        return False
    if any(p in text for p in _HALLUCINATION_PHRASES):
        return True
    if ft in _STOCK_LINES:
        return not (pass_ == "vad" and mean_prob >= 0.8)
    return False


def _repeat_unit(t: str) -> tuple[str, int]:
    """Smallest 1–2 char unit *t* is a pure repetition of, and its count."""
    for n in (1, 2):
        if len(t) >= 2 * n and len(t) % n == 0 and t == t[:n] * (len(t) // n):
            return t[:n], len(t) // n
    return t, 1


def classify(seg: dict, *, vocals_p95_db: float | None = None) -> dict:
    """Return ``{"content": ..., "reasons": [...]}`` for one ASR segment."""
    text = seg.get("text") or ""
    sweep = seg.get("pass") == "sweep"
    conf = seg.get("asr_conf")
    nsp = seg.get("no_speech_prob")
    alp = seg.get("avg_logprob")
    cr = seg.get("compression_ratio")
    dur = max(1e-3, float(seg.get("end", 0)) - float(seg.get("start", 0)))
    ft = fold(text)
    vis = _visible_len(text)
    reasons: list[str] = []

    def out(kind: str, why: str) -> dict:
        return {"content": kind, "reasons": reasons + [why]}

    # 7. nothing audible where the text supposedly was
    if vocals_p95_db is not None and vocals_p95_db < CONTENT_ENERGY_FLOOR_DBFS:
        return out("drop", f"energy {vocals_p95_db:.0f} dBFS")
    if not vis:
        return out("nonlexical", "no visible text")
    # 1. stock phrases — a real high-confidence "thank you" in an interview survives
    if any(p in text for p in _HALLUCINATION_PHRASES):
        return out("drop", "boilerplate")                 # YouTube closings are never said on set
    if ft.rstrip("。") in _STOCK_LINES:
        if sweep or conf is None or conf < 0.8:
            return out("drop", "stock phrase")
        reasons.append("stock phrase kept (asr_conf ≥ 0.8, vad pass)")
    # 2. no CJK at all: Latin garbage.  Digits / units alone are fine ("3cm")
    if not _CJK.search(text):
        if re.fullmatch(r"[\d０-９.,%％cmCMkgKG\s]+", text.strip() or "x"):
            return out("speech", "numeric")
        return out("drop", "no CJK")
    # 3. vocalisations before loops: a transcribed moan (「アーッ、アーッ…」「あーーーー」) has a high
    #    compression ratio too, and dropping it silenced 52 lines / 167 s of the first film instead of
    #    keeping the original voice
    unit, n = _repeat_unit(ft)
    if is_nonlexical(text) or (n >= 2 and is_nonlexical(unit)):
        return out("nonlexical", f"vocalisation ×{n}" if n >= 2 else "interjection only")
    if (cr is not None and cr > 2.4) or n >= CONTENT_REPEAT_DROP:
        return out("drop", f"repetition ×{n}" if n >= CONTENT_REPEAT_DROP else f"compression {cr:.2f}")
    # 4. impossible speaking rate
    if vis / dur > CONTENT_MAX_CPS and dur >= 0.3:
        return out("drop", f"{vis / dur:.0f} chars/s")
    # 5. sweep-only extreme scores
    if sweep and nsp is not None and alp is not None and nsp > CONTENT_NSP_DROP and alp < CONTENT_LOGPROB_DROP:
        return out("drop", f"nsp {nsp:.2f} / logprob {alp:.2f}")
    # 8. moans, sighs, laughs
    if is_nonlexical(text):
        return out("nonlexical", "interjection only")
    # 6. cross-decode evidence (sweep windows decoded on both sources)
    alt = seg.get("alt_text")
    if sweep and alt is not None:
        fa = fold(alt)
        if fa:
            sim = difflib.SequenceMatcher(None, ft, fa).ratio()
            if sim >= CONTENT_AGREE_MIN:
                reasons.append(f"confirmed by second decode ({sim:.2f})")
                return {"content": "speech", "reasons": reasons, "confirmed": True}
            # A conflicting second decode is a veto only when (a) it is itself believable — the vocals
            # decode is often the hallucination (「ご視聴ありがとうございました」 against a clean 0.94-confidence
            # line) — and (b) the line is weak on its own scores.  Measured on the short films: the bare
            # "decodes disagree" rule removed 12 real lines of test_2 and one of output_test.
            alt_junk = bool(raw_hallucination(alt, pass_="sweep")) or not _CJK.search(alt) or is_nonlexical(alt)
            weak = (conf is None or conf < CONTENT_WEAK_CONF) and (alp is None or alp < CONTENT_WEAK_LOGPROB)
            if sim < CONTENT_CONFLICT_MAX and weak and not alt_junk:
                return out("drop", f"decodes disagree ({sim:.2f}) and the line is weak")
        elif nsp is not None and nsp >= 0.5:
            return out("drop", "only one decode heard text, nsp ≥ 0.5")
    if sweep and vis <= 2 and (nsp or 0) > 0.5:
        return out("nonlexical", "sweep fragment")
    return out("speech", "content")


def classify_segments(segments: list[dict], *, vocals: str | None = None,
                      log=None) -> list[dict]:
    """Tag every segment (``content``, ``content_reasons``, ``keep_original``,
    ``no_lipsync``) and remove the drops.  *vocals* enables the energy rule."""
    p95 = None
    if vocals:
        import numpy as np
        from ai_movie.diarize import _load_mono16k
        y = _load_mono16k(vocals)
        frame = 320                                    # 20 ms
        n = len(y) // frame
        lv = 20 * np.log10(np.sqrt(np.mean(y[:n * frame].reshape(n, frame) ** 2, axis=1)) + 1e-9)

        def p95(a: float, b: float) -> float | None:
            i, j = int(a * 50), max(int(a * 50) + 1, int(b * 50))
            seg = lv[i:j]
            return float(np.percentile(seg, 95)) if seg.size else None
    # the same folded text in ≥ 3 segments within 30 s is a decoder loop, whatever each copy scores
    folded = [fold(s.get("text", "")) for s in segments]
    looped = set()
    for i, ft in enumerate(folded):
        if len(ft) < 4:
            continue
        same = [j for j in range(len(segments)) if folded[j] == ft and abs(float(segments[j]["start"]) - float(segments[i]["start"])) <= 30]
        if len(same) >= 3:
            looped.update(same)
    kept = []
    n_drop = n_nl = 0
    for i, s in enumerate(segments):
        e = p95(float(s["start"]), float(s["end"])) if p95 else None
        r = classify(s, vocals_p95_db=e)
        if i in looped and r["content"] != "nonlexical":
            r = {"content": "drop", "reasons": r["reasons"] + ["repeated line (loop)"]}
        if r["content"] == "drop":
            n_drop += 1
            if log:
                log(f"  content: dropped [{float(s['start']):.1f}s] {s.get('text', '')!r} — {r['reasons'][-1]}")
            continue
        s = dict(s, content=r["content"], content_reasons=r["reasons"])
        if r.get("confirmed"):
            s["sweep_confirmed"] = True
        if r["content"] == "nonlexical":
            s["keep_original"] = True
            s["no_lipsync"] = True
            n_nl += 1
        kept.append(s)
    if log and (n_drop or n_nl):
        log(f"  content: {n_drop} dropped, {n_nl} kept as original voice, {len(kept) - n_nl} to dub")
    return kept
