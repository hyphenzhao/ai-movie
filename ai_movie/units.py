"""Sentence units: translate what was said, not how the segmenter cut it.

The segmenter caps a segment at ``ASR_MAX_SEGMENT_CHARS`` (24) characters,
so one spoken sentence often lands in two or three segments.  Translating
each fragment on its own turns 「かんなちゃんが思うおちんちん」+「の大きさ」
into two complete — and wrong — Chinese sentences.  Measured against the
burned-in subtitles of output_test, 20 of 87 sentences were split this way.

Everything downstream (TTS, compact, fit, mix, faces, lip-sync, QC, the web
editor) indexes segments by position, so the fix must not change the
segment count.  Instead a *unit* groups adjacent segments that belong to one
utterance, the unit is translated once, and its Chinese is split back onto
the member segments.

The flag rules at the bottom are shared by the translation polish pass and
``scripts/eval_against_subs.py`` so the metric and the fix can never drift.
"""

from __future__ import annotations

import re

from ai_movie.config import (
    ASR_SENTENCE_END,
    UNIT_CONTINUOUS_GAP,
    UNIT_MAX_CHARS,
    UNIT_MAX_DUR,
    UNIT_MAX_GAP,
)

_ZH_BREAK = "。！？!?…，、,；;：:"
_ZH_END = "。！？!?…"
_ASCII_WORD = re.compile(r"[0-9A-Za-z０-９Ａ-Ｚａ-ｚ.．%％]")
_KANA = re.compile(r"[぀-ゟ゠-ヿ]")
_PUNCT = re.compile(r"[、。，．！？!?…·・\-—～~「」『』（）()\s　,.]")


def visible_len(text: str | None) -> int:
    return len(_PUNCT.sub("", text or ""))


# ── grouping ────────────────────────────────────────────────────────

_CONNECTIVE = re.compile(
    r"(?:けども|けれど|けど|から|ので|のに|たら|って|ながら|し|て|で|と|が|ば|[、，,])$")


def joins(prev: dict, seg: dict, *,
          continuous_gap: float = UNIT_CONTINUOUS_GAP,
          max_gap: float = UNIT_MAX_GAP) -> bool:
    """Does *seg* continue the utterance that *prev* left unfinished?

    Never across sentence-final punctuation.  Otherwise either:

    * **no pause at all** (gap < ``continuous_gap``): the segmenter's
      character cap cut through running speech, often mid-word
      (「20セ|ンチ」「だった|ら」).  The speaker label is ignored when one
      side is a ≤2-character scrap, because diarization mislabels those; or
    * a short pause (gap < ``max_gap``) after a clause connective
      (けど/から/て/…, or a comma): the grammar says the sentence goes on.
    """
    text_prev = (prev.get("text") or "").rstrip()
    if not text_prev or text_prev[-1] in ASR_SENTENCE_END:
        return False
    gap = float(seg["start"]) - float(prev["end"])
    same = prev.get("speaker") == seg.get("speaker")
    scrap = min(visible_len(text_prev), visible_len(seg.get("text"))) <= 2
    if gap < continuous_gap and (same or scrap):
        return True
    return same and gap < max_gap and bool(_CONNECTIVE.search(text_prev))


def group_units(segments: list[dict], *,
                continuous_gap: float = UNIT_CONTINUOUS_GAP,
                max_gap: float = UNIT_MAX_GAP,
                max_dur: float = UNIT_MAX_DUR,
                max_chars: int = UNIT_MAX_CHARS) -> list[list[int]]:
    """Return runs of segment indices that form one utterance each (see
    :func:`joins`), capped at ``max_dur`` seconds and ``max_chars``."""
    units: list[list[int]] = []
    for i, seg in enumerate(segments):
        if units:
            cur = units[-1]
            joined_chars = sum(visible_len(segments[j].get("text")) for j in cur) \
                + visible_len(seg.get("text"))
            if (joins(segments[cur[-1]], seg,
                      continuous_gap=continuous_gap, max_gap=max_gap)
                    and float(seg["end"]) - float(segments[cur[0]]["start"]) <= max_dur
                    and joined_chars <= max_chars):
                cur.append(i)
                continue
        units.append([i])
    return units


# ── splitting a unit's translation back onto its segments ──────────

def _word_bounds(zh: str) -> set[int]:
    """Character offsets that fall between Chinese words.

    A proportional cut lands mid-word surprisingly often — the Sakura A/B
    produced 「假设有一根小 | 鸡鸡…」 and 「确实啊顶 | 到了深处」, and each
    half is then synthesised as its own TTS line.  jieba is optional; without
    it every offset outside a digit/Latin run counts as a boundary.
    """
    try:
        import logging
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import jieba
        jieba.setLogLevel(logging.WARNING)
        bounds, pos = set(), 0
        for tok in jieba.cut(zh):
            pos += len(tok)
            bounds.add(pos)
        return bounds
    except Exception:                                   # noqa: BLE001
        return {i for i in range(1, len(zh))
                if not (_ASCII_WORD.match(zh[i - 1]) and _ASCII_WORD.match(zh[i]))}


def split_translation(zh: str, weights: list[int], *,
                      min_chars: int = 2) -> list[str] | None:
    """Split *zh* into ``len(weights)`` pieces proportional to *weights*.

    Each boundary goes to the nearest Chinese break character within ±30 %
    of the ideal position; failing that, to the nearest word boundary, so a
    word (and a number like 「20厘米」) is never cut in half.  Returns None
    when any piece would be shorter than *min_chars*, so the caller can fall
    back to translating the fragments individually.
    """
    n = len(weights)
    zh = (zh or "").strip()
    if n <= 1:
        return [zh]
    if len(zh) < n * min_chars:
        return None
    words = _word_bounds(zh)
    total = sum(max(w, 1) for w in weights)
    cuts: list[int] = []
    acc = 0
    lo_bound = 0
    for w in weights[:-1]:
        acc += max(w, 1)
        ideal = round(len(zh) * acc / total)
        window = max(1, round(len(zh) * 0.3 * max(w, 1) / total))
        best = None
        for d in range(0, window + 1):
            for pos in (ideal + d, ideal - d):
                # a cut at `pos` means zh[pos-1] ends the left piece
                if lo_bound < pos < len(zh) and zh[pos - 1] in _ZH_BREAK:
                    best = pos
                    break
            if best is not None:
                break
        if best is None:
            options = [b for b in words if lo_bound < b < len(zh)]
            if not options:
                return None
            best = min(options, key=lambda b: (abs(b - ideal), b))
        cuts.append(best)
        lo_bound = best
    pieces = []
    prev = 0
    for c in cuts + [len(zh)]:
        pieces.append(zh[prev:c].strip())
        prev = c
    if any(visible_len(p) < min_chars for p in pieces):
        return None
    return pieces


# ── flags shared by the polish pass and the evaluation ─────────────

# Second/third person only.  Supplying 我 for a dropped first-person subject
# is normal, correct Chinese; every wrong attribution found in v3.0.0 used
# 你/她/他 (「我可能喜欢上你了」「她说可以了」「不是这种感觉的她」).
_ZH_PRONOUN = re.compile(r"[你您他她]")
_JA_PRONOUN = re.compile(
    r"私|わたし|あたし|僕|ぼく|俺|おれ|自分|うち|あなた|あんた|君|きみ|お前|おまえ|"
    r"彼女|彼|かれ|かのじょ|我々|みんな|皆")
_JA_QUESTION = re.compile(r"[?？]\s*$|(?:か|の|かな|かい)\s*$")
_ZH_QUESTION = re.compile(r"[?？]|[吗么嘛]\s*[。…]?\s*$")


def has_kana(text: str | None) -> bool:
    return bool(_KANA.search(text or ""))


def unsupported_pronoun(ja: str | None, zh: str | None) -> bool:
    """The Chinese addresses or names a person the Japanese never mentions.

    Japanese drops subjects; a translator that has to guess fills in 你/她
    and is often wrong (「好きなのかも」→「我可能喜欢上你了」).  Flagging —
    not rejecting — keeps legitimate additions for the polish pass to judge.
    """
    return bool(_ZH_PRONOUN.search(zh or "")) and not _JA_PRONOUN.search(ja or "")


def question_mismatch(ja: str | None, zh: str | None) -> bool:
    ja_q = bool(_JA_QUESTION.search((ja or "").strip()))
    zh_q = bool(_ZH_QUESTION.search((zh or "").strip()))
    return ja_q != zh_q


def length_off(ja: str | None, zh: str | None, lo: float = 0.3, hi: float = 2.5) -> bool:
    a, b = visible_len(ja), visible_len(zh)
    if not a or not b:
        return bool(a) != bool(b)
    return not (lo <= b / a <= hi)


def flag_line(ja: str | None, zh: str | None, *, split_fallback: bool = False) -> list[str]:
    """Film-independent reasons a translated unit deserves a second look."""
    flags = []
    if unsupported_pronoun(ja, zh):
        flags.append("F1_pronoun")
    if question_mismatch(ja, zh):
        flags.append("F2_question")
    if has_kana(zh):
        flags.append("F3_kana")
    if length_off(ja, zh):
        flags.append("F4_length")
    if split_fallback:
        flags.append("F5_split_fallback")
    return flags


_EDIT_PUNCT = set("，。！？!?…、,.：:；;“”\"'～~ ")
_INSERTABLE = set("你您我他她它们咱的了吗呢吧啊嘛呀哦") | _EDIT_PUNCT
_DELETABLE = _INSERTABLE | set("说是")


def polish_edit_ok(draft: str, cand: str, flags: list[str]) -> bool:
    """Is *cand* an acceptable correction of *draft* for these *flags*?

    Character overlap alone let a dolphin-mixtral "fix" turn 电视剧 into 电影
    (5 of 6 characters shared).  For pronoun and question-tone flags the
    only legitimate edits are removing or swapping pronouns, particles and
    punctuation — plus deleting an invented 「她说」 — so anything touching
    another character is refused.  A kana flag needs a real translation of
    the leftover word, so it gets the looser overlap test instead.
    """
    import difflib

    draft, cand = (draft or "").strip(), (cand or "").strip()
    if not cand or cand == draft or has_kana(cand) or "\n" in cand:
        return False
    if "F3_kana" in flags:
        cjk = [c for c in cand if "一" <= c <= "鿿"]
        # a kana word collapses into fewer hanzi, so compare without the kana
        kept = max(len(_KANA.sub("", draft)), 1)
        if not cjk or not (0.6 <= len(cand) / kept <= 1.6 + len(draft) / kept):
            return False
        return sum(1 for c in cjk if c in set(draft)) / len(cjk) >= 0.5
    # Deleting can only ever remove something the Japanese did not have, so
    # any deletion is allowed; inserting or substituting is how a "fix"
    # invents content (电视剧 → 电影 deletes 视剧 *and* inserts 影), so only
    # function words may be inserted.
    sm = difflib.SequenceMatcher(None, draft, cand, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal" and any(c not in _INSERTABLE for c in cand[j1:j2]):
            return False
    # …but over-deleting leaves broken Chinese ("因为有她在背后支持着我" →
    # "因为有支持着"), so keep at least half the line and refuse a dangling tail.
    if len(cand) < 0.55 * len(draft):
        return False
    return cand.rstrip("。！？!?…，、,") [-1:] not in set("着地得把被和与在给对从让")


def sentence_ends(text: str | None) -> int:
    """Count sentence terminators, treating a trailing run as one."""
    return len(re.findall(r"[。！？!?…]+", text or ""))


# ── non-lexical vocalisations ───────────────────────────────────────
# Moans, sighs, laughs ("あっ", "んん…", "はぁはぁ", "ふふ").  They are dubbed like any other line, but
# there is no articulation worth re-drawing, so the face plan leaves those frames alone (no lip-sync,
# hence nothing for the enhance pass to restore).
_NONLEX_BASE = set("あいうえおんはひふへほ")
_NONLEX_WORDS = {"はい", "いいえ", "いえ", "いい", "ええ", "うん", "ううん", "おい", "あい", "へえ", "ほう",
                 "はあい", "いいえい", "おお", "ほほう", "あう", "いう", "おう", "おはよう"}
_SMALL = str.maketrans("ぁぃぅぇぉゃゅょゎ", "あいうえおやゆよわ")


def is_nonlexical(ja: str) -> bool:
    """True when *ja* is only interjection sounds (no word content)."""
    t = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in (ja or ""))
    t = t.translate(_SMALL)
    base = "".join(c for c in t if "ぁ" <= c <= "ゖ" or "一" <= c <= "鿿" or c.isalnum())
    base = base.replace("っ", "")
    if not base:
        return True
    if base in _NONLEX_WORDS:
        return False
    return all(c in _NONLEX_BASE for c in base)
