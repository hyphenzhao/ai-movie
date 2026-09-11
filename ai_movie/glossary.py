"""Terminology extraction and injection for translation.

The reference run rendered the performer's name カンナ as 「镰鼬」 (a yōkai) —
the translator had no idea it was a name, and because every segment was
translated in isolation it was free to render it differently each time.  A
glossary fixes both halves of that: it tells the model what the proper nouns
are, and it pins one rendering across the whole video.

Candidates are mined from the transcript with plain string work first (no
model), then a single LLM call proposes translations for the top-N.  A
user-editable seed file always wins over anything the model produced.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Callable

from ai_movie.config import (
    GLOSSARY_MAX_TERMS,
    GLOSSARY_MIN_COUNT,
    GLOSSARY_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_SAKURA_MODEL,
)

# Katakana runs are where foreign words, stage names and product names live.
_KATAKANA = re.compile(r"[ァ-ヴー]{2,}")
# Kanji/kana followed by an honorific is almost always a personal name.
_HONORIFIC = re.compile(r"([一-龥ぁ-んァ-ヴー]{2,5})(さん|ちゃん|くん|君|様)")
# Latin/alphanumeric tokens: brands, model names ("S1").
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9]{1,}")

# Grammatical katakana that is never a term worth pinning.
_STOP = {
    "ソウ", "コレ", "ソレ", "アレ", "ドレ", "ナニ", "デス", "マス", "ネエ",
    "ホント", "ホントウ", "チョット", "ヤッパリ", "ドウ", "アト", "ソウデス",
    "エッチ", "セックス",
}


def mine_candidates(segments: list[dict], *,
                    min_count: int = GLOSSARY_MIN_COUNT,
                    max_terms: int = GLOSSARY_MAX_TERMS) -> list[dict]:
    """Frequency-ranked term candidates with one example line each."""
    counts: dict[str, int] = {}
    example: dict[str, str] = {}

    def bump(term: str, line: str) -> None:
        term = term.strip()
        if len(term) < 2 or term in _STOP:
            return
        counts[term] = counts.get(term, 0) + 1
        example.setdefault(term, line)

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        for m in _HONORIFIC.finditer(text):
            bump(m.group(1), text)
        for m in _KATAKANA.finditer(text):
            bump(m.group(0), text)
        for m in _LATIN.finditer(text):
            bump(m.group(0), text)

    items = [{"ja": t, "count": c, "example": example[t]}
             for t, c in counts.items() if c >= min_count]
    # A name that appears with an honorific is worth pinning even once.
    for seg in segments:
        for m in _HONORIFIC.finditer((seg.get("text") or "")):
            t = m.group(1)
            if counts.get(t, 0) < min_count and \
                    not any(i["ja"] == t for i in items):
                items.append({"ja": t, "count": counts.get(t, 1),
                              "example": seg.get("text", "")})
    items.sort(key=lambda x: -x["count"])
    return items[:max_terms]


def load_seed(path: str | Path | None = None) -> dict[str, dict]:
    """User-maintained glossary; these entries override extraction."""
    p = Path(path or GLOSSARY_PATH)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:                            # noqa: BLE001
        print(f"[glossary] cannot read {p}: {exc}", file=sys.stderr)
        return {}
    out: dict[str, dict] = {}
    for k, v in (raw or {}).items():
        if isinstance(v, str):
            out[k] = {"zh": v, "kind": "seed", "source": "user"}
        elif isinstance(v, dict) and v.get("zh"):
            out[k] = {**v, "source": "user"}
    return out


def extract_glossary(
    segments: list[dict],
    *,
    model: str | None = None,
    base_url: str | None = None,
    max_terms: int = GLOSSARY_MAX_TERMS,
    min_count: int = GLOSSARY_MIN_COUNT,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, dict]:
    """Ask the LLM to translate the mined candidates.  One call, no retries."""
    from ai_movie.translator import _call_ollama_chat

    cands = mine_candidates(segments, min_count=min_count, max_terms=max_terms)
    if not cands:
        return {}

    model = model or OLLAMA_SAKURA_MODEL
    base_url = base_url or OLLAMA_BASE_URL

    listing = "\n".join(
        f"{i}. {c['ja']}（出现 {c['count']} 次）  例：{c['example'][:40]}"
        for i, c in enumerate(cands))
    system = (
        "你在为一段日语视频建立中文译名对照表。我会给出候选词及其出现的例句。\n"
        "对每个候选词判断它是什么，并给出**全片统一**的中文译法：\n"
        "  - 人名/艺名 → 音译（如 カンナ→坎娜），绝不要意译成其他词\n"
        "  - 品牌/厂牌/作品名 → 保留原样或通用中文译名\n"
        "  - 俚语/俗语 → 最口语的中文说法\n"
        "  - 普通词汇、语气词、无需固定的词 → 直接跳过，不要收进表里\n"
        "严格只输出一个 JSON 对象，形如 "
        '{"カンナ": {"zh": "坎娜", "kind": "name"}, ...}，不要任何解释。'
    )
    if progress_cb:
        progress_cb(f"术语抽取：{len(cands)} 个候选词")

    try:
        raw = _call_ollama_chat(
            model, [{"role": "system", "content": system},
                    {"role": "user", "content": listing}],
            base_url, timeout=900,
            options={"num_predict": 64 * max(8, len(cands)),
                     "temperature": 0.1})
    except Exception as exc:                            # noqa: BLE001
        print(f"[glossary] LLM unavailable ({exc}) — seed only", file=sys.stderr)
        return {}

    obj = _parse_json_object(raw)
    if not obj:
        print("[glossary] LLM output was not a usable JSON object",
              file=sys.stderr)
        return {}

    counts = {c["ja"]: c["count"] for c in cands}
    out: dict[str, dict] = {}
    for k, v in obj.items():
        if isinstance(v, str):
            v = {"zh": v}
        if not isinstance(v, dict) or not str(v.get("zh") or "").strip():
            continue
        out[k] = {"zh": str(v["zh"]).strip(),
                  "kind": v.get("kind") or "term",
                  "count": counts.get(k, 0), "source": "auto"}
    return out


def _parse_json_object(raw: str) -> dict | None:
    """Extract the outermost JSON object from a model reply."""
    raw = re.sub(r"```(?:json)?\s*", "", raw or "")
    raw = raw.replace("```", "").strip()
    a, b = raw.find("{"), raw.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        obj = json.loads(raw[a:b + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def build_glossary(
    segments: list[dict],
    *,
    model: str | None = None,
    base_url: str | None = None,
    seed_path: str | Path | None = None,
    auto: bool | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, dict]:
    """Seed file merged over auto-extraction (user entries win)."""
    from ai_movie.config import GLOSSARY_AUTO_EXTRACT

    if auto is None:
        auto = GLOSSARY_AUTO_EXTRACT

    terms: dict[str, dict] = {}
    if auto:
        try:
            terms.update(extract_glossary(segments, model=model,
                                          base_url=base_url,
                                          progress_cb=progress_cb))
        except Exception as exc:                        # noqa: BLE001
            print(f"[glossary] extraction failed: {exc}", file=sys.stderr)
    terms.update(load_seed(seed_path))

    if progress_cb:
        progress_cb(f"术语表：{len(terms)} 条")
    return terms


def format_for_prompt(glossary: dict[str, dict], texts: list[str]) -> str:
    """Render only the entries actually present in *texts*.

    Sending the whole table every batch wastes context and invites the model
    to drag in unrelated terms.
    """
    if not glossary:
        return ""
    blob = "\n".join(texts)
    hits = [(k, v) for k, v in glossary.items() if k and k in blob]
    if not hits:
        return ""
    return "；".join(f"{k}={v['zh']}" for k, v in hits)


_HONORIFIC_SUFFIX = "(?:さん|ちゃん|くん|君|様)?"
_HIRAGANA = "\u3041-\u309f"
_HIRAGANA_SET = {chr(c) for c in range(0x3041, 0x30a0)}


def protect_terms(text: str,
                  glossary: dict[str, dict]) -> tuple[str, dict[str, str]]:
    """Replace pinned terms with placeholders that survive translation.

    Three approaches were measured on the reference interview, where the
    performer's name came back as 坎娜 / 小卡娜 / 小香菜 / 小勘娜 across four
    lines:

    * Instructing the model ("固定译名：かんな=坎娜") — ignored; Sakura is a
      completion-style model and does not follow side instructions.
    * Rewriting the source to already contain the Chinese
      (「S1専属セット坎娜です」) — the model *re-transliterates* it, producing
      卡娜.
    * Replacing the term with a bracketed marker 「【N1】」 — passes through
      verbatim, so the pinned rendering can be restored afterwards.  (Bare
      Latin markers do not survive: ``NAME1`` came back as 「姓名1」 and
      ``Xq1`` as ``XQ1``.)

    Returns ``(protected_text, {placeholder: chinese})``.

    Matching is honorific-aware and refuses to fire inside a longer kana word:
    「かんな」 is a name in 「かんなちゃん」 but not in 「わかんない」 ("don't
    know"), where substituting would corrupt the sentence.
    """
    if not glossary or not text:
        return text, {}
    out = text
    mapping: dict[str, str] = {}
    n = 0
    for ja in sorted(glossary, key=len, reverse=True):
        zh = (glossary[ja] or {}).get("zh")
        if not zh or ja not in out:
            continue
        pat = _term_pattern(ja)
        if not pat.search(out):
            continue
        n += 1
        token = f"【N{n}】"
        out = pat.sub(token, out)
        mapping[token] = zh
    return out, mapping


def restore_terms(text: str, mapping: dict[str, str]) -> str:
    """Put the pinned translations back where the placeholders landed."""
    if not mapping or not text:
        return text
    out = text
    for token, zh in mapping.items():
        out = out.replace(token, zh)
        # Models occasionally re-punctuate the marker; accept the bare form.
        out = re.sub(r"[【\[(]\s*" + re.escape(token.strip("【】")) + r"\s*[】\])]",
                     zh, out)
    return out


def _term_pattern(ja: str) -> "re.Pattern":
    esc = re.escape(ja)
    # An honorific confirms a name, so match regardless of what follows.
    # Without one, only guard the *hiragana* terms: a katakana or kanji term
    # followed by a particle (ドラマを) is a normal reading, whereas a hiragana
    # term followed by more hiragana (わかんない) is the middle of another word.
    alts = [esc + "(?:さん|ちゃん|くん|君|様)"]
    if ja and ja[-1] in _HIRAGANA_SET:
        alts.append(esc + "(?![" + _HIRAGANA + "])")
    else:
        alts.append(esc)
    return re.compile("|".join(alts))


def apply_to_source(text: str, glossary: dict[str, dict]) -> str:
    """Substitute pinned translations directly into the Japanese source.

    Kept for callers that want the simple form; :func:`protect_terms` is the
    reliable one (see its docstring for why).
    """
    if not glossary or not text:
        return text
    out = text
    # Longest terms first so 「かんなちゃん」 wins over 「かんな」.
    for ja in sorted(glossary, key=len, reverse=True):
        zh = (glossary[ja] or {}).get("zh")
        if not zh or ja not in out:
            continue
        new_out, n = _term_pattern(ja).subn(lambda m: zh, out)
        if n:
            out = new_out
    return out


def check_consistency(glossary: dict[str, dict], segments: list[dict],
                      translations: list[str]) -> dict:
    """How often each glossary term's Chinese rendering actually appears.

    Reported in the acceptance run: a term present in the source but missing
    from the translation means the pin did not take.
    """
    rows = []
    for k, v in (glossary or {}).items():
        zh = v.get("zh") or ""
        occ = hit = 0
        for seg, tr in zip(segments, translations):
            if k in (seg.get("text") or ""):
                occ += 1
                if zh and zh in (tr or ""):
                    hit += 1
        if occ:
            rows.append({"ja": k, "zh": zh, "occurrences": occ, "applied": hit,
                         "rate": round(hit / occ, 3)})
    rows.sort(key=lambda r: r["rate"])
    total_occ = sum(r["occurrences"] for r in rows)
    total_hit = sum(r["applied"] for r in rows)
    return {"terms": rows,
            "overall": round(total_hit / total_occ, 3) if total_occ else 1.0}
