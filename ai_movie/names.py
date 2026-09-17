"""Performer names: pin the Chinese rendering, feed Whisper the reading.

`asset/names/` holds what `scripts/fetch_name_glossary.py` pulled from
Wikidata.  This module decides *which* of those names a given film may use,
because injecting all of them would be actively harmful:

* Many rows are stage names or nicknames pointing at a different full name
  (「あずみ」→上原杏美, 「イヴ」→神代弓子, 「アッコ」→中村晃子).  Pinning
  those would rewrite every unrelated Azumi in any film.
* Some readings are ordinary words — 「ひとみ」 (pupil/eyes), 「つぐみ」
  (a thrush) — so a blanket pin would corrupt normal dialogue.
* Whisper's ``initial_prompt`` holds only a couple of hundred tokens, while
  the Japanese list alone has ~2 000 surface forms.

So a name is **pinnable** only in its canonical family+given spelling (known
from the kana field carrying a space), aliases are hotword-only, and callers
select per film — either from the cast list or from names the transcript
already contains.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES_DIR = ROOT / "asset" / "names"

_SPACES = re.compile(r"[\s　]+")
# Readings that are also everyday words; never pin these, whatever the source.
_COMMON_WORDS = {
    "あい", "ひとみ", "つぐみ", "そら", "ひかる", "めぐみ", "のぞみ", "さくら",
    "ゆめ", "みなみ", "かおり", "まなみ", "りん", "あゆみ", "まりあ", "あすか",
    "いちご", "すず", "もも", "かな", "はな", "ゆき", "みく", "あん",
}


def load_rows(countries: tuple[str, ...] = ("jp", "kr", "vn", "th")) -> list[dict]:
    rows: list[dict] = []
    for cc in countries:
        path = NAMES_DIR / f"{cc}.json"
        if path.exists():
            rows += json.loads(path.read_text(encoding="utf-8"))
    return rows


def _canonical_forms(row: dict) -> list[str]:
    """Spellings safe to pin: the full name in kanji and in kana.

    A kana reading with a space ("ほしの ひかる") means Wikidata knows both
    the family and the given name, so the label really is a full name rather
    than a one-word stage name.
    """
    kana = row.get("kana") or ""
    if " " not in kana and "　" not in kana:
        return []
    forms = []
    for form in (row.get("native"), kana, _SPACES.sub("", kana)):
        form = _SPACES.sub("", form or "")
        if len(form) >= 3 and form not in _COMMON_WORDS:
            forms.append(form)
    return list(dict.fromkeys(forms))


def all_surface_forms(row: dict) -> list[str]:
    """Every spelling worth giving Whisper as a hotword (aliases included)."""
    out = [row.get("native") or "", *(row.get("native_aliases") or [])]
    kana = row.get("kana") or ""
    if kana:
        out += [kana, _SPACES.sub("", kana)]
    forms = {_SPACES.sub("", o) for o in out if o}
    return [f for f in forms if len(f) >= 2]


def pinnable(rows: list[dict] | None = None) -> dict[str, dict]:
    """surface form → {zh, qid, …} for names whose Chinese rendering is safe
    to force.  Forms claimed by two different people are dropped, not
    guessed."""
    rows = rows if rows is not None else load_rows()
    pins: dict[str, dict] = {}
    clashes: set[str] = set()
    for row in rows:
        zh = (row.get("zh") or "").strip()
        if not zh or zh.isascii():          # no agreed Chinese rendering
            continue
        for form in _canonical_forms(row):
            prev = pins.get(form)
            if prev and prev["zh"] != zh:
                clashes.add(form)
                continue
            pins[form] = {"zh": zh, "kind": "name", "qid": row.get("qid", ""),
                          "country": row.get("country", ""),
                          "gender": row.get("gender", "")}
    for form in clashes:
        pins.pop(form, None)
    return pins


def select_for_segments(segments: list[dict], *,
                        rows: list[dict] | None = None,
                        text_key: str = "text") -> dict[str, dict]:
    """Pins for names the transcript actually contains.

    This is the safe way to use the list after ASR: a name nobody said can
    never be pinned, so the glossary stays as small as the film needs.  It
    cannot rescue a name the recogniser got wrong — that needs the reading up
    front, via :func:`hotword_prompt`.
    """
    joined = "".join(_SPACES.sub("", (s.get(text_key) or "")) for s in segments)
    return {form: meta for form, meta in pinnable(rows).items() if form in joined}


def select_by_name(query: str, *, rows: list[dict] | None = None,
                   limit: int = 10) -> list[dict]:
    """Look a performer up by any spelling (cast list, product page, filename)."""
    raw = (query or "").strip()
    q = _SPACES.sub("", raw)
    if not q:
        return []
    out = []
    for row in rows if rows is not None else load_rows():
        forms = all_surface_forms(row)
        # the romaji comparison keeps the spaces the query came with:
        # "Sora Aoi" must match en="Sora Aoi", not the space-stripped form
        if any(q in f or f in q for f in forms) \
                or raw.lower() in (row.get("en") or "").lower():
            out.append(row)
        if len(out) >= limit:
            break
    return out


def hotword_prompt(rows: list[dict], *, language: str = "ja",
                   max_chars: int = 200) -> str:
    """An ``initial_prompt`` naming this film's cast.

    Whisper conditions on the prompt text, so it must stay short and must
    read like the transcript's own language; a long list drifts into being
    echoed back as output.
    """
    forms: list[str] = []
    for row in rows:
        kana = _SPACES.sub("", row.get("kana") or "")
        native = _SPACES.sub("", row.get("native") or "")
        forms += [f for f in (native, kana) if f]
    forms = list(dict.fromkeys(forms))
    if not forms:
        return ""
    lead = {"ja": "出演者：", "ko": "출연자: ", "vi": "Diễn viên: ", "th": "นักแสดง: "}
    text = lead.get(language, "") + "、".join(forms)
    return text[:max_chars]
