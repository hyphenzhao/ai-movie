#!/usr/bin/env python
"""Build a performer-name glossary from Wikidata (no hand-typed names).

Names are the single worst class of ASR error in this pipeline — v3.0.0 heard
瀬戸 (せと) as セット and the translator duly wrote 「套装」 (a clothing set).
A name list fixes that twice over: as Whisper hotwords so the reading is
recognised, and as glossary pins so the Chinese rendering is forced.

The data comes from Wikidata rather than being written out by hand, because a
wrong kana reading injected as a hotword is worse than no hotword at all.
Every row is a queryable fact with a Q-id to check: label per language, the
kana reading (P1814), aliases, gender (P21) and birth year (P569).  Re-run it
to refresh; it is the same query for every country, so Korean, Vietnamese and
Thai performers come from the same code path as Japanese ones.

    python scripts/fetch_name_glossary.py --country jp           # fetch one
    python scripts/fetch_name_glossary.py --country jp,kr,vn,th  # all
    python scripts/fetch_name_glossary.py --build                # → glossary + hotwords

Output (all under asset/names/, gitignored data, small enough to commit):
    <cc>.json            raw rows, one per performer
    glossary_seed.json   ja/ko/vi/th surface form → pinned Chinese
    hotwords.json        per-language reading lists for ASR_INITIAL_PROMPT
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "asset" / "names"
ENDPOINT = "https://query.wikidata.org/sparql"
UA = "ai-movie-glossary/1.0 (local dubbing pipeline; contact: repo owner)"

# occupation → "pornographic actor"; country of citizenship per market.
OCCUPATION = "wd:Q488111"          # pornographic actor
# Mainstream screen actors, kept in their own files: the end goal is adult
# *advertising* translation, where a well-known face may be a regular actor.
# ~13.7k people have both a Chinese and a native label (jp 8140, kr 4236,
# th 1142, vn 169), so this stays a separate, opt-in library.
MAINSTREAM_OCCUPATION = "wd:Q33999"
COUNTRIES = {
    "jp": ("wd:Q17", "ja"),
    "kr": ("wd:Q884", "ko"),
    "vn": ("wd:Q881", "vi"),
    "th": ("wd:Q869", "th"),
}
PAGE = 500
PAGE_PAUSE = 3.0                   # between pages, so WDQS does not throttle us

QUERY = """
SELECT ?p ?native ?kana ?zh ?zhHans ?zhCn ?en ?gender ?born
       (GROUP_CONCAT(DISTINCT ?nativeAlias; separator="|") AS ?nativeAliases)
       (GROUP_CONCAT(DISTINCT ?zhAlias; separator="|") AS ?zhAliases)
WHERE {
  ?p wdt:P106 %(occ)s ; wdt:P27 %(country)s .
  OPTIONAL { ?p rdfs:label ?native FILTER(lang(?native)="%(lang)s") }
  OPTIONAL { ?p rdfs:label ?zh     FILTER(lang(?zh)="zh") }
  OPTIONAL { ?p rdfs:label ?zhHans FILTER(lang(?zhHans)="zh-hans") }
  OPTIONAL { ?p rdfs:label ?zhCn   FILTER(lang(?zhCn)="zh-cn") }
  OPTIONAL { ?p rdfs:label ?en     FILTER(lang(?en)="en") }
  OPTIONAL { ?p wdt:P1814 ?kana }
  OPTIONAL { ?p wdt:P21/rdfs:label ?gender FILTER(lang(?gender)="en") }
  OPTIONAL { ?p wdt:P569 ?bornDate BIND(YEAR(?bornDate) AS ?born) }
  OPTIONAL { ?p skos:altLabel ?nativeAlias FILTER(lang(?nativeAlias)="%(lang)s") }
  OPTIONAL { ?p skos:altLabel ?zhAlias     FILTER(lang(?zhAlias)="zh") }
}
GROUP BY ?p ?native ?kana ?zh ?zhHans ?zhCn ?en ?gender ?born
ORDER BY ?p
LIMIT %(limit)d OFFSET %(offset)d
"""


MAINSTREAM_QUERY = """
SELECT ?p ?native ?kana ?zh ?zhHans ?zhCn ?en ?gender ?born WHERE {
  ?p wdt:P106 %(occ)s ; wdt:P27 %(country)s ; rdfs:label ?zh .
  FILTER(lang(?zh)="zh")
  OPTIONAL { ?p rdfs:label ?native FILTER(lang(?native)="%(lang)s") }
  OPTIONAL { ?p rdfs:label ?zhHans FILTER(lang(?zhHans)="zh-hans") }
  OPTIONAL { ?p rdfs:label ?zhCn   FILTER(lang(?zhCn)="zh-cn") }
  OPTIONAL { ?p rdfs:label ?en     FILTER(lang(?en)="en") }
  OPTIONAL { ?p wdt:P1814 ?kana }
  OPTIONAL { ?p wdt:P21/rdfs:label ?gender FILTER(lang(?gender)="en") }
  OPTIONAL { ?p wdt:P569 ?bornDate BIND(YEAR(?bornDate) AS ?born) }
}
ORDER BY ?p
LIMIT %(limit)d OFFSET %(offset)d
"""


def sparql(query: str, tries: int = 6) -> list[dict]:
    url = f"{ENDPOINT}?{urllib.parse.urlencode({'query': query})}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/sparql-results+json", "User-Agent": UA})
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))["results"]["bindings"]
        except Exception as exc:                        # noqa: BLE001
            last = exc
            code = getattr(exc, "code", None)
            # 429 means we are the problem: back off far harder than for a 502
            time.sleep((60 if code == 429 else 15) * (attempt + 1))
    raise SystemExit(f"Wikidata query failed after {tries} tries: {last}")


def fetch_country(cc: str, *, occupation: str = OCCUPATION,
                  page: int = PAGE, save_to: Path | None = None,
                  query: str | None = None) -> list[dict]:
    """Page through one country.  Mainstream actors are 10x the rows of
    performers and WDQS answers the aliases aggregate with 502s at 500/page,
    so callers pass a smaller page and a file to flush into as it goes."""
    country, lang = COUNTRIES[cc]
    rows: list[dict] = []
    offset = 0
    while True:
        q = (query or QUERY) % {"occ": occupation, "country": country,
                                "lang": lang, "limit": page, "offset": offset}
        batch = sparql(q)
        for b in batch:
            def val(key: str) -> str:
                return (b.get(key) or {}).get("value", "").strip()

            qid = val("p").rsplit("/", 1)[-1]
            row = {
                "qid": qid,
                "country": cc,
                "lang": lang,
                "native": val("native"),
                "kana": val("kana"),
                "zh": val("zhHans") or val("zhCn") or val("zh"),
                "zh_raw": val("zh"),
                "en": val("en"),
                "gender": val("gender"),
                "born": val("born"),
                "native_aliases": [a for a in val("nativeAliases").split("|") if a],
                "zh_aliases": [a for a in val("zhAliases").split("|") if a],
            }
            if row["native"] or row["en"]:
                rows.append(row)
        if save_to is not None:
            save_to.write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                               encoding="utf-8")
        if len(batch) < page:
            break
        offset += page
        print(f"  {cc}: {len(rows)} rows so far…", flush=True)
        time.sleep(PAGE_PAUSE)
    return rows


# ── surface forms an ASR/translation pipeline will actually meet ────

_SPACES = re.compile(r"[\s　]+")


def ja_surface_forms(row: dict) -> list[str]:
    """Every spelling of a Japanese name the transcript might contain.

    Whisper writes names without the space Wikidata puts between family and
    given name, and it picks kanji or kana unpredictably, so both go in.
    """
    out = [row["native"], *row["native_aliases"]]
    kana = row.get("kana") or ""
    if kana:
        out += [kana, _SPACES.sub("", kana)]
    return [t for t in {_SPACES.sub("", o) for o in out if o} if len(t) >= 2]


def build_outputs(countries: list[str]) -> tuple[dict, dict]:
    glossary: dict[str, dict] = {}
    hotwords: dict[str, list[str]] = {}
    for cc in countries:
        path = OUT_DIR / f"{cc}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        words: list[str] = []
        for row in rows:
            zh = row.get("zh") or ""
            forms = ja_surface_forms(row) if cc == "jp" else \
                [t for t in {row["native"], *row["native_aliases"]} if t and len(t) >= 2]
            words += [w for w in forms if w]
            if not zh:
                continue          # no agreed Chinese rendering → hotword only
            for form in forms:
                prev = glossary.get(form)
                if prev and prev["zh"] != zh:
                    prev.setdefault("ambiguous", []).append(zh)
                    continue
                glossary[form] = {"zh": zh, "kind": "name", "qid": row["qid"],
                                  "country": cc, "gender": row.get("gender", "")}
        hotwords[cc] = sorted(set(words))
    return glossary, hotwords


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", default="jp",
                    help="comma-separated: " + ",".join(COUNTRIES))
    ap.add_argument("--mainstream", action="store_true",
                    help="fetch mainstream screen actors into <cc>_mainstream.json instead")
    ap.add_argument("--build", action="store_true",
                    help="(re)build glossary_seed.json + hotwords.json from cached rows")
    ap.add_argument("--fetch", action="store_true", help="query Wikidata (default unless --build)")
    args = ap.parse_args()

    ccs = [c.strip() for c in args.country.split(",") if c.strip() in COUNTRIES]
    if not ccs:
        raise SystemExit(f"--country must name one of {list(COUNTRIES)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mainstream:
        for cc in ccs:
            print(f"fetching mainstream {cc}…", flush=True)
            dest = OUT_DIR / f"{cc}_mainstream.json"
            rows = fetch_country(cc, occupation=MAINSTREAM_OCCUPATION,
                                 page=300, save_to=dest, query=MAINSTREAM_QUERY)
            rows = [r for r in rows if r["zh"] and not r["zh"].isascii()]
            dest.write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            print(f"  {cc}: {len(rows)} actors with a Chinese name "
                  f"→ {OUT_DIR / f'{cc}_mainstream.json'}")
        return 0

    if args.fetch or not args.build:
        for cc in ccs:
            print(f"fetching {cc}…", flush=True)
            rows = fetch_country(cc)
            (OUT_DIR / f"{cc}.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            with_zh = sum(1 for r in rows if r["zh"])
            with_kana = sum(1 for r in rows if r["kana"])
            print(f"  {cc}: {len(rows)} performers, {with_zh} with a Chinese name, "
                  f"{with_kana} with a kana reading → {OUT_DIR / f'{cc}.json'}")

    glossary, hotwords = build_outputs(list(COUNTRIES))
    (OUT_DIR / "glossary_seed.json").write_text(
        json.dumps(glossary, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_DIR / "hotwords.json").write_text(
        json.dumps(hotwords, ensure_ascii=False, indent=1), encoding="utf-8")
    amb = sum(1 for v in glossary.values() if v.get("ambiguous"))
    print(f"glossary: {len(glossary)} surface forms ({amb} ambiguous, left unpinned) "
          f"→ {OUT_DIR / 'glossary_seed.json'}")
    print("hotwords: " + ", ".join(f"{k}={len(v)}" for k, v in hotwords.items())
          + f" → {OUT_DIR / 'hotwords.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
