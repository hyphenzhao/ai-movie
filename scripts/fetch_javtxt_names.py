#!/usr/bin/env python
"""Performer name triples (日本語 / 中文 / English) from javtxt.com.

Wikidata knows only the famous and the historic — the performer in our own
test footage is not in it, and it carries no Chinese rendering for half the
rows it does have.  javtxt's actress pages publish exactly the three fields
this pipeline needs, as plain facts:

    演员(女优): めぐり          ← Japanese stage name (h1)
    中文名： 藤浦惠  英文名： Meguri  别名： 藤浦めぐ

Only `/actress/<id>` pages are fetched — never film pages, never images —
and only those four name fields are kept.  Requests are serialised with a
delay and every page is cached on disk, so a re-run costs the site nothing.
robots.txt (checked 2026-09-17) carries no User-agent/Disallow rules and sets
no content signals.

    python scripts/fetch_javtxt_names.py --pages 10      # index pages to walk
    python scripts/fetch_javtxt_names.py --pages 10 --refresh
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "asset" / "names" / "javtxt.json"
CACHE = ROOT / "workspace" / "_cache" / "javtxt"
BASE = "https://javtxt.com"
UA = "Mozilla/5.0 (X11; Linux x86_64) ai-movie-glossary/1.0"
DELAY = 1.5                     # seconds between requests

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[\s　]+")


def clean(fragment: str) -> str:
    return _WS.sub(" ", html_mod.unescape(_TAG.sub(" ", fragment))).strip()


def fetch(path: str, *, refresh: bool = False) -> str | None:
    """GET *path*, using the on-disk cache unless *refresh*."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^0-9A-Za-z]+", "_", path).strip("_") or "root"
    cached = CACHE / f"{key}.html"
    if cached.exists() and not refresh:
        return cached.read_text(encoding="utf-8")
    req = urllib.request.Request(BASE + path, headers={"User-Agent": UA})
    body = None
    for attempt in range(5):
        try:
            body = urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "replace")
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == 4:
                raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # this host's uplink drops DNS for stretches; the cache means a
            # resumed run re-fetches only what is still missing
            print(f"    retry {attempt + 1}/5 after {type(exc).__name__}", flush=True)
            if attempt == 4:
                return None
        time.sleep(10 * (attempt + 1))
    if body is None:
        return None
    cached.write_text(body, encoding="utf-8")
    time.sleep(DELAY)
    return body


def parse_actress(page: str, actress_id: str) -> dict | None:
    """Pull only the name fields out of one actress page.

    The page states them as definition pairs — ``<dt>英文名： </dt><dd>…</dd>``
    — so read those rather than pattern-matching flattened text: the fields
    present vary per performer (many have an English name but no Chinese
    one), and a text regex silently returned empty for all of them.
    """
    body = re.sub(r"<script.*?</script>", "", page, flags=re.S)
    m = re.search(r"<h1[^>]*>(.*?)</h1>", body, flags=re.S)
    ja = clean(m.group(1)) if m else ""
    ja = re.sub(r"^演员\s*\(女优\)\s*[:：]\s*", "", ja)

    pairs = {}
    for dt, dd in re.findall(r"<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>", body, flags=re.S):
        key = clean(dt).rstrip(":：").strip()
        if key:
            pairs[key] = clean(dd)

    zh, en = pairs.get("中文名", ""), pairs.get("英文名", "")
    alias = pairs.get("别名", "") or pairs.get("別名", "")
    if not ja and not zh:
        return None
    return {"id": actress_id, "ja": ja, "zh": zh, "en": en,
            "aliases": [a.strip() for a in re.split(r"[,，、/]", alias) if a.strip()],
            "born": pairs.get("生日", ""),
            "source": f"{BASE}/actress/{actress_id}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=int, default=10, help="index pages to walk (50 each)")
    ap.add_argument("--refresh", action="store_true", help="ignore the on-disk cache")
    ap.add_argument("--limit", type=int, default=0, help="stop after N performers")
    args = ap.parse_args()

    ids: list[str] = []
    for page in range(1, args.pages + 1):
        body = fetch(f"/top-actresses?page={page}", refresh=args.refresh)
        if not body:
            break
        found = [i for i in dict.fromkeys(re.findall(r"/actress/(\d+)", body))]
        new = [i for i in found if i not in ids]
        print(f"index page {page}: {len(found)} links, {len(new)} new", flush=True)
        if not new:
            break
        ids += new
    if args.limit:
        ids = ids[:args.limit]

    rows: dict[str, dict] = {}
    if OUT.exists():
        rows = {r["id"]: r for r in json.loads(OUT.read_text(encoding="utf-8"))}
    for n, actress_id in enumerate(ids, 1):
        if actress_id in rows and not args.refresh:
            continue
        body = fetch(f"/actress/{actress_id}", refresh=args.refresh)
        if not body:
            continue
        row = parse_actress(body, actress_id)
        if row:
            rows[actress_id] = row
        if n % 25 == 0:
            print(f"  {n}/{len(ids)} performers…", flush=True)

    out = sorted(rows.values(), key=lambda r: int(r["id"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    trip = sum(1 for r in out if r["ja"] and r["zh"] and r["en"])
    print(f"{len(out)} performers ({trip} with all three names) → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
