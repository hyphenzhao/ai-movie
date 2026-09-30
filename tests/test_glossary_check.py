"""glossary.check_consistency counts term occurrences boundary-aware (no GPU, no Ollama).

The acceptance gate B3 read 0.90 / 0.857 on the release films because a plain
substring test counted 「かんな」 inside 「わかんない」 ("don't know") — a line
no enforce model can ever "fix" because the name is not in it.

    .venv/bin/python tests/test_glossary_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie.glossary import check_consistency                 # noqa: E402

GLOSS = {"かんな": {"zh": "坎娜", "kind": "name"},
         "カンナ": {"zh": "坎娜", "kind": "name"},
         "ドラマ": {"zh": "短剧", "kind": "term"}}


def _rows(res):
    return {r["ja"]: r for r in res["terms"]}


def test_wakannai_is_not_an_occurrence():
    segs = [{"text": "わかんない"}, {"text": "もらってないとわかんないかも"},
            {"text": "かんなちゃんが思う"}, {"text": "いつものかんなちゃんじゃない、"}]
    zh = ["不知道", "不实际体验过可能不知道", "小坎娜这么想。", "虽然不是平时的小坎娜，"]
    res = check_consistency(GLOSS, segs, zh)
    r = _rows(res)["かんな"]
    # the two 「わかんない」 lines are not the name: 2 occurrences, both pinned
    assert (r["occurrences"], r["applied"], r["rate"]) == (2, 2, 1.0), r
    assert res["overall"] == 1.0, res


def test_real_miss_still_counts():
    segs = [{"text": "かんなちゃん。"}, {"text": "カンナです。"}, {"text": "ドラマをやりました。"}]
    zh = ["小卡娜。", "我是坎娜。", "我演了电视剧。"]
    res = check_consistency(GLOSS, segs, zh)
    r = _rows(res)
    assert (r["かんな"]["occurrences"], r["かんな"]["applied"]) == (1, 0)
    assert (r["カンナ"]["occurrences"], r["カンナ"]["applied"]) == (1, 1)
    # a katakana term followed by a particle is a normal reading, so it counts
    assert (r["ドラマ"]["occurrences"], r["ドラマ"]["applied"]) == (1, 0)
    assert res["overall"] == round(1 / 3, 3)
    # worst term first
    assert res["terms"][0]["rate"] == 0.0 and res["terms"][-1]["rate"] == 1.0


def test_absent_or_empty_terms():
    res = check_consistency(GLOSS, [{"text": "こんにちは"}], ["你好"])
    assert res == {"terms": [], "overall": 1.0}
    assert check_consistency({}, [{"text": "かんな"}], ["坎娜"])["overall"] == 1.0
    # an empty key can never match
    res = check_consistency({"": {"zh": "x"}}, [{"text": "かんな"}], ["坎娜"])
    assert res["terms"] == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
