"""Glossary term matching (v3.4 E8) and the E12 flag clean-up.

    .venv/bin/python tests/test_glossary_terms.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie import glossary as G  # noqa: E402


def test_hiragana_term_is_not_matched_inside_another_word():
    pat = G._term_pattern("かんな")
    assert not pat.search("わかんないよ")                  # measured false hit (output_test, SONE-846)
    assert pat.search("かんなさん、来て")                   # honorific confirms the name
    assert pat.search("かんな、来て")
    assert pat.search("かんな！")


def test_hiragana_guard_is_what_it_was():
    # The guard is deliberately conservative: a hiragana term followed by more
    # hiragana is treated as the middle of a word, particles included.  E8
    # does not change that — it fixes the two functions that ignored it.
    assert not G._term_pattern("かんな").search("かんなに")
    assert G._term_pattern("カンナ").search("カンナに")           # katakana: no guard


def test_key_ending_in_honorific_matches_before_a_particle():
    # かねちゃん followed by に: the old look-ahead refused it (SONE-846_p14:
    # substring 1 hit, pattern 0); an honorific already confirms the name.
    for key in ("かねちゃん", "田中さん", "たけしくん", "山田君", "お嬢様"):
        pat = G._term_pattern(key)
        assert pat.search(key + "にまた会った"), key
        assert pat.search(key), key
    assert not G._term_pattern("かねちゃん").search("かねさん")


def test_check_consistency_uses_word_boundaries():
    gl = {"かんな": {"zh": "栞奈"}, "かねちゃん": {"zh": "小兼"}}
    segs = [{"text": "わかんないよ"}, {"text": "かんなさん"}, {"text": "かねちゃんにまた"}, {"text": "何もない"}]
    zh = ["不知道啦", "栞奈小姐", "又见到小兼了", "什么都没有"]
    res = G.check_consistency(gl, segs, zh)
    rows = {r["ja"]: r for r in res["terms"]}
    assert rows["かんな"]["occurrences"] == 1 and rows["かんな"]["applied"] == 1
    assert rows["かねちゃん"]["occurrences"] == 1 and rows["かねちゃん"]["applied"] == 1
    assert res["overall"] == 1.0
    # the raw-substring reading would have charged わかんない as a miss
    assert G.check_consistency({"かんな": {"zh": "栞奈"}}, segs[:1], zh[:1])["overall"] == 1.0


def test_format_for_prompt_only_lists_real_hits():
    gl = {"かんな": {"zh": "栞奈"}, "かねちゃん": {"zh": "小兼"}}
    out = G.format_for_prompt(gl, ["わかんないよ", "かねちゃんにまた"])
    assert "かねちゃん" in out and "かんな" not in out.replace("かねちゃん", "")


def test_flag_line_has_no_split_fallback_flag():
    from ai_movie.units import flag_line
    assert "split_fallback" not in inspect.signature(flag_line).parameters
    assert flag_line("今日はいい天気ですね", "今天天气真好呢") == []
    assert all(not f.startswith("F5") for f in flag_line("元気？", "很好。"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
