"""content.classify: hallucinations drop, vocalisations keep the original, lines stay (no GPU).

    .venv/bin/python tests/test_content.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie.content import classify          # noqa: E402
from ai_movie.units import is_nonlexical       # noqa: E402


def seg(text, dur=2.0, **kw):
    return {"text": text, "start": 10.0, "end": 10.0 + dur, **kw}


def kind(*a, **k):
    return classify(seg(*a, **k))["content"]


def test_stock_phrases():
    assert kind("ご清聴ありがとうございました。", **{"pass": "sweep"}) == "drop"
    assert kind("ありがとうございました", asr_conf=0.4) == "drop"
    assert kind("ありがとうございます。", asr_conf=0.92) == "speech"      # a real thank-you in the interview
    assert kind("ご視聴ありがとうございました", asr_conf=0.95) == "drop"   # YouTube boilerplate is never real here


def test_latin_and_numbers():
    assert kind("inse Philaname", **{"pass": "sweep"}) == "drop"
    assert kind("3cm") == "speech"
    assert kind("15センチ") == "speech"


def test_loops_and_rate():
    assert kind("あああああああああ", 0.5) == "nonlexical"                 # a held vowel is a sound, not a loop
    assert kind("ぐふぐふぐふぐふ", 2.0) == "nonlexical"                  # a laugh, however long: original voice
    assert kind("はあはあ", 1.5) == "nonlexical"
    assert kind("そうそうそう", 1.0) == "speech"
    assert kind("気持ちいい", 2.0, compression_ratio=2.9) == "drop"
    assert kind("今日はいい天気ですねとても暖かい", 0.4) == "drop"        # 15 chars in 0.4 s


def test_transcribed_moans_keep_the_voice_even_with_high_compression():
    for t in ("アーッ、アーッ、アーッ、アーッ", "あーーーーーーー", "はぁはぁ…はぁ", "んっ、んっ、んっ、んっ、んっ"):
        assert kind(t, 2.0, compression_ratio=3.1) == "nonlexical", t
    assert kind("気持ちいい気持ちいい気持ちいい気持ちいい気持ちいい", 2.0, compression_ratio=3.1) == "drop"   # a real word looping


def test_nonlexical_and_energy():
    assert kind("グーグー") == "nonlexical"
    assert kind("ハァハァ") == "nonlexical"
    assert kind("んっ…あっ") == "nonlexical"
    assert kind("だめ") == "speech"
    assert kind("もっと") == "speech"
    assert classify(seg("気持ちいい"), vocals_p95_db=-60.0)["content"] == "drop"
    assert classify(seg("気持ちいい"), vocals_p95_db=-40.0)["content"] == "speech"


def test_sweep_cross_decode():
    s = {"pass": "sweep"}
    assert classify(seg("奥まで入っちゃうよ", asr_conf=0.3, alt_text="奥まで入っちゃうよ。", **s)).get("confirmed")
    assert kind("奥まで入っちゃうよ", asr_conf=0.3, avg_logprob=-1.1, alt_text="ご飯食べました", **s) == "drop"
    # a confident line is not vetoed by a conflicting second decode …
    assert kind("毎日仕事中電話しすぎはい会議だって遅刻するし", asr_conf=0.94, avg_logprob=-0.18, alt_text="どうする", **s) == "speech"
    # … and a second decode that is itself a stock phrase vetoes nothing
    assert kind("下に座っていいのかな", asr_conf=0.46, avg_logprob=-0.86, alt_text="おやすみなさい", **s) == "speech"
    assert kind("よろしくお願いします", asr_conf=0.7) == "speech"
    assert kind("奥まで入っちゃうよ", alt_text="", no_speech_prob=0.7, **s) == "drop"
    assert kind("奥まで入っちゃうよ", alt_text="", no_speech_prob=0.2, **s) == "speech"
    assert kind("奥まで入っちゃうよ", no_speech_prob=0.9, avg_logprob=-1.5, **s) == "drop"
    assert kind("奥まで入っちゃうよ", no_speech_prob=0.9, avg_logprob=-0.6, **s) == "speech"


def test_is_nonlexical_extension():
    for t in ("あっ", "んん…", "はぁはぁ", "グーグー", "ハァハァ", "ふふふ", "ぐぐ", "", "…"):
        assert is_nonlexical(t), t
    for t in ("はい", "うん", "いいえ", "だめ", "いや", "もっと", "ねえ", "気持ちいい", "あの、すみません", "いく", "はーい"):
        assert not is_nonlexical(t), t


def test_held_vowel_dominates():
    # a moan window decodes as the held vowel plus a hallucinated tail; the run decides, not the window score
    for t in ("ああああああああああああああ兄ちゃん…", "ぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁぁとりあえず", "ー" * 40 + "アイク、アイク、アイク",
              "アアアアアアアアアアアアアアアアお前もごらん", "ああああああ ああああああたーーーー", "ここまでああああああああああああ",
              "あ゛ぁぁぁぁぁぁ", "でいっ ああああああ"):
        assert kind(t, 3.0, compression_ratio=30) == "nonlexical", t
    # real lines with a run: below RUN_MIN or below the fraction — at their real window ratios (< 2.4)
    assert kind("あぁ、あぁ、あぁあぁ、痛い、痛い", 3.0, compression_ratio=2.37) == "speech"
    assert kind("ああああ北までいるよ", 2.0, compression_ratio=1.668) == "speech"
    assert kind("っ ああああああ今までやってきた中", 3.0, compression_ratio=2.372) == "speech"
    assert kind("あああ大", 1.0) == "speech"                                    # 3/4: too short a run
    for t in ("コーヒー", "えーっと", "ーーーーーーーーアイク、アイク"):            # last: 8/14 < 0.7
        assert not is_nonlexical(t), t
    # the run rule is opt-out for evidence use (20/25 = 0.8 of the line is the held vowel)
    assert is_nonlexical("あ" * 20 + "寝てきたよ")
    assert not is_nonlexical("あ" * 20 + "寝てきたよ", held_run=False)


def test_screams():
    for t in ("うわー", "うわっ", "うわぁぁぁぁ", "きゃー", "きゃああ"):
        assert kind(t, 1.0) == "nonlexical", t
    for t in ("いいわ", "ひやひや", "お客", "きゃく"):
        assert kind(t, 1.0) == "speech", t


def test_whitelist_beats_repeat_unit():
    # the old `is_nonlexical(unit)` half let a repeated whitelist word bypass the whitelist
    for t in ("いい", "いい?", "ええ", "はーい", "おお"):
        assert kind(t, 1.0) == "speech", t
    assert kind("そうそうそうそう", 1.0) == "drop"                    # repetition ×4 of a word
    assert kind("ぐふぐふぐふぐふ", 1.0) == "nonlexical"
    # staccato moans spell a whitelist word once っ / ー are stripped; the repetition keeps them nonlexical
    for t in ("いっいっ", "えっ、えっ", "おっおっ", "いーいー", "えーえー"):
        assert kind(t, 1.0) == "nonlexical", t
    # …but a whitelist word merely ending in っ is still the word (SONE-846 p13 248.9 s 「はいっ」)
    for t in ("はいっ", "あいっ!", "うんっ"):
        assert kind(t, 1.0) == "speech", t


def test_alt_held_run_is_not_junk():
    s = {"pass": "sweep", "asr_conf": 0.498, "avg_logprob": -1.031}
    weak = "何してるんだよトお前お前死んじゃった"
    # a held run *with a tail* as the second decode says the window looped: the veto stands
    assert kind(weak, 5.0, alt_text="あ" * 300 + "寝てきたよおまわししちゃった", **s) == "drop"
    # a pure vocalisation as the second decode is junk and vetoes nothing (existing semantics)
    assert kind(weak, 5.0, alt_text="あああああ", **s) == "speech"


def test_rule_order_keeps_cr_for_words():
    assert kind("気持ちいい", 2.0, compression_ratio=2.9) == "drop"
    assert kind("気持ちいい" * 5, 2.0, compression_ratio=3.1) == "drop"
    assert kind("胸が痛い", 1.0, compression_ratio=3.08) == "drop"                 # a T=1.0 window sample
    assert kind("あーーーー", 2.0, compression_ratio=55) == "nonlexical"
    assert kind("ん", 0.8, compression_ratio=3.56) == "nonlexical"
    assert kind("気持ちいい", 2.0, compression_ratio=2.4) == "speech"              # the threshold is exclusive


def test_looped_indices_is_text_and_time_only():
    from ai_movie.content import looped_indices
    segs = [seg("ご飯食べました", 1.0), seg("ご飯食べました", 1.0), seg("ご飯食べました", 1.0), seg("違う話", 1.0)]
    for i, s in enumerate(segs):
        s["start"], s["end"] = 10.0 + 5 * i, 11.0 + 5 * i
    assert looped_indices(segs) == {0, 1, 2}
    segs[2]["start"] = 100.0                              # the third copy is > 30 s away from the first
    assert looped_indices(segs) == set()
    assert looped_indices([seg("あっ")] * 5) == set()      # < 4 folded chars never counts


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
