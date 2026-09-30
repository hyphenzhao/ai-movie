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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
