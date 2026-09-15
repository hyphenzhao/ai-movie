"""Sentence units, split-back and the film-independent flag rules (no GPU).

    .venv/bin/python tests/test_units.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ai_movie import units as U                 # noqa: E402


def seg(start, end, text, speaker="S0"):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def test_joins_continuous_speech():
    # the 24-char cap cut running speech mid-word: no pause, no end mark
    assert U.joins(seg(0, 2, "普通くらいだった"), seg(2.0, 3, "ら全然気持ちいい"))
    # a real pause after an unfinished clause without a connective stays apart
    assert not U.joins(seg(0, 2, "かんなちゃんが思う"), seg(3.3, 4, "おちんちん"))


def test_joins_never_crosses_sentence_end():
    assert not U.joins(seg(0, 1, "カンナちゃん。"), seg(1.0, 2, "見てますね。"))
    assert not U.joins(seg(0, 1, "違うかな?"), seg(1.0, 2, "握るとどう?"))


def test_joins_connective_short_pause():
    assert U.joins(seg(0, 1, "緊張したけれど、"), seg(1.2, 2, "一応先生と生徒役"))
    assert U.joins(seg(0, 1, "撮影しつつなんですけど"), seg(1.2, 2, "今回"))
    # the connective rule needs the same speaker
    assert not U.joins(seg(0, 1, "けど", "S0"), seg(1.2, 2, "今回はね", "S1"))
    # and a short pause, not a long one
    assert not U.joins(seg(0, 1, "けど"), seg(1.5, 2, "今回"))


def test_joins_scrap_ignores_speaker_label():
    # diarization mislabels 1-2 character scraps; continuous speech still joins
    assert U.joins(seg(0, 1, "ってことだも", "S1"), seg(1.0, 1.2, "ん", "S0"))


def test_group_units_caps():
    segs = [seg(i, i + 1.0, "あいうえおかきくけこさしすせそ") for i in range(6)]
    units = U.group_units(segs, max_chars=40)
    assert all(sum(U.visible_len(segs[i]["text"]) for i in u) <= 40 for u in units)
    units = U.group_units(segs, max_dur=2.5)
    assert all(segs[u[-1]]["end"] - segs[u[0]]["start"] <= 2.5 for u in units)


def test_split_prefers_punctuation_and_keeps_words():
    pieces = U.split_translation("嗯，差不多就是这样的感觉吧。一般在12到15厘米左右", [12, 14])
    assert pieces == ["嗯，差不多就是这样的感觉吧。", "一般在12到15厘米左右"]
    pieces = U.split_translation("假设有一根小鸡鸡和一根大鸡巴吧。嗯。", [9, 20])
    assert pieces and "".join(pieces) == "假设有一根小鸡鸡和一根大鸡巴吧。嗯。"
    assert not any(p.startswith("鸡鸡") for p in pieces), pieces
    assert U.split_translation("好", [3, 3]) is None          # too short → fallback
    assert U.split_translation("只有一句", [5]) == ["只有一句"]


def test_split_never_cuts_numbers():
    for w in ([1, 1], [3, 7], [7, 3]):
        pieces = U.split_translation("这个大概是20厘米乘15厘米那么大吧", w)
        assert pieces is not None
        joined = "|".join(pieces)
        assert "2|0" not in joined and "1|5" not in joined, joined


def test_flags():
    assert "F1_pronoun" in U.flag_line("好きなのかも。", "我可能喜欢上你了。")
    assert "F1_pronoun" not in U.flag_line("私は生徒?", "我是学生？")
    assert "F1_pronoun" not in U.flag_line("ドラマをやりました。", "我演过电视剧了。")
    assert "F2_question" in U.flag_line("相性はどうだった?", "相性很好。")
    assert "F3_kana" in U.flag_line("見て", "看ください")
    assert "F4_length" in U.flag_line("すごい嬉しいです", "好厉害我真的非常非常非常开心而且特别特别期待接下来的拍摄呢")
    assert U.flag_line("気持ちよかった?", "舒服吗？") == []


def test_polish_edit_guard():
    ok, F1, F2, F3 = U.polish_edit_ok, ["F1_pronoun"], ["F2_question"], ["F3_kana"]
    # content words must not change, even when most characters are shared
    assert not ok("我演过电视剧了。", "我演过电影了。", F1)
    assert not ok("不过呢，我有点事想问小香菜就是了", "小香菜，我有个问题想问坎娜。", F1)
    assert not ok("她说可以了。", "【前文】", F1)
    # removing an invented pronoun / attribution is exactly what F1 is for
    assert ok("我可能喜欢上你了。", "可能喜欢上了。", F1)
    assert ok("你在看我呢。", "在看呢。", F1)
    assert ok("她说可以了。", "可以了。", F1)
    assert ok("相性很好。", "相性很好吗？", F2)
    assert ok("看ください", "请看", F3)
    assert not ok("好的", "好的", F1)                      # unchanged is not a fix


def test_units_on_v300_output_test():
    """On the real v3.0.0 transcript: units only ever merge neighbours, every
    segment is covered exactly once, and merges stay a minority."""
    state = ROOT / "workspace" / "_archive_v3.0.0" / "output_test" / "state.json"
    if not state.exists():
        print("  (skipped: no v3.0.0 archive)")
        return
    segs = json.loads(state.read_text(encoding="utf-8"))["asr"]["segments"]
    units = U.group_units(segs)
    flat = [i for u in units for i in u]
    assert flat == list(range(len(segs)))
    assert all(b - a == 1 for u in units for a, b in zip(u, u[1:]))
    multi = sum(1 for u in units if len(u) > 1)
    assert 0 < multi <= len(segs) // 4, multi


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
