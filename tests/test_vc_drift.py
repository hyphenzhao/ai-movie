"""VC content-drift judge (vc_guard.fold_zh / text_sim / judge_content) and the WhisperClips
transcript cache — pure functions on fake transcripts, no GPU, no model load.

    .venv/bin/python tests/test_vc_drift.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie.vc_guard import fold_zh, judge_content, text_sim     # noqa: E402


def R(auto: str, lang: str = "zh", zh: str | None = None) -> dict:
    """A WhisperClips-shaped reading."""
    return {"auto": {"text": auto, "language": lang}, "zh": ({"text": zh} if zh is not None else None)}


def test_fold_zh():
    assert fold_zh("小雞雞哦——！") == "小鸡鸡哦"                      # t2s + punctuation
    assert text_sim("快點快點", "快点快点") == 1.0
    assert fold_zh("１０㎝") == "10cm"                                 # NFKC full-width → ASCII
    assert fold_zh("ママ") == "まま"                                    # katakana → hiragana
    assert fold_zh("……——！") == "" and text_sim("……", "好的") == 0.0   # empty after fold → 0
    assert text_sim("把腿伸直", "把腿先举") == 0.5                     # the documented boundary case


def test_judge_content_rules():
    # intact conversion: judged, ok, score 1
    d = judge_content(R("我们扮演的是老师和学生"), R("我们扮演的是老师和学生"), "我们扮演的是老师和学生。")
    assert d["judged"] and d["ok"] and d["score"] == 1.0 and d["baseline"] == 1.0
    # language flip on a 3-char line: script_flip (forced-zh reading does not rescue it)
    d = judge_content(R("Tchau, tchau.", "pt", zh="桥桥"), R("不妙啊"), "不妙啊")
    assert d["judged"] and not d["ok"] and d["reason"] == "script_flip"
    # flip on a 2-char line is still caught — hard rules ignore the length gate
    d = judge_content(R("Вот так.", "ru", zh="我说"), R("好大"), "好大")
    assert not d["ok"] and d["reason"] == "script_flip"
    # …but a forced-zh re-decode that reads the line rescues it (verify_dub's rule: the audio was Chinese)
    d = judge_content(R("Hao da", "en", zh="好大"), R("好大"), "好大")
    assert d["ok"]
    # 2-char line whose conversion reads as CJK but differs: the flip test was applied and passed,
    # the drift rule does not apply (ratio quantised) → judged, ok
    d = judge_content(R("好 吞吐"), R("好疼"), "好疼")
    assert d["judged"] and d["ok"] and d["reason"] == ""
    # boundary: 0.50 passes at min_sim 0.5 and fails at 0.6
    assert judge_content(R("把腿先举"), R("把腿伸直"), "把腿伸直")["ok"]
    d = judge_content(R("把腿先举"), R("把腿伸直"), "把腿伸直", min_sim=0.6)
    assert not d["ok"] and d["reason"] == "drift_0.50"
    # word drift
    d = judge_content(R("打书法"), R("好舒服啊"), "好舒服啊")
    assert d["judged"] and not d["ok"] and d["reason"].startswith("drift_0.") and d["score"] < 0.5
    # 1-char intended text, conversion still CJK → drift not applied (ratio quantised); hard rules passed
    d = judge_content(R("嗯"), R("嗯"), "嗯。")
    assert d["ok"] and d["reason"] == ""
    # 1-char line that flipped ("Hmm.") is caught by the hard rule
    d = judge_content(R("Hmm.", "en", zh="哼"), R("嗯"), "嗯。")
    assert not d["ok"] and d["reason"] == "script_flip"
    # …and a 1-char conversion that decodes to nothing is not judged at all
    assert not judge_content(R(""), R("嗯"), "嗯。")["judged"]
    # v1 heard as non-CJK garbage → nothing can be judged, not even a flip
    d = judge_content(R("Booga.", "en"), R("Фига!", "ru"), "哥哥……")
    assert not d["judged"] and d["ok"] and d["baseline"] is None and d["gate"] == "no_v1_cjk"
    # v1's forced-zh reading counts only when it reads the intended text (verify_dub's rule)
    assert judge_content(R("Booga.", "en"), R("Back off", "en", zh="百口"), "好疼")["gate"] == "no_v1_cjk"
    d = judge_content(R("Booga.", "en", zh="不噶"), R("Hao teng", "en", zh="好疼"), "好疼")
    assert d["reason"] == "script_flip" and d["heard_v1"] == "好疼"
    # baseline gate: v1 heard as Chinese but unlike the intended text → drift rule not applied
    # (the flip test still ran and passed)
    d = judge_content(R("保鑒塊"), R("跑较快"), "好像快。")
    assert d["ok"] and d["gate"] == "baseline" and d["reason"] == ""
    assert judge_content(R("好 吞吐"), R("好疼"), "好疼")["gate"] == "short"
    # max-with-want term: Whisper mis-heard v1 but heard the conversion
    d = judge_content(R("标的风格"), R("中文字幕提供"), "啾的风格")
    assert d["ok"] and d["score"] >= 0.5
    # kana leak with a clean v1; a single kana is not a leak
    d = judge_content(R("はい大丈夫", "ja"), R("好的没问题"), "好的没问题")
    assert not d["ok"] and d["reason"] == "kana_leak"
    assert judge_content(R("好的没问题ん"), R("好的没问题"), "好的没问题")["ok"]
    # empty conversion reading on a judgeable line → drift_0.00, not script_flip
    d = judge_content(R(""), R("到此为止怎么样"), "到此为止怎么样？")
    assert not d["ok"] and d["reason"] == "drift_0.00"
    # plain-string readings work too (fakes in the guard tests)
    assert judge_content("这么多", "这么多", "这么多")["ok"]
    assert not judge_content("我们都要", "这么多", "这么多")["ok"]


def test_calibration_sample_reproduces():
    """The v3.3 paired rows the thresholds were set on (design + review recomputed them identically)."""
    rows = [("这么多", "这么多", "我们都要", False), ("好疼先等等啊", "好 糖先等等啊", "我唐仙懂得啊", False),
            ("大肉棒", "大肉棒", "大萝卜", False), ("射出来射出来", "射出来射出来", "秀出來", False),
            ("变得好大了呢", "变得好大了呢", "电的好大的呢", True), ("疼得要命", "疼得要命", "腾力要命", True),
            ("那家伙……", "那家伙", "另一家伙", True), ("让坎娜过来", "让卡纳过来", "让你看到过来", True)]
    for want, v1, conv, ok in rows:
        d = judge_content(R(conv), R(v1), want)
        assert d["judged"] and d["ok"] == ok, (want, conv, d)


def test_whisper_clips_cache_and_regime():
    """Dedupe, content-keyed cache (path-independent), error entries not cached, forced-zh bookkeeping."""
    from ai_movie.asr import WhisperClips
    d = Path(tempfile.mkdtemp(prefix="vcdrift_"))
    a, b = d / "a.wav", d / "try1" / "chunks" / "a_copy.wav"
    b.parent.mkdir(parents=True)
    a.write_bytes(b"RIFF" + bytes(200))
    b.write_bytes(a.read_bytes())                        # same content, different path (retry dir)
    c = d / "c.wav"; c.write_bytes(b"RIFF" + bytes(range(100)))
    calls = []

    def fake_decode(paths):
        calls.append(list(paths))
        out = {}
        for p in paths:
            if p.endswith("missing.wav"):
                out[p] = {"auto": {"text": "", "language": None, "error": "FileNotFoundError: x"}, "zh": None}
            else:
                out[p] = {"auto": {"text": "好的", "language": "zh"}, "zh": None}
        return out

    wc = WhisperClips(model_size="tiny", cache=d / "cache.json", batch=1)
    wc._decode_paths = fake_decode
    r = wc.transcribe([str(a), str(a), str(c), str(d / "missing.wav")])
    assert set(r) == {str(a), str(c), str(d / "missing.wav")}
    assert calls == [[str(a), str(c), str(d / "missing.wav")]]              # one call, deduped
    assert (d / "cache.json").exists()
    entries = json.loads((d / "cache.json").read_text())["entries"]
    assert len(entries) == 2                                              # the error entry is not cached
    # a second instance reading the cache: the retry-dir copy of `a` is a hit by content, c too
    wc2 = WhisperClips(model_size="tiny", cache=d / "cache.json")
    wc2._decode_paths = fake_decode
    r2 = wc2.transcribe([str(b), str(c)])
    assert r2[str(b)]["auto"]["text"] == "好的" and len(calls) == 1      # no decode at all
    # a different decode regime (beam) is a different key
    wc3 = WhisperClips(model_size="tiny", cache=d / "cache.json", beam=1)
    wc3._decode_paths = fake_decode
    wc3.transcribe([str(c)])
    assert len(calls) == 2
    assert wc3.batch == 1 and wc.beam == 5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
