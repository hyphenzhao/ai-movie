"""Pronoun agreement with the reference Chinese (no GPU, no film needed).

    .venv/bin/python tests/test_pronoun_agreement.py

The metric behind release gate H5 and eval_long L7: 你(含您)/他/她 present in
the reference vs present in our line, per scored cue.  Unlike the old
``pronoun_units`` count it gets *worse* when a pronoun the reference has is
deleted, which is the failure the shipped v3.3 output actually shows.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_against_subs as subs                 # noqa: E402

spec = importlib.util.spec_from_file_location("eval_long", ROOT / "scripts" / "eval_long.py")
el = importlib.util.module_from_spec(spec); spec.loader.exec_module(el)     # type: ignore[union-attr]


def g(ref, ours, cue=0, start=0.0):
    return {"cue": cue, "start": start, "ref_zh": ref, "ours_zh": ours}


def test_pronoun_classes():
    assert subs.pronoun_classes("你好") == {"你"}
    assert subs.pronoun_classes("您好") == {"你"}                 # 您 is the polite 你
    assert subs.pronoun_classes("他和她") == {"他", "她"}
    assert subs.pronoun_classes("我们走吧") == set()                # first person is not scored
    assert subs.pronoun_classes("") == set() and subs.pronoun_classes(None) == set()


def test_agreement_counts():
    groups = [
        g("所有的粉丝都在看著你哦", "大家的粉丝都在看着哦"),       # missing 你
        g("她说可以了", "你说可以了"),                             # extra 你 AND missing 她
        g("好厉害", "我可能喜欢上你了"),                           # extra 你
        g("你扮演的是学生", "您扮演的是学生"),                     # agree (您 = 你)
        g("是的", "没错"),                                          # agree (no pronouns)
        g("你好", ""),                                              # not scored: no line of ours
        g("", "你好"),                                              # not scored: no reference
        {"cue": None, "start": 9.0, "ours_zh": "嗯"},               # loose run without ref_zh key
    ]
    pa = subs.pronoun_agreement(groups)
    assert pa["scored"] == 5, pa
    assert pa["agree"] == 2 and pa["extra"] == 2 and pa["missing"] == 2, pa
    assert pa["mismatch"] == 4
    ex = {e["ref"]: e for e in pa["examples"]}
    assert ex["她说可以了"]["extra"] == "你" and ex["她说可以了"]["missing"] == "她"
    assert ex["所有的粉丝都在看著你哦"]["extra"] == "" and ex["所有的粉丝都在看著你哦"]["missing"] == "你"


def test_deletion_cannot_improve_it():
    # the old F1 count falls when 你 is deleted; this metric rises
    before = subs.pronoun_agreement([g("你扮演的是学生", "你演的是学生")])
    after = subs.pronoun_agreement([g("你扮演的是学生", "演的是学生")])
    assert before["mismatch"] == 0 and after["mismatch"] == 1


def test_summarise_includes_it_only_with_reference():
    segs = [{"start": 0.0, "end": 1.0, "text": "見てますね", "text_translated": "在看着哦", "speaker": "S0"}]
    groups = subs.build_groups(segs, [{"id": 0, "ja": "見てますね", "start": 0.0, "end": 1.0}])
    assert "pronoun_agreement" not in subs.summarise(groups, segs)
    subs.attach_reference_zh(groups, [(0.0, 1.0, "都在看著你哦")])
    s = subs.summarise(groups, segs)
    assert s["pronoun_agreement"]["scored"] == 1 and s["pronoun_agreement"]["missing"] == 1
    # the report renders the section
    assert "人称代词与参考译文" in subs.render("x", groups, segs, s, 5)


def test_eval_long_l7_pairs_by_overlap():
    truth = [
        {"id": 0, "start": 10.0, "end": 12.0, "zh": "你出演了第一部电视剧"},
        {"id": 1, "start": 20.0, "end": 21.0, "zh": "她说可以了"},
        {"id": 2, "start": 30.0, "end": 31.0, "zh": "没有人说话"},          # no line overlaps: not scored
        {"id": 3, "start": 40.0, "end": 41.0, "zh": ""},                   # cue without Chinese: skipped
    ]
    segs = [
        {"t0": 9.0, "t1": 10.2, "text_translated": "第一次的"},             # 0.2 s overlap: below the floor
        {"t0": 10.2, "t1": 11.5, "text_translated": "我演了短剧。"},          # scored, missing 你
        {"t0": 20.1, "t1": 20.9, "text_translated": "她说可以了"},           # agree
        {"t0": 40.0, "t1": 41.0, "text_translated": "你好"},
    ]
    pa = el.pronoun_check(truth, segs)
    assert pa["scored"] == 2 and pa["agree"] == 1 and pa["missing"] == 1 and pa["extra"] == 0, pa
    # a different text key scores the drafts of an A/B arm on the same cues
    for s in segs:
        s["draft"] = s["text_translated"].replace("我演了", "你演了")
    assert el.pronoun_check(truth, segs, key="draft")["missing"] == 0
    assert el.L7_MAX_MISMATCH == 48      # shipped v3.3: 9 extra + 39 missing on 445 cues


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
