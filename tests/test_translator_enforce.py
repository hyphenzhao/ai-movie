"""enforce_glossary / compact_translation on the shared instruct model (no GPU, no Ollama).

Ollama is replaced by a fake ``urllib.request.urlopen`` (the module
attribute — _call_ollama_chat imports the module inside the function), so
the tests see exactly the request body the pipeline would send: the
thinking-capable model must always get ``think: false`` and the rewrite
channels ``repeat_penalty 1.0``.  The edit guard, the candidate cleaning and
the report plumbing (segment index, one CSV per row shape) are pure.

    .venv/bin/python tests/test_translator_enforce.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie import config as C                                # noqa: E402
from ai_movie import translator as T                            # noqa: E402

GLOSS = {"かんな": {"zh": "坎娜", "kind": "name"},
         "カンナ": {"zh": "坎娜", "kind": "name"},
         "ドラマ": {"zh": "短剧", "kind": "term"},
         "メラネス": {"zh": "梅拉尼斯", "kind": "name"}}


class FakeOllama:
    """``urlopen`` stand-in: answers /api/chat from *reply(prompt)* and keeps every body."""

    def __init__(self, reply):
        self.reply = reply
        self.bodies: list[dict] = []

    def __call__(self, req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        self.bodies.append(body)
        prompt = body["messages"][-1]["content"]
        content = self.reply(prompt)
        if isinstance(content, Exception):
            raise content
        payload = json.dumps({"message": {"role": "assistant", "content": content}}).encode("utf-8")
        return contextlib.closing(io.BytesIO(payload))


@contextlib.contextmanager
def fake_ollama(reply):
    fake = FakeOllama(reply)
    orig = urllib.request.urlopen
    urllib.request.urlopen = fake
    try:
        yield fake
    finally:
        urllib.request.urlopen = orig


def _line(prompt: str, label: str) -> str:
    return prompt.split(label, 1)[1].splitlines()[0].strip()


# ── config ────────────────────────────────────────────────────────

def test_one_instruct_model_for_the_three_rewrite_uses():
    assert C.COMPACT_MODEL == C.OLLAMA_POLISH_MODEL
    assert C.GLOSSARY_ENFORCE_MODEL == C.OLLAMA_POLISH_MODEL
    assert "dolphin" not in C.COMPACT_MODEL
    # eviction behaviour unchanged: the instruct model keeps Sakura resident
    assert C.OLLAMA_MODEL_SIZE_GB[C.OLLAMA_POLISH_MODEL] < C.OLLAMA_EXCLUSIVE_ABOVE_GB
    assert "dolphin-mixtral:8x7b" in C.OLLAMA_MODEL_SIZE_GB       # smoke-test fallback stays known


# ── the edit guard ────────────────────────────────────────────────

def test_guard_accepts_every_documented_wrong_rendering():
    for wrong in ("卡娜", "加奈", "卡恩娜", "小蓝华", "镰鼬", "小勘娜", "小卡娜"):
        zh = f"我是S1专属套装的{wrong}。"
        assert T._enforce_edit_ok(zh, "我是S1专属套装的坎娜。", ["坎娜"]) == "accepted", wrong
    # the name alone: the old _char_overlap guard returned 0.0 here (nothing left after the pin)
    assert T._enforce_edit_ok("卡娜", "坎娜", ["坎娜"]) == "accepted"
    assert T._enforce_edit_ok("小卡娜。", "坎娜。", ["坎娜"]) == "accepted"
    assert T._enforce_edit_ok("我演了电视剧。", "我演了短剧。", ["短剧"]) == "accepted"
    # re-punctuation only is not a rewrite
    assert T._enforce_edit_ok("卡娜来了", "坎娜来了。", ["坎娜"]) == "accepted"
    # two occurrences when the source names the term twice
    zh, cand = "这边的美拉尼斯，那边的美拉尼斯", "这边的梅拉尼斯，那边的梅拉尼斯"
    assert T._enforce_edit_ok(zh, cand, ["梅拉尼斯"], {"梅拉尼斯": 2}) == "accepted"
    assert T._enforce_edit_ok(zh, cand, ["梅拉尼斯"], {"梅拉尼斯": 1}) == "rejected_rewrote"


def test_guard_rejects_rewrites_and_missing_pins():
    ok = T._enforce_edit_ok
    assert ok("小卡娜来了", "小卡娜来了", ["坎娜"]) == "rejected_no_pin"      # unchanged
    assert ok("小卡娜来了", "", ["坎娜"]) == "rejected_no_pin"
    assert ok("我演了电视剧。", "我演了电影。", ["短剧"]) == "rejected_no_pin"
    # 电视剧 → 电影 style substitution beside the pin (5/6 chars shared passed _char_overlap)
    assert ok("小卡娜来了", "坎娜走了", ["坎娜"]) == "rejected_rewrote"
    assert ok("我演了电视剧。", "我演了短剧，很好看。", ["短剧"]) == "rejected_rewrote"
    assert ok("我是S1专属套装的卡娜。", "我是S1专属的坎娜。", ["坎娜"]) == "rejected_rewrote"
    assert ok("她说小卡娜来了", "坎娜来了", ["坎娜"]) == "rejected_rewrote"    # deleted 她说 too
    assert ok("小卡娜来了", "坎娜坎娜来了", ["坎娜"]) == "rejected_rewrote"     # duplicated pin
    assert ok("卡娜来了", "坎娜ちゃん来了", ["坎娜"]) == "rejected_rewrote"    # kana leak


def test_candidate_cleaning():
    assert T._clean_enforce_candidate("改正后：「我是坎娜。」") == "我是坎娜。"
    assert T._clean_enforce_candidate("“我是坎娜。”\n解释：因为术语表") == "我是坎娜。"
    assert T._clean_enforce_candidate("输出: 我是坎娜。") == "我是坎娜。"
    assert T._clean_enforce_candidate("") == ""


# ── enforce_glossary end to end (fake Ollama) ─────────────────────

SEGS = [{"text": "S1専属セットカンナです。", "seg_idx": 6},
        {"text": "わかんない", "seg_idx": 73},                    # not the name: never attempted
        {"text": "かんなちゃんが思う", "seg_idx": 54},
        {"text": "ドラマをやりました。", "seg_idx": 15},
        {"text": "カンナちゃん。", "seg_idx": 8}]
ZH = ["我是S1专属套装的卡娜。", "不知道", "小坎娜这么想。", "我演了电视剧。", "小加奈。"]


def test_enforce_sends_think_false_and_reports_segment_index():
    def reply(prompt):
        line = _line(prompt, "待改正：")
        if "电视剧" in line:
            return "「我演了电影。」"                                # wrong term → rejected
        return "改正后：" + line.replace("卡娜", "坎娜").replace("加奈", "坎娜")

    rows: list[dict] = []
    with fake_ollama(reply) as fake:
        out = T.enforce_glossary(SEGS, ZH, GLOSS, model="m", base_url="http://x", report=rows)
    assert out == ["我是S1专属套装的坎娜。", "不知道", "小坎娜这么想。", "我演了电视剧。", "小坎娜。"]
    assert [b["think"] for b in fake.bodies] == [False, False, False]
    for b in fake.bodies:
        assert b["options"]["repeat_penalty"] == 1.0 and b["options"]["temperature"] == 0.0
        assert b["options"]["num_predict"] >= 64
        assert "不要加引号" in b["messages"][-1]["content"]
    assert [(r["idx"], r["status"]) for r in rows] == [
        (6, "accepted"), (15, "rejected_no_pin"), (8, "accepted")]
    assert rows[0]["pins"] == "カンナ→坎娜" and rows[0]["before"] == ZH[0]
    assert rows[0]["candidate"] == "我是S1专属套装的坎娜。"
    # every row has the same columns (one DictWriter per file)
    assert all(list(r) == list(rows[0]) for r in rows)


def test_enforce_error_row_and_option_override():
    def reply(prompt):
        return TimeoutError("boom")

    rows: list[dict] = []
    with fake_ollama(reply) as fake:
        out = T.enforce_glossary(SEGS[:1], ZH[:1], GLOSS, model="m", base_url="http://x",
                                 report=rows, options={"repeat_penalty": 1.15})
    assert out == ZH[:1]
    assert rows[0]["status"] == "error" and "TimeoutError" in rows[0]["candidate"]
    assert fake.bodies[0]["options"]["repeat_penalty"] == 1.15
    # no report list → same behaviour, nothing recorded
    with fake_ollama(reply):
        assert T.enforce_glossary(SEGS[:1], ZH[:1], GLOSS, model="m", base_url="http://x") == ZH[:1]
    # idx falls back to the position when the caller did not tag segments
    rows = []
    with fake_ollama(lambda p: _line(p, "待改正：").replace("卡娜", "坎娜")):
        T.enforce_glossary([{"text": SEGS[0]["text"]}], ZH[:1], GLOSS, model="m",
                           base_url="http://x", report=rows)
    assert rows[0]["idx"] == 0 and rows[0]["status"] == "accepted"


def test_compact_sends_think_false():
    def reply(prompt):
        return "「我演了短剧」\n（解释）"

    with fake_ollama(reply) as fake:
        cand = T.compact_translation("ドラマをやりました。", "我之前演了一部短剧哦。", 5,
                                     glossary=GLOSS, model="m", base_url="http://x")
    assert cand == "我演了短剧"
    assert fake.bodies[0]["think"] is False
    assert fake.bodies[0]["options"]["repeat_penalty"] == 1.0
    # an empty reply (what a thinking model returns without think=False) → None, not a crash
    with fake_ollama(lambda p: ""):
        assert T.compact_translation("ドラマをやりました。", "我之前演了一部短剧哦。", 5,
                                     glossary=GLOSS, model="m", base_url="http://x") is None


# ── run_pipeline plumbing ─────────────────────────────────────────

def _load_run_pipeline():
    spec = importlib.util.spec_from_file_location("run_pipeline", ROOT / "scripts" / "run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                  # type: ignore[union-attr]
    return mod


class FakeTranslator:
    """translate_segments stand-in: one enforce row per segment, keyed by seg_idx."""

    def __init__(self, answers):
        self.answers = answers
        self.calls: list[list[dict]] = []

    def translate_segments(self, segments, *, engine, glossary, report=None,
                           enforce_report=None, progress_cb=None):
        self.calls.append(segments)
        for s in segments:
            if enforce_report is not None:
                enforce_report.append({"idx": s.get("seg_idx"), "ja": s["text"], "pins": "",
                                       "before": "", "candidate": "", "status": "accepted"})
        return [self.answers[s["text"]] for s in segments]


def test_translate_by_units_tags_segment_indexes():
    rp = _load_run_pipeline()
    segs = [{"start": 0.0, "end": 1.0, "text": "あっ", "speaker": "S0", "keep_original": True},
            {"start": 1.0, "end": 2.0, "text": "緊張したけれど、", "speaker": "S0"},
            {"start": 2.1, "end": 3.0, "text": "一応先生と生徒役", "speaker": "S0"},
            {"start": 4.0, "end": 5.0, "text": "はい", "speaker": "S1"},
            {"start": 5.1, "end": 6.0, "text": "そう", "speaker": "S1"}]
    units = [[0], [1, 2], [3, 4]]
    answers = {"あっ": "", "緊張したけれど、一応先生と生徒役": "虽然紧张，但还是老师和学生的角色",
               "はいそう": "嗯。",                                   # too short to split → fallback
               "はい": "是", "そう": "对"}
    tr = FakeTranslator(answers)
    polish_rows: list[dict] = []
    unit_rows: list[dict] = []
    enforce_rows: list[dict] = []
    out = rp._translate_by_units(tr, segs, units, "sakura+qwen", {}, polish_rows, unit_rows,
                                 enforce_rows)
    assert out[0] == "" and out[3] == "是" and out[4] == "对"
    # the kept-original unit was never sent; the unit and the fallback segments were
    assert [s["seg_idx"] for s in tr.calls[0]] == [1, 3]
    assert [s["seg_idx"] for s in tr.calls[1]] == [3, 4]
    assert [r["idx"] for r in enforce_rows] == [1, 3, 3, 4]
    # pseudo segments never leak into the stored segments
    assert all("seg_idx" not in s for s in segs)
    assert [s.get("unit_id") for s in segs] == [0, 1, 1, 2, 2]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
