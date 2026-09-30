"""Sakura draft prompt: speaker tags, prior-turn context, echo handling (no Ollama, no GPU).

    .venv/bin/python tests/test_translate_context.py

``translator._call_ollama_chat`` is replaced by a fake that records every
request, so the tests pin the exact messages each configuration sends:
the v3.3 request with the defaults, the block-append context with
SAKURA_CTX_APPEND, and what happens when the model echoes a tag or a source
line.  Config knobs are monkeypatched on ``ai_movie.config`` — the
translator imports them lazily inside the functions.
"""
from __future__ import annotations

import copy
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie import config as C                 # noqa: E402
from ai_movie import translator as T             # noqa: E402
from ai_movie.units import polish_edit_ok        # noqa: E402

ASK = "将以下日文翻译为口语化中文：\n"


def seg(i, text, speaker="S0", gender="female", overlap=0.0):
    return {"start": float(i), "end": i + 1.0, "text": text, "speaker": speaker,
            "gender": gender, "overlap": overlap}


def dialogue(n=10):
    out = []
    for i in range(n):
        spk, g = ("S0", "female") if i % 2 == 0 else ("S1", "male")
        out.append(seg(i, f"台詞{i}です", spk, g))
    return out


class Fake:
    """Records requests; answers by a function of the target line or a queue."""

    def __init__(self, reply=None, queue=None):
        self.calls: list[list[dict]] = []
        self.reply = reply or (lambda target: "译" + target[2:-2])   # 台詞3です → 译3
        self.queue = list(queue or [])
        self.lock = threading.Lock()

    def __call__(self, model, messages, base_url, timeout=600, options=None, think=None):
        with self.lock:
            self.calls.append(copy.deepcopy(messages))
            if self.queue:
                return self.queue.pop(0)
        target = messages[-1]["content"].split(ASK)[-1]
        target = target.lstrip("[［【").split("]")[-1] if target.startswith("[") else target
        return self.reply(target)


def with_config(**over):
    """Run _sakura_translate under temporary config values; return (out, trace, fake)."""
    def run(segments, fake):
        saved = {k: getattr(C, k) for k in over}
        orig = T._call_ollama_chat
        T._call_ollama_chat = fake
        for k, v in over.items():
            setattr(C, k, v)
        try:
            trace: list[dict] = []
            out = T._sakura_translate(segments, model="m", base_url="http://x", glossary=None, trace=trace)
            return out, trace, fake
        finally:
            T._call_ollama_chat = orig
            for k, v in saved.items():
                setattr(C, k, v)
    return run


def assistant_turns(msgs):
    return [m["content"] for m in msgs if m["role"] == "assistant"]


def request_for(fake, i):
    """The (last) request whose target line is unit *i*."""
    hits = [m for m in fake.calls if m[-1]["content"].endswith(f"台詞{i}です")]
    assert hits, i
    return hits[-1]


# ── builders ───────────────────────────────────────────────────────

def test_speaker_tag_overlap_rule():
    assert T._speaker_tag({"speaker": "S0", "gender": "female"}) == "[S0女]"
    assert T._speaker_tag({"speaker": "S1", "tts_gender": "male", "overlap": 0.5}) == "[S1男]"
    # above OSD_SEED_EXCLUDE the *label* is unreliable, so no id at all
    assert T._speaker_tag({"speaker": "S0", "gender": "female", "overlap": 0.8}) == ""
    assert T._speaker_tag({"speaker": "S0", "gender": "female", "overlap": "0.9"}) == ""
    assert T._speaker_tag({"gender": "female"}) == ""
    assert T._speaker_tag({"speaker": "S2", "gender": "unknown"}) == "[S2]"


def test_sakura_messages_shape():
    segs = dialogue(8)
    segs[5]["overlap"] = 0.8
    prior = [(T._speaker_tag(segs[j]), segs[j]["text"], f"译{j}") for j in range(1, 5)]
    msgs = T._sakura_messages(segs[5], prior, glossary=None, tags=True)
    assert [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * 4 + ["user"]
    assert msgs[0]["content"] == C.OLLAMA_SAKURA_TRANSLATE_PROMPT + "\n" + C.OLLAMA_SAKURA_TAG_RULE
    users = [m["content"] for m in msgs if m["role"] == "user"]
    assert users[:4] == [f"{ASK}[S1男]台詞1です", f"{ASK}[S0女]台詞2です",
                         f"{ASK}[S1男]台詞3です", f"{ASK}[S0女]台詞4です"]
    assert assistant_turns(msgs) == ["译1", "译2", "译3", "译4"]      # never tagged
    assert users[-1] == f"{ASK}台詞5です"                              # overlapped: no speaker id
    # exactly one Japanese line per user message, the target last; no 「A」→「B」 block anywhere
    for u in users:
        assert u.count(ASK) == 1 and "\n" not in u.split(ASK)[1]
    assert not any("→" in m["content"] for m in msgs if m["role"] != "system")   # (the prompt says 日文→中文)
    assert not any(T._looks_like_context_echo(m["content"]) for m in msgs)
    # tags off: byte-identical to the v3.3 request
    msgs = T._sakura_messages(segs[5], prior, glossary=None, tags=False)
    assert msgs[0]["content"] == C.OLLAMA_SAKURA_TRANSLATE_PROMPT
    assert not any("[S" in m["content"] for m in msgs)
    assert [m["content"] for m in msgs if m["role"] == "user"][0] == f"{ASK}台詞1です"
    # glossary head stays in front of the ask, as before
    gl = {"台詞": {"zh": "台词", "kind": "term"}}
    msgs = T._sakura_messages(segs[4], [], glossary=gl, tags=True)
    assert msgs[-1]["content"].startswith("固定译名：") and msgs[-1]["content"].endswith(f"{ASK}[S0女]台詞4です")


def test_strip_tag_echo():
    assert T._strip_tag_echo("[S0女]我回来了") == ("我回来了", False)
    assert T._strip_tag_echo("【S1 男】：我回来了") == ("我回来了", False)
    assert T._strip_tag_echo("［S2］我回来了") == ("我回来了", False)
    assert T._strip_tag_echo("我回来了") == ("我回来了", False)
    assert T._strip_tag_echo("我[S1男]回来了") == ("我[S1男]回来了", True)
    assert T._strip_tag_echo("") == ("", False)


def test_echo_detection_needs_kana_and_a_source():
    assert T._sakura_echo("台詞3です", ["台詞3です"])
    assert T._sakura_echo("台詞3です。", ["台詞3です"])          # punctuation folded
    assert not T._sakura_echo("译3", ["台詞3です"])              # Chinese: never an echo
    assert not T._sakura_echo("台詞4です", ["台詞3です"])         # kana but not a source
    assert not T._sakura_echo("", ["台詞3です"])


# ── context semantics ──────────────────────────────────────────────

def test_default_config_matches_v33():
    """Defaults: tags off, last two finished pairs, frozen per block of four."""
    run = with_config(SAKURA_SPEAKER_TAGS=False, SAKURA_CTX_BEFORE=2, SAKURA_CTX_APPEND=False,
                      SAKURA_CTX_BLOCK=4, OLLAMA_SAKURA_CONCURRENCY=4)
    out, trace, fake = run(dialogue(10), Fake())
    assert out == [f"译{i}" for i in range(10)]
    assert len(fake.calls) == 10
    assert assistant_turns(request_for(fake, 0)) == []
    assert assistant_turns(request_for(fake, 3)) == []                # same block as 0: nothing yet
    assert assistant_turns(request_for(fake, 4)) == ["译2", "译3"]
    assert assistant_turns(request_for(fake, 7)) == ["译2", "译3"]    # frozen for the block
    assert assistant_turns(request_for(fake, 8)) == ["译6", "译7"]
    for msgs in fake.calls:
        assert msgs[0]["content"] == C.OLLAMA_SAKURA_TRANSLATE_PROMPT
        assert not any("[S" in m["content"] for m in msgs)
    assert [r["n_before"] for r in sorted(trace, key=lambda r: r["idx"])] == [0, 0, 0, 0, 2, 2, 2, 2, 2, 2]
    assert all(r["retry_bare"] == 0 and r["empty"] == 0 for r in trace)


def test_block_append_context():
    """SAKURA_CTX_APPEND: every finished unit of the block becomes a real turn."""
    run = with_config(SAKURA_SPEAKER_TAGS=True, SAKURA_CTX_BEFORE=4, SAKURA_CTX_APPEND=True,
                      SAKURA_CTX_BLOCK=4)
    out, trace, fake = run(dialogue(12), Fake())
    assert out == [f"译{i}" for i in range(12)]
    assert assistant_turns(request_for(fake, 4)) == ["译0", "译1", "译2", "译3"]
    assert assistant_turns(request_for(fake, 6)) == [f"译{j}" for j in range(6)]
    assert assistant_turns(request_for(fake, 7)) == [f"译{j}" for j in range(7)]     # 4 + 3
    assert assistant_turns(request_for(fake, 9)) == [f"译{j}" for j in range(4, 9)]  # snapshot 4..7 + 8
    # within a block the next request is a pure append to the previous one (the cached prefix)
    for i in (4, 5, 6, 8, 9, 10):
        a, b = request_for(fake, i), request_for(fake, i + 1)
        assert b[:len(a)] == a and b[len(a)]["content"] == f"译{i}" and len(b) == len(a) + 2
    # user lines carry the tag of their speaker; assistant lines never do
    msgs = request_for(fake, 6)
    users = [m["content"] for m in msgs if m["role"] == "user"]
    assert users[0] == f"{ASK}[S0女]台詞0です" and users[1] == f"{ASK}[S1男]台詞1です"
    assert users[-1] == f"{ASK}[S0女]台詞6です"
    assert not any("[S" in t for t in assistant_turns(msgs))
    # the ask appears exactly once per user message: never a multi-line look-around block
    for m in fake.calls:
        for u in (x["content"] for x in m if x["role"] == "user"):
            assert u.count(ASK) == 1 and "\n" not in u.split(ASK)[1]
    assert [r["n_before"] for r in sorted(trace, key=lambda r: r["idx"])] == [0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7]


def test_exact_sliding_when_block_is_one():
    run = with_config(SAKURA_SPEAKER_TAGS=False, SAKURA_CTX_BEFORE=4, SAKURA_CTX_APPEND=True,
                      SAKURA_CTX_BLOCK=1)
    out, trace, fake = run(dialogue(8), Fake())
    assert assistant_turns(request_for(fake, 7)) == ["译3", "译4", "译5", "译6"]
    assert assistant_turns(request_for(fake, 2)) == ["译0", "译1"]
    # prefix is the system prompt only once the window is full: that is the 3× prompt-eval cost
    a, b = request_for(fake, 6), request_for(fake, 7)
    assert a[1] != b[1]


def test_empty_source_and_failed_units_do_not_enter_context():
    run = with_config(SAKURA_SPEAKER_TAGS=True, SAKURA_CTX_BEFORE=4, SAKURA_CTX_APPEND=True,
                      SAKURA_CTX_BLOCK=4)
    segs = dialogue(6)
    segs[2]["text"] = ""
    fake = Fake(reply=lambda t: "" if t == "台詞3です" else "译" + t[2:-2])
    out, trace, fake = run(segs, fake)
    assert out[2] == "" and out[3] == ""
    assert assistant_turns(request_for(fake, 5)) == ["译0", "译1", "译4"]
    rows = {r["idx"]: r for r in trace}
    assert rows[2]["empty"] == 0 and rows[2]["retry_bare"] == 0          # nothing to ask: not a failure
    assert rows[3]["empty"] == 1 and rows[3]["retry_bare"] == 1          # asked twice, still empty


# ── echo / leak handling ───────────────────────────────────────────

def test_tag_echo_and_bare_retry():
    run = with_config(SAKURA_SPEAKER_TAGS=True, SAKURA_CTX_BEFORE=4, SAKURA_CTX_APPEND=True,
                      SAKURA_CTX_BLOCK=4)
    # 1. leading tag echoed → stripped, no retry, no leak
    out, trace, fake = run(dialogue(1), Fake(queue=["[S0女]我回来了"]))
    assert out == ["我回来了"] and len(fake.calls) == 1
    assert trace[0]["tag_leak"] == 0 and trace[0]["retry_bare"] == 0
    # 2. the target source echoed (kana) → one bare retry, second reply wins
    out, trace, fake = run(dialogue(1), Fake(queue=["台詞0です", "我回来了"]))
    assert out == ["我回来了"] and len(fake.calls) == 2
    assert trace[0]["retry_bare"] == 1 and trace[0]["echo"] == 0 and trace[0]["empty"] == 0
    assert not any("[S" in m["content"] for m in fake.calls[1])         # retry is the untagged request
    assert fake.calls[1][0]["content"] == C.OLLAMA_SAKURA_TRANSLATE_PROMPT
    # 3. echo twice → counted, but the kana line is kept (the F3 polish job can still rescue it;
    #    a blank line would be silent)
    out, trace, fake = run(dialogue(1), Fake(queue=["台詞0です", "台詞0です"]))
    assert out == ["台詞0です"] and trace[0]["empty"] == 0 and trace[0]["echo"] == 1
    # 4. tag leaked mid-line → retry; a leak that survives the retry is dropped, never spoken
    out, trace, fake = run(dialogue(1), Fake(queue=["我[S1男]回来了", "我回来了"]))
    assert out == ["我回来了"] and trace[0]["retry_bare"] == 1 and trace[0]["tag_leak"] == 0
    out, trace, fake = run(dialogue(1), Fake(queue=["我[S1男]回来了", "我[S1男]回来了"]))
    assert out == [""] and trace[0]["tag_leak"] == 1 and trace[0]["empty"] == 1
    # 5. empty twice → "" with empty=1
    out, trace, fake = run(dialogue(1), Fake(queue=["", ""]))
    assert out == [""] and trace[0]["empty"] == 1 and trace[0]["retry_bare"] == 1
    # 6. an echo of a *prior* source line is caught too
    out, trace, fake = run(dialogue(2), Fake(queue=["译0", "台詞0です", "译1"]))
    assert out == ["译0", "译1"] and trace[1]["retry_bare"] == 1


def test_untagged_path_has_no_retry():
    run = with_config(SAKURA_SPEAKER_TAGS=False, SAKURA_CTX_BEFORE=2, SAKURA_CTX_APPEND=False,
                      SAKURA_CTX_BLOCK=4)
    out, trace, fake = run(dialogue(1), Fake(queue=["", "我回来了"]))
    assert out == [""] and len(fake.calls) == 1 and trace[0]["empty"] == 1 and trace[0]["retry_bare"] == 0
    # v3.3 behaviour on an echo: the kana line is kept for the F3 polish, only counted here
    out, trace, fake = run(dialogue(1), Fake(queue=["台詞0です", "我回来了"]))
    assert out == ["台詞0です"] and len(fake.calls) == 1 and trace[0]["echo"] == 1


def test_multi_line_reply_is_counted_and_first_line_kept():
    run = with_config(SAKURA_SPEAKER_TAGS=True, SAKURA_CTX_BEFORE=4, SAKURA_CTX_APPEND=True,
                      SAKURA_CTX_BLOCK=4)
    out, trace, fake = run(dialogue(1), Fake(queue=["我回来了\n欢迎回来"]))
    assert out == ["我回来了"] and trace[0]["n_lines"] == 2 and trace[0]["retry_bare"] == 0


def test_translate_segments_passes_trace_through():
    import inspect
    assert "trace" in inspect.signature(T.translate_segments).parameters
    assert "trace" in inspect.signature(T._sakura_translate).parameters


# ── polish guard fixture (design test 5, corrected) ─────────────────

def test_polish_edit_ok_pronoun_deletion():
    # pure deletion of an invented 你 is accepted …
    assert polish_edit_ok("我可能喜欢上你了。", "我可能喜欢上了。", ["F1_pronoun"])
    # … a 你→上 substitution is not (the original fixture was this and would fail)
    assert not polish_edit_ok("我可能喜欢你了。", "我可能喜欢上了。", ["F1_pronoun"])
    # swapping the person is allowed; the guard does not force deletion
    assert polish_edit_ok("她说可以了", "你说可以了", ["F1_pronoun"])
    # a tag leaked into the candidate is refused (S/0/女 are not insertable)
    assert not polish_edit_ok("我回来了", "[S0女]我回来了", ["F1_pronoun"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
