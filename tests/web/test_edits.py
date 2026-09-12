"""Web edit semantics against a synthetic workspace (no GPU, no pipeline).

    .venv/bin/python -m pytest tests/web -q      (or run directly)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_pipeline as rp                       # noqa: E402
from ai_movie.web import projects as P          # noqa: E402
from ai_movie.web import edits as E             # noqa: E402

NAME = "webtest_tmp"


def _fp(body: dict) -> dict:
    body = dict(body)
    body["hash"] = rp._fp_hash(body)
    return body


def make_state() -> dict:
    segs = [{"start": 1.0, "end": 2.0, "text": "おはよう", "speaker": "S0", "gender": "female",
             "tts_gender": "female", "asr_conf": 0.9, "speaker_conf": 1.0},
            {"start": 3.0, "end": 4.5, "text": "どうも", "speaker": "S0", "gender": "female",
             "tts_gender": "female", "asr_conf": 0.8, "speaker_conf": 1.0}]
    tr = [dict(s, text_translated=t) for s, t in zip(segs, ["早上好", "你好"])]
    tts = [dict(s, audio=None) for s in tr]
    fit = [dict(s, audio_fit=None, fit_ratio=1.0) for s in tts]
    st = {"_video": "/nonexistent.mp4",
          "asr": {"segments": segs, "diarization": {"speakers": {"S0": {"gender": "female"}}}, "language": "ja"},
          "glossary": {"カンナ": {"zh": "坎娜", "kind": "name"}},
          "translate": {"segments": tr, "variants": {"sakura": ["早上好", "你好"]}, "chosen": "sakura"},
          "tts": {"segments": tts, "refs": {}, "quality": {}, "ok": 2},
          "fit": {"segments": fit}}
    fps = {}
    prev = {}
    for s in ("asr", "glossary", "translate", "tts", "fit"):
        up = {d: prev[d] for d in rp.STEP_DEPS[s] if d in prev}
        fps[s] = _fp({"v": 1, "input": "x", "cfg": {}, "code": {}, "files": {}, "up": up, "extra": {}})
        prev[s] = fps[s]["hash"]
    st["_fp"] = fps
    return st


def setup():
    E.SEED_PATH_OVERRIDE = P.workdir(NAME) / "glossary_seed.json"
    d = P.workdir(NAME)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    rp.save_state(P.state_path(NAME), make_state())


def teardown():
    shutil.rmtree(P.workdir(NAME), ignore_errors=True)


def _valid(state, step):
    fps = state["_fp"]
    body = dict(fps[step])
    return body["hash"] == rp._fp_hash(body)


def _up_matches(state, step):
    fps = state["_fp"]
    return all(fps[dep]["hash"] == h for dep, h in fps[step]["up"].items())


def test_translation_edit():
    setup()
    try:
        E.set_translation(NAME, 1, "你好呀")
        st = P.load_state(NAME)
        assert st["translate"]["segments"][1]["text_translated"] == "你好呀"
        assert st["fit"]["segments"][1]["text_translated"] == "你好呀"
        assert st["_edits"]["translate"] == 1
        assert st["_fp"]["translate"]["extra"]["_edits"] == 1
        assert _valid(st, "translate")
        assert not _up_matches(st, "tts"), "tts must go STALE after a translation edit"
        assert list((P.webdir(NAME) / "state_backups").glob("*.json"))
        # a second edit bumps again
        E.set_translation(NAME, 0, "早安")
        assert P.load_state(NAME)["_edits"]["translate"] == 2
    finally:
        teardown()


def test_speaker_edit_mints_and_propagates():
    setup()
    try:
        r = E.set_speaker(NAME, 0, gender="male")
        assert r["speaker"] == "S1", r
        st = P.load_state(NAME)
        assert st["asr"]["diarization"]["speakers"]["S1"]["gender"] == "male"
        for k in ("translate", "tts", "fit"):
            assert st[k]["segments"][0]["speaker"] == "S1"
            assert st[k]["segments"][0]["gender"] == "male"
        assert st["_edits"]["asr"] == 1
        assert _up_matches(st, "translate"), "translate is kept valid (text unchanged)"
        assert _up_matches(st, "glossary")
        assert not _up_matches(st, "tts"), "tts must go STALE (voice routing changed)"
    finally:
        teardown()


def test_glossary_apply():
    setup()
    try:
        r = E.set_glossary(NAME, {"カンナ": {"zh": "康娜"}, "早上": {"zh": "早晨"}}, apply_to_translation=True)
        st = P.load_state(NAME)
        assert st["glossary"]["カンナ"]["zh"] == "康娜"
        # old rendering 坎娜 did not occur in the text, so no change; 早上 was not in old glossary → no replace
        assert r["applied"] == 0
        assert _up_matches(st, "translate")
    finally:
        teardown()


def test_media_guard():
    assert P.safe_media_path("/etc/passwd") is None
    assert P.safe_media_path(str(ROOT / "workspace" / ".." / "ai_movie" / "config.py")) is None


if __name__ == "__main__":
    for fn in (test_translation_edit, test_speaker_edit_mints_and_propagates, test_glossary_apply, test_media_guard):
        fn()
        print("ok", fn.__name__)
