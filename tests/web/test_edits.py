"""Web edit semantics against a synthetic workspace (no GPU, no pipeline).

    .venv/bin/python tests/web/test_edits.py      (or via pytest)

The synthetic projects live in a temp directory (``P.WORKSPACE`` is repointed):
``workspace/`` holds the live release data and must not be touched.  Covers the
fingerprint effect of every edit, the edit log (``edits.jsonl``: who / when /
before / after / consumers), the ``-NNN`` backups, and the ``runs.py`` viewer.
"""

from __future__ import annotations

import atexit
import io
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_pipeline as rp                       # noqa: E402
from ai_movie.web import projects as P          # noqa: E402
from ai_movie.web import edits as E             # noqa: E402
import runs                                     # noqa: E402  (scripts/runs.py viewer)

TMP_WS = Path(tempfile.mkdtemp(prefix="webtest_ws_"))
P.WORKSPACE = TMP_WS
runs.WS = TMP_WS
atexit.register(lambda: shutil.rmtree(TMP_WS, ignore_errors=True))

NAME = "webtest_tmp"
FILM = "webtest_film"
CHUNK = f"{FILM}_p01"          # chunk without an enrol fingerprint (legacy branch)
CHUNK2 = f"{FILM}_p02"         # chunk with an enrol fingerprint


def _fp(body: dict) -> dict:
    body = dict(body)
    body["hash"] = rp._fp_hash(body)
    return body


def make_state(with_enrol: bool = False) -> dict:
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
    stages = ["asr", "glossary", "translate"] + (["enrol"] if with_enrol else []) + ["tts", "fit"]
    for s in stages:
        up = {d: prev[d] for d in rp.STEP_DEPS[s] if d in prev}
        fps[s] = _fp({"v": 1, "input": "x", "cfg": {}, "code": {}, "files": {}, "up": up, "extra": {}})
        prev[s] = fps[s]["hash"]
    st["_fp"] = fps
    return st


def setup(name: str = NAME, state: dict | None = None):
    E.SEED_PATH_OVERRIDE = P.workdir(name) / "glossary_seed.json"
    d = P.workdir(name)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    rp.save_state(P.state_path(name), state or make_state())


def teardown(*names: str):
    for n in names or (NAME,):
        shutil.rmtree(P.workdir(n), ignore_errors=True)


def _valid(state, step):
    fps = state["_fp"]
    body = dict(fps[step])
    return body["hash"] == rp._fp_hash(body)


def _up_matches(state, step):
    fps = state["_fp"]
    return all(fps[dep]["hash"] == h for dep, h in fps[step]["up"].items())


def _log(name: str = NAME, film: bool = False) -> list[dict]:
    p = (P.WORKSPACE / name / E.EDIT_LOG_NAME)
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def _backups(name: str = NAME) -> list[Path]:
    return sorted((P.workdir(name) / "web" / "state_backups").glob("*.json"))


# ── fingerprint semantics (pre-existing) ──

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
        assert _backups()
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


# ── edit log ──

def test_translation_logged():
    st0 = make_state()
    st0["tts"]["segments"][1]["text_translated_full"] = "你好，请多关照"     # compact-stage original lives downstream
    setup(state=st0)
    try:
        E.set_translation(NAME, 1, "你好呀", who="tester")
        rows = _log()
        assert len(rows) == 1
        e = rows[0]
        assert e["kind"] == "set_translation" and e["stage"] == "translate" and e["idx"] == 1
        assert e["before"] == {"text_translated": "你好", "text_translated_full": "你好，请多关照"}, e["before"]
        assert e["after"] == {"text_translated": "你好呀"}
        assert e["speaker"] == {"before": "S0", "after": "S0"} and e["who"] == "tester"
        assert e["edits_n"] == 1 and e["restamped"] is True and e["consumers"] == ["tts"]
        assert e["text"] == "どうも" and e["variant"] == "sakura" and e["stages_touched"] == ["tts", "fit"]
        assert "t" in e and len(e["at"]) == 19
        # the backup named in the record exists and is the PRE-edit state
        bk = P.workdir(NAME) / "web" / "state_backups" / e["backup"]
        assert bk.exists() and bk.name.endswith("-001.json"), bk
        assert json.loads(bk.read_text())["translate"]["segments"][1]["text_translated"] == "你好"
        assert "text_translated_full" not in P.load_state(NAME)["tts"]["segments"][1]
        E.set_translation(NAME, 0, "早安")
        rows = _log()
        assert len(rows) == 2 and rows[1]["edits_n"] == 2 and rows[1]["who"] == "cli"
        assert rows[1]["before"]["text_translated_full"] is None
    finally:
        teardown()


def test_speaker_edit_logged():
    setup()
    try:
        E.set_speaker(NAME, 0, gender="male", who="tester")
        e = _log()[-1]
        assert e["kind"] == "set_speaker" and e["stage"] == "asr" and e["idx"] == 0
        assert e["speaker"] == {"before": "S0", "after": "S1"}
        assert e["before"]["gender"] == "female" and e["after"]["gender"] == "male"
        assert e["after"]["tts_gender"] == "male"
        assert e["minted"] == ["S1"] and e["requested"] == {"speaker": None, "gender": "male"}
        assert e["propagated"] == ["translate", "tts", "fit"] and e["not_propagated"] == []
        assert e["consumers"] == ["tts"] and e["keep_valid"] == ["glossary", "translate"]
        assert e["restamped"] is True and e["edits_n"] == 1
        E.set_speaker(NAME, 1, speaker="S1")
        e2 = _log()[-1]
        assert e2["requested"] == {"speaker": "S1", "gender": None} and e2["minted"] == []
        assert e2["speaker"] == {"before": "S0", "after": "S1"} and e2["edits_n"] == 2
    finally:
        teardown()


def test_propagate_returns_relabelled_stages_and_skips_drift():
    st = make_state()
    st["fit"]["segments"][1]["start"] = 3.3           # drifted line: must not be relabelled
    src = dict(st["asr"]["segments"][1], speaker="S9", gender="male", tts_gender="male")
    done = E._propagate_labels(st, 1, src)
    assert done == ["translate", "tts"], done
    assert st["fit"]["segments"][1]["speaker"] == "S0"
    assert st["tts"]["segments"][1]["speaker"] == "S9"


def test_add_speaker_and_binding_logged():
    setup()
    try:
        r = E.add_speaker(NAME, "male", who="tester")
        e = _log()[-1]
        assert e["kind"] == "add_speaker" and e["stage"] == "asr" and e["idx"] is None
        assert e["before"] is None and e["after"] == {"speaker": r["speaker"], "gender": "male"}
        assert e["speaker"] == {"before": None, "after": "S1"}
        assert e["restamped"] is False and e["consumers"] == [] and e["backup"]
        assert "_edits" not in P.load_state(NAME)
        E.set_face_binding(NAME, {"S0": 3, "S1": None}, who="tester")
        e = _log()[-1]
        assert e["kind"] == "set_face_binding" and e["stage"] == "faces"
        assert e["before"] == {"faces_bind": None} and e["after"] == {"faces_bind": "S0=3,S1=none"}
        assert e["restamped"] is False and e["consumers"] == ["faces"] and e["backup"] is None
        assert e["binding"] == {"S0": 3, "S1": None} and e["edits_n"] is None
        assert P.load_options(NAME)["faces_bind"] == "S0=3,S1=none"
        E.set_face_binding(NAME, {"S0": "none"})
        e = _log()[-1]
        assert e["before"] == {"faces_bind": "S0=3,S1=none"} and e["after"] == {"faces_bind": "S0=none"}
        try:
            E.set_face_binding(NAME, {"S0": "abc"})
            raise AssertionError("non-integer track id must be rejected")
        except E.EditError as exc:
            assert exc.code == 400
        assert len(_log()) == 3, "a rejected edit writes no record"
    finally:
        teardown()


def test_glossary_logged():
    setup()
    try:
        # non-apply: diff recorded, translate is the consumer
        E.set_glossary(NAME, {"カンナ": {"zh": "康娜"}, "早上": {"zh": "早晨"}}, who="tester")
        e = _log()[-1]
        assert e["kind"] == "set_glossary" and e["stage"] == "glossary"
        assert e["added"] == ["早上"] and e["removed"] == [] and e["changed"] == [["カンナ", "坎娜", "康娜"]]
        assert e["before"] == {"カンナ": "坎娜"} and e["after"] == {"カンナ": "康娜", "早上": "早晨"}
        assert e["n_before"] == 1 and e["n_after"] == 2 and e["apply_to_translation"] is False
        assert e["restamped"] is True and e["restamped_stages"] == ["glossary"] and e["consumers"] == ["translate"]
        assert e["seed_written"] == 2 and e["seed_path"] == str(E.SEED_PATH_OVERRIDE)
        assert e["edits_n"] == 1 and e["backup"]
        # apply with a rendering that occurs in the text: translate restamped, tts is the consumer
        E.set_glossary(NAME, {"カンナ": {"zh": "康娜"}, "早上": {"zh": "早"}}, apply_to_translation=True)
        e = _log()[-1]
        assert e["changed"] == [["早上", "早晨", "早"]] and e["replacements"] == {"早晨": "早"}
        assert e["applied"] == 0 and e["changed_idx"] == [] and e["consumers"] == []
        assert e["restamped_stages"] == ["glossary"]
        st = P.load_state(NAME)
        st["translate"]["segments"][0]["text_translated"] = "早好"       # uses the current rendering 早
        rp.save_state(P.state_path(NAME), st)
        E.set_glossary(NAME, {"カンナ": {"zh": "康娜"}, "早上": {"zh": "早上"}}, apply_to_translation=True)
        e = _log()[-1]
        assert e["replacements"] == {"早": "早上"}
        assert e["applied"] == 1 and e["changed_idx"] == [0] and e["consumers"] == ["tts"]
        assert e["restamped_stages"] == ["glossary", "translate"] and e["restamped"] is True
        assert P.load_state(NAME)["translate"]["segments"][0]["text_translated"] == "早上好"
        # removed term
        E.set_glossary(NAME, {"カンナ": {"zh": "康娜"}})
        e = _log()[-1]
        assert e["removed"] == ["早上"] and e["before"] == {"早上": "早上"} and e["after"] == {}
    finally:
        teardown()


def _setup_film():
    d = P.WORKSPACE / FILM
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "profiles" ).mkdir()
    (d / "profiles" / "ref_P0.wav").write_bytes(b"RIFF")
    (d / "profiles" / "ref_P0_alt0.wav").write_bytes(b"RIFF")
    doc = {"version": 2, "film": FILM, "profiles": {
        "P0": {"gender": "female", "f0_median": 246.0, "seconds": 100.0, "ref_audio": "profiles/ref_P0.wav",
               "sources": [{"chunk": CHUNK, "speaker": "S0"}], "default_for_gender": True, "manual": False},
        "P1": {"gender": "female", "f0_median": 230.0, "seconds": 20.0, "ref_audio": None,
               "sources": [{"chunk": CHUNK2, "speaker": "S0"}], "default_for_gender": False, "manual": False}}}
    (d / "profiles.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    setup(CHUNK)
    setup(CHUNK2, make_state(with_enrol=True))


def test_speaker_profile_logged():
    _setup_film()
    try:
        E.set_speaker_profile(CHUNK, "S0", "P1", who="tester")
        e = _log(CHUNK)[-1]
        assert e["kind"] == "set_speaker_profile" and e["stage"] == "enrol" and e["idx"] is None
        assert e["before"] == {"profile": None} and e["after"] == {"profile": {"profile": "P1", "how": "manual", "manual": True}}
        assert e["speaker"] == {"before": "S0", "after": "S0"} and e["film"] == FILM
        assert e["restamped"] is False and e["consumers"] == ["v2"], "legacy chunk: no enrol fingerprint"
        assert e["backup"] and e["edits_n"] == 0
        st = P.load_state(CHUNK)
        assert st["enrol"]["speaker_profile"]["S0"]["profile"] == "P1" and "_edits" not in st
        # a chunk with an enrol fingerprint: restamped, translate/compact kept valid
        E.set_speaker_profile(CHUNK2, "S0", "P1")
        e = _log(CHUNK2)[-1]
        assert e["restamped"] is True and e["edits_n"] == 1 and e["consumers"] == ["v2"]
        st = P.load_state(CHUNK2)
        assert st["_edits"]["enrol"] == 1 and _up_matches(st, "translate")
        E.set_speaker_profile(CHUNK2, "S0", None)
        e = _log(CHUNK2)[-1]
        assert e["before"]["profile"]["profile"] == "P1" and e["after"] == {"profile": None} and e["requested"] is None
        # unknown profile / speaker: nothing written, no backup consumed
        n_bk = len(_backups(CHUNK2))
        for bad in (("S9", "P0"), ("S0", "P9")):
            try:
                E.set_speaker_profile(CHUNK2, *bad)
                raise AssertionError(bad)
            except E.EditError as exc:
                assert exc.code == 404
        assert len(_backups(CHUNK2)) == n_bk and len(_log(CHUNK2)) == 2
    finally:
        teardown(CHUNK, CHUNK2, FILM)


def test_update_profile_logged():
    _setup_film()
    try:
        from ai_movie.profiles import assignment_signature
        pj = P.WORKSPACE / FILM / "profiles.json"
        sig0 = assignment_signature(pj)
        # 1st edit: name only, but manual False→True changes the assignment signature → enrol@chunks
        E.update_profile(FILM, "P0", {"name": "Kanna"}, who="tester")
        rows = _log(FILM)
        assert len(rows) == 1 and not (P.WORKSPACE / CHUNK / E.EDIT_LOG_NAME).exists(), "film-level log"
        e = rows[0]
        assert e["kind"] == "update_profile" and e["stage"] == "profiles" and e["who"] == "tester"
        assert e["before"]["name"] is None and e["after"]["name"] == "Kanna"
        assert e["before"]["manual"] is False and e["after"]["manual"] is True
        assert e["before"]["version"] == 2 and e["after"]["version"] == 3
        assert e["signature_before"] == sig0 and e["signature_after"] != sig0
        assert e["chunks"] == [CHUNK, CHUNK2] and e["consumers"] == [f"enrol@{CHUNK}", f"enrol@{CHUNK2}"]
        assert e["restamped"] is False and e["edits_n"] is None and e["merged_into"] is None and e["deleted"] == []
        bk = P.WORKSPACE / FILM / "profiles_backups" / e["backup"]
        assert bk.exists() and json.loads(bk.read_text())["version"] == 2, "backup is the pre-edit file"
        # 2nd edit: ref_audio only → signature unchanged → consumer is v2 only
        E.update_profile(FILM, "P0", {"ref_audio": "profiles/ref_P0_alt0.wav"})
        e = _log(FILM)[-1]
        assert e["before"]["ref_audio"] == "profiles/ref_P0.wav" and e["after"]["ref_audio"] == "profiles/ref_P0_alt0.wav"
        assert e["signature_before"] == e["signature_after"] and e["consumers"] == ["v2"]
        # missing reference: rejected before the backup is taken
        n_bk = len(list((P.WORKSPACE / FILM / "profiles_backups").glob("*.json")))
        try:
            E.update_profile(FILM, "P0", {"ref_audio": "profiles/nope.wav"})
            raise AssertionError("missing ref must be rejected")
        except E.EditError as exc:
            assert exc.code == 404
        assert len(list((P.WORKSPACE / FILM / "profiles_backups").glob("*.json"))) == n_bk
        # merge: the deleted profile is named, sources folded, default carried
        doc = E.update_profile(FILM, "P0", {"merge_into": "P1"})
        e = _log(FILM)[-1]
        assert e["merged_into"] == "P1" and e["deleted"] == ["P0"] and e["speaker"] == {"before": "P0", "after": "P1"}
        assert "P0" not in doc["profiles"] and doc["profiles"]["P1"]["default_for_gender"] is True
        assert e["after"]["n_sources"] == 2 and e["after"]["seconds"] == 120.0
        assert e["consumers"][0].startswith("enrol@")
        assert doc["version"] == 5
    finally:
        teardown(CHUNK, CHUNK2, FILM)


def test_backup_suffix_and_no_orphans():
    setup()
    try:
        E.set_translation(NAME, 0, "早安")
        E.set_translation(NAME, 0, "早安呀")
        E.set_translation(NAME, 0, "早安啊")
        bks = _backups()
        assert len(bks) == 3, bks
        assert len({b.name for b in bks}) == 3 and all(b.name[-9:-5].startswith("-0") for b in bks)
        names = [e["backup"] for e in _log()]
        assert names == [b.name for b in bks], "log points at distinct, existing backups"
        # rejected edits do not consume backups nor write records
        for call in ((E.set_translation, NAME, 0, "  "), (E.set_translation, NAME, 9, "x"),
                     (E.set_speaker, NAME, 0, "S7", None), (E.set_speaker, NAME, 0, None, None),
                     (E.set_speaker, NAME, 0, None, "other"), (E.add_speaker, NAME, "other")):
            try:
                call[0](*call[1:])
                raise AssertionError(call)
            except E.EditError:
                pass
        assert len(_backups()) == 3 and len(_log()) == 3
        # legacy state (no fingerprint for the stage): 409 before anything is written
        st = P.load_state(NAME)
        del st["_fp"]["translate"]
        rp.save_state(P.state_path(NAME), st)
        try:
            E.set_translation(NAME, 1, "x")
            raise AssertionError("legacy state must be refused")
        except E.EditError as exc:
            assert exc.code == 409
        assert len(_backups()) == 3 and len(_log()) == 3
        # retention: the newest 20 stay
        st = make_state()
        rp.save_state(P.state_path(NAME), st)
        for i in range(22):
            E.set_translation(NAME, 0, f"第{i}版")
        assert len(_backups()) == 20
    finally:
        teardown()


def test_clip_and_read_edits():
    long = "喵" * 800
    flag: list = []
    c = E._clip({"a": long, "b": list(range(150)), "c": {"x": 1}}, flag)
    assert len(c["a"]) == E._MAX_TEXT + 1 and c["a"].endswith("…") and len(c["b"]) == E._MAX_LIST
    assert c["c"] == {"x": 1} and flag
    assert E._clip("short") == "short" and E._clip(None) is None and E._clip(3.5) == 3.5
    setup()
    try:
        E.set_translation(NAME, 0, long, who="a")
        E.set_translation(NAME, 1, "你好呀", who="b")
        E.set_speaker(NAME, 1, gender="male", who="a")
        e = _log()[0]
        assert e["truncated"] is True and len(e["after"]["text_translated"]) == E._MAX_TEXT + 1
        assert P.load_state(NAME)["translate"]["segments"][0]["text_translated"] == long, "state keeps the full text"
        assert [r["kind"] for r in E.read_edits(NAME)] == ["set_translation", "set_translation", "set_speaker"]
        assert [r["idx"] for r in E.read_edits(NAME, stage="translate")] == [0, 1]
        assert [r["who"] for r in E.read_edits(NAME, who="a")] == ["a", "a"]
        assert [r["kind"] for r in E.read_edits(NAME, idx=1)] == ["set_translation", "set_speaker"]
        assert len(E.read_edits(NAME, n=1)) == 1 and E.read_edits(NAME, n=1)[0]["kind"] == "set_speaker"
        assert E.read_edits("webtest_nonexistent") == []
        # a corrupt line is skipped, not fatal
        with open(P.WORKSPACE / NAME / E.EDIT_LOG_NAME, "a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        assert len(E.read_edits(NAME)) == 3
    finally:
        teardown()


def test_runs_viewer():
    setup()
    try:
        E.set_translation(NAME, 1, "你好呀", who="lan")
        E.set_speaker(NAME, 0, gender="male", who="lan")
        E.set_glossary(NAME, {"カンナ": {"zh": "康娜"}, "早上": {"zh": "早晨"}}, who="lan")
        E.set_face_binding(NAME, {"S0": 3, "S1": None}, who="lan")
        rows = runs.read_edits(NAME)
        assert len(rows) == 4
        s = runs.edits_summary(rows, since_epoch=None)
        assert s["n"] == 4 and s["by_stage"] == {"translate": 1, "asr": 1, "glossary": 1, "faces": 1}
        assert s["last_who"] == "lan" and s["since_last_run"] == 0
        assert rows[0]["t"] <= rows[1]["t"] <= rows[2]["t"] <= rows[3]["t"]
        spaced = [dict(r, t=1000.0 + i) for i, r in enumerate(rows)]       # edits can share a millisecond
        s = runs.edits_summary(spaced, since_epoch=1001.0)
        assert s["since_last_run"] == 2 and s["consumers_since"] == ["faces", "translate"]
        out = io.StringIO()
        with redirect_stdout(out):
            runs.show_edits(NAME)
        text = out.getvalue()
        assert "「你好」 → 「你好呀」" in text and "speaker S0→S1" in text and "minted S1" in text
        assert "+1 ~1 -0" in text and "faces_bind none → S0=3,S1=none" in text
        assert "⇒ tts" in text and "⇒ translate" in text and "⇒ faces" in text
        assert "undo" in text and "state_backups" in text and "cp " in text
        assert "WARNING" not in text, "no run started after the edits"
        out = io.StringIO()
        with redirect_stdout(out):
            runs.show_edits(NAME, stage="asr", who="lan")
        text = out.getvalue()
        assert text.count("\n") == 4, text                      # header + 1 record + undo (2 lines)
        assert "speaker S0→S1" in text and "你好呀" not in text
        cp_line = next(ln for ln in text.splitlines() if ln.strip().startswith("cp "))
        assert _log()[2]["backup"] in cp_line and _log()[1]["backup"] not in cp_line, \
            "undo hint names the newest state edit of the whole log (glossary), not the filtered one"
        out = io.StringIO()
        with redirect_stdout(out):
            runs.show_edits(NAME, who="nobody")
        assert "no manual edits" in out.getvalue()
        # show_project prints the edit line even without run records …
        out = io.StringIO()
        with redirect_stdout(out):
            runs.show_project(NAME)
        assert "edits: 4 (asr 1, faces 1, glossary 1, translate 1), last" in out.getvalue(), out.getvalue()
        assert "after the newest run" not in out.getvalue()
        # … and, with a run that started after the edits, counts them and warns in the undo hint
        rdir = P.workdir(NAME) / "runs" / "20990101-000000-1"
        rdir.mkdir(parents=True)
        (rdir / "manifest.json").write_text(json.dumps({"run_id": "20990101-000000-1", "name": NAME,
                                                        "started": "2099-01-01 00:00:00", "status": "running"}))
        (P.workdir(NAME) / "runs" / "index.jsonl").write_text(json.dumps({
            "run_id": "20990101-000000-1", "started": "2099-01-01 00:00:00", "status": "ok", "steps": ["translate"],
            "ran": ["translate"], "commit": "abc", "dirty": False, "seconds": 1.0}) + "\n")
        out = io.StringIO()
        with redirect_stdout(out):
            runs.show_project(NAME)
        assert "0 after the newest run started (20990101-000000-1 2099-01-01 00:00:00)" in out.getvalue(), out.getvalue()
        (rdir / "manifest.json").write_text(json.dumps({"run_id": "20000101-000000-1", "name": NAME,
                                                        "started": "2000-01-01 00:00:00", "status": "running"}))
        out = io.StringIO()
        with redirect_stdout(out):
            runs.show_project(NAME)
        assert "4 after the newest run started" in out.getvalue() and "⇒ to redo: faces,translate,tts" in out.getvalue()
        out = io.StringIO()
        with redirect_stdout(out):
            runs.show_edits(NAME)
        assert "WARNING" not in out.getvalue()
        (rdir / "manifest.json").write_text(json.dumps({"run_id": "20990101-000000-1", "name": NAME,
                                                        "started": "2099-01-01 00:00:00", "status": "running"}))
        out = io.StringIO()
        with redirect_stdout(out):
            runs.show_edits(NAME)
        assert "WARNING: run 20990101-000000-1 started" in out.getvalue()
    finally:
        teardown()


def test_server_passes_who():
    """The routes forward Remote-User (or "lan") as ``who``; the GET route reads the log."""
    import warnings
    warnings.filterwarnings("ignore", message=".*httpx.*")          # starlette's TestClient nag
    from fastapi.testclient import TestClient
    from ai_movie.web import server as S
    setup()
    try:
        c = TestClient(S.app)
        r = c.put(f"/api/projects/{NAME}/segments/1/translation", json={"text": "你好呀"},
                  headers={"Remote-User": "haifeng"})
        assert r.status_code == 200, r.text
        r = c.put(f"/api/projects/{NAME}/segments/0/speaker", json={"gender": "male"})
        assert r.status_code == 200, r.text
        r = c.put(f"/api/projects/{NAME}/faces/binding", json={"binding": {"S0": 1}})
        assert r.status_code == 200, r.text
        r = c.put(f"/api/projects/{NAME}/segments/1/translation", json={"text": ""})
        assert r.status_code == 400
        assert [e["who"] for e in _log()] == ["haifeng", "lan", "lan"]
        r = c.get(f"/api/projects/{NAME}/edits?n=2")
        assert r.status_code == 200 and [e["kind"] for e in r.json()] == ["set_speaker", "set_face_binding"]
        r = c.get(f"/api/projects/{NAME}/edits?stage=translate")
        assert [e["who"] for e in r.json()] == ["haifeng"]
        assert c.get("/api/projects/webtest_nonexistent/edits").status_code == 404
        assert c.get("/api/films/webtest_nonexistent/edits").json() == []
    finally:
        teardown()


def test_v2_stale_after_profile_pin():
    """derive_status: a profile pin (enrol restamp) makes v2 stale once vc_deps records enrol;
    an older record (fit/compose only) still compares on its own keys."""
    st = make_state(with_enrol=True)
    fps = st["_fp"]
    fps["compose"] = _fp({"v": 1, "input": "x", "cfg": {}, "code": {}, "files": {}, "up": {}, "extra": {}})
    video = TMP_WS / "v2.mp4"
    video.write_bytes(b"\0")
    st["vc"] = {"video": str(video)}
    raw = {s: {"status": "valid", "reasons": []} for s in rp.ALL_STEPS}
    st["_web"] = {"vc_deps": {k: fps[k]["hash"] for k in P.VC_DEPS}}
    assert P.derive_status(NAME, raw, st)["v2"]["status"] == "done"
    rp.restamp_after_edit(st, "enrol", keep_valid=["glossary", "translate", "compact"])
    assert P.derive_status(NAME, raw, st)["v2"]["status"] == "stale"
    st["_web"] = {"vc_deps": {k: fps[k]["hash"] for k in ("fit", "compose")}}     # pre-F5 record
    assert P.derive_status(NAME, raw, st)["v2"]["status"] == "done"
    assert P.VC_DEPS == ("fit", "compose", "enrol")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
