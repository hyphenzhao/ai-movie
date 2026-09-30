"""Voice-consistency metric (ai_movie/voice_consistency.py): gates on synthetic ECAPA-like vectors,
row collection from a state, eval_long / eval_pipeline / qc wiring.  numpy only — the one real-data
smoke test embeds 30 clips on CPU and skips itself when the model or the clips are absent.

    .venv/bin/python tests/test_voice_consistency.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

for _k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
    os.environ[_k] = ""                                  # the smoke test must never touch the GPU

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie import voice_consistency as vcm           # noqa: E402

DIM = vcm.DIM


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)     # type: ignore[union-attr]
    return mod


# ── synthetic vectors ──────────────────────────────────────────────

def _unit(v):
    return v / np.linalg.norm(v)


def rand_unit(rng):
    return _unit(rng.standard_normal(DIM))


def around(c, n, rng, sigma=0.07):
    """n unit vectors around c; sigma 0.07 in 192-d gives a cosine distance ≈ 0.28 (the built-in floor)."""
    X = c + sigma * rng.standard_normal((n, DIM))
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def at_distance(c, d, rng):
    """A unit vector at cosine distance d from c."""
    u = rand_unit(rng)
    u = _unit(u - (u @ c) * c)
    cos = 1.0 - d
    return _unit(cos * c + np.sqrt(1 - cos * cos) * u)


def mk_rows(n, *, key="P0", gender="female", chunk=None, vc=True, dur=3.0, reason="converted",
            speaker="S0", has_ref=True, i0=0, detail=None):
    return [{"i": i0 + k, "chunk": chunk, "speaker": speaker, "gender": gender, "key": key, "vc": vc,
             "reason": reason, "detail": detail, "dur": dur, "dur_heard": dur, "start": 10.0 * (i0 + k),
             "end": 10.0 * (i0 + k) + dur, "wav2": f"w2_{i0 + k}.wav", "wav1": f"w1_{i0 + k}.wav",
             "text": "台词", "has_ref": has_ref} for k in range(n)]


def cfg(**over):
    c = dict(vcm.default_cfg())
    c.update(over)
    return c


def gate_map(rows_g):
    return {r["id"]: r for r in rows_g}


def one_voice_film(rng, n_per_chunk=(20, 20, 20), spread=0.7):
    A, rows, E2, E1 = rand_unit(rng), [], [], []
    B = at_distance(A, spread, rng)
    i0 = 0
    for c, n in enumerate(n_per_chunk, start=1):
        rows += mk_rows(n, chunk=c, i0=i0); i0 += n
        E2.append(around(A, n, rng)); E1.append(around(B, n, rng, 0.06))
    return A, B, rows, np.concatenate(E2), np.concatenate(E1)


# ── 1–7: the gates ─────────────────────────────────────────────────

def test_one_voice_passes():
    rng = np.random.default_rng(1)
    A, B, rows, E2, E1 = one_voice_film(rng)
    s = vcm.summarize(rows, E2, E1, cfg=cfg())
    g = gate_map(vcm.gates(s, cfg(), film=True))
    st = s["groups"]["P0"]["stats"]
    assert 0.2 < st["median_long"] < 0.4, st
    assert abs(st["median_excess"]) < 0.1, st
    for gid in ("V1[P0]", "V2[P0]", "V4[P0]", "V5[P0]"):
        assert g[gid]["ok"] is True, g[gid]
    assert g["V3[P0]"]["ok"] is None and "listening" in g["V3[P0]"]["note"]
    assert "V6[P0]" not in g                              # no profile vector → no V6
    assert s["groups"]["P0"]["fallback"]["share_sec"] == 0.0
    assert all(v["gated"] and v["d_loo"] < 0.15 for v in s["groups"]["P0"]["chunks"].values()), s["groups"]["P0"]["chunks"]


def test_two_references_fail_chunk_spread():
    """Chunks 1–2 on reference A, chunk 3 on B (0.7 apart): V4-LOO trips; the absolute/relative median
    barely moves (a 1/5 mix shifts it < 0.02), which is why V2 is not the two-reference detector."""
    rng = np.random.default_rng(2)
    A = rand_unit(rng); B = at_distance(A, 0.7, rng); Bi = at_distance(A, 0.7, rng)
    rows = mk_rows(20, chunk=1) + mk_rows(20, chunk=2, i0=20) + mk_rows(10, chunk=3, i0=40) + mk_rows(3, chunk=4, i0=50)
    E2 = np.concatenate([around(A, 20, rng), around(A, 20, rng), around(B, 10, rng), around(A, 3, rng)])
    E1 = around(Bi, 53, rng, 0.06)
    s = vcm.summarize(rows, E2, E1, cfg=cfg())
    g = gate_map(vcm.gates(s, cfg(), film=True))
    ch = s["groups"]["P0"]["chunks"]
    assert ch["3"]["gated"] and ch["3"]["d_loo"] > 0.5, ch
    assert ch["4"]["gated"] is False and ch["4"]["d_loo"] is not None, ch      # 3 lines: reported, not gated
    assert g["V4[P0]"]["ok"] is False, g["V4[P0]"]
    assert "chunk 3" in g["V4[P0]"]["value"]
    assert g["V2[P0]"]["ok"] is True, g["V2[P0]"]                              # median excess < 0.15 still
    # …and the same mix would pass a self-inclusive centroid at the half-angle: that is the bug LOO avoids
    c_all = vcm.centroid(E2[:50]); c3 = vcm.centroid(E2[40:50])
    assert vcm.cosd(c3, c_all) < ch["3"]["d_loo"] - 0.1


def test_noop_conversion_fails_shift():
    rng = np.random.default_rng(3)
    A = rand_unit(rng)
    rows = mk_rows(20)
    E1 = around(A, 20, rng)
    s = vcm.summarize(rows, E1.copy(), E1, cfg=cfg())
    g = gate_map(vcm.gates(s, cfg(), film=False))
    assert s["groups"]["P0"]["shift"] < 0.01
    assert g["V5[P0]"]["ok"] is False, g["V5[P0]"]
    assert s["groups"]["P0"]["stats"]["n_unchanged"] == 20
    assert all(ln["d_v1"] == 0.0 for ln in s["lines"])


def test_fallback_share_by_seconds():
    rng = np.random.default_rng(4)
    A = rand_unit(rng); B = at_distance(A, 0.7, rng)
    conv = mk_rows(10)                                   # 30 s converted
    fb = [dict(mk_rows(1, vc=False, dur=2.0, reason="guard", i0=10, detail="pitch_jump_0.6")[0]),
          dict(mk_rows(1, vc=False, dur=0.5, reason="short", i0=11)[0]),
          dict(mk_rows(1, vc=False, dur=1.0, reason="error", i0=12, detail="vc failed")[0]),
          dict(mk_rows(1, vc=False, dur=0.5, reason="other", i0=13)[0])]
    rows = conv + fb
    E2 = np.concatenate([around(A, 10, rng), around(B, 4, rng)])
    E1 = around(B, 14, rng)
    s = vcm.summarize(rows, E2, E1, cfg=cfg())
    f = s["groups"]["P0"]["fallback"]
    assert f["lines"] == 4 and abs(f["seconds"] - 4.0) < 1e-6
    assert abs(f["share_sec"] - 4.0 / 34.0) < 1e-3 and abs(f["share_lines"] - 4 / 14) < 1e-3
    assert f["by_reason"] == {"error": 1, "guard": 1, "other": 1, "short": 1}
    g = gate_map(vcm.gates(s, cfg(), film=False))
    assert g["V1[P0]"]["ok"] is False and "11.8%" in g["V1[P0]"]["value"], g["V1[P0]"]
    # by seconds, not lines: the same four lines at 0.3 s each pass
    for r in fb:
        r["dur_heard"] = 0.3
    s2 = vcm.summarize(rows, E2, E1, cfg=cfg())
    assert gate_map(vcm.gates(s2, cfg(), film=False))["V1[P0]"]["ok"] is True
    # non-converted lines: d(v2, v1) is reported as the pairing sanity value
    assert s["max_dv1_nonvc"] is not None


def test_short_lines_feed_centroid_not_median():
    rng = np.random.default_rng(5)
    A = rand_unit(rng); B = at_distance(A, 0.9, rng); Bi = at_distance(A, 0.7, rng)
    long_rows = mk_rows(20)
    E2_long = around(A, 20, rng)
    E1 = around(Bi, 30, rng)
    s0 = vcm.summarize(long_rows, E2_long, E1[:20], cfg=cfg())
    rows = long_rows + mk_rows(10, dur=1.0, i0=20)      # 10 short converted lines on a far voice
    E2 = np.concatenate([E2_long, around(B, 10, rng)])
    s1 = vcm.summarize(rows, E2, E1, cfg=cfg())
    g0, g1 = s0["groups"]["P0"], s1["groups"]["P0"]
    assert g1["n_long"] == 20 and g1["stats"]["n_short"] == 10
    assert g1["stats"]["outlier_share"] == 0.0            # short lines cannot be outliers
    assert vcm.cosd(g0["_c_conv"], g1["_c_conv"]) > 0.05  # …but they moved the centroid
    assert g1["stats"]["median_short"] > g1["stats"]["median_long"]


def test_not_judged_below_min_lines():
    rng = np.random.default_rng(6)
    A = rand_unit(rng); B = at_distance(A, 0.7, rng)
    rows = mk_rows(5)
    s = vcm.summarize(rows, around(A, 5, rng), around(B, 5, rng), cfg=cfg())
    for r in vcm.gates(s, cfg(), film=True):
        assert r["ok"] is None, r
        assert ("need 8" in (r["note"] or "")) or (r["gate"] == "V4" and r["note"] == "no gated chunk"), r
    # a voice without a reference clip is built-in by design: never judged, even with many lines
    rows = mk_rows(30, vc=False, reason="no_ref", has_ref=False, key="P1", gender="male")
    s = vcm.summarize(rows, np.full((30, DIM), np.nan, np.float32), around(B, 30, rng), cfg=cfg())
    for r in vcm.gates(s, cfg(), film=True):
        assert r["ok"] is None and "built-in by design" in r["note"], r


def test_profile_gain():
    rng = np.random.default_rng(7)
    A = rand_unit(rng); B = at_distance(A, 0.7, rng)
    rows = mk_rows(20)
    E2, E1 = around(A, 20, rng), around(B, 20, rng)
    near_A = at_distance(A, 0.4, rng)                    # d(conv)≈0.4 vs d(builtin)≈0.7+ → gain ≈ +0.3
    s = vcm.summarize(rows, E2, E1, profile_vecs={"P0": near_A}, cfg=cfg())
    g = gate_map(vcm.gates(s, cfg(), film=False))
    assert g["V6[P0]"]["ok"] is True and s["groups"]["P0"]["profile"]["gain"] > 0.1, g["V6[P0]"]
    near_B = at_distance(B, 0.3, rng)
    s = vcm.summarize(rows, E2, E1, profile_vecs={"P0": near_B}, cfg=cfg())
    assert gate_map(vcm.gates(s, cfg(), film=False))["V6[P0]"]["ok"] is False
    s = vcm.summarize(rows, E2, E1, profile_vecs={"P9": near_A}, cfg=cfg())
    assert "V6[P0]" not in gate_map(vcm.gates(s, cfg(), film=False))


# ── 8–13: rows from a state ────────────────────────────────────────

def _state():
    fit = [{"start": 0.0, "end": 2.0, "fit_end": 1.6, "speaker": "S0", "gender": "female", "text_translated": "一", "audio_fit": "f0.wav"},
           {"start": 3.0, "end": 4.0, "fit_end": 3.9, "speaker": "S0", "gender": "female", "text_translated": "二", "audio_fit": "f1.wav"},
           {"start": 5.0, "end": 6.0, "fit_end": 5.5, "speaker": "S0", "gender": "female", "text_translated": "三", "audio_fit": "f2.wav"},
           {"start": 7.0, "end": 8.0, "fit_end": 7.9, "speaker": "S1", "gender": "male", "text_translated": "四", "audio_fit": "f3.wav"},
           {"start": 9.0, "end": 10.0, "fit_end": 9.9, "speaker": "S0", "gender": "female", "text_translated": "五", "audio_fit": "f4.wav", "keep_original": True},
           {"start": 11.0, "end": 12.0, "fit_end": 11.9, "speaker": "S0", "gender": "female", "text_translated": "", "audio_fit": "f5.wav"},
           {"start": 13.0, "end": 15.0, "fit_end": 14.5, "speaker": "S0", "gender": "female", "text_translated": "七", "audio_fit": "f6.wav"},
           {"start": 16.0, "end": 16.5, "fit_end": 16.4, "speaker": "S0", "gender": "female", "text_translated": "八", "audio_fit": "f7.wav"}]
    vc = [dict(s, audio_fit=f"v{i}.wav") for i, s in enumerate(fit)]
    vc[0]["vc"] = True
    vc[1]["vc"] = False; vc[1]["vc_guard"] = "pitch_jump_0.6"
    vc[2]["vc"] = False                                   # old state: no vc_skip → wav duration decides
    vc[3]["vc"] = False                                   # S1 has no reference
    vc[6]["vc"] = False; vc[6]["vc_skip"] = "too short to convert safely"
    vc[7]["vc"] = False                                   # slot 0.5 s but the wav is 1.0 s → "other"
    return {"asr": {"diarization": {"speakers": {"S0": {"gender": "female"}, "S1": {"gender": "male"}}}},
            "fit": {"segments": fit},
            "vc": {"segments": vc, "refs": {"S0": {"ref_audio": "/r.wav", "gender": "female"}}, "converted": 1}}


def test_collect_lines():
    st = _state()
    durs = {"f2.wav": 0.5, "f7.wav": 1.0}
    rows = vcm.collect_lines(st, {}, dur_of=lambda p: durs.get(p, 2.0))
    assert [r["i"] for r in rows] == [0, 1, 2, 3, 6, 7]  # keep_original and empty text skipped
    by = {r["i"]: r for r in rows}
    assert by[0]["wav2"] == "v0.wav" and by[0]["wav1"] == "f0.wav"        # paired by index
    assert by[0]["key"] == "gender:female" and by[3]["key"] == "gender:male"
    assert by[0]["reason"] == "converted" and by[0]["vc"]
    assert by[1]["reason"] == "guard" and by[1]["detail"] == "pitch_jump_0.6"
    assert by[2]["reason"] == "short"                     # wav 0.5 s < 0.7 although the slot is 1.0 s
    assert by[3]["reason"] == "no_ref" and not by[3]["has_ref"]
    assert by[6]["reason"] == "short" and by[6]["detail"].startswith("too short")
    assert by[7]["reason"] == "other"                     # slot 0.5 s but the fitted wav is 1.0 s
    assert abs(by[0]["dur_heard"] - 1.6) < 1e-9 and abs(by[0]["dur"] - 2.0) < 1e-9
    rows = vcm.collect_lines(st, {"S0": "P0"}, dur_of=lambda p: 2.0)
    assert {r["key"] for r in rows if r["speaker"] == "S0"} == {"P0"}


def test_gate_ids_stable():
    assert vcm.GATE_IDS == ("V1", "V2", "V3", "V4", "V5", "V6")
    rng = np.random.default_rng(9)
    A, B, rows, E2, E1 = one_voice_film(rng)
    ids = [r["id"] for r in vcm.gates(vcm.summarize(rows, E2, E1, cfg=cfg()), cfg(), film=True)]
    assert ids == ["V1[P0]", "V2[P0]", "V3[P0]", "V4[P0]", "V5[P0]"], ids
    el = _load("eval_long")
    assert [(a, b) for a, b, _ in el.L3_VOICE_GATES] == [("L3b", "V2"), ("L3c", "V4"), ("L3d", "V1"), ("L3e", "V5"), ("L3f", "V6")]
    assert not hasattr(el, "L3_MIN_MEDIAN") and not hasattr(el, "L3_MAX_SPREAD")


def test_short_reason_uses_wav_duration_not_slot():
    st = _state()
    seg = st["vc"]["segments"][2]
    seg["start"], seg["end"] = 5.0, 9.0                   # a 4 s slot…
    st["fit"]["segments"][2]["start"], st["fit"]["segments"][2]["end"] = 5.0, 9.0
    rows = {r["i"]: r for r in vcm.collect_lines(st, {}, dur_of=lambda p: 0.4 if p == "f2.wav" else 2.0)}
    assert rows[2]["reason"] == "short"                   # …whose fitted wav is 0.4 s
    rows = {r["i"]: r for r in vcm.collect_lines(st, {}, dur_of=None)}
    assert rows[2]["reason"] == "other"                   # unknown duration → never guessed from the slot


def test_v1_only_chunk_rows_count_in_v1():
    st = _state()
    del st["vc"]
    rows = vcm.collect_lines(st, {"S0": "P0"}, chunk=7, v1_only=True)
    assert rows and all(not r["vc"] and r["reason"] == "chunk_v1_only" and r["wav2"] == r["wav1"] for r in rows)
    assert all(r["chunk"] == 7 for r in rows)
    rng = np.random.default_rng(11)
    A = rand_unit(rng); B = at_distance(A, 0.7, rng)
    good = mk_rows(20, chunk=1)
    allrows = good + rows
    E2 = np.concatenate([around(A, 20, rng), around(B, len(rows), rng)])
    E1 = around(B, len(allrows), rng)
    s = vcm.summarize(allrows, E2, E1, profile_has_ref={"P0": True}, cfg=cfg())
    f = s["groups"]["P0"]["fallback"]
    assert f["by_reason"] == {"chunk_v1_only": len([r for r in rows if r["key"] == "P0"])}
    assert f["seconds"] > 0 and s["groups"]["P0"]["has_ref"]


def test_no_refs_all_gates_none():
    st = _state()
    st["vc"]["refs"] = {}
    for s in st["vc"]["segments"]:
        s["vc"] = False
    rows = vcm.collect_lines(st, {}, dur_of=lambda p: 2.0)
    assert {r["reason"] for r in rows} == {"no_ref"}
    E = np.full((len(rows), DIM), np.nan, np.float32)
    for r in vcm.gates(vcm.summarize(rows, E, E, cfg=cfg()), cfg(), film=False):
        assert r["ok"] is None and "no reference" in r["note"], r
    # a voice with a reference where nothing converted: judged as "no conversion attempted"
    rows = mk_rows(12, vc=False, reason="error")
    E12 = np.full((12, DIM), np.nan, np.float32)
    for r in vcm.gates(vcm.summarize(rows, E12, E12, cfg=cfg()), cfg(), film=False):
        assert r["ok"] is None and "no conversion attempted" in r["note"], r


def test_gender_key_maps_to_default_profile():
    st = _state()
    doc = {"profiles": {"P0": {"gender": "female", "default_for_gender": True, "ref_audio": "r.wav"},
                        "P1": {"gender": "male", "default_for_gender": True, "ref_audio": None},
                        "P2": {"gender": "female"}}}
    assert vcm.profile_map(st, doc) == {"S0": "P0", "S1": "P1"}
    st["enrol"] = {"speaker_profile": {"S0": {"profile": "P2", "score": 0.3}}}
    assert vcm.profile_map(st, doc) == {"S0": "P2", "S1": "P1"}          # the enrol assignment wins
    assert vcm.profile_map(st, None) == {"S0": "P2"}                     # no document: no default
    st["enrol"] = {}
    st["vc"]["refs"]["S0"]["profile"] = "P0"
    assert vcm.profile_map(st, None) == {"S0": "P0"}                     # what run_vc_version recorded


# ── 14–16: wiring ──────────────────────────────────────────────────

def test_eval_long_none_is_note_and_stale_is_note():
    el = _load("eval_long")
    rep = {"chunks": {"1": {"vc_sig": "a"}, "2": {"vc_sig": "b"}},
           "gates": [{"id": "V2[P0]", "gate": "V2", "key": "P0", "ok": True, "value": "+0.06", "note": None},
                     {"id": "V4[P0]", "gate": "V4", "key": "P0", "ok": False, "value": "max 0.5", "note": None},
                     {"id": "V1[P0]", "gate": "V1", "key": "P0", "ok": True, "value": "3%", "note": None},
                     {"id": "V5[P0]", "gate": "V5", "key": "P0", "ok": None, "value": "—", "note": "only 3 lines"},
                     {"id": "V1[P1]", "gate": "V1", "key": "P1", "ok": None, "value": "100%", "note": "no reference"}],
           "notes": ["p07: delivered without a cloned version"]}
    checks, notes = [], []
    el.voice_consistency_checks("X", checks, notes, report=rep, current_sigs={"1": "a", "2": "b"})
    ids = {c["id"]: c for c in checks}
    assert set(ids) == {"L3b", "L3c", "L3d"}, ids           # V5/V6 not judged → not a check
    assert ids["L3c"]["ok"] is False and ids["L3b"]["ok"] is True and ids["L3d"]["ok"] is True
    assert all(isinstance(c["ok"], bool) for c in checks)
    assert any(n.startswith("L3e: not judged") for n in notes), notes
    assert any("voice consistency: p07" in n for n in notes)
    checks, notes = [], []
    el.voice_consistency_checks("X", checks, notes, report=rep, current_sigs={"1": "a", "2": "CHANGED"})
    assert checks == [] and any("stale" in n and "['2']" in n for n in notes), notes
    checks, notes = [], []
    el.voice_consistency_checks("no_such_film_zz", checks, notes)
    assert checks == [] and any("not measured" in n for n in notes)


def test_pairing_guard():
    st = _state()
    assert vcm.pairing_problem(st) is None
    bad = json.loads(json.dumps(st)); bad["fit"]["segments"].pop()
    assert "stale" in vcm.pairing_problem(bad)
    bad = json.loads(json.dumps(st)); bad["fit"]["segments"][0]["text_translated"] = "改"
    assert "text_translated" in vcm.pairing_problem(bad)
    assert vcm.pairing_problem({"fit": st["fit"]}) == "state has no vc segments"
    # the CLI refuses (exit 2, never a failure) before touching any file
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "state.json"
        sp.write_text(json.dumps(bad), encoding="utf-8")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "voice_consistency.py"), str(sp), "--no-write"],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 2, (r.returncode, r.stdout[-400:], r.stderr[-400:])
        assert not (Path(td) / "deliverables").exists()
    sig = vcm.vc_signature(st)
    st["vc"]["segments"][0]["audio_fit"] = "recloned.wav"
    assert vcm.vc_signature(st) != sig


def test_json_roundtrip_reaches_build_qc_warn():
    from ai_movie.qc import build_qc
    st = _state()
    st["vc"]["consistency"] = {"lines": {0: {"outlier": True, "d_self": 0.61, "long": True},
                                         1: {"outlier": False, "d_self": 0.2, "long": True}}}
    st = json.loads(json.dumps(st))                       # int keys become "0" / "1"
    assert list(st["vc"]["consistency"]["lines"]) == ["0", "1"]
    q = build_qc(st, key="vc", plan={}, osd={})
    reasons = {r["idx"]: r["reasons"] for r in q["segments"]}
    assert "voice_outlier>0.55" in reasons[0] and "voice_outlier" not in reasons[1], reasons
    assert q["segments"][0]["voice_d_self"] == 0.61
    assert q["thresholds"]["voice_outlier"] == 0.55
    assert "voice_outlier>0.55" in q["summary"]["reasons"]


def test_eval_pipeline_c7_rows():
    ep = _load("eval_pipeline")
    rep = ep.Report()
    ep.eval_voice_consistency({"vc": {}}, rep)
    assert rep.rows[0]["key"] == "C7" and rep.rows[0]["ok"] is None and "not measured" in rep.rows[0]["value"]
    rep = ep.Report()
    st = {"vc": {"consistency": {
        "gates": [{"id": "V1[P0]", "gate": "V1", "key": "P0", "ok": True, "value": "3%", "note": None},
                  {"id": "V2[P0]", "gate": "V2", "key": "P0", "ok": True, "value": "+0.06", "note": None},
                  {"id": "V3[P0]", "gate": "V3", "key": "P0", "ok": None, "value": "2/50", "note": "not gated"},
                  {"id": "V5[P0]", "gate": "V5", "key": "P0", "ok": False, "value": "0.1", "note": None},
                  {"id": "V6[P0]", "gate": "V6", "key": "P0", "ok": None, "value": "—", "note": "only 3 lines"}],
        "lines": {"4": {"key": "P0", "d_self": 0.7, "long": True}, "9": {"key": "P0", "d_self": 0.6, "long": True},
                  "2": {"key": "P0", "d_self": 0.9, "long": False}}}}}
    ep.eval_voice_consistency(st, rep)
    rows = {r["key"]: r for r in rep.rows}
    assert rows["C7a[P0]"]["ok"] is True and rows["C7b[P0]"]["ok"] is True
    assert rows["C7c[P0]"]["ok"] is False and rows["C7e[P0]"]["ok"] is None
    assert rows["C7d[P0]"]["ok"] is None and "#4 0.70" in rows["C7d[P0]"]["value"] and "#2" not in rows["C7d[P0]"]["value"]
    ar = _load("accept_release")
    assert ar.check_rows(rep.rows) == {"C7a": True, "C7b": True, "C7c": False}


# ── 17: real data (skips itself when the model or the clips are missing) ────────────

def test_real_reference_switch_smoke():
    from ai_movie import diarize
    base = ROOT / "workspace" / "SONE-846" / "profiles" / "real_P0"
    if not (diarize.ecapa_available() and (base / "c0").is_dir() and (base / "c1").is_dir() and (base / "c2").is_dir()):
        print("  (skipped: ECAPA model or real_P0 clips absent)")
        return
    embs = {}
    for c in ("c0", "c1", "c2"):
        wavs = sorted(str(w) for w in (base / c).glob("seg_*.wav"))
        e, keep = diarize.embed_files(wavs, device="cpu")
        assert len(e) >= 8, (c, len(e))
        embs[c] = e
    c = cfg(min_rest_lines=8)                             # two 10-line "chunks": a smaller LOO pool than a film
    for other, expect in (("c1", True), ("c2", False)):
        rows = mk_rows(len(embs["c0"]), chunk=1) + mk_rows(len(embs[other]), chunk=2, i0=100)
        E2 = np.concatenate([embs["c0"], embs[other]])
        s = vcm.summarize(rows, E2, np.full_like(E2, np.nan), cfg=c)
        g = gate_map(vcm.gates(s, c, film=True))["V4[P0]"]
        assert g["ok"] is expect, (other, g)
    d01 = vcm.cosd(vcm.centroid(embs["c0"]), vcm.centroid(embs["c1"]))
    d02 = vcm.cosd(vcm.centroid(embs["c0"]), vcm.centroid(embs["c2"]))
    assert d01 < 0.35 < d02, (d01, d02)                   # measured 0.20 / 0.92


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
