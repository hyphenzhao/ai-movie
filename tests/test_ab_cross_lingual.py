"""ab_cross_lingual's pure helpers: sampling, the marker, timing/consistency stats, the
preference threshold and the pre-registered decision (no GPU, no film needed).

    .venv/bin/python tests/test_ab_cross_lingual.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("ab_cross_lingual", ROOT / "scripts" / "ab_cross_lingual.py")
ab = importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)     # type: ignore[union-attr]


def test_marker_and_text_rules():
    assert ab.job_text("早上好") == "You are a helpful assistant.<|endofprompt|>早上好"
    assert ab.job_text(ab.job_text("早上好")) == ab.job_text("早上好")           # idempotent
    assert ab.has_ascii("我是S1专属套装坎娜。") and ab.has_ascii("一般在12到15厘米左右") and not ab.has_ascii("早上好？")
    assert ab.visible_chars("嗯? 好大") == 3 and ab.visible_chars("。") == 0
    # the worker's collapse floor counts hanzi only, so the English prefix does not raise it
    assert abs(ab.min_take_dur(ab.job_text("早上好")) - (3 * 0.10 + 0.3)) < 1e-9


def test_stratify_bins_spread_and_judgeable_first():
    assert ab.dur_bin(0.5) == 0 and ab.dur_bin(0.7) == 1 and ab.dur_bin(1.49) == 1 and ab.dur_bin(2.5) == 3 and ab.dur_bin(0.1) is None
    cands = []
    k = 0
    for dur in (0.4, 1.0, 2.0, 3.0):
        for u in range(30):
            cands.append({"index": k, "unit_id": u, "dur": dur, "judgeable": u % 3 != 0}); k += 1
    lines = ab.stratify(cands, 10)
    assert len(lines) == 40 and [sum(1 for l in lines if l["bin"] == b) for b in range(4)] == [10, 10, 10, 10]
    assert all(l["judgeable"] for l in lines)                    # enough judgeable lines → none unjudgeable
    units = sorted(l["unit_id"] for l in lines if l["bin"] == 1)
    assert units[0] < 5 and units[-1] > 24                       # spread over the chunk, not the first ten
    assert len(set(l["index"] for l in lines)) == 40
    # a thin bin fills with unjudgeable lines and never duplicates
    thin = [{"index": i, "unit_id": i, "dur": 0.5, "judgeable": i < 4} for i in range(7)]
    got = ab.stratify(thin, 10)
    assert len(got) == 7 and sum(1 for l in got if l["judgeable"]) == 4 and len(set(l["index"] for l in got)) == 7


def test_speedup_and_consistency_stats():
    s = ab.speedup_stats([1.0, 1.3, 1.7, 2.0], [1.0, 1.0, 1.0, 1.0], 1.25, 1.60)
    assert s["n"] == 4 and abs(s["gt_soft"] - 0.75) < 1e-9 and abs(s["gt_hard"] - 0.5) < 1e-9
    assert ab.speedup_stats([], [], 1.25, 1.6)["gt_soft"] is None
    rng = np.random.default_rng(1)
    base = rng.normal(size=192); base /= np.linalg.norm(base)
    tight = [(base + 0.05 * rng.normal(size=192)) for _ in range(20)]
    tight = [e / np.linalg.norm(e) for e in tight]
    loose = [(base + 0.6 * rng.normal(size=192)) for _ in range(20)]
    loose = [e / np.linalg.norm(e) for e in loose]
    ct, cl = ab.consistency(tight), ab.consistency(loose)
    assert ct["p10"] > cl["p10"] and ct["spread"] < cl["spread"] and ct["min"] <= ct["p10"]
    assert ab.consistency(tight[:2])["p10"] is None


def test_pref_threshold_is_a_binomial_test():
    assert ab.pref_threshold(30) == 20          # 18/30 has p ≈ 0.18: not evidence
    assert ab.pref_threshold(40) == 26
    assert ab.pref_threshold(10) == 9
    assert ab.pref_threshold(0) == 1            # no votes can never pass


def _arm(**kw):
    base = {"kana_lines": 0, "non_zh_lines": 0, "overlap_median": 1.0, "guard_ok_rate": 1.0, "ratio_pass_rate": 1.0,
            "sim_median": 0.55, "consistency": {"p10": 0.70}, "speedup": {"gt_soft": 0.05, "gt_hard": 0.025}, "collapsed": 0}
    base.update(kw)
    return base


def _metrics(A, B, A2=None, base_soft=0.05, base_hard=0.025):
    arms = {"A": A, "B": B}
    if A2:
        arms["A2"] = A2
    return {"arms": arms, "baseline": {"speedup": {"gt_soft": base_soft, "gt_hard": base_hard}}}


def test_decide_requires_every_condition():
    B = _arm(sim_median=0.50, consistency={"p10": 0.70})
    A = _arm(sim_median=0.56, consistency={"p10": 0.66})           # +0.06 timbre, p10 within 0.05
    prefs = {"votes": {f"{i:02d}": ("A" if i < 26 else "B") for i in range(40)}}
    d = ab.decide(_metrics(A, B), prefs)
    assert d["verdict"] == "adopt_cross_lingual" and all(d["conditions"].values()), d
    # without listening the verdict cannot be a win
    d0 = ab.decide(_metrics(A, B), None)
    assert d0["verdict"] == "keep_sft_vc" and d0["conditions"]["e_preference"] is None
    # 25/40 is not evidence
    weak = {"votes": {f"{i:02d}": ("A" if i < 25 else "B") for i in range(40)}}
    assert ab.decide(_metrics(A, B), weak)["conditions"]["e_preference"] is False
    # one kana line, a timbre margin below 0.05, a p10 more than 0.05 behind, one collapsed take,
    # or 3 more guard failures than B (> 5 % of 40) each sink A
    for bad in ({"kana_lines": 1}, {"sim_median": 0.54}, {"consistency": {"p10": 0.64}}, {"collapsed": 1},
                {"guard_ok_rate": 0.92}, {"non_zh_lines": 1}, {"overlap_median": 0.9}):
        assert ab.decide(_metrics(_arm(**dict(A, **bad)), B), prefs)["verdict"] == "keep_sft_vc", bad
    # 2 more guard failures than B (5 % of 40) is tolerated
    assert ab.decide(_metrics(_arm(**dict(A, guard_ok_rate=0.95)), B), prefs)["verdict"] == "adopt_cross_lingual"


def test_decide_timing_is_relative_to_the_sft_baseline():
    B = _arm(sim_median=0.50)
    A = _arm(sim_median=0.56, consistency={"p10": 0.70}, speedup={"gt_soft": 0.10, "gt_hard": 0.05})
    prefs = {"votes": {f"{i:02d}": "A" for i in range(40)}}
    assert ab.decide(_metrics(A, B, base_soft=0.05, base_hard=0.025), prefs)["conditions"]["d_timing"] is True
    assert ab.decide(_metrics(A, B, base_soft=0.04, base_hard=0.025), prefs)["conditions"]["d_timing"] is False
    assert ab.decide(_metrics(A, B, base_soft=0.05, base_hard=0.02), prefs)["conditions"]["d_timing"] is False
    # the design's absolute 30 % / 10 % would have passed this arm; the baseline-relative rule does not
    loose = _arm(sim_median=0.56, speedup={"gt_soft": 0.25, "gt_hard": 0.08})
    assert ab.decide(_metrics(loose, B), prefs)["conditions"]["d_timing"] is False


def test_alt_reference_is_a_finding_not_a_winner():
    B = _arm(sim_median=0.50)
    A = _arm(sim_median=0.40)                                   # A loses on timbre
    A2 = _arm(sim_median=0.60, guard_ok_rate=1.0)               # A′ would have won
    prefs = {"votes": {f"{i:02d}": "A" for i in range(40)}}
    d = ab.decide(_metrics(A, B, A2), prefs)
    assert d["verdict"] == "keep_sft_vc" and d["alt_reference_finding"]["alt_ref_better_timbre"] is True
    assert ab.decide(_metrics(A, B), prefs)["alt_reference_finding"] is None
    assert ab.decide({"arms": {"A": A}}, prefs)["verdict"] == "keep_sft_vc"


def test_listening_kit_key_matches_files():
    import tempfile
    import soundfile as sf
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        sr = 16000
        a = np.sin(np.arange(sr) * 2 * np.pi * 220 / sr).astype(np.float32) * 0.3     # 1.0 s, fits a 2 s slot → copied
        b = np.sin(np.arange(sr) * 2 * np.pi * 440 / sr).astype(np.float32) * 0.3
        sf.write(d / "a.wav", a, sr); sf.write(d / "b.wav", b, sr)
        lines = [{"index": 7, "zh": "早上好", "ja": "おはよう", "slot": 2.0, "dur": 1.0, "b_raw": str(d / "b.wav"), "b_fit": str(d / "b.wav")},
                 {"index": 3, "zh": "你好", "ja": "こんにちは", "slot": 2.0, "dur": 1.0, "b_raw": str(d / "b.wav"), "b_fit": str(d / "b.wav")}]
        ab.listening_kit(lines, {7: str(d / "a.wav"), 3: str(d / "a.wav")}, d / "kit")
        key = __import__("json").loads((d / "kit" / "key.json").read_text())
        assert set(key) == {"00", "01"} and key["00"]["index"] == 3 and key["01"]["index"] == 7    # sorted by index
        for p, k in key.items():
            for letter in ("X", "Y"):
                got, _ = sf.read(d / "kit" / "pairs" / f"{p}_{letter}.wav", dtype="float32")
                assert np.allclose(got, a if k[letter] == "A" else b, atol=1e-3), (p, letter, k)   # PCM_16 round trip
            assert {k["X"], k["Y"]} == {"A", "B"}
        assert (d / "kit" / "pairs.csv").exists() and (d / "kit" / "README.md").exists()
        prefs = __import__("json").loads((d / "kit" / "prefs.json").read_text())
        assert set(prefs["votes"]) == {"00", "01"}


def test_prefs_map_letters_to_arms():
    key = {"00": {"X": "A", "Y": "B", "index": 5}, "01": {"X": "B", "Y": "A", "index": 9}, "02": {"X": "A", "Y": "B", "index": 11}}
    votes = ab.prefs_to_arms({"votes": {"00": "x", "01": "X", "02": "=", "03": "Y"}}, key)["votes"]
    assert votes == {"00": "A", "01": "B", "02": "=", "03": ""}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
