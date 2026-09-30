"""run_vc_version._merge_attempts: per-speaker choice, failed attempts never win (no GPU).

    .venv/bin/python tests/test_vc_attempts.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("run_vc_version", ROOT / "scripts" / "run_vc_version.py")
rv = importlib.util.module_from_spec(spec); spec.loader.exec_module(rv)     # type: ignore[union-attr]


def attempt(k, per_line, ref):
    """per_line: {i: (vc, ok|None)} → (k, items, stats, ref_set)."""
    items = {i: {"audio": f"a{k}_{i}.wav", "vc": vc} for i, (vc, _) in per_line.items()}
    verdicts = {i: {"ok": ok, "judged": True, "reason": "" if ok else "lost_x"} for i, (vc, ok) in per_line.items() if vc and ok is not None}
    return (k, items, {"verdicts": verdicts, "checked": len(verdicts), "rejected": sum(1 for v in verdicts.values() if not v["ok"])},
            {"A": {"ref_audio": f"refA{ref}.wav"}, "B": {"ref_audio": f"refB{ref}.wav"}})


def test_per_speaker_choice_and_failed_attempt_loses():
    segs = [{"speaker": "A"}] * 4 + [{"speaker": "B"}] * 4
    a0 = attempt(0, {0: (True, True), 1: (True, True), 2: (True, True), 3: (True, False),      # A: 1/4 rejected
                     4: (True, False), 5: (True, False), 6: (True, True), 7: (True, False)}, "0")   # B: 3/4 rejected
    a1 = attempt(1, {0: (True, False), 1: (True, False), 2: (True, True), 3: (True, True),      # A: 2/4 (worse)
                     4: (True, True), 5: (True, True), 6: (True, True), 7: (True, False)}, "1")     # B: 1/4 (better)
    a2 = attempt(2, {i: (False, None) for i in range(8)}, "2")                                   # worker died: nothing judged
    items, refs, stats, chosen = rv._merge_attempts([a0, a1, a2], segs)
    assert chosen == {"A": 0, "B": 1}
    assert refs["A"]["ref_audio"] == "refA0.wav" and refs["B"]["ref_audio"] == "refB1.wav"
    assert items[0]["audio"] == "a0_0.wav" and items[4]["audio"] == "a1_4.wav"
    assert stats["checked"] == 8 and stats["rejected"] == 2
    # only the dead attempt available → it is still chosen (there is nothing else), with checked 0
    items, refs, stats, chosen = rv._merge_attempts([a2], segs)
    assert chosen == {"A": 2, "B": 2} and stats["checked"] == 0


def test_merge_carries_chosen_verdicts_and_drift():
    """The merged stats carry each speaker's verdicts from ITS chosen attempt (f0/voiced/drift reach the
    report) and aggregate the content judge across attempts."""
    segs = [{"speaker": "A"}] * 2 + [{"speaker": "B"}] * 2
    a0 = attempt(0, {0: (True, True), 1: (True, True), 2: (True, False), 3: (True, False)}, "0")
    a1 = attempt(1, {0: (True, False), 1: (True, False), 2: (True, True), 3: (True, True)}, "1")
    for k, (_k, _it, st, _r) in enumerate((a0, a1)):
        for i, v in st["verdicts"].items():
            v["f0_conv"] = 200 + 10 * k + i
            v["drift"] = {"judged": True, "ok": v["ok"], "score": 1.0 if v["ok"] else 0.2, "reason": "" if v["ok"] else "drift_0.20"}
        st["drift"] = {"enabled": True, "units": 4, "judged": 4, "rejected": 2, "seconds": 3.0 + k}
    items, refs, stats, chosen = rv._merge_attempts([a0, a1], segs)
    assert chosen == {"A": 0, "B": 1}
    assert stats["verdicts"][0]["f0_conv"] == 200 and stats["verdicts"][2]["f0_conv"] == 212   # A from try0, B from try1
    assert all(stats["verdicts"][i]["ok"] for i in range(4)) and stats["rejected"] == 0
    d = stats["drift"]
    assert d["enabled"] and d["judged"] == 4 and d["rejected"] == 0 and d["seconds"] == 7.0
    assert [a["attempt"] for a in d["attempts"]] == [0, 1]
    # an attempt whose judge crashed is recorded, not fatal
    a1[2]["drift"]["error"] = "RuntimeError: no GPU"
    assert rv._merge_attempts([a0, a1], segs)[2]["drift"]["error"] == "RuntimeError: no GPU"


def test_pin_to_v1_one_stretch_and_bitexact_fallback():
    """pin_to_v1: a converted line is stretched once to v1's sample count (fit_ratio = v1's, vc_pin_ratio =
    have/target, vc_len_ratio = have/source); a vc=False line is v1's fitted wav copied bit-exact; fit_end is
    measured from the pinned file and equals v1's."""
    import tempfile
    import numpy as np
    import soundfile as sf
    sr = 24000
    d = Path(tempfile.mkdtemp(prefix="vcpin_"))

    def tone(sec, hz, name):
        t = np.arange(int(sec * sr)) / sr
        p = d / name
        sf.write(p, (0.3 * np.sin(2 * np.pi * hz * t)).astype("float32"), sr)
        return str(p)

    v1 = [{"start": 3.0, "end": 3.8, "audio": tone(1.0, 220, "nat0.wav"), "audio_fit": tone(0.8, 220, "fit0.wav"),
           "fit_ratio": 1.25, "fit_end": 3.8, "overrun": 0.1},
          {"start": 4.0, "end": 5.0, "audio": tone(1.0, 220, "nat1.wav"), "audio_fit": tone(1.0, 220, "fit1.wav"),
           "fit_ratio": 1.0, "fit_end": 5.0, "overrun": 0},
          {"start": 6.0, "end": 6.5, "audio": tone(0.5, 220, "nat2.wav")}]                 # no audio_fit → untouched
    conv0 = tone(1.05, 230, "conv0.wav")
    items = {0: {"audio": conv0, "vc": True, "source": v1[0]["audio"], "source_dur": 1.0},
             1: {"audio": v1[1]["audio"], "vc": False, "source": v1[1]["audio"], "source_dur": 1.0, "skipped": "x"},
             2: {"audio": None, "vc": False}}
    segs = [dict(s, audio=(items[i]["audio"] if items[i].get("vc") else s.get("audio_fit"))) for i, s in enumerate(v1)]

    def stretch(src, dst, ratio):
        y, r = sf.read(str(src), dtype="float32")
        n = int(round(len(y) / ratio))
        sf.write(str(dst), y[(np.arange(n) * (len(y) - 1) / max(n - 1, 1)).astype(int)], r)

    rv.pin_to_v1(segs, v1, d / "fitted", items, source_key="audio", stretch=stretch)
    a, _ = sf.read(segs[0]["audio_fit"], dtype="float32")
    assert len(a) == round(0.8 * sr)
    assert segs[0]["fit_ratio"] == 1.25 and abs(segs[0]["vc_pin_ratio"] - 1.3125) < 1e-3 and abs(segs[0]["vc_len_ratio"] - 1.05) < 1e-3
    assert segs[0]["fit_end"] == 3.8 and segs[0]["overrun"] == 0.1
    b, _ = sf.read(segs[1]["audio_fit"], dtype="float32")
    ref, _ = sf.read(v1[1]["audio_fit"], dtype="float32")
    assert np.array_equal(b, ref) and segs[1]["vc_pin_ratio"] == 1.0 and segs[1]["vc_len_ratio"] is None
    assert segs[1]["fit_end"] == 5.0 and segs[1]["fit_ratio"] == 1.0
    assert "audio_fit" not in segs[2] and "vc_pin_ratio" not in segs[2]
    assert all(abs(s["fit_end"] - v1[i]["fit_end"]) == 0 for i, s in enumerate(segs) if "fit_end" in s)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
