"""vc_guard.judge_line: lost voicing, octave collapse and out-of-band outputs fall back (no GPU).

    .venv/bin/python tests/test_vc_guard.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie.vc_guard import guard_lines, judge_line     # noqa: E402

SR = 16000


def voiced(sec: float, hz: float, seed: int = 0) -> np.ndarray:
    """A pulse train with harmonics — pYIN sees it as confidently voiced at *hz*."""
    t = np.arange(int(sec * SR)) / SR
    y = sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 8))
    return (0.3 * y / np.abs(y).max()).astype("float32")


def noise(sec: float) -> np.ndarray:
    return (0.05 * np.random.default_rng(1).standard_normal(int(sec * SR))).astype("float32")


def wav(d: Path, name: str, y: np.ndarray) -> str:
    p = d / name
    sf.write(p, y, SR)
    return str(p)


def test_judge_line():
    d = Path(tempfile.mkdtemp(prefix="vcguard_"))
    v1 = wav(d, "v1.wav", voiced(2.0, 220))
    assert judge_line(wav(d, "ok.wav", voiced(2.1, 230)), v1, "female")["ok"]
    r = judge_line(wav(d, "noise.wav", noise(2.0)), v1, "female")
    assert not r["ok"] and r["reason"].startswith(("lost_voicing", "unvoiced"))
    r = judge_line(wav(d, "octave.wav", voiced(2.0, 110)), v1, "female")
    assert not r["ok"] and (r["reason"].startswith("band_female") or r["reason"].startswith("pitch_jump"))
    r = judge_line(wav(d, "up.wav", voiced(2.0, 310)), v1, "female")     # 1.41× = a jump even inside the band
    assert not r["ok"] and r["reason"].startswith("pitch_jump")
    assert judge_line(wav(d, "male.wav", voiced(2.0, 120)), wav(d, "v1m.wav", voiced(2.0, 125)), "male")["ok"]


def test_short_baseline_is_not_judged_and_chunks_drop_whole():
    d = Path(tempfile.mkdtemp(prefix="vcguard_"))
    v1_short = wav(d, "v1s.wav", voiced(0.25, 220))           # ~15 voiced frames: below the judging floor
    r = judge_line(wav(d, "cs.wav", noise(0.3)), v1_short, "female")
    assert r["ok"] and not r["judged"]
    v1 = [{"audio_fit": wav(d, f"v1_{i}.wav", voiced(1.5, 220))} for i in range(4)]
    segs = [{"gender": "female"} for _ in range(4)]
    items = {0: {"audio": wav(d, "c0.wav", voiced(1.5, 225)), "vc": True, "chunk": [0, 1]},
             1: {"audio": wav(d, "c1.wav", noise(1.5)), "vc": True, "chunk": [0, 1]},
             2: {"audio": wav(d, "c2.wav", voiced(1.5, 225)), "vc": True},
             3: {"audio": wav(d, "c3.wav", voiced(1.5, 110)), "vc": True}}
    st = guard_lines(segs, items, v1)
    assert st["rejected"] == 2 and st["dropped"] == 3                  # 1 fails → its chunk mate 0 goes too; 3 fails alone
    assert items[0]["vc"] is False and items[0]["guard"] == "chunk_member" and items[2]["vc"] is True
    assert list(st["reasons"]) and all(k in ("lost", "unvoiced", "band", "pitch") for k in st["reasons"])


def test_guard_lines_rewrites_items():
    d = Path(tempfile.mkdtemp(prefix="vcguard_"))
    v1 = [{"audio_fit": wav(d, f"v1_{i}.wav", voiced(1.5, 220))} for i in range(3)]
    segs = [{"gender": "female"} for _ in range(3)]
    items = {0: {"audio": wav(d, "c0.wav", voiced(1.5, 225)), "vc": True},
             1: {"audio": wav(d, "c1.wav", noise(1.5)), "vc": True},
             2: {"audio": v1[2]["audio_fit"], "vc": False}}
    st = guard_lines(segs, items, v1)
    assert st["checked"] == 2 and st["rejected"] == 1
    assert items[1]["vc"] is False and items[1]["audio"] == v1[1]["audio_fit"] and items[1]["guard"]
    assert items[0]["vc"] is True
    assert st["drift"] == {"enabled": False, "units": 0, "judged": 0, "rejected": 0}     # transcribe=None: pitch only


def test_guard_source_key_judges_natural_line_but_falls_back_to_fitted():
    """source_key="audio": the baseline is v1['audio'] (what was converted); the fallback is still audio_fit."""
    d = Path(tempfile.mkdtemp(prefix="vcguard_"))
    # natural take is voiced at 220 Hz for 1.0 s; the fitted wav is noise — judging against the fitted one
    # would reject a good conversion, judging against the natural one accepts it
    v1 = [{"audio": wav(d, "nat0.wav", voiced(1.0, 220)), "audio_fit": wav(d, "fit0.wav", noise(0.8))},
          {"audio": wav(d, "nat1.wav", voiced(1.0, 220)), "audio_fit": wav(d, "fit1.wav", voiced(0.8, 220))}]
    segs = [{"gender": "female"}, {"gender": "female"}]
    items = {0: {"audio": wav(d, "c0.wav", voiced(1.05, 225)), "vc": True},
             1: {"audio": wav(d, "c1.wav", noise(1.05)), "vc": True}}
    st = guard_lines(segs, items, v1, source_key="audio")
    assert st["checked"] == 2 and st["rejected"] == 1
    assert items[0]["vc"] is True
    assert items[1]["vc"] is False and items[1]["audio"] == v1[1]["audio_fit"]     # fallback = fitted, not natural
    # default key keeps today's behaviour: line 0 judged against the noise fit → not judged (no voiced baseline)
    items2 = {0: {"audio": items[0]["audio"], "vc": True}}
    st2 = guard_lines(segs, items2, v1)
    assert st2["checked"] == 0 and items2[0]["vc"] is True


def test_guard_lines_drift_chunk_and_single():
    """Content verdicts fold into the pitch verdicts: a drifted chunk drops whole, a flipped single drops,
    a line the judge cannot hear stays pitch-judged only, the pitch reason wins when both fail."""
    d = Path(tempfile.mkdtemp(prefix="vcguard_"))
    v1 = [{"audio_fit": wav(d, f"v1_{i}.wav", voiced(1.5, 220))} for i in range(6)]
    segs = [{"gender": "female", "text_translated": t}
            for t in ("把腿伸直", "好舒服啊", "不妙啊", "哥哥……", "这么多", "疼得要命")]
    chunk_src = wav(d, "chunk_0000_0001.wav", voiced(3.0, 220))
    chunk_out = wav(d, "seg_1000001.wav", voiced(3.1, 225))
    items = {0: {"audio": wav(d, "c0.wav", voiced(1.5, 225)), "vc": True, "chunk": [0, 1],
                 "chunk_src": chunk_src, "chunk_out": chunk_out},
             1: {"audio": wav(d, "c1.wav", voiced(1.5, 225)), "vc": True, "chunk": [0, 1],
                 "chunk_src": chunk_src, "chunk_out": chunk_out},
             2: {"audio": wav(d, "c2.wav", voiced(1.5, 225)), "vc": True},          # script flip
             3: {"audio": wav(d, "c3.wav", voiced(1.5, 225)), "vc": True},          # v1 not heard → pitch only
             4: {"audio": wav(d, "c4.wav", voiced(1.5, 110)), "vc": True},          # pitch fails AND drifts
             5: {"audio": wav(d, "c5.wav", voiced(1.5, 225)), "vc": True}}          # intact
    T = {items[0]["audio"]: "把腿伸直", v1[0]["audio_fit"]: "把腿伸直",
         items[1]["audio"]: "打书法", v1[1]["audio_fit"]: "好舒服啊",                    # member 1 drifted
         chunk_out: "把腿伸直。打书法", chunk_src: "把腿伸直。好舒服啊",                   # chunk level: 0.53 → passes (dilution)
         items[2]["audio"]: {"auto": {"text": "Tchau, tchau.", "language": "pt"}, "zh": {"text": "桥桥"}},
         v1[2]["audio_fit"]: "不妙啊",
         items[3]["audio"]: "Booga.", v1[3]["audio_fit"]: {"auto": {"text": "Фига!", "language": "ru"}, "zh": None},
         items[4]["audio"]: "我们都要", v1[4]["audio_fit"]: "这么多",
         items[5]["audio"]: "疼得要命", v1[5]["audio_fit"]: "疼得要命"}
    calls = []
    conv_paths = {i: it["audio"] for i, it in items.items()}          # guard_lines rewrites rejected items

    def transcribe(paths):
        calls.append(list(paths))
        return {p: T[p] for p in paths}

    st = guard_lines(segs, items, v1, transcribe=transcribe)
    assert len(calls) == 1 and len(calls[0]) == len(set(calls[0]))                  # one call, deduped
    assert set(calls[0]) == set(T)                                                    # every unit path, once
    v = st["verdicts"]
    assert v[1]["reason"].startswith("drift_") and not v[1]["ok"]
    # the chunk-level reading passes (0.53: the intact mate dilutes the drift) — the member level catches it
    assert v[0]["ok"] and v[0]["drift"]["score"] == 1.0 and 0.5 <= v[0]["drift_chunk"]["score"] < 0.6
    assert items[0]["vc"] is False and items[0]["guard"] == "chunk_member"           # chunk mate dropped whole
    assert items[1]["vc"] is False and items[1]["guard"].startswith("drift_")
    assert v[2]["reason"] == "script_flip" and items[2]["vc"] is False
    assert v[3]["drift"]["gate"] == "no_v1_cjk" and v[3]["ok"] and items[3]["vc"] is True
    assert not v[4]["ok"] and v[4]["reason"].startswith(("band_", "pitch_jump")) and v[4]["drift"]["reason"].startswith("drift_")
    assert v[5]["ok"] and items[5]["vc"] is True
    assert st["checked"] == 6 and st["rejected"] == 3 and st["dropped"] == 4
    assert st["reasons"]["drift"] == 1 and st["reasons"]["script"] == 1
    assert st["drift"]["units"] == 7 and st["drift"]["rejected"] == 3 and st["drift"]["judged"] == 6   # 6 lines judged + chunk unit judged, line 3 not
    # a crashing transcriber never fails the guard: pitch verdicts stand, error recorded
    items_b = {4: {"audio": conv_paths[4], "vc": True}, 5: {"audio": conv_paths[5], "vc": True}}

    def boom(paths):
        raise RuntimeError("no GPU")
    st_b = guard_lines(segs, items_b, v1, transcribe=boom)
    assert st_b["drift"]["error"].startswith("RuntimeError") and st_b["checked"] == 2 and st_b["rejected"] == 1


def test_guard_drift_chunk_verdict_drops_members():
    """A chunk whose whole-utterance reading drifted drops every member even when each member's own reading is short."""
    d = Path(tempfile.mkdtemp(prefix="vcguard_"))
    v1 = [{"audio_fit": wav(d, f"v1_{i}.wav", voiced(0.4, 220))} for i in range(2)]      # too short for pitch
    segs = [{"gender": "female", "text_translated": "好大"}, {"gender": "female", "text_translated": "好疼"}]
    src, out = wav(d, "chunk.wav", voiced(1.3, 220)), wav(d, "conv.wav", voiced(1.3, 225))
    items = {i: {"audio": wav(d, f"c{i}.wav", voiced(0.4, 225)), "vc": True, "chunk": [0, 1],
                 "chunk_src": src, "chunk_out": out} for i in range(2)}
    T = {items[0]["audio"]: "好大", items[1]["audio"]: "好疼", v1[0]["audio_fit"]: "好大", v1[1]["audio_fit"]: "好疼",
         out: "网络电脑", src: "好大。好疼"}
    st = guard_lines(segs, items, v1, transcribe=lambda ps: {p: T[p] for p in ps})
    assert st["verdicts"][0]["reason"].startswith("drift_") and st["verdicts"][0]["reason"].endswith("_chunk")
    assert items[0]["vc"] is False and items[1]["vc"] is False and st["dropped"] == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
