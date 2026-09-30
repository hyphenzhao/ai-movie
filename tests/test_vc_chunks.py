"""Chunked voice conversion: grouping, chunk layout and split-back (no GPU).

    .venv/bin/python tests/test_vc_chunks.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie import tts as T                   # noqa: E402

SR = 22050


def tone(sec: float, hz: float) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype("float32")


def test_groups_join_short_lines_to_same_speaker_neighbours():
    segs = [{"start": 0, "end": 0.4, "speaker": "S0"},
            {"start": 0.5, "end": 1.4, "speaker": "S0"},
            {"start": 1.5, "end": 2.0, "speaker": "S0"},
            {"start": 2.1, "end": 2.4, "speaker": "S1"},
            {"start": 9.0, "end": 9.3, "speaker": "S1"}]
    conv = {i: ("x", "refA" if i < 3 else "refB", d)
            for i, d in enumerate([0.4, 0.9, 0.5, 0.3, 0.3])}
    assert T._vc_chunks(segs, conv, min_seconds=0.7, target_seconds=1.5) == [[0, 1, 2], [3, 4]]
    # a long line stays alone unless its neighbour is too short to convert
    conv2 = {0: ("x", "r", 2.0), 1: ("x", "r", 1.8), 2: ("x", "r", 0.4)}
    segs2 = [{"start": i, "end": i + 1, "speaker": "S0"} for i in range(3)]
    assert T._vc_chunks(segs2, conv2, min_seconds=0.7, target_seconds=1.5) == [[0], [1, 2]]
    # never across a speaker change or a different reference
    conv3 = {0: ("x", "rA", 0.3), 1: ("x", "rB", 0.3)}
    segs3 = [{"start": 0, "end": 0.3, "speaker": "S0"}, {"start": 0.4, "end": 0.7, "speaker": "S0"}]
    assert T._vc_chunks(segs3, conv3, min_seconds=0.7, target_seconds=1.5) == [[0], [1]]


def test_chunk_roundtrip_splits_at_inserted_silence():
    d = Path(tempfile.mkdtemp(prefix="vcchunk_"))
    lens = [0.4, 0.9, 0.5]
    srcs = []
    for n, (sec, hz) in enumerate(zip(lens, (220, 330, 440))):
        p = d / f"src{n}.wav"
        sf.write(p, tone(sec, hz), SR)
        srcs.append(str(p))
    layout = T._write_vc_chunk(srcs, d / "chunk.wav")
    assert layout and len(layout["bounds"]) == 3
    # pretend the converter resampled to 24 kHz and stretched by 3 %
    y, _ = sf.read(d / "chunk.wav", dtype="float32")
    n_out = int(len(y) * 24000 / SR * 1.03)
    y2 = y[(np.arange(n_out) * (len(y) - 1) / (n_out - 1)).astype(int)]
    sf.write(d / "conv.wav", y2, 24000)
    pieces = T._split_vc_chunk(str(d / "conv.wav"), {"members": [10, 11, 12], **layout}, d)
    assert all(pieces)
    for p, sec in zip(pieces, lens):
        got = sf.info(p).duration
        assert abs(got - sec * 1.03) < 0.08, (p, got, sec)
        assert sf.info(p).samplerate == 24000


def test_mixed_sample_rates_refuse_chunking():
    d = Path(tempfile.mkdtemp(prefix="vcchunk_"))
    a, b = d / "a.wav", d / "b.wav"
    sf.write(a, tone(0.5, 220), 22050)
    sf.write(b, tone(0.5, 220), 24000)
    assert T._write_vc_chunk([str(a), str(b)], d / "chunk.wav") is None


def test_audio_bound_counts_separators():
    """A staccato run (every line < min_seconds) keeps appending under the 12 s span rule; the audio bound
    Σdur + 0.45·(n−1) splits it once the chunk would exceed max_audio_seconds."""
    n = 14                                                   # 14 × 0.6 s lines inside an 11 s span
    segs = [{"start": i * 0.78, "end": i * 0.78 + 0.6, "speaker": "S0"} for i in range(n)]
    conv = {i: ("x", "r", 0.6) for i in range(n)}
    assert T._vc_chunks(segs, conv, min_seconds=0.7, target_seconds=1.5) == [list(range(n))]   # 8.4 s + 5.85 s = 14.25 s ≤ 20
    groups = T._vc_chunks(segs, conv, min_seconds=0.7, target_seconds=1.5, max_audio_seconds=7.0)
    assert [len(g) for g in groups] == [7, 7]                # 7 × 0.6 + 6 × 0.45 = 6.9 ≤ 7 < 8th line
    assert sum(groups, []) == list(range(n))
    # grouping is identical to the old rule when every chunk is small
    segs2 = [{"start": 0, "end": 0.4, "speaker": "S0"}, {"start": 0.5, "end": 1.4, "speaker": "S0"},
             {"start": 1.5, "end": 2.0, "speaker": "S0"}]
    conv2 = {i: ("x", "r", dd) for i, dd in enumerate([0.4, 0.9, 0.5])}
    assert (T._vc_chunks(segs2, conv2, min_seconds=0.7, target_seconds=1.5)
            == T._vc_chunks(segs2, conv2, min_seconds=0.7, target_seconds=1.5, max_audio_seconds=1e9) == [[0, 1, 2]])


def test_run_vc_conversion_source_key_and_fallbacks(monkeypatch=None):
    """With source_key="audio" the fake worker receives the natural takes (told apart by tone), a segment
    without `audio` falls back to audio_fit, a > 29 s source is skipped, every result carries source/source_dur,
    and chunk members carry chunk_src/chunk_out.  The worker and the Ollama eviction are monkeypatched:
    run_vc_conversion evicts every resident Ollama model first, which would hit a concurrently running pipeline."""
    from ai_movie import translator
    d = Path(tempfile.mkdtemp(prefix="vcconv_"))
    hz = {"audio": 220, "audio_fit": 440}

    def mk(name, key, sec):
        p = d / f"{name}_{key}.wav"
        sf.write(p, tone(sec, hz[key]), SR)
        return str(p)

    segs = [{"start": 0.0, "end": 2.0, "speaker": "S0", "text_translated": "第一句",
             "audio": mk("s0", "audio", 2.0), "audio_fit": mk("s0", "audio_fit", 1.6)},
            {"start": 2.5, "end": 4.0, "speaker": "S0", "text_translated": "第二句",       # no natural take
             "audio_fit": mk("s1", "audio_fit", 1.5)},
            {"start": 5.0, "end": 5.4, "speaker": "S3", "text_translated": "短",             # short → chunked with 3
             "audio": mk("s2", "audio", 0.4), "audio_fit": mk("s2", "audio_fit", 0.4)},
            {"start": 5.5, "end": 6.0, "speaker": "S3", "text_translated": "也短",
             "audio": mk("s3", "audio", 0.5), "audio_fit": mk("s3", "audio_fit", 0.5)},
            {"start": 7.0, "end": 40.0, "speaker": "S0", "text_translated": "太长",
             "audio": mk("s4", "audio", 31.0), "audio_fit": mk("s4", "audio_fit", 31.0)},
            {"start": 41.0, "end": 42.0, "speaker": "S1", "text_translated": "无参考",
             "audio": mk("s5", "audio", 1.0), "audio_fit": mk("s5", "audio_fit", 1.0)}]
    received: dict[int, str] = {}

    def fake_isolated(seg_texts, model_choice, ref_audio, ref_text, method, output_dir, *,
                      progress_cb=None, cancel_check=None, seg_refs=None, seg_sources=None):
        out = {}
        for idx, _text in seg_texts:
            src = seg_sources[idx]
            received[idx] = src
            dst = Path(output_dir) / f"seg_{idx + 1:04d}.wav"
            y, sr = sf.read(src, dtype="float32")
            sf.write(dst, y, sr)                                  # "conversion" = copy
            out[idx] = {"audio": str(dst)}
        return out

    orig_iso, orig_free = T.run_isolated_synthesis, translator.free_gpu_for_local_work
    T.run_isolated_synthesis = fake_isolated
    translator.free_gpu_for_local_work = lambda *a, **k: None
    try:
        ref = mk("ref", "audio", 3.0)
        res = T.run_vc_conversion(segs, {"S0": {"ref_audio": ref}, "S3": {"ref_audio": ref}}, d / "out", source_key="audio")
    finally:
        T.run_isolated_synthesis, translator.free_gpu_for_local_work = orig_iso, orig_free

    def dominant_hz(path):
        y, sr = sf.read(path, dtype="float32")
        spec = np.abs(np.fft.rfft(y))
        return np.fft.rfftfreq(len(y), 1 / sr)[int(np.argmax(spec))]

    assert res[0]["vc"] and abs(dominant_hz(received[0]) - 220) < 5          # natural take was converted
    assert res[0]["source"] == segs[0]["audio"] and abs(res[0]["source_dur"] - 2.0) < 0.01
    assert res[1]["vc"] and abs(dominant_hz(received[1]) - 440) < 5          # fell back to audio_fit
    assert res[1]["source"] == segs[1]["audio_fit"]
    assert res[2]["vc"] and res[3]["vc"] and res[2]["chunk"] == [2, 3]        # short lines chunked
    assert res[2]["chunk_src"] == res[3]["chunk_src"] and Path(res[2]["chunk_src"]).exists()
    assert res[2]["chunk_out"] == res[3]["chunk_out"] and Path(res[2]["chunk_out"]).exists()
    assert res[2]["source"] == segs[2]["audio"]
    assert res[4]["vc"] is False and res[4]["skipped"].startswith("source > 29") and 4 not in received
    assert res[5]["vc"] is False and res[5]["source"] == segs[5]["audio"] and res[5]["source_dur"] == 1.0
    # default source key comes from config (audio); audio_fit reproduces v3.3's input
    T.run_isolated_synthesis = fake_isolated
    translator.free_gpu_for_local_work = lambda *a, **k: None
    received.clear()
    try:
        res2 = T.run_vc_conversion(segs[:1], {"S0": {"ref_audio": ref}}, d / "out2", source_key="audio_fit")
    finally:
        T.run_isolated_synthesis, translator.free_gpu_for_local_work = orig_iso, orig_free
    assert abs(dominant_hz(received[0]) - 440) < 5 and res2[0]["source"] == segs[0]["audio_fit"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
