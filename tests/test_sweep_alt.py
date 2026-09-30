"""The sweep's second decoder (asr.AnimeWhisper / _sweep_alt_texts / _pick_alt_audio): selection,
alt_by plumbing onto words → pieces → segments, batching, failure fallback, memory release — all
with fake decoders, no GPU, no weights.

    .venv/bin/python tests/test_sweep_alt.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ai_movie import asr as A                                  # noqa: E402
from ai_movie import segmenter                                 # noqa: E402
from ai_movie.asr import (AnimeWhisper, _collect, _pick_alt_audio,   # noqa: E402
                          _resolve_alt_source, _sweep_alt_texts, _sweep_pass)

SR = 16000
ANIME_DIR = ROOT / "models" / "anime-whisper"


# ── fakes ────────────────────────────────────────────────────────

def _t0(chunk) -> float:
    """Audio arrays in these tests hold their own sample index, so a slice knows where it starts."""
    return round(float(chunk[0]) / SR, 3)


def _fake_primary(canned: dict):
    """A `_transcribe_sweep` stand-in: canned {window start: [(rel_start, rel_end, text), ...]}."""
    def fn(model, chunk, language, errors):
        segs = canned.get(_t0(chunk))
        if segs is None:
            return {"segments": []}
        out = []
        for a, b, text in segs:
            words = [{"word": ch, "start": a + (b - a) * k / len(text), "end": a + (b - a) * (k + 1) / len(text),
                      "probability": 0.5} for k, ch in enumerate(text)]
            out.append({"start": a, "end": b, "text": text, "words": words,
                        "no_speech_prob": 0.3, "avg_logprob": -0.9, "compression_ratio": 1.0})
        return {"segments": out}
    return fn


class FakeAnime:
    name = "anime"

    def __init__(self, texts: dict | None = None, fail_times: int = 0, fail_all: bool = False):
        self.texts, self.calls, self.closed = texts or {}, [], False
        self.fail_left, self.fail_all, self.disabled = fail_times, fail_all, False
        self.revision, self.dir = "deadbeef", "/fake/anime"

    def transcribe_batch(self, chunks):
        self.calls.append([_t0(c) for c in chunks])
        if self.fail_all or self.fail_left > 0:
            self.fail_left -= 1
            raise RuntimeError("HIP out of memory")
        return [self.texts.get(_t0(c), f"anime@{_t0(c)}") for c in chunks]

    def close(self):
        self.closed = True


def _audio(seconds: float, offset: int = 0):
    return (np.arange(int(seconds * SR), dtype=np.float64) + offset).astype(np.float32)


# VAD spans of a 30 s file; _sweep_windows (pad 0.3 s, index-valued audio → every frame above the
# floor) turns them into the windows 0–5.3, 7.7–20.3 and 24.7–30
SPANS = [{"start": 5.0, "end": 8.0}, {"start": 20.0, "end": 25.0}]
WIN_STARTS = [0.0, 7.7, 24.7]
CANNED = {0.0: [(1.0, 2.0, "きつい")], 7.7: [(0.8, 3.3, "もっと強くして"), (11.8, 12.3, "ね")]}   # 24.7: silence


# ── pure helpers ─────────────────────────────────────────────────

def test_generate_kwargs_text_only_no_prompt():
    g = AnimeWhisper.generate_kwargs()
    assert g["language"] == "ja" and g["task"] == "transcribe"
    assert g["num_beams"] == 1 and g["do_sample"] is False and g["return_timestamps"] is False
    assert g["no_repeat_ngram_size"] == 5 and g["max_new_tokens"] + 4 <= 448
    for k in ("prompt_ids", "initial_prompt", "prompt_condition_type", "return_token_timestamps"):
        assert k not in g, k


def test_supports_word_timestamps_rejects_inherited_heads():
    if (ANIME_DIR / "generation_config.json").exists():
        real = json.loads((ANIME_DIR / "generation_config.json").read_text())
    else:
        real = {"alignment_heads": [[7, 0], [10, 17], [12, 18], [13, 12], [16, 1], [17, 14],
                                    [19, 11], [21, 4], [24, 1], [25, 6]]}
    assert AnimeWhisper.supports_word_timestamps(real, decoder_layers=2) is False
    assert AnimeWhisper.supports_word_timestamps(real, decoder_layers=32) is True
    assert AnimeWhisper.supports_word_timestamps({"alignment_heads": [[0, 3], [1, 7]]}, decoder_layers=2) is True
    assert AnimeWhisper.supports_word_timestamps({}, decoder_layers=2) is False


def test_resolve_alt_source():
    assert _resolve_alt_source("whisper", "auto") == "other"          # v3.3: the other source is the point
    assert _resolve_alt_source("anime", "auto") == "same"             # a second model hears the same audio
    assert _resolve_alt_source("anime", "other") == "other"
    assert _resolve_alt_source("whisper", "same") == "same"


def test_pick_alt_audio_keeps_todays_rule():
    mix = _audio(10.0)
    assert _pick_alt_audio(mix, None, "same") is mix
    assert _pick_alt_audio(mix, None, "other") is None
    longer = _audio(10.5)
    got = _pick_alt_audio(mix, longer, "other")
    assert got is not None and len(got) == len(mix)                   # ≤ 1 s longer → truncated
    assert _pick_alt_audio(mix, _audio(9.8), "other") is None         # shorter within tolerance → None (as before)
    assert _pick_alt_audio(mix, _audio(11.5), "other") is None        # ≥ 1 s off → None
    assert _pick_alt_audio(mix, _audio(9.8), "same") is mix           # "same" ignores the other track entirely


def test_read_revision_from_hf_metadata():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        assert AnimeWhisper.read_revision(d) == "local"
        (d / "REVISION").write_text("abc123\n")
        assert AnimeWhisper.read_revision(d) == "abc123"
        meta = d / ".cache" / "huggingface" / "download"
        meta.mkdir(parents=True)
        (meta / "config.json.metadata").write_text("22e2008a8182b357da3922a6308d095008f72973\n8e16\n1790778881.6\n")
        assert AnimeWhisper.read_revision(d) == "22e2008a8182b357da3922a6308d095008f72973"
        (meta / "model.safetensors.metadata").write_text("ffff\n")
        assert AnimeWhisper.read_revision(d) == "ffff"                # the weights' own record wins


# ── word / piece / segment plumbing ──────────────────────────────

def test_collect_puts_alt_by_on_segment_and_words():
    res = _fake_primary(CANNED)(None, _audio(5.0), "ja", [])
    segs, words = [], []
    _collect(res, 0.0, "sweep", segs, words, alt_text="", alt_by="anime")
    assert segs[0]["alt_text"] == "" and segs[0]["alt_by"] == "anime"
    assert all(w["alt"] == "" and w["alt_by"] == "anime" for w in words)
    segs, words = [], []
    _collect(res, 0.0, "sweep", segs, words, alt_text=None, alt_by="anime")     # decoder failed
    assert "alt_text" not in segs[0] and segs[0]["alt_by"] == "anime"
    assert all("alt" not in w and w["alt_by"] == "anime" for w in words)
    segs, words = [], []
    _collect(res, 0.0, "vad", segs, words)
    assert "alt_by" not in segs[0] and all("alt_by" not in w for w in words)


def test_flush_carries_empty_alt_and_alt_by():
    buf = [{"w": "き", "s": 1.0, "e": 1.3, "p": 0.5, "pass": "sweep", "alt": "", "alt_by": "anime"},
           {"w": "つ", "s": 1.3, "e": 1.6, "p": 0.5, "pass": "sweep", "alt": "", "alt_by": "anime"},
           {"w": "い", "s": 1.6, "e": 2.0, "p": 0.5, "pass": "sweep", "alt": "", "alt_by": "anime"}]
    d = segmenter._flush(buf, "S0")
    assert d["pass"] == "sweep" and d["alt_text"] == "" and d["alt_by"] == "anime"    # "" is evidence, not absence
    buf2 = [dict(w, alt=None) for w in buf]
    for w in buf2:
        del w["alt"]
    d2 = segmenter._flush(buf2, "S0")
    assert "alt_text" not in d2 and d2["alt_by"] == "anime"                           # decoder failed: who, not what
    d3 = segmenter._flush([dict(w, pass_="vad") for w in buf], "S0")                  # a vad word list
    assert "alt_by" in d3 or d3.get("pass") == "sweep"


def test_finalize_segments_copies_alt_by():
    res = _fake_primary(CANNED)(None, _audio(5.0), "ja", [])
    segs, words = [], []
    _collect(res, 0.0, "sweep", segs, words, alt_text="んっ…はぁっ", alt_by="anime")
    out = A._finalize_segments(segs, words, source="x.wav")
    assert out and out[0]["alt_by"] == "anime" and out[0]["alt_text"] == "んっ…はぁっ" and out[0]["pass"] == "sweep"


# ── _sweep_alt_texts / _sweep_pass with fakes ────────────────────

def test_sweep_pass_anime_alt_by_and_window_records():
    A_prev = A._transcribe_sweep
    A._transcribe_sweep = _fake_primary(CANNED)
    try:
        audio, alt = _audio(30.0), _audio(30.0)
        dec = FakeAnime(texts={0.0: "きつい…んっ", 7.7: ""})
        segs, words, errors = [], [], []
        n_win, n_txt, recs = _sweep_pass(None, audio, SPANS, "ja", segs, words, errors,
                                         alt_audio=alt, alt_decoder=dec)
        assert (n_win, n_txt) == (3, 2) and not errors
        assert [r["start"] for r in recs] == WIN_STARTS                    # the real window computation ran
        assert dec.calls == [[0.0, 7.7]]                                   # one batched call for the 2 text windows
        assert [r["text"] for r in recs] == ["きつい", "もっと強くしてね", ""]  # every window recorded, silent one too
        assert [r["alt_text"] for r in recs] == ["きつい…んっ", "", None]
        assert [r["alt_by"] for r in recs] == ["anime", "anime", None]
        assert all(w["alt_by"] == "anime" for w in words) and all(s["alt_by"] == "anime" for s in segs)
        by_win = {s["start"]: s for s in segs}
        assert by_win[1.0]["alt_text"] == "きつい…んっ" and by_win[8.5]["alt_text"] == ""
        assert 19.5 in by_win and by_win[19.5]["alt_text"] == ""           # 「ね」 19.5–20.0: midpoint 19.75 < 20 → kept
    finally:
        A._transcribe_sweep = A_prev


def test_sweep_pass_words_inside_vad_span_are_dropped():
    A_prev = A._transcribe_sweep
    canned = {7.7: [(11.3, 13.3, "ねえ")]}                        # 19.0–21.0 straddles the span starting at 20.0
    A._transcribe_sweep = _fake_primary(canned)
    try:
        segs, words, errors = [], [], []
        _sweep_pass(None, _audio(30.0), SPANS, "ja", segs, words, errors,
                    alt_audio=_audio(30.0), alt_decoder=FakeAnime())
        assert [w["w"] for w in words] == ["ね"]                  # 「え」 (20.0–21.0, midpoint 20.5) belongs to the VAD pass
        assert segs == []                                        # the segment's midpoint (20.0) is inside the span
    finally:
        A._transcribe_sweep = A_prev


def test_sweep_pass_whisper_mode_unchanged():
    A_prev = A._transcribe_sweep
    canned = dict(CANNED)
    canned[0.0 + 1_000_000 / SR] = [(0.0, 1.0, "ゆっくり休ませてください")]          # the alt track's window 0
    A._transcribe_sweep = _fake_primary(canned)
    try:
        segs, words, errors = [], [], []
        recs = _sweep_pass(None, _audio(30.0), SPANS, "ja", segs, words, errors,
                           alt_audio=_audio(30.0, offset=1_000_000))[2]
        assert recs[0]["alt_text"] == "ゆっくり休ませてください" and recs[0]["alt_by"] == "whisper"
        assert recs[1]["alt_text"] == "" and recs[1]["alt_by"] == "whisper"         # large-v3 heard nothing there
        assert recs[2]["alt_text"] is None and recs[2]["alt_by"] is None            # no text: nothing to compare
        assert all(w["alt_by"] == "whisper" for w in words)
        segs, words = [], []
        recs = _sweep_pass(None, _audio(30.0), SPANS, "ja", segs, words, errors)[2]
        assert recs[0]["alt_text"] is None and recs[0]["alt_by"] is None            # no alt audio: no second decode
        assert all("alt_by" not in w and "alt" not in w for w in words)
    finally:
        A._transcribe_sweep = A_prev


def test_alt_texts_batching_and_failure_fallback():
    A_prev = A._transcribe_sweep
    A._transcribe_sweep = lambda model, chunk, language, errors: {"segments": [{"text": f"v3@{_t0(chunk)}"}]}
    try:
        wins = [{"start": float(k * 2), "end": float(k * 2 + 1)} for k in range(10)]
        alt = _audio(21.0)
        dec = FakeAnime()
        out = _sweep_alt_texts(None, alt, wins, "ja", [], alt_decoder=dec, batch=4)
        assert [len(c) for c in dec.calls] == [4, 4, 2] and all(by == "anime" for _, by in out)
        # one failing batch: those windows show who failed, the rest still go to anime
        errors = []
        dec = FakeAnime(fail_times=1)
        out = _sweep_alt_texts(None, alt, wins, "ja", errors, alt_decoder=dec, batch=4)
        assert out[:4] == [(None, "anime")] * 4 and out[4] == ("anime@8.0", "anime") and len(errors) == 1
        assert not dec.disabled
        # three failures: large-v3 takes over the rest of the file, decoder marked disabled
        errors = []
        dec = FakeAnime(fail_all=True)
        out = _sweep_alt_texts(None, alt, wins, "ja", errors, alt_decoder=dec, batch=2)
        assert out[:6] == [(None, "anime")] * 6 and len(dec.calls) == 3 and dec.disabled
        assert out[6:] == [(f"v3@{w['start']}", "whisper") for w in wins[6:]]
        assert len(errors) == 4 and "large-v3 decodes the remaining 4" in errors[-1]
        # a disabled decoder is not retried
        out = _sweep_alt_texts(None, alt, wins[:2], "ja", [], alt_decoder=dec, batch=2)
        assert out == [("v3@0.0", "whisper"), ("v3@2.0", "whisper")] and len(dec.calls) == 3
        # cancel: nothing after the cancel point
        n = {"k": 0}

        def cancel():
            n["k"] += 1
            return n["k"] > 1
        out = _sweep_alt_texts(None, alt, wins, "ja", [], alt_decoder=FakeAnime(), batch=3, cancel_check=cancel)
        assert out[:3] != [(None, None)] * 3 and out[3:] == [(None, None)] * 7
        assert _sweep_alt_texts(None, None, wins, "ja", [], alt_decoder=FakeAnime()) == [(None, None)] * 10
    finally:
        A._transcribe_sweep = A_prev


def test_load_alt_decoder_never_raises():
    errors = []
    assert A._load_alt_decoder("whisper", errors) is None and errors == []
    assert A._load_alt_decoder("bogus", errors) is None and "unknown" in errors[0]
    prev = A.AnimeWhisper
    try:
        class Boom:
            def __init__(self):
                raise FileNotFoundError("weights missing")
        A.AnimeWhisper = Boom
        errors = []
        assert A._load_alt_decoder("anime", errors) is None
        assert errors and "anime" in errors[0] and "weights missing" in errors[0]
    finally:
        A.AnimeWhisper = prev


# ── selection through _transcribe_whisper_gpu (whisper / torch stubbed) ────────

def _run_gpu_path(sweep_alt: str, loader, alt_audio="/fake/vocals.wav", vocals="/fake/vocals.wav",
                  alt_source="auto"):
    import whisper

    saved = (whisper.load_model, whisper.load_audio, A._vad_detect, A._transcribe_chunk, A._transcribe_sweep,
             A._load_alt_decoder, A._get_audio_duration)
    tracks = {"/fake/mix.wav": _audio(30.0), "/fake/vocals.wav": _audio(30.0, offset=1_000_000)}

    class FakeModel:
        def to(self, device):
            return self
    A_prev_exists = Path.exists
    try:
        whisper.load_model = lambda size: FakeModel()
        whisper.load_audio = lambda p: tracks[str(p)]
        A._vad_detect = lambda audio, **kw: SPANS
        A._transcribe_chunk = lambda model, chunk, language, errors: {"segments": [
            {"start": 0.5, "end": 2.0, "text": "はい", "no_speech_prob": 0.1, "avg_logprob": -0.3, "compression_ratio": 1.0,
             "words": [{"word": "は", "start": 0.5, "end": 1.0, "probability": 0.9},
                       {"word": "い", "start": 1.0, "end": 2.0, "probability": 0.9}]}]}
        # windows the real _sweep_windows derives from SPANS (pad 0.3 s): 0–5.3, 7.7–20.3, 24.7–30
        canned = {0.0: [(1.0, 2.0, "きつい")], 7.7: [(0.8, 3.3, "もっと強くして")],
                  1_000_000 / SR: [(0.0, 1.0, "v3 on vocals")]}          # the vocals track's window 0
        A._transcribe_sweep = _fake_primary(canned)
        A._load_alt_decoder = loader
        A._get_audio_duration = lambda p: 30.0
        Path.exists = lambda self: str(self) in tracks or A_prev_exists(self)
        return A._transcribe_whisper_gpu(
            [Path("/fake/mix.wav")], "ja", "large-v3", None, None, None, None, None,
            diarize=False, vocals_path=vocals, sweep=True, alt_audio=alt_audio,
            sweep_alt=sweep_alt, sweep_alt_source=alt_source)
    finally:
        (whisper.load_model, whisper.load_audio, A._vad_detect, A._transcribe_chunk, A._transcribe_sweep,
         A._load_alt_decoder, A._get_audio_duration) = saved
        Path.exists = A_prev_exists


def test_gpu_path_selects_anime_records_and_frees_it():
    made = []

    def loader(name, errors):
        assert name == "anime"
        d = FakeAnime(texts={0.0: "きつい…", 7.7: "もっと…んっ"})
        made.append(d)
        return d
    res = _run_gpu_path("anime", loader)
    assert len(made) == 1 and made[0].closed                                   # loaded once, freed on the way out
    e = res[0]
    assert e["sweep_alt"]["name"] == "anime" and e["sweep_alt"]["source"] == "same"
    assert e["sweep_alt"]["revision"] == "deadbeef" and e["sweep_alt"]["fallback"] is False
    assert made[0].calls == [[0.0, 7.7]]                                       # heard the MIX (sample index 0), not the vocals
    assert [(w["start"], w["alt_by"]) for w in e["sweep_windows"]] == [(0.0, "anime"), (7.7, "anime"), (24.7, None)]
    sweep_segs = [s for s in e["segments"] if s.get("pass") == "sweep"]
    assert sweep_segs and all(s["alt_by"] == "anime" for s in sweep_segs)
    assert next(s for s in sweep_segs if s["text"] == "きつい")["alt_text"] == "きつい…"
    assert next(s for s in sweep_segs if s["text"] == "もっと強くして")["alt_text"] == "もっと…んっ"
    assert "chunk_errors" not in e


def test_gpu_path_anime_same_mode_needs_no_alt_file():
    made = []

    def loader(name, errors):
        d = FakeAnime()
        made.append(d)
        return d
    res = _run_gpu_path("anime", loader, alt_audio=None)                        # untrusted vocals: step_asr passes None
    e = res[0]
    assert e["sweep_alt"]["alt_audio"] is True and made[0].calls == [[0.0, 7.7]]
    assert all(w["alt_by"] == "anime" for w in e["sweep_windows"] if w["text"])


def test_gpu_path_whisper_mode_and_anime_fallback():
    res = _run_gpu_path("whisper", lambda name, errors: (_ for _ in ()).throw(AssertionError("must not load")))
    e = res[0]
    assert e["sweep_alt"]["name"] == "whisper" and e["sweep_alt"]["source"] == "other"
    assert e["sweep_windows"][0]["alt_text"] == "v3 on vocals" and e["sweep_windows"][0]["alt_by"] == "whisper"

    def failing_loader(name, errors):
        errors.append("sweep alt (anime) unavailable, large-v3 alt used: FileNotFoundError: weights missing")
        return None
    res = _run_gpu_path("anime", failing_loader)
    e = res[0]
    assert e["sweep_alt"]["name"] == "whisper" and e["sweep_alt"]["requested"] == "anime"
    assert any("weights missing" in x for x in e["chunk_errors"])
    assert e["sweep_alt"]["source"] == "same" and e["sweep_windows"][0]["alt_by"] == "whisper"
    assert e["sweep_windows"][0]["alt_text"] == "きつい"                      # large-v3 re-read the same (mix) audio

    def oom_loader(name, errors):
        return FakeAnime(fail_all=True)
    res = _run_gpu_path("anime", oom_loader)                   # one batch (2 text windows) fails: not yet a fallback
    e = res[0]
    assert e["sweep_alt"]["name"] == "anime" and e["sweep_alt"]["fallback"] is False
    assert any("HIP out of memory" in x for x in e["chunk_errors"])
    assert [(w["alt_text"], w["alt_by"]) for w in e["sweep_windows"] if w["text"]] == [(None, "anime")] * 2
    sweep_segs = [s for s in e["segments"] if s.get("pass") == "sweep"]
    assert sweep_segs and all(s["alt_by"] == "anime" and "alt_text" not in s for s in sweep_segs)


# ── fingerprint: effective value, not the default's repr ─────────

def test_fingerprint_extra_effective_sweep_alt():
    import run_pipeline as P
    from ai_movie import config as cfg
    ap = P.build_parser()
    a0 = P._args_extra("asr", ap.parse_args(["x.mp4"]))
    assert a0["sweep_alt"] == "whisper" and a0["sweep_alt_source"] == "other" and "sweep_alt_params" not in a0
    a1 = P._args_extra("asr", ap.parse_args(["x.mp4", "--sweep-alt", "anime"]))
    assert a1["sweep_alt"] == "anime" and a1["sweep_alt_source"] == "same"
    assert set(a1["sweep_alt_params"]) == {"ASR_ANIME_DTYPE", "ASR_ANIME_NO_REPEAT_NGRAM",
                                           "ASR_ANIME_MAX_NEW_TOKENS", "ASR_ANIME_ATTN"}
    prev = cfg.ASR_SWEEP_ALT_DECODER
    try:
        cfg.ASR_SWEEP_ALT_DECODER = "anime"                      # after adoption: no flag, same fingerprint
        assert P._args_extra("asr", ap.parse_args(["x.mp4"])) == a1
    finally:
        cfg.ASR_SWEEP_ALT_DECODER = prev
    assert "ASR_SWEEP_ALT_DECODER" not in P.STEP_CONFIG["asr"]
    for d in ("ai_movie.asr._pick_alt_audio", "ai_movie.asr._resolve_alt_source",
              "ai_movie.asr._sweep_alt_texts", "ai_movie.asr.AnimeWhisper"):
        assert d in P.STEP_CODE["asr"] and P._source_hash(d) != "missing", d


# ── the real transformers call path on a tiny random model (CPU) ──

def test_transformers_generate_accepts_our_kwargs():
    if not (ANIME_DIR / "config.json").exists() or not (ANIME_DIR / "vocab.json").exists():
        print("  (skipped: models/anime-whisper config/tokenizer not present)")
        return
    try:
        import torch
        from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor
    except Exception as exc:                                    # noqa: BLE001
        print(f"  (skipped: {exc})")
        return
    d = str(ANIME_DIR)
    proc = WhisperProcessor.from_pretrained(d, local_files_only=True)
    cfg = WhisperConfig.from_pretrained(d, local_files_only=True)
    assert cfg.decoder_layers == 2 and cfg.num_mel_bins == 128
    gen_cfg = GenerationConfig.from_pretrained(d, local_files_only=True)
    assert not AnimeWhisper.supports_word_timestamps(gen_cfg.to_dict(), cfg.decoder_layers)
    cfg.encoder_layers, cfg.d_model, cfg.encoder_attention_heads, cfg.decoder_attention_heads = 1, 64, 4, 4
    cfg.encoder_ffn_dim = cfg.decoder_ffn_dim = 128
    torch.manual_seed(0)
    aw = AnimeWhisper.__new__(AnimeWhisper)                     # bypass the loader: tiny random weights instead
    aw.proc, aw.device, aw.dtype = proc, torch.device("cpu"), torch.float32
    aw.model = WhisperForConditionalGeneration(cfg).eval()
    aw.model.generation_config = gen_cfg
    aw.model.generation_config.forced_decoder_ids = None
    aw.gen = dict(AnimeWhisper.generate_kwargs(), max_new_tokens=6)
    out = aw.transcribe_batch([np.zeros(SR * 3, dtype=np.float32), np.zeros(SR * 20, dtype=np.float32)])
    assert isinstance(out, list) and len(out) == 2 and all(isinstance(t, str) for t in out)
    aw.close()
    assert aw.model is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
