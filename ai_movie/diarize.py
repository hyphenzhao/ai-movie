"""Speaker diarization: who spoke when, and is that voice male or female.

Replaces the old ``tts.build_ecapa_gender_map`` path, which hard-coded
``num_speakers=2``, *discarded* the speaker label and kept only a gender, ran
a second Silero VAD of its own, and — worst — was unreachable in the default
configuration, so gender was really decided by a per-segment F0 median.  On a
44-second segment containing both an interviewer and an interviewee that
median is meaningless, which is why the baseline run labelled all 35 segments
"female".

How it works, and why
---------------------
Three signals, in order of trust, each covering the previous one's blind spot.
The ordering is not a guess: all three were implemented and measured against a
hand-labelled subset of the 390 s reference interview.

1. **Pitch (leads).**  A two-component GMM over log-F0 of the *separated
   vocals* lands on 123 Hz (28 % of units) and 246 Hz (72 %) — a clean
   male/female split.  Music has to be removed first; on the raw mixdown the
   modes collapse.

2. **Channel characteristics (fills the gaps).**  Pitch fails on exactly the
   segments that matter most: a short off-mic question yields *no* voiced
   frames at all (「どうだった?」、「難しいもんね。」), and pYIN octave-doubles a
   quiet male voice into the female band (「どうでしたか?」 measured 224 Hz).
   So units whose pitch is unmeasurable or near the boundary are decided by a
   logistic classifier over log-mel mean/std, seeded from the confident units.
   This works because the interviewee is on a close lavalier and the
   interviewer is off-mic across the room — a large, stable difference.

3. **ECAPA embeddings (last).**  Used *only* to split multiple speakers of the
   same gender.  They are deliberately not used for the male/female decision:
   speaker embeddings are trained to be channel-*invariant*, which throws away
   signal 2, and on this recording agglomerative clustering over them produced
   clusters with median F0 of 212 / 236 / 309 Hz — three variants of the same
   woman, with the man nowhere.

Measured on the hand-labelled subset: baseline 0/10 male segments correct
(everything was labelled female); pitch alone 12/18; pitch + channel 15/18.
The residual errors are segments with neither measurable pitch nor a decisive
channel signature.  Because no acoustic method is exact here, every run also
exports one demo WAV per speaker so a human can confirm the labelling in half
a minute, and the speaker column of ``01_speakers.csv`` is editable.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from ai_movie.config import (
    DIARIZE_AHC_THRESHOLD,
    DIARIZE_DEVICE,
    DIARIZE_GENDER_HZ,
    DIARIZE_GENDER_REL_MIN_HZ,
    DIARIZE_MAX_SPEAKERS,
    DIARIZE_PERIOD,
    DIARIZE_UNCERTAIN_DIST,
    DIARIZE_WINDOW,
    TTS_VOCALS_RMS_MIN_RATIO,
)

_ECAPA_DIR = Path(__file__).parent.parent / "models" / "speechbrain-ecapa"
_SR = 16000

_encoder = None
_enc_lock = threading.Lock()


# ── model ──────────────────────────────────────────────────────────

def ecapa_available() -> bool:
    return (_ECAPA_DIR / "embedding_model.ckpt").exists() and \
           (_ECAPA_DIR / "hyperparams.yaml").exists()


def _load_encoder(device: str = DIARIZE_DEVICE):
    """Load the local ECAPA-TDNN speaker encoder (singleton)."""
    global _encoder
    if _encoder is not None:
        return _encoder
    with _enc_lock:
        if _encoder is not None:
            return _encoder
        if not ecapa_available():
            raise RuntimeError(
                f"ECAPA speaker model not found at {_ECAPA_DIR}. "
                f"Expected embedding_model.ckpt + hyperparams.yaml."
            )
        from speechbrain.inference.speaker import EncoderClassifier
        _encoder = EncoderClassifier.from_hparams(
            source=str(_ECAPA_DIR),
            savedir=str(_ECAPA_DIR),
            run_opts={"device": device},
        )
    return _encoder


# ── audio helpers ──────────────────────────────────────────────────

def _load_mono16k(path: str | Path) -> np.ndarray:
    import librosa
    import soundfile as sf

    a, sr = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != _SR:
        a = librosa.resample(a, orig_sr=sr, target_sr=_SR)
    return a.astype(np.float32)


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else 0.0


def _pick_embed_source(orig: np.ndarray, vocals: np.ndarray | None,
                       spans: list[tuple[float, float]]) -> np.ndarray:
    """Use the separated vocals unless separation ate the voice.

    Demucs/UVR routinely suppress a male voice to near-silence; embedding
    that would collapse two speakers into one cluster.  Compare total RMS
    over the speech spans and fall back to the original when the vocals
    track lost more than ``TTS_VOCALS_RMS_MIN_RATIO`` of the energy.
    """
    if vocals is None:
        return orig
    n = min(len(orig), len(vocals))
    if n == 0:
        return orig
    o_e, v_e = 0.0, 0.0
    for s, e in spans:
        a, b = int(s * _SR), min(int(e * _SR), n)
        if b <= a:
            continue
        o_e += float(np.sum(orig[a:b] ** 2))
        v_e += float(np.sum(vocals[a:b] ** 2))
    if o_e <= 0:
        return orig
    ratio = (v_e / o_e) ** 0.5
    if ratio < TTS_VOCALS_RMS_MIN_RATIO:
        print(f"[diarize] vocals RMS is {ratio:.2f}x the original — "
              f"separation suppressed speech, embedding the original instead",
              file=sys.stderr)
        return orig
    return vocals[:n]


# ── VAD ────────────────────────────────────────────────────────────

def _speech_spans(audio: np.ndarray,
                  spans: list[dict] | None) -> list[tuple[float, float]]:
    """Normalise caller-supplied VAD spans, or run Silero if none given."""
    if spans:
        return [(float(s["start"]), float(s["end"])) for s in spans]

    from ai_movie.asr import _vad_detect
    from ai_movie.config import (
        ASR_VAD_MIN_SILENCE_DURATION_MS,
        ASR_VAD_MIN_SPEECH_DURATION_MS,
        ASR_VAD_SPEECH_PAD_MS,
        ASR_VAD_THRESHOLD,
    )
    import torch

    detected = _vad_detect(
        torch.from_numpy(audio),
        threshold=ASR_VAD_THRESHOLD,
        min_silence_duration_ms=ASR_VAD_MIN_SILENCE_DURATION_MS,
        min_speech_duration_ms=ASR_VAD_MIN_SPEECH_DURATION_MS,
        speech_pad_ms=ASR_VAD_SPEECH_PAD_MS,
    )
    if detected:
        return [(float(s["start"]), float(s["end"])) for s in detected]
    return [(0.0, len(audio) / _SR)]


# ── embeddings ─────────────────────────────────────────────────────

def embed_windows(
    audio: np.ndarray,
    spans: list[tuple[float, float]],
    *,
    window: float = DIARIZE_WINDOW,
    period: float = DIARIZE_PERIOD,
    device: str = DIARIZE_DEVICE,
    batch: int = 32,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window ECAPA embeddings.

    Returns ``(embeddings [N,192] L2-normalised, times [N,2])``.
    """
    import torch

    enc = _load_encoder(device)
    win_n = int(window * _SR)
    hop_n = int(period * _SR)

    clips: list[np.ndarray] = []
    times: list[tuple[float, float]] = []
    for s, e in spans:
        a, b = int(s * _SR), min(int(e * _SR), len(audio))
        if b - a < int(0.6 * _SR):          # too short to identify a voice
            continue
        pos = a
        while pos + win_n <= b:
            clips.append(audio[pos:pos + win_n])
            times.append((pos / _SR, (pos + win_n) / _SR))
            pos += hop_n
        # Tail: keep a final (shorter) window so short turns aren't lost.
        if pos < b and (b - a) >= int(0.6 * _SR):
            tail = audio[max(a, b - win_n):b]
            if len(tail) >= int(0.6 * _SR):
                clips.append(tail)
                times.append((max(a, b - win_n) / _SR, b / _SR))

    if not clips:
        return np.zeros((0, 192), np.float32), np.zeros((0, 2), np.float32)

    embs: list[np.ndarray] = []
    for i in range(0, len(clips), batch):
        chunk = clips[i:i + batch]
        maxlen = max(len(c) for c in chunk)
        padded = np.zeros((len(chunk), maxlen), np.float32)
        lens = np.zeros(len(chunk), np.float32)
        for j, c in enumerate(chunk):
            padded[j, :len(c)] = c
            lens[j] = len(c) / maxlen
        with torch.no_grad():
            out = enc.encode_batch(torch.from_numpy(padded),
                                   torch.from_numpy(lens))
        embs.append(out.squeeze(1).cpu().numpy())
        if progress_cb and (i // batch) % 8 == 0:
            progress_cb(f"说话人特征提取 {min(i + batch, len(clips))}/{len(clips)}")

    emb = np.concatenate(embs).astype(np.float32)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    return emb, np.asarray(times, np.float32)


def embed_units(
    audio: np.ndarray,
    units: list[tuple[float, float]],
    *,
    device: str = DIARIZE_DEVICE,
    min_dur: float = 0.6,
    max_dur: float = 8.0,
    batch: int = 16,
) -> tuple[np.ndarray, list[int]]:
    """One embedding per unit, with the indices of the units that produced them.

    Unlike :func:`embed_windows` this keeps a strict correspondence with the
    caller's unit list (short units are reported as missing rather than
    silently dropped), which is what seeded assignment needs.
    """
    import torch

    enc = _load_encoder(device)
    keep: list[int] = []
    clips: list[np.ndarray] = []
    for i, (s, e) in enumerate(units):
        a, b = int(s * _SR), min(int(e * _SR), len(audio))
        if b - a < int(min_dur * _SR):
            continue
        if (b - a) > int(max_dur * _SR):        # centre-crop very long units
            mid = (a + b) // 2
            half = int(max_dur * _SR) // 2
            a, b = mid - half, mid + half
        keep.append(i)
        clips.append(audio[a:b])

    if not clips:
        return np.zeros((0, 192), np.float32), []

    embs: list[np.ndarray] = []
    for i in range(0, len(clips), batch):
        chunk = clips[i:i + batch]
        maxlen = max(len(c) for c in chunk)
        padded = np.zeros((len(chunk), maxlen), np.float32)
        lens = np.zeros(len(chunk), np.float32)
        for j, c in enumerate(chunk):
            padded[j, :len(c)] = c
            lens[j] = len(c) / maxlen
        with torch.no_grad():
            out = enc.encode_batch(torch.from_numpy(padded),
                                   torch.from_numpy(lens))
        embs.append(out.squeeze(1).cpu().numpy())

    emb = np.concatenate(embs).astype(np.float32)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    return emb, keep


# ── clustering ─────────────────────────────────────────────────────

def estimate_num_speakers(emb: np.ndarray,
                          max_speakers: int = DIARIZE_MAX_SPEAKERS) -> int:
    """Eigengap heuristic on the normalised Laplacian of cosine affinity."""
    n = len(emb)
    if n < 4:
        return 1
    sim = emb @ emb.T
    aff = np.clip((sim + 1.0) / 2.0, 0.0, 1.0)
    np.fill_diagonal(aff, 0.0)

    deg = aff.sum(axis=1)
    deg[deg <= 0] = 1e-9
    dinv = 1.0 / np.sqrt(deg)
    lap = np.eye(n) - (aff * dinv[:, None]) * dinv[None, :]

    try:
        vals = np.linalg.eigvalsh(lap)
    except np.linalg.LinAlgError:
        return 1
    vals = np.sort(vals)[:min(max_speakers + 1, n)]
    gaps = np.diff(vals)
    if len(gaps) == 0:
        return 1
    k = int(np.argmax(gaps)) + 1
    return max(1, min(k, max_speakers))


def cluster(emb: np.ndarray, num_speakers: int | None,
            threshold: float = DIARIZE_AHC_THRESHOLD) -> np.ndarray:
    """Agglomerative clustering on cosine distance."""
    from sklearn.cluster import AgglomerativeClustering

    if len(emb) == 0:
        return np.zeros(0, np.int32)
    if num_speakers == 1 or len(emb) < 3:
        return np.zeros(len(emb), np.int32)

    if num_speakers:
        model = AgglomerativeClustering(
            n_clusters=int(num_speakers), metric="cosine", linkage="average")
    else:
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold,
            metric="cosine", linkage="average")
    return model.fit_predict(emb).astype(np.int32)


# ── turns ──────────────────────────────────────────────────────────

def _labels_to_turns(labels: np.ndarray, times: np.ndarray,
                     *, min_turn: float = 0.4) -> list[dict]:
    """Collapse per-window labels into contiguous speaker turns."""
    if len(labels) == 0:
        return []
    turns: list[dict] = []
    cur = {"start": float(times[0][0]), "end": float(times[0][1]),
           "speaker": int(labels[0])}
    for i in range(1, len(labels)):
        s, e = float(times[i][0]), float(times[i][1])
        if int(labels[i]) == cur["speaker"] and s <= cur["end"] + 0.35:
            cur["end"] = max(cur["end"], e)
        else:
            turns.append(cur)
            cur = {"start": s, "end": e, "speaker": int(labels[i])}
    turns.append(cur)

    # Drop micro-turns produced by a single noisy window, absorbing them
    # into whichever neighbour they touch.
    cleaned: list[dict] = []
    for t in turns:
        if cleaned and (t["end"] - t["start"]) < min_turn and \
                cleaned[-1]["speaker"] != t["speaker"]:
            cleaned[-1]["end"] = max(cleaned[-1]["end"], t["end"])
            continue
        if cleaned and cleaned[-1]["speaker"] == t["speaker"]:
            cleaned[-1]["end"] = max(cleaned[-1]["end"], t["end"])
            continue
        cleaned.append(t)
    return cleaned


# ── pitch track ────────────────────────────────────────────────────

_F0_HOP = 256
_F0_FRAME = 2048
_F0_FMIN = 60.0
_F0_FMAX = 400.0
_F0_VOICED_PROB = 0.25

_f0_cache: dict[str, tuple] = {}


def pitch_track(audio: np.ndarray, cache_key: str | None = None) -> tuple:
    """Whole-file pYIN pitch track.

    Returns ``(f0, ok, fps)`` where *ok* is a boolean mask of frames with a
    trustworthy pitch estimate.  One pass over 390 s takes ~11 s, so this is
    computed once and reused for every unit.
    """
    import librosa

    if cache_key and cache_key in _f0_cache:
        return _f0_cache[cache_key]

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio, fmin=_F0_FMIN, fmax=_F0_FMAX, sr=_SR,
        frame_length=_F0_FRAME, hop_length=_F0_HOP,
    )
    ok = voiced_flag & (voiced_prob > _F0_VOICED_PROB) & ~np.isnan(f0)
    out = (f0, ok, _SR / _F0_HOP)
    if cache_key:
        _f0_cache[cache_key] = out
    return out


def unit_f0(f0: np.ndarray, ok: np.ndarray, fps: float,
            start: float, end: float, *, min_frames: int = 6) -> float | None:
    """Median F0 over ``[start, end)``, or ``None`` if too few voiced frames."""
    a, b = int(start * fps), min(int(end * fps), len(f0))
    if b <= a:
        return None
    m = ok[a:b]
    if int(m.sum()) < min_frames:
        return None
    return float(np.median(f0[a:b][m]))


def split_by_pitch(values: list[float], weights: list[float] | None = None
                   ) -> tuple[float | None, dict]:
    """Decide whether *values* (Hz) contain both a male and a female mode.

    Returns ``(cut_hz, info)``.  ``cut_hz`` is ``None`` when the recording
    looks single-gender, in which case the caller labels everything by the
    absolute threshold instead.
    """
    from sklearn.mixture import GaussianMixture

    vals = np.asarray([v for v in values if v and v > 0], dtype=np.float64)
    info: dict = {"n": int(len(vals))}
    if len(vals) < 12:
        info["reason"] = "too few voiced units"
        return None, info

    lf = np.log(vals).reshape(-1, 1)
    try:
        gmm = GaussianMixture(2, n_init=5, random_state=0).fit(lf)
    except Exception:                                   # noqa: BLE001
        info["reason"] = "gmm failed"
        return None, info

    order = np.argsort(gmm.means_.ravel())
    lo_hz = float(np.exp(gmm.means_.ravel()[order[0]]))
    hi_hz = float(np.exp(gmm.means_.ravel()[order[1]]))
    lo_w = float(gmm.weights_[order[0]])
    hi_w = float(gmm.weights_[order[1]])
    info.update({"low_hz": round(lo_hz, 1), "high_hz": round(hi_hz, 1),
                 "low_w": round(lo_w, 3), "high_w": round(hi_w, 3)})

    # Two genders only if the modes are far apart, both have real support,
    # and they straddle the male/female band.  Otherwise this is one voice
    # whose pitch simply varies (or pYIN octave errors).
    ratio = hi_hz / max(lo_hz, 1e-6)
    if ratio < 1.35:
        info["reason"] = f"modes too close (ratio {ratio:.2f})"
        return None, info
    if min(lo_w, hi_w) < 0.05:
        info["reason"] = f"minor mode too small ({min(lo_w, hi_w):.3f})"
        return None, info
    if lo_hz > 185.0 or hi_hz < DIARIZE_GENDER_HZ:
        info["reason"] = f"modes not straddling ({lo_hz:.0f}/{hi_hz:.0f} Hz)"
        return None, info

    # Cut where the two log-normal components are equally likely.
    grid = np.linspace(np.log(lo_hz), np.log(hi_hz), 400).reshape(-1, 1)
    post = gmm.predict_proba(grid)[:, order[0]]
    idx = int(np.argmin(np.abs(post - 0.5)))
    cut = float(np.exp(grid[idx, 0]))
    info["cut_hz"] = round(cut, 1)
    return cut, info


# ── gender ─────────────────────────────────────────────────────────

def _speaker_f0(audio: np.ndarray, turns: list[dict], speaker: int,
                *, max_seconds: float = 30.0) -> float | None:
    """Median F0 over up to *max_seconds* of this speaker's loudest audio."""
    import librosa

    spans = [t for t in turns if t["speaker"] == speaker]
    if not spans:
        return None
    scored = []
    for t in spans:
        a, b = int(t["start"] * _SR), min(int(t["end"] * _SR), len(audio))
        if b - a < int(0.8 * _SR):
            continue
        scored.append((_rms(audio[a:b]), a, b))
    if not scored:
        return None
    scored.sort(reverse=True)

    chunks, total = [], 0.0
    for _, a, b in scored:
        chunks.append(audio[a:b])
        total += (b - a) / _SR
        if total >= max_seconds:
            break
    clip = np.concatenate(chunks)

    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            clip, fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")), sr=_SR)
    except Exception:                                   # noqa: BLE001
        return None
    valid = f0[(voiced_prob > 0.7) & voiced_flag]
    valid = valid[~np.isnan(valid)]
    if len(valid) < 20:
        return None
    return float(np.median(valid))


def _assign_genders(f0_by_spk: dict[int, float | None]) -> dict[int, str]:
    """Absolute threshold, with a relative fallback when it fails.

    The absolute 165 Hz cut is unreliable on a single recording (mic, codec,
    and speaking style all shift F0).  When every measurable speaker lands on
    the same side of it but their medians are clearly separated, split them
    relatively instead — this is what rescues the "everyone is female" case.
    """
    measured = {k: v for k, v in f0_by_spk.items() if v is not None}
    genders: dict[int, str] = {}

    if not measured:
        return {k: "female" for k in f0_by_spk}

    for k, v in measured.items():
        genders[k] = "female" if v >= DIARIZE_GENDER_HZ else "male"

    if len(measured) >= 2 and len(set(genders.values())) == 1:
        lo_k = min(measured, key=lambda k: measured[k])
        hi_k = max(measured, key=lambda k: measured[k])
        spread = measured[hi_k] - measured[lo_k]
        if spread >= DIARIZE_GENDER_REL_MIN_HZ:
            mid = (measured[hi_k] + measured[lo_k]) / 2.0
            print(f"[diarize] all speakers fell on one side of "
                  f"{DIARIZE_GENDER_HZ:.0f} Hz (spread {spread:.0f} Hz) — "
                  f"splitting relatively at {mid:.0f} Hz", file=sys.stderr)
            for k, v in measured.items():
                genders[k] = "female" if v >= mid else "male"

    # Unmeasurable speakers inherit the majority label.
    if len(genders) < len(f0_by_spk):
        fallback = max(set(genders.values()), key=list(genders.values()).count)
        for k in f0_by_spk:
            genders.setdefault(k, fallback)
    return genders


# ── public API ─────────────────────────────────────────────────────

def _units_from(spans: list[tuple[float, float]],
                segments: list[dict] | None,
                *, window: float = 1.5, hop: float = 0.75) -> list[tuple[float, float]]:
    """Units to label: ASR segments when available, else sliding windows.

    ASR segments are much better units than fixed windows — they already end
    at pauses and punctuation, so a short interviewer question tends to be
    its own unit instead of being averaged into the reply that follows it.
    """
    if segments:
        units = [(float(s["start"]), float(s["end"])) for s in segments
                 if float(s["end"]) > float(s["start"])]
        if units:
            return units
    units = []
    for s, e in spans:
        pos = s
        while pos < e:
            units.append((pos, min(pos + window, e)))
            pos += hop
    return units


def _smooth_labels(units: list[tuple[float, float]], labels: list[str],
                   *, min_run: float = 0.8) -> list[str]:
    """Remove single-unit label flips shorter than *min_run* seconds."""
    out = list(labels)
    i = 0
    n = len(out)
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        dur = units[j - 1][1] - units[i][0]
        isolated = (i > 0 and j < n and out[i - 1] == out[j])
        if dur < min_run and isolated:
            for k in range(i, j):
                out[k] = out[i - 1]
        i = j
    return out


def diarize_file(
    audio_path: str | Path,
    *,
    vocals_path: str | Path | None = None,
    num_speakers: int | None = None,
    speech_spans: list[dict] | None = None,
    segments: list[dict] | None = None,
    device: str = DIARIZE_DEVICE,
    progress_cb: Callable[[str], None] | None = None,
    overlap_regions: list | None = None,
) -> dict:
    """Diarize one audio file.

    Gender comes from a two-component GMM over log-F0 of the units (see the
    module docstring for why pitch leads here).  ECAPA embeddings are then
    used only to split multiple speakers *within* one gender.

    Parameters
    ----------
    segments:
        ASR segments to use as labelling units.  Strongly preferred over
        sliding windows — they already end at pauses.
    vocals_path:
        Separated vocals.  Pitch is measured here (music wrecks pYIN);
        it is rejected automatically if separation gutted the speech.

    Returns::

        {"turns":    [{"start", "end", "speaker"}],
         "speakers": {"S0": {"gender", "f0_median", "total_speech", "n_turns"}},
         "num_speakers": int, "backend": str, "pitch": {...}}
    """
    def _say(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    audio = _load_mono16k(audio_path)
    spans = _speech_spans(audio, speech_spans)

    vocals = None
    if vocals_path and Path(vocals_path).exists():
        try:
            vocals = _load_mono16k(vocals_path)
        except Exception:                               # noqa: BLE001
            vocals = None
    pitch_audio = _pick_embed_source(audio, vocals, spans)

    units = _units_from(spans, segments)
    if not units:
        return {"turns": [], "speakers": {}, "num_speakers": 0,
                "backend": "f0"}

    _say("音高分析中…")
    f0, ok, fps = pitch_track(pitch_audio, cache_key=str(audio_path))
    u_f0 = [unit_f0(f0, ok, fps, s, e) for s, e in units]

    voiced = [(v, e - s) for v, (s, e) in zip(u_f0, units) if v]
    cut, pinfo = split_by_pitch([v for v, _ in voiced], [w for _, w in voiced])

    chan_info: dict = {}
    if cut is None:
        # Single-gender recording (or unusable pitch) — fall back to the
        # absolute threshold so we still produce a sane label.
        med = float(np.median([v for v, _ in voiced])) if voiced else 0.0
        gender_all = "female" if med >= DIARIZE_GENDER_HZ else "male"
        genders = [gender_all] * len(units)
        _say(f"单一性别（中位 F0 {med:.0f} Hz → {gender_all}）")
    else:
        genders = []
        last = None
        for v in u_f0:
            if v is None:
                genders.append(last or "female")
            else:
                g = "female" if v >= cut else "male"
                genders.append(g)
                last = g
        _say(f"音高分离：男 {pinfo['low_hz']:.0f} Hz / 女 {pinfo['high_hz']:.0f} Hz"
             f"（切分 {cut:.0f} Hz）")

        # Pitch alone mislabels units it could not measure — a short, off-mic
        # question yields no voiced frames at all, and pYIN octave-doubles a
        # quiet male voice into the female band.  Let a classifier seeded from
        # the confident units decide those, using channel characteristics.
        confs = []
        for (s_, e_), val in zip(units, u_f0):
            fa, fb = int(s_ * fps), min(int(e_ * fps), len(ok))
            confs.append(_acoustic_conf(val, cut,
                                        int(ok[fa:fb].sum()) if fb > fa else 0))
        # Units where two people speak (OSD) measure a mixture: they are
        # neither seeds for the channel classifier nor eligible for its
        # override — their pitch label stands, flagged for review downstream.
        exclude: set[int] = set()
        if overlap_regions:
            from ai_movie.config import OSD_SEED_EXCLUDE
            from ai_movie.osd import unit_overlap
            u_ov = unit_overlap(overlap_regions, units)
            exclude = {i for i, r in enumerate(u_ov) if r > OSD_SEED_EXCLUDE}
            for i in exclude:
                confs[i] = 0.0
            if exclude:
                _say(f"重叠语音：{len(exclude)} 个单元不参与声道分类")
        genders, chan_info = _refine_with_channel(pitch_audio, units,
                                                  genders, confs,
                                                  exclude=exclude)
        if chan_info.get("overridden"):
            _say(f"声道特征修正 {chan_info['overridden']} 段"
                 f"（交叉验证 {chan_info.get('cv_accuracy')}）")
        elif chan_info.get("reason"):
            _say(f"声道特征修正跳过：{chan_info['reason']}")

        genders = _smooth_labels(units, genders)

    # ── split same-gender speakers with ECAPA (optional refinement) ──
    labels = list(genders)
    sub_info: dict = {}
    if num_speakers is None or num_speakers > len(set(genders)):
        try:
            labels, sub_info = _split_same_gender(
                pitch_audio, units, genders, num_speakers=num_speakers,
                device=device, progress_cb=progress_cb)
        except Exception as exc:                        # noqa: BLE001
            print(f"[diarize] same-gender split skipped: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)

    # ── name speakers by total speech time (S0 = most talkative) ────
    totals: dict[str, float] = {}
    for lab, (s, e) in zip(labels, units):
        totals[lab] = totals.get(lab, 0.0) + (e - s)
    order = sorted(totals, key=lambda k: -totals[k])
    naming = {lab: f"S{i}" for i, lab in enumerate(order)}

    turns: list[dict] = []
    for lab, (s, e) in zip(labels, units):
        name = naming[lab]
        if turns and turns[-1]["speaker"] == name and s <= turns[-1]["end"] + 0.35:
            turns[-1]["end"] = max(turns[-1]["end"], e)
        else:
            turns.append({"start": round(s, 2), "end": round(e, 2),
                          "speaker": name})

    speakers: dict[str, dict] = {}
    for lab, name in naming.items():
        vals = [v for v, l in zip(u_f0, labels) if l == lab and v]
        speakers[name] = {
            "gender": lab.split("/")[0],
            "f0_median": round(float(np.median(vals)), 1) if vals else None,
            "total_speech": round(totals[lab], 2),
            "n_turns": sum(1 for t in turns if t["speaker"] == name),
        }

    _say("说话人分离完成：" + "、".join(
        f"{k2}={v2['gender']}({v2['total_speech']:.0f}s)"
        for k2, v2 in speakers.items()))

    # Per-unit detail: needed by the dialogue refinement pass, which has to
    # know how much acoustic evidence backs each individual label.
    unit_rows = []
    for (s, e), lab, val in zip(units, labels, u_f0):
        fa, fb = int(s * fps), min(int(e * fps), len(ok))
        unit_rows.append({
            "start": round(s, 2), "end": round(e, 2),
            "speaker": naming[lab],
            "gender": lab.split("/")[0],
            "f0": round(val, 1) if val else None,
            "voiced": int(ok[fa:fb].sum()) if fb > fa else 0,
            "conf": round(_acoustic_conf(val, cut,
                                         int(ok[fa:fb].sum()) if fb > fa else 0), 2),
        })

    if overlap_regions:
        from ai_movie.osd import overlap_ratio as _ovr
        for row in unit_rows:
            row["overlap"] = round(_ovr(overlap_regions, row["start"], row["end"]), 3)

    return {
        "turns": turns,
        "speakers": speakers,
        "num_speakers": len(speakers),
        "backend": "f0+channel" + ("+ecapa" if sub_info else "")
                   + ("+osd" if overlap_regions else ""),
        "pitch": pinfo,
        "cut_hz": cut,
        "subsplit": sub_info,
        "channel": chan_info,
        "units": unit_rows,
        "overlap_regions": list(overlap_regions or []),
    }


def channel_features(audio: np.ndarray,
                     units: list[tuple[float, float]]) -> tuple[np.ndarray, list[int]]:
    """Per-unit log-mel mean+std — a *channel* fingerprint, not a voice one.

    ECAPA embeddings are trained to be invariant to recording conditions,
    which throws away the single most reliable cue in an interview: the
    interviewee is on a close lavalier while the interviewer is off-mic
    across the room.  Raw log-mel statistics keep that difference, and on
    the reference recording a classifier seeded from confident-pitch units
    reaches 89 % cross-validated accuracy on those seeds.
    """
    import librosa

    X: list[np.ndarray] = []
    keep: list[int] = []
    for i, (s, e) in enumerate(units):
        a, b = int(s * _SR), min(int(e * _SR), len(audio))
        if b - a < int(0.4 * _SR):
            continue
        mel = librosa.feature.melspectrogram(
            y=audio[a:b], sr=_SR, n_fft=1024, hop_length=256, n_mels=40)
        db = librosa.power_to_db(mel)
        X.append(np.concatenate([db.mean(1), db.std(1)]))
        keep.append(i)
    if not X:
        return np.zeros((0, 80), np.float32), []
    return np.asarray(X, np.float32), keep


def _refine_with_channel(
    audio: np.ndarray,
    units: list[tuple[float, float]],
    genders: list[str],
    confs: list[float],
    *,
    min_conf: float = 0.45,
    min_seeds: int = 6,
    exclude: set[int] | None = None,
) -> tuple[list[str], dict]:
    """Re-label low-pitch-confidence units with a seeded channel classifier.

    ``exclude``: unit indices (overlapped speech) that are neither seeds nor
    override candidates.

    Units whose pitch is unambiguous become training seeds; the classifier
    then decides the ones where pYIN found too few voiced frames (a short,
    off-mic question) or landed near the male/female boundary (an octave
    error).  Confident pitch measurements are never overridden.
    """
    info: dict = {}
    seeds_m = [i for i, (g, c) in enumerate(zip(genders, confs))
               if g == "male" and c >= 0.65]
    seeds_f = [i for i, (g, c) in enumerate(zip(genders, confs))
               if g == "female" and c >= 0.65]
    info["seeds"] = {"male": len(seeds_m), "female": len(seeds_f)}
    if len(seeds_m) < min_seeds or len(seeds_f) < min_seeds:
        info["reason"] = "not enough confident seeds on both sides"
        return genders, info

    X, keep = channel_features(audio, units)
    if len(X) < 2 * min_seeds:
        info["reason"] = "too few usable units"
        return genders, info
    pos = {u: k for k, u in enumerate(keep)}

    tr_m = [pos[i] for i in seeds_m if i in pos]
    tr_f = [pos[i] for i in seeds_f if i in pos]
    if len(tr_m) < min_seeds or len(tr_f) < min_seeds:
        info["reason"] = "seeds dropped by feature extraction"
        return genders, info

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xs = np.vstack([X[tr_m], X[tr_f]])
    ys = np.array([0] * len(tr_m) + [1] * len(tr_f))
    # class_weight="balanced" is essential here: the on-mic speaker supplies
    # most of the confident seeds (41 vs 6 on the reference recording), and an
    # unweighted fit simply learns to answer "female".
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=0.3, max_iter=2000,
                                           class_weight="balanced"))
    try:
        cv = float(cross_val_score(clf, Xs, ys,
                                   cv=min(5, len(tr_m), len(tr_f))).mean())
    except Exception:                                   # noqa: BLE001
        cv = 0.0
    info["cv_accuracy"] = round(cv, 3)
    if cv < 0.75:
        info["reason"] = f"classifier too weak (cv {cv:.2f})"
        return genders, info

    clf.fit(Xs, ys)
    p_female = clf.predict_proba(X)[:, 1]

    out = list(genders)
    changed = 0
    exclude = exclude or set()
    for i in range(len(units)):
        if confs[i] >= min_conf or i not in pos or i in exclude:
            continue
        pf = float(p_female[pos[i]])
        if pf < 0.35 or pf > 0.65:              # only act when it is decisive
            g = "female" if pf >= 0.5 else "male"
            if g != out[i]:
                changed += 1
            out[i] = g
    info["overridden"] = changed
    return out, info


def _acoustic_conf(f0_val: float | None, cut: float | None,
                   voiced: int) -> float:
    """How much to trust this unit's pitch-based gender label (0–1).

    Two things make a label untrustworthy: too few voiced frames to measure
    anything, and a median sitting right on the male/female boundary.  Both
    are common for a short, off-mic question — exactly the segments the
    dialogue pass needs to override.
    """
    if f0_val is None or not cut or voiced <= 0:
        return 0.0
    # Frame support saturates at 15 voiced frames (~0.24 s of voicing at the
    # 62.5 fps pitch rate) — enough for a stable median.  A higher bar would
    # mark every off-mic speaker as unreliable: on the reference recording the
    # interviewer's median voiced-frame count per segment is 6.
    support = min(1.0, voiced / 15.0)
    # Distance from the boundary in octaves.  The scale is deliberately wide
    # (0.6 octave for full confidence) because pYIN's octave-doubling errors
    # land ~0.3 octave above the cut — treating those as confident would lock
    # in exactly the mistakes the channel classifier exists to fix.
    dist = abs(np.log2(max(f0_val, 1e-6) / cut))
    margin = min(1.0, dist / 0.6)
    return float(support * margin)


def _split_same_gender(
    audio: np.ndarray,
    units: list[tuple[float, float]],
    genders: list[str],
    *,
    num_speakers: int | None,
    device: str,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[list[str], dict]:
    """Split each gender group into multiple speakers when embeddings justify it.

    Conservative on purpose: a wrong split creates a phantom speaker with its
    own cloned voice, which is worse than merging two same-gender people.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    out = list(genders)
    info: dict = {}

    for g in sorted(set(genders)):
        idx = [i for i, x in enumerate(genders) if x == g]
        # Need enough long units to trust an embedding split.
        usable = [i for i in idx if units[i][1] - units[i][0] >= 2.0]
        if len(usable) < 12:
            continue

        clips_spans = [(units[i][0], units[i][1]) for i in usable]
        emb, _times = embed_windows(
            audio, clips_spans, window=2.0, period=2.0, device=device)
        if len(emb) != len(usable) or len(emb) < 12:
            continue

        best_k, best_sil, best_lab = 1, -1.0, None
        for k in range(2, min(4, len(emb) // 6) + 1):
            lab = AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage="average").fit_predict(emb)
            sizes = [int((lab == u).sum()) for u in set(lab.tolist())]
            if min(sizes) < max(3, int(0.15 * len(emb))):
                continue
            sil = float(silhouette_score(emb, lab, metric="cosine"))
            if sil > best_sil:
                best_k, best_sil, best_lab = k, sil, lab

        info[g] = {"candidates": len(emb), "best_k": best_k,
                   "silhouette": round(best_sil, 3)}
        if best_lab is None or best_sil < 0.30:
            continue

        centroids = {}
        for u in sorted(set(best_lab.tolist())):
            c = emb[best_lab == u].mean(axis=0)
            centroids[u] = c / max(np.linalg.norm(c), 1e-9)

        assign = {i: int(l) for i, l in zip(usable, best_lab)}
        last = None
        for i in idx:
            if i in assign:
                last = assign[i]
            out[i] = f"{g}/{last if last is not None else 0}"
        info[g]["applied"] = True

    return out, info


def refine_speakers_with_dialogue(
    segments: list[dict],
    diar: dict,
    *,
    model: str | None = None,
    base_url: str | None = None,
    scene_hint: str | None = None,
    min_conf: float = 0.45,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Fix speaker attribution using the dialogue structure of the transcript.

    Pitch alone gets the *identity* of each speaker right (who is the man,
    who is the woman) but mis-assigns individual lines when a question is
    short, off-mic, or unvoiced — measured on the reference interview,
    「どうだった?」 and 「難しいもんね。」 carry no usable voiced frames at all,
    and pYIN octave-doubles two more of the interviewer's questions into the
    female band.

    A transcript, however, makes those lines obvious: they are questions and
    back-channels, and the interviewee never asks them.  This pass asks the
    local LLM to label each line by role, then *only* overrides labels whose
    acoustic confidence is below ``min_conf``.  Confident pitch evidence
    always wins, so the LLM can never rewrite a clearly-voiced line.

    Returns an updated diarization dict (turns rebuilt); the original is not
    modified.  Returns *diar* unchanged if the recording is single-gender or
    the LLM is unavailable.

    NOT enabled by default.  On the gfx1151/ROCm box this was developed on,
    the only local model strong enough for the task (gpt-oss-120b) loads onto
    the GPU but its llama-server never becomes available, and the smaller
    models degenerate rather than follow the output format.  The channel
    classifier in :func:`_refine_with_channel` covers the same failure cases
    acoustically and always terminates.
    """
    from ai_movie.config import OLLAMA_BASE_URL, OLLAMA_GPTOSS_MODEL
    from ai_movie.translator import _call_ollama_chat, _parse_json_array

    def _say(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    speakers = diar.get("speakers") or {}
    units = diar.get("units") or []
    if len(speakers) < 2 or not units or not segments:
        return diar

    genders = {v.get("gender") for v in speakers.values()}
    if not {"male", "female"} <= genders:
        return diar

    model = model or OLLAMA_GPTOSS_MODEL
    base_url = base_url or OLLAMA_BASE_URL

    # Map each unit to the ASR segment it came from (they are the same list
    # when diarization ran with segments=, but be defensive).
    by_time = {round(float(u["start"]), 2): u for u in units}
    rows = []
    for i, seg in enumerate(segments):
        u = by_time.get(round(float(seg.get("start", 0.0)), 2))
        rows.append({
            "i": i,
            "text": (seg.get("text") or "").strip(),
            "unit": u,
            "conf": float(u["conf"]) if u else 0.0,
            "gender": (u or {}).get("gender") or "female",
        })

    fem_name = next(k for k, v in speakers.items() if v.get("gender") == "female")
    male_name = next(k for k, v in speakers.items() if v.get("gender") == "male")

    scene = scene_hint or (
        "一位男性主持人（画外音，负责提问、引导、附和）与一位女性受访者"
        "（负责回答、自述经历）的访谈。"
    )
    system = (
        "你在为一段日语访谈标注说话人。场景：" + scene + "\n"
        "判断依据：疑问句、引导句、附和（「〜だよね」「〜もんね」「なるほど」「そっか」）"
        "通常来自主持人 M；第一人称叙述、回答、感想通常来自受访者 F。\n"
        "注意：有些句子我已经用声音特征确定了说话人，我会在行首标出 [M] 或 [F]，"
        "这些行的标注不要改变，请把它们当作可靠参照来推断其余行。\n"
        "严格只输出一个 JSON 字符串数组，长度等于台词条数，每项是 \"M\" 或 \"F\"，"
        "不要输出任何解释。"
    )

    numbered = []
    for r in rows:
        anchor = ""
        if r["conf"] >= 0.75:
            anchor = "[M] " if r["gender"] == "male" else "[F] "
        numbered.append(f"{r['i']}\t{anchor}{r['text']}")
    user = (f"共 {len(rows)} 句，按时间顺序：\n" + "\n".join(numbered))

    _say("对话结构分析中（LLM）…")
    try:
        raw = _call_ollama_chat(
            model,
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            base_url, timeout=1800,
        )
    except Exception as exc:                            # noqa: BLE001
        print(f"[diarize] dialogue refinement unavailable: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return diar

    labels, err = _parse_json_array(raw, len(rows))
    if labels is None:
        print(f"[diarize] dialogue refinement rejected: {err}", file=sys.stderr)
        return diar

    llm_gender = ["male" if str(x).strip().upper().startswith("M") else "female"
                  for x in labels]

    changed = 0
    agree = 0
    final: list[str] = []
    for r, g_llm in zip(rows, llm_gender):
        g_ac = r["gender"]
        if g_ac == g_llm:
            agree += 1
        if r["conf"] >= min_conf:
            final.append(g_ac)                # trust the pitch measurement
        else:
            if g_llm != g_ac:
                changed += 1
            final.append(g_llm)

    out = dict(diar)
    out["turns"] = _rebuild_turns(segments, final, fem_name, male_name)
    out["units"] = [
        {**(r["unit"] or {"start": float(segments[r["i"]]["start"]),
                          "end": float(segments[r["i"]]["end"])}),
         "gender": g, "speaker": male_name if g == "male" else fem_name}
        for r, g in zip(rows, final)
    ]
    out["backend"] = str(diar.get("backend", "f0")) + "+dialogue"
    out["dialogue"] = {
        "model": model,
        "agreement": round(agree / max(1, len(rows)), 3),
        "overridden": changed,
        "anchors": sum(1 for r in rows if r["conf"] >= 0.75),
    }

    # Recompute per-speaker totals from the corrected turns.
    for name, info in out["speakers"].items():
        t = [x for x in out["turns"] if x["speaker"] == name]
        info["total_speech"] = round(sum(x["end"] - x["start"] for x in t), 2)
        info["n_turns"] = len(t)

    _say(f"对话结构修正：一致率 {out['dialogue']['agreement']:.0%}，"
         f"修正 {changed} 句")
    return out


def _rebuild_turns(segments: list[dict], genders: list[str],
                   fem_name: str, male_name: str) -> list[dict]:
    """Turn a per-segment gender list back into contiguous speaker turns."""
    turns: list[dict] = []
    for seg, g in zip(segments, genders):
        name = male_name if g == "male" else fem_name
        s, e = float(seg["start"]), float(seg["end"])
        if turns and turns[-1]["speaker"] == name and s <= turns[-1]["end"] + 0.35:
            turns[-1]["end"] = max(turns[-1]["end"], e)
        else:
            turns.append({"start": round(s, 2), "end": round(e, 2),
                          "speaker": name})
    return turns


def speech_density(clip: np.ndarray, *, frame: float = 0.02,
                   rel_thresh: float = 0.12) -> float:
    """Fraction of a clip that is actually speech rather than internal pause.

    Zero-shot cloning copies *prosody* along with timbre, so a reference made
    of four short utterances separated by pauses yields a dub that drawls:
    measured on the reference recording, a 9 s / 22 %-voiced prompt produced
    0.42–0.78 s per Chinese character against a natural ~0.22.
    """
    if len(clip) < int(frame * _SR):
        return 0.0
    n = int(frame * _SR)
    trimmed = clip[:len(clip) - len(clip) % n].reshape(-1, n)
    energy = np.sqrt((trimmed ** 2).mean(axis=1))
    peak = float(np.percentile(energy, 95))
    if peak <= 1e-6:
        return 0.0
    return float((energy > rel_thresh * peak).mean())


def compress_silences(clip: np.ndarray, *, max_gap: float = 0.18,
                      frame: float = 0.02, rel_thresh: float = 0.12
                      ) -> np.ndarray:
    """Shorten internal pauses so the cloned voice does not inherit them.

    Pauses are capped, not removed — deleting them entirely makes the prompt
    sound clipped and hurts the clone.
    """
    n = int(frame * _SR)
    if len(clip) < n * 3:
        return clip
    usable = clip[:len(clip) - len(clip) % n]
    frames = usable.reshape(-1, n)
    energy = np.sqrt((frames ** 2).mean(axis=1))
    peak = float(np.percentile(energy, 95))
    if peak <= 1e-6:
        return clip
    loud = energy > rel_thresh * peak
    keep = np.ones(len(frames), dtype=bool)
    max_frames = max(1, int(max_gap / frame))

    i = 0
    while i < len(loud):
        if loud[i]:
            i += 1
            continue
        j = i
        while j < len(loud) and not loud[j]:
            j += 1
        run = j - i
        if run > max_frames and i > 0 and j < len(loud):     # internal pause
            drop = run - max_frames
            keep[i + max_frames // 2: i + max_frames // 2 + drop] = False
        i = j

    out = frames[keep].reshape(-1)
    tail = clip[len(usable):]
    return np.concatenate([out, tail]) if len(tail) else out


def extract_speaker_references(
    diar: dict,
    segments: list[dict],
    audio_path: str | Path,
    *,
    vocals_path: str | Path | None = None,
    out_dir: str | Path,
    min_dur: float | None = None,
    max_dur: float | None = None,
    target_dur: float | None = None,
    n_alternatives: int = 2,
    overlap_regions: list | None = None,
) -> dict[str, dict]:
    """Pick one clean reference clip per speaker for zero-shot voice cloning.

    ``overlap_regions`` (OSD): windows with more than ``OSD_REF_MAX_OVERLAP``
    of their span in overlapped speech are rejected outright — two voices
    make no timbre reference.

    Candidates are runs of consecutive ASR segments belonging to one speaker,
    so the clip always comes with its exact Japanese transcript — which is
    what CosyVoice's ``inference_zero_shot`` needs as ``prompt_text``.

    Selection is deliberately picky: a breathy or laughter-only reference
    makes *every* synthesized line breathy, and a reference containing two
    voices produces a chimera. Returns ``{speaker: {...}}``.
    """
    import soundfile as sf
    from ai_movie.config import (
        TTS_REF_MAX_DURATION, TTS_REF_MIN_DURATION,
        TTS_REF_MIN_VOICED_RATIO, TTS_REF_TARGET_DURATION,
    )
    from ai_movie.utils import ensure_dir

    min_dur = TTS_REF_MIN_DURATION if min_dur is None else min_dur
    max_dur = TTS_REF_MAX_DURATION if max_dur is None else max_dur
    target_dur = TTS_REF_TARGET_DURATION if target_dur is None else target_dur

    out_dir = ensure_dir(Path(out_dir))
    cut_hz = diar.get("cut_hz")
    orig = _load_mono16k(audio_path)
    voc = None
    if vocals_path and Path(vocals_path).exists():
        try:
            voc = _load_mono16k(vocals_path)
        except Exception:                               # noqa: BLE001
            voc = None

    f0, ok, fps = pitch_track(voc if voc is not None else orig,
                              cache_key=str(audio_path))

    # ── build candidate runs (consecutive same-speaker segments) ────
    by_spk: dict[str, list[list[dict]]] = {}
    run: list[dict] = []
    for seg in sorted(segments, key=lambda s: float(s.get("start", 0.0))):
        spk = seg.get("speaker") or ""
        if not spk or not (seg.get("text") or "").strip():
            run = []
            continue
        if run and run[-1].get("speaker") == spk and \
                float(seg["start"]) - float(run[-1]["end"]) < 0.25:
            run.append(seg)
        else:
            run = [seg]
        by_spk.setdefault(spk, [])
        # Register every suffix of the run so we can pick the best length.
        for i in range(len(run)):
            window = run[i:]
            dur = float(window[-1]["end"]) - float(window[0]["start"])
            if min_dur <= dur <= max_dur:
                by_spk[spk].append(list(window))

    src = voc if voc is not None else orig
    src_name = "vocals" if voc is not None else "original"

    # Quality gates are *relative to each speaker*, not absolute.  An
    # off-mic interviewer can have a voiced-frame ratio around 0.06 where the
    # close-mic interviewee sits near 0.35; an absolute bar calibrated on the
    # latter rejects every clip the former ever produced, leaving him with no
    # reference at all.  We reject only what is unusable in principle
    # (too short, silent, clipped) and rank the rest within the speaker.
    def _score_window(window: list[dict], *,
                      min_voiced: float = 0.0,
                      min_rms: float = 0.006,
                      want_gender: str | None = None) -> dict | None:
        s = float(window[0]["start"])
        e = float(window[-1]["end"])
        dur = e - s
        a, b = int(s * _SR), min(int(e * _SR), len(src))
        if b - a < int(0.5 * _SR):
            return None
        if overlap_regions:
            from ai_movie.config import OSD_REF_MAX_OVERLAP
            from ai_movie.osd import overlap_ratio as _ovr
            if _ovr(overlap_regions, s, e) > OSD_REF_MAX_OVERLAP:
                return None

        clip = src[a:b]
        rms = _rms(clip)
        peak = float(np.abs(clip).max()) if len(clip) else 0.0
        if rms < min_rms or peak >= 0.999:
            return None

        fa, fb = int(s * fps), min(int(e * fps), len(ok))
        voiced_ratio = float(ok[fa:fb].mean()) if fb > fa else 0.0
        if voiced_ratio < min_voiced:
            return None

        # The reference defines the cloned timbre, so it must not be the
        # *other* speaker leaking in through a mis-labelled segment.  Reject
        # any window whose own measurable pitch contradicts the gender we are
        # cloning — this is cheap insurance against a diarization slip
        # producing a male voice cloned from a woman.
        win_f0 = None
        if fb > fa and int(ok[fa:fb].sum()) >= 8:
            win_f0 = float(np.median(f0[fa:fb][ok[fa:fb]]))
        if want_gender and cut_hz and win_f0 is not None:
            own = "female" if win_f0 >= cut_hz else "male"
            if own != want_gender:
                return None

        # Residual = how much non-vocal energy sits under this span.
        residual = 0.0
        if voc is not None:
            n = min(len(orig), len(voc), b)
            if n > a:
                residual = _rms(orig[a:n] - voc[a:n]) / max(rms, 1e-6)

        text = "".join((w.get("text") or "").strip() for w in window)
        asr_conf = float(np.mean([float(w.get("asr_conf", 0.0) or 0.0)
                                  for w in window]))
        density = speech_density(clip)
        score = (
            - abs(dur - target_dur) * 0.30       # prefer ~target length
            + min(density, 0.95) * 5.0           # prefer continuous speech
            + min(voiced_ratio, 0.9) * 2.0       # prefer sustained voicing
            + min(rms, 0.20) * 4.0               # prefer loud, clear speech
            - min(residual, 2.0) * 1.2           # penalise music/noise bleed
            + asr_conf * 1.5                     # penalise uncertain speech
            - (0.8 if len(text) < 6 else 0.0)    # need real content
        )
        return {"start": round(s, 2), "end": round(e, 2),
                "duration": round(dur, 2), "text": text,
                "voiced_ratio": round(voiced_ratio, 2),
                "density": round(density, 2),
                "rms": round(rms, 4), "residual": round(residual, 2),
                "score": round(score, 3), "src": src_name,
                "f0": round(win_f0, 1) if win_f0 else None,
                "_span": (a, b)}

    all_by_spk: dict[str, list[dict]] = {}
    for seg in segments:
        spk = seg.get("speaker") or ""
        if spk and (seg.get("text") or "").strip():
            all_by_spk.setdefault(spk, []).append(seg)

    results: dict[str, dict] = {}
    for spk in sorted(all_by_spk):
        # Calibrate the voicing gate on this speaker's own distribution.
        want_gender = (diar.get("speakers", {}).get(spk, {}) or {}).get("gender")
        probe = [r for r in (_score_window([s], want_gender=want_gender)
                             for s in all_by_spk[spk])
                 if r is not None]
        if not probe:
            print(f"[diarize] no usable reference audio for {spk}",
                  file=sys.stderr)
            continue
        vr = sorted(r["voiced_ratio"] for r in probe)
        min_voiced = max(0.05, 0.6 * vr[int(len(vr) * 0.75)])
        rms_med = float(np.median([r["rms"] for r in probe]))
        min_rms = max(0.006, 0.35 * rms_med)

        scored = [r for r in (_score_window(w, min_voiced=min_voiced,
                                            min_rms=min_rms,
                                            want_gender=want_gender)
                              for w in by_spk.get(spk, []))
                  if r is not None]
        # Prefer dense prompts, but only if any exist — a sparse prompt is
        # still far better than none.
        from ai_movie.config import TTS_REF_MIN_DENSITY
        dense = [r for r in scored if r["density"] >= TTS_REF_MIN_DENSITY]
        if dense:
            scored = dense
        best = max(scored, key=lambda r: r["score"]) if scored else None

        if best is not None and best["duration"] >= min_dur:
            a, b = best.pop("_span")
            wav = src[a:b]
            pieces = [(best["start"], best["end"])]
        else:
            # No single run is long enough — a short-turn speaker such as an
            # off-mic interviewer.  Concatenate that speaker's best individual
            # utterances instead; zero-shot cloning only needs the prompt text
            # to match the prompt audio, and it does when we join both.
            singles = [r for r in probe
                       if r["voiced_ratio"] >= min_voiced and r["rms"] >= min_rms]
            if not singles:
                singles = probe
            singles.sort(key=lambda r: -r["score"])
            gap = np.zeros(int(0.12 * _SR), dtype=np.float32)
            chunks: list[np.ndarray] = []
            texts: list[str] = []
            pieces = []
            total = 0.0
            for r in singles:
                if total >= target_dur:
                    break
                a, b = r["_span"]
                chunks.append(src[a:b])
                chunks.append(gap)
                texts.append(r["text"])
                pieces.append((r["start"], r["end"]))
                total += r["duration"] + 0.12
            if total < 1.5:
                print(f"[diarize] reference for {spk} too short ({total:.1f}s)",
                      file=sys.stderr)
                continue
            wav = np.concatenate(chunks)
            base = singles[0]
            best = {
                "start": pieces[0][0], "end": pieces[-1][1],
                "duration": round(len(wav) / _SR, 2),
                "text": "".join(texts),
                "voiced_ratio": round(float(np.mean(
                    [r["voiced_ratio"] for r in singles[:len(pieces)]])), 2),
                "rms": base["rms"], "residual": base["residual"],
                "score": round(base["score"], 3), "src": src_name,
                "assembled": True,
            }
            best.pop("_span", None)

        best.pop("_span", None)

        # Keep the runners-up so the caller can choose by *measured* clone
        # quality instead of by proxy score alone (see
        # tts.select_best_reference).  Heuristics get this wrong: on the 390 s
        # reference the top-scoring female clip was a flat recitation of
        # measurements ("20cm 15cm...") that cloned at 0.389 similarity, below
        # the acceptance threshold, while other clips of the same speaker
        # reached 0.6+.
        alts: list[dict] = []
        seen = {round(float(best.get("start", -1)), 2)}
        for r in sorted((scored or probe), key=lambda r: -r["score"]):
            key = round(float(r["start"]), 2)
            if key in seen or r["duration"] < min_dur * 0.6:
                continue
            seen.add(key)
            r = dict(r)
            r.pop("_span", None)
            alts.append(r)
            if len(alts) >= n_alternatives:
                break

        # CosyVoice caps prompt audio at 30 s; ours is far below that, but
        # trim defensively in case the caller loosened max_dur.
        if len(wav) > int(max_dur * _SR):
            wav = wav[:int(max_dur * _SR)]
            best["duration"] = round(len(wav) / _SR, 2)

        # Cap internal pauses: the clone copies the prompt's pacing.
        wav = compress_silences(wav)
        best["duration"] = round(len(wav) / _SR, 2)
        best["density"] = round(speech_density(wav), 2)

        wav_path = out_dir / f"ref_{spk}.wav"
        sf.write(str(wav_path), wav, _SR)
        (out_dir / f"ref_{spk}.txt").write_text(best["text"], encoding="utf-8")

        best["ref_audio"] = str(wav_path)
        best["ref_text"] = best["text"]
        best["pieces"] = pieces
        best["gender"] = want_gender

        # Materialise the alternatives too, so trying one costs only a probe.
        kept = []
        for k, alt in enumerate(alts):
            a2 = int(alt["start"] * _SR)
            b2 = min(int(alt["end"] * _SR), len(src))
            if b2 - a2 < int(1.5 * _SR):
                continue
            w2 = compress_silences(src[a2:b2])
            p2 = out_dir / f"ref_{spk}_alt{k}.wav"
            sf.write(str(p2), w2, _SR)
            alt["ref_audio"] = str(p2)
            alt["ref_text"] = alt["text"]
            alt["duration"] = round(len(w2) / _SR, 2)
            kept.append(alt)
        if kept:
            best["alternatives"] = kept
        results[spk] = best

    return results


def speaker_at(turns: list[dict] | None, start: float, end: float,
               fallback: str | None = None) -> str | None:
    """Re-exported from :mod:`ai_movie.segmenter` for convenience."""
    from ai_movie.segmenter import speaker_at as _impl
    return _impl(turns, start, end, fallback)


def similarity(a_wav: str | Path, b_wav: str | Path,
               *, device: str = DIARIZE_DEVICE) -> float:
    """Cosine similarity between the speaker embeddings of two clips.

    Used to verify that a cloned TTS segment actually sounds like the
    speaker it was cloned from.
    """
    import torch

    enc = _load_encoder(device)
    out = []
    for p in (a_wav, b_wav):
        a = _load_mono16k(p)
        if len(a) < int(0.5 * _SR):
            return 0.0
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(a[None, :])).squeeze().cpu().numpy()
        out.append(e / max(np.linalg.norm(e), 1e-9))
    return float(np.dot(out[0], out[1]))


def assign_speaker_for_gender(diar: dict, seg: dict, gender: str,
                              *, minted_by: str = "manual") -> str:
    """Point *seg* at a speaker whose gender is *gender*; mint one if none exists.

    Shared by ``scripts/review_speakers.py --apply`` and the web editor.
    With exactly one same-gender speaker the segment is re-pointed to it.
    With none, a new ``S<k>`` is minted (an off-mic speaker with too few
    voiced frames to form a pitch mode gets merged away by the diarizer;
    leaving the old id would bind these lines to the *other* speaker's face).
    With two or more candidates there is no basis to choose, so the id is
    left as-is unless it already matches.  Returns the speaker id.
    """
    spks = diar.setdefault("speakers", {})
    same_g = [spk for spk, meta in spks.items() if meta.get("gender") == gender]
    cur = seg.get("speaker")
    if len(same_g) == 1:
        seg["speaker"] = same_g[0]
    elif not same_g:
        new_id = f"S{max((int(k[1:]) for k in spks if k.startswith('S') and k[1:].isdigit()), default=-1) + 1}"
        spks[new_id] = {"gender": gender, "f0_median": None, "total_speech": 0.0,
                        "n_turns": 0, "synthesized_by": minted_by}
        seg["speaker"] = new_id
    elif cur not in same_g:
        seg["speaker"] = same_g[0]
    seg["gender"] = gender
    seg["tts_gender"] = gender
    return seg["speaker"]
