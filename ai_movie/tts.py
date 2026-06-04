"""Text-to-speech synthesis using CosyVoice.

Two synthesis modes (selectable per-run):
  "gender"  — detect speaker gender from the source vocals, then use the
               matching CosyVoice-300M-SFT built-in speaker ("中文女" / "中文男").
               Requires CosyVoice-300M-SFT in models/; falls back to
               CosyVoice2-0.5B zero-shot if SFT model is absent.
  "clone"   — extract the highest-energy 8 s speech segment from the source
               vocals and use it for cross-lingual voice cloning via
               CosyVoice2-0.5B.  Quality depends on Demucs separation.

IMPORTANT: CosyVoice uses internal threading for LLM inference and
must be called from the main thread.
"""

import sys
import threading
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import soundfile as sf

from ai_movie.config import WORKSPACE_DIR
from ai_movie.utils import ensure_dir

# ── model paths ──────────────────────────────────────────────────
_COSYVOICE_SRC = Path(__file__).parent.parent / "models" / "CosyVoice"
_MATCHA_SRC    = _COSYVOICE_SRC / "third_party" / "Matcha-TTS"
_SFT_MODEL_DIR = Path(__file__).parent.parent / "models" / "CosyVoice-300M-SFT"
_ZS_MODEL_DIR  = Path(__file__).parent.parent / "models" / "CosyVoice2-0.5B"

# ── SFT speaker IDs ───────────────────────────────────────────────
_SFT_FEMALE_SPK = "中文女"
_SFT_MALE_SPK   = "中文男"

# ── fallback reference voices (CosyVoice2 zero-shot) ─────────────
_ASSET_DIR       = _COSYVOICE_SRC / "asset"
_FEMALE_REF_WAV  = _ASSET_DIR / "zero_shot_prompt.wav"
_FEMALE_REF_TEXT = "希望你以后能够做的比我还好呦。"
_MALE_REF_WAV    = _ASSET_DIR / "cross_lingual_prompt.wav"

# F0 threshold (Hz) separating male from female
_GENDER_THRESHOLD_HZ = 165.0

# ── model cache ───────────────────────────────────────────────────
_model    = None
_is_sft   = False          # True when CosyVoice-300M-SFT is loaded
_lock     = threading.Lock()


def _load_model():
    global _model, _is_sft
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        if str(_COSYVOICE_SRC) not in sys.path:
            sys.path.insert(0, str(_COSYVOICE_SRC))
        if str(_MATCHA_SRC) not in sys.path:
            sys.path.insert(0, str(_MATCHA_SRC))
        from cosyvoice.cli.cosyvoice import AutoModel
        if _SFT_MODEL_DIR.exists() and (_SFT_MODEL_DIR / "llm.pt").exists():
            _model  = AutoModel(model_dir=str(_SFT_MODEL_DIR))
            _is_sft = True
        else:
            _model  = AutoModel(model_dir=str(_ZS_MODEL_DIR))
            _is_sft = False


# ── public helpers ────────────────────────────────────────────────

def _pyin_gender(clip: np.ndarray, sr: int) -> str | None:
    """Return 'female'/'male' from a mono float32 clip, or None if inconclusive.

    Uses frames with voicing probability > 0.7 to balance noise suppression
    against sensitivity.  Returns None when fewer than 2 frames are available.
    """
    import librosa
    if len(clip) < sr * 1.5:          # need ≥ 1.5 s for reliable estimate
        return None
    f0, voiced_flag, voiced_prob = librosa.pyin(
        clip,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
    )
    high_conf = voiced_prob > 0.7
    valid = f0[high_conf & voiced_flag]
    if len(valid) < 2:
        return None
    return "female" if float(np.median(valid)) >= _GENDER_THRESHOLD_HZ else "male"


def detect_gender(audio_path: str | Path) -> str:
    """Estimate speaker gender for an entire audio file.

    Analyses a centred 20 s window. Falls back to ``'female'`` when
    no voiced frames are found.
    """
    a, sr = sf.read(str(audio_path))
    if a.ndim > 1:
        a = a.mean(axis=1)
    clip_len = min(len(a), sr * 20)
    mid = len(a) // 2
    start = max(0, mid - clip_len // 2)
    clip = a[start: start + clip_len].astype(np.float32)
    return _pyin_gender(clip, sr) or "female"


def detect_gender_from_segment(
    seg: dict,
    fallback: str = "female",
    vocals_path: str | Path | None = None,
) -> str:
    """Detect speaker gender from a single translated segment.

    Parameters
    ----------
    seg:
        Segment dict with ``source``, ``start``, ``end`` fields.
    fallback:
        Gender to return when the clip is too short or detection fails.
    vocals_path:
        If provided, use this Demucs-separated vocals file instead of
        ``seg['source']``.  The same ``[start, end]`` timestamps apply.
        Vocals tracks yield more high-confidence voiced frames because
        background music is removed, giving better F0 estimates.
    """
    source = str(vocals_path) if vocals_path and Path(vocals_path).exists() \
             else seg.get("source")
    start  = seg.get("start", 0.0)
    end    = seg.get("end")
    if not source or not Path(source).exists() or end is None:
        return fallback
    a, sr = sf.read(str(source))
    if a.ndim > 1:
        a = a.mean(axis=1)
    s0 = int(start * sr)
    s1 = int(end   * sr)
    clip = a[s0:s1].astype(np.float32)
    return _pyin_gender(clip, sr) or fallback


# ── Algorithm 2: global F0 cache ─────────────────────────────────
_f0_cache: dict[str, tuple] = {}   # audio_path → (f0_array, voiced_prob, sr)


def _load_global_f0(audio_path: str | Path) -> tuple:
    """Load and cache the full-track F0 arrays for *audio_path*."""
    import librosa
    key = str(audio_path)
    if key not in _f0_cache:
        a, sr = sf.read(key)
        if a.ndim > 1:
            a = a.mean(axis=1)
        f0, voiced_flag, voiced_prob = librosa.pyin(
            a.astype(np.float32),
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=sr,
        )
        _f0_cache[key] = (f0, voiced_flag, voiced_prob, sr)
    return _f0_cache[key]


def detect_gender_global_f0(seg: dict, fallback: str = "female") -> str:
    """Gender detection using a pre-computed whole-track F0 array.

    Computes F0 for the entire source audio once (cached), then looks up
    the frames that fall within [seg['start'], seg['end']].  More stable
    than re-running pyin per segment because background-music frames are
    filtered globally.
    """
    source = seg.get("source")
    start  = seg.get("start", 0.0)
    end    = seg.get("end")
    if not source or not Path(source).exists() or end is None:
        return fallback
    try:
        f0, voiced_flag, voiced_prob, sr = _load_global_f0(source)
        # Hop size used by librosa.pyin default (512 samples)
        hop = 512
        frame_start = int(start * sr / hop)
        frame_end   = int(end   * sr / hop)
        seg_f0   = f0[frame_start:frame_end]
        seg_vf   = voiced_flag[frame_start:frame_end]
        seg_vp   = voiced_prob[frame_start:frame_end]
        high_conf = seg_vp > 0.7
        valid = seg_f0[high_conf & seg_vf]
        if len(valid) < 2:
            return fallback
        return "female" if float(np.median(valid)) >= _GENDER_THRESHOLD_HZ else "male"
    except Exception:
        return fallback


# ── Algorithm 3: ECAPA speaker diarization ───────────────────────

def build_ecapa_gender_map(
    audio_path: str | Path,
    num_speakers: int = 2,
    progress_cb=None,
) -> dict:
    """Run speaker diarization and return a gender map for each speaker label.

    Uses ``simple_diarizer`` (silero-VAD + SpeechBrain ECAPA-TDNN +
    spectral clustering).  All models are loaded from local disk — no
    internet required after first download.

    Returns
    -------
    dict with keys:
      ``"segments"``: list of ``{start, end, label}`` dicts
      ``"gender"``:   dict mapping speaker label → ``'male'``/``'female'``
    """
    import torch
    import tempfile
    import librosa as _librosa

    # Ensure torch.hub uses local cache (for silero-vad)
    _hub_dir = Path(__file__).parent.parent / "models" / "torch_hub"
    _hub_dir.mkdir(exist_ok=True)
    torch.hub.set_dir(str(_hub_dir))

    from simple_diarizer.diarizer import Diarizer

    if progress_cb:
        progress_cb("加载说话人日志模型…")

    diar = Diarizer(embed_model="ecapa", cluster_method="sc")

    # Resample to 16kHz mono WAV for diarizer
    a, sr = sf.read(str(audio_path))
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        a = _librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000

    if progress_cb:
        progress_cb("运行说话人分割（约30-60秒）…")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, a, sr)
        tmp = f.name
    try:
        diar_segs = diar.diarize(tmp, num_speakers=num_speakers)
    finally:
        Path(tmp).unlink(missing_ok=True)

    if progress_cb:
        progress_cb("用 F0 标注说话人性别…")

    # Label each speaker cluster with F0
    speaker_f0: dict = {}
    for ds in diar_segs:
        spk = ds["label"]
        s0 = int(ds["start"] * sr)
        s1 = int(ds["end"]   * sr)
        clip = a[s0:s1].astype(np.float32)
        g = _pyin_gender(clip, sr)
        if g is not None:
            speaker_f0.setdefault(spk, []).append(g)

    gender_map: dict[int, str] = {}
    for spk, labels in speaker_f0.items():
        female_count = labels.count("female")
        gender_map[spk] = "female" if female_count >= len(labels) / 2 else "male"

    # Fallback: if only one speaker found, label by overall F0
    if not gender_map:
        g = detect_gender(audio_path)
        gender_map[0] = g

    return {"segments": diar_segs, "gender": gender_map}


def lookup_ecapa_gender(
    seg: dict,
    ecapa_result: dict,
    fallback: str = "female",
) -> str:
    """Return the gender for a segment using pre-computed ECAPA diarization."""
    diar_segs = ecapa_result.get("segments", [])
    gender_map = ecapa_result.get("gender", {})
    if not diar_segs:
        return fallback
    mid = (seg.get("start", 0) + seg.get("end", 0)) / 2
    best_spk = None
    best_overlap = 0.0
    seg_start = seg.get("start", 0)
    seg_end   = seg.get("end",   0)
    for ds in diar_segs:
        overlap = min(ds["end"], seg_end) - max(ds["start"], seg_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_spk = ds["label"]
    if best_spk is None:
        # Fallback: nearest midpoint
        best_spk = min(diar_segs, key=lambda d: abs((d["start"]+d["end"])/2 - mid))["label"]
    return gender_map.get(best_spk, fallback)


def extract_best_speech_segment(
    audio_path: str | Path,
    out_path: str | Path,
    duration: int = 8,
) -> str:
    """Extract the highest-RMS speech segment as a clean reference clip.

    Slides a ``duration``-second window across the audio and picks the
    window with the highest RMS energy (loudest clear speech).
    Writes the result to *out_path* and returns its path string.
    """
    a, sr = sf.read(str(audio_path))
    if a.ndim > 1:
        a = a.mean(axis=1)

    target_len = sr * duration
    if len(a) <= target_len:
        sf.write(str(out_path), a, sr)
        return str(out_path)

    hop = max(1, sr // 2)  # 0.5 s steps
    best_rms, best_start = -1.0, 0
    for start in range(0, len(a) - target_len, hop):
        rms = float(np.sqrt(np.mean(a[start: start + target_len] ** 2)))
        if rms > best_rms:
            best_rms, best_start = rms, start

    sf.write(str(out_path), a[best_start: best_start + target_len], sr)
    return str(out_path)


def trim_reference_audio(reference_audio: str | Path, max_seconds: int = 10) -> str:
    """Trim reference audio to at most *max_seconds* (≤ 30 s CosyVoice limit).

    Writes a ``*.ref10s.wav`` file next to the source and returns its path.
    Falls back to the original path on any error.
    """
    src = Path(reference_audio)
    dst = src.with_name(src.stem + ".ref10s.wav")
    try:
        audio_np, sr = sf.read(str(src))
        clip_len = min(len(audio_np), sr * max_seconds)
        sf.write(str(dst), audio_np[:clip_len], sr)
        return str(dst)
    except Exception:
        return str(src)


def prepare_reference(
    vocals_path: str | Path,
    mode: Literal["gender", "clone"] = "gender",
    cache_dir: str | Path | None = None,
) -> tuple[str, str | None, str]:
    """Derive the speaker / reference and synthesis method for one session.

    Returns
    -------
    (spk_or_ref, ref_text_or_None, method)

    When *method* is ``'sft'``, *spk_or_ref* is a speaker ID string
    (e.g. ``'中文女'``) and *ref_text* is None.
    When *method* is ``'zero_shot'`` or ``'cross_lingual'``, *spk_or_ref*
    is a file path to reference audio.
    """
    cache_dir = ensure_dir(Path(cache_dir) if cache_dir else WORKSPACE_DIR / "synthesized")
    gender    = detect_gender(vocals_path)

    if mode == "gender":
        if _is_sft:
            # Best path: SFT built-in speaker, no reference audio needed
            spk = _SFT_FEMALE_SPK if gender == "female" else _SFT_MALE_SPK
            return spk, None, "sft"
        # Fallback: zero-shot / cross-lingual with bundled reference audio
        if gender == "female" and _FEMALE_REF_WAV.exists():
            return str(_FEMALE_REF_WAV), _FEMALE_REF_TEXT, "zero_shot"
        if _MALE_REF_WAV.exists():
            return str(_MALE_REF_WAV), None, "cross_lingual"
        return trim_reference_audio(vocals_path), None, "cross_lingual"

    # mode == "clone" — extract best speech segment from source vocals
    best_seg = str(cache_dir / "ref_best_segment.wav")
    extract_best_speech_segment(vocals_path, best_seg)
    return best_seg, None, "cross_lingual"


def call_tts(
    model,
    text: str,
    spk_or_ref: str,
    ref_text: str | None,
    method: str,
) -> np.ndarray:
    """Run one CosyVoice inference call and return a float32 numpy array.

    Parameters
    ----------
    method:
        ``'sft'``           — *spk_or_ref* is a speaker ID; uses inference_sft.
        ``'zero_shot'``     — *spk_or_ref* is a reference audio path.
        ``'cross_lingual'`` — *spk_or_ref* is a reference audio path.
    """
    chunks = []
    if method == "sft":
        for gen in model.inference_sft(text, spk_or_ref, stream=False):
            chunks.append(gen["tts_speech"].squeeze(0).cpu().numpy())
    elif method == "zero_shot" and ref_text:
        for gen in model.inference_zero_shot(text, ref_text, spk_or_ref, stream=False):
            chunks.append(gen["tts_speech"].squeeze(0).cpu().numpy())
    else:
        for gen in model.inference_cross_lingual(text, spk_or_ref, stream=False):
            chunks.append(gen["tts_speech"].squeeze(0).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def synthesize(
    segments: list[dict],
    reference_audio: str | Path,
    output_dir: str | Path | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    mode: Literal["gender", "clone"] = "gender",
) -> list[dict]:
    """Generate Chinese speech for translated segments.

    Must be called from the **main thread** (CosyVoice internal threading).

    Parameters
    ----------
    mode:
        ``'gender'`` — clear bundled voice matched to detected gender (default).
        ``'clone'``  — voice cloning from the best segment of *reference_audio*.
    """
    _load_model()

    if output_dir is None:
        output_dir = ensure_dir(WORKSPACE_DIR / "synthesized")
    else:
        output_dir = ensure_dir(Path(output_dir))

    ref_audio, ref_text, method = prepare_reference(
        reference_audio, mode=mode, cache_dir=output_dir
    )

    total = len(segments)
    results: list[dict] = list(segments)
    per_segment_gender = _is_sft and mode == "gender"
    last_gender = detect_gender(reference_audio)  # whole-track baseline

    for i, seg in enumerate(results):
        if cancel_check and cancel_check():
            break

        text = seg.get("text_translated", "").strip()
        if not text:
            results[i]["audio"] = None
            if progress_cb:
                progress_cb(i + 1, total)
            continue

        # Per-segment speaker selection (SFT gender mode only)
        if per_segment_gender:
            # Use seg['source'] (original demuxed audio) — NOT Demucs vocals,
            # which suppresses the male voice and corrupts F0 detection.
            seg_gender = detect_gender_from_segment(seg, fallback=last_gender)
            last_gender = seg_gender
            spk = _SFT_FEMALE_SPK if seg_gender == "female" else _SFT_MALE_SPK
            seg_ref, seg_ref_text, seg_method = spk, None, "sft"
        else:
            seg_ref, seg_ref_text, seg_method = ref_audio, ref_text, method

        try:
            audio_np = call_tts(_model, text, seg_ref, seg_ref_text, seg_method)
            out_path = str(output_dir / f"seg_{i + 1:04d}.wav")
            sf.write(out_path, audio_np, _model.sample_rate)
            results[i]["audio"] = out_path
            results[i]["tts_gender"] = seg_gender if per_segment_gender else last_gender
        except Exception as exc:
            results[i]["audio"] = None
            results[i]["tts_error"] = str(exc)

        if progress_cb:
            progress_cb(i + 1, total)

    return results
