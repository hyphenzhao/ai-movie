"""Centralized configuration management."""

import sys
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent

# Workspace for intermediate files
WORKSPACE_DIR = ROOT_DIR / "workspace"

# Project save files
PROJECTS_DIR = ROOT_DIR / "projects"

# Supported video formats
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".flv", ".m4v"}

# ── Font configuration (cross-platform CJK) ────────────────────

# Default CJK font per platform (used before Tk is initialized).
_CJK_FONT_DEFAULT = {
    "win32":  "Microsoft YaHei",
    "darwin": "PingFang SC",
}.get(sys.platform, "Noto Sans CJK SC")

# Monospace font per platform.
_MONO_FONT_DEFAULT = {
    "win32":  "Consolas",
    "darwin": "Menlo",
}.get(sys.platform, "DejaVu Sans Mono")

# UI symbol font (play/pause/etc).
_SYMBOL_FONT_DEFAULT = {
    "win32":  "Segoe UI",
    "darwin": "Helvetica",
}.get(sys.platform, "DejaVu Sans")


def get_cjk_font(tk_root=None) -> str:
    """Return the best available CJK font on this system.

    Call after Tk is initialized for font-family detection;
    falls back to a platform-default otherwise.
    """
    if tk_root is not None:
        try:
            import tkinter.font as tkfont
            available = set(tkfont.families(root=tk_root))
            candidates = [
                "Microsoft YaHei",        # Windows
                "PingFang SC",            # macOS
                "Noto Sans CJK SC",       # Linux (preferred)
                "WenQuanYi Micro Hei",    # Linux (fallback)
                "Noto Sans SC",
                "WenQuanYi Zen Hei",
                "Source Han Sans SC",
            ]
            for f in candidates:
                if f in available:
                    return f
        except Exception:
            pass
    return _CJK_FONT_DEFAULT


CJK_FONT = _CJK_FONT_DEFAULT
MONO_FONT = _MONO_FONT_DEFAULT
SYMBOL_FONT = _SYMBOL_FONT_DEFAULT


def init_fonts(tk_root) -> None:
    """Detect best available fonts once Tk is running.

    Call this early in ``App.__init__`` to update the module-level
    ``CJK_FONT``, ``MONO_FONT``, ``SYMBOL_FONT`` globals.
    """
    global CJK_FONT, MONO_FONT, SYMBOL_FONT
    CJK_FONT = get_cjk_font(tk_root)
    # Rough monospace/symbol fallbacks on CJF font selection
    if sys.platform == "win32":
        MONO_FONT, SYMBOL_FONT = "Consolas", "Segoe UI"
    elif sys.platform == "darwin":
        MONO_FONT, SYMBOL_FONT = "Menlo", "Helvetica"
    else:
        MONO_FONT, SYMBOL_FONT = "DejaVu Sans Mono", "DejaVu Sans"

# ── ASR settings ─────────────────────────────────────────────

# CPU fallback: faster-whisper / CTranslate2 model.
# Can be a HuggingFace model name ("large-v3") or a local path.
ASR_MODEL_SIZE = str(ROOT_DIR / "models" / "faster-whisper-large-v3")

# GPU backends (DirectML, WSL+ROCm): openai-whisper model name or .pt path.
# Use "large-v3" for auto-download from OpenAI CDN.
ASR_OPENAI_WHISPER_MODEL = "large-v3"

# ── VAD (Voice Activity Detection) settings ──────────────────
# Applied to openai-whisper GPU path to prevent hallucination loops
# (especially critical for Japanese) and improve sentence segmentation.
#
# Speech probability threshold (0.0–1.0).  Lower = more sensitive.
# Japanese conversational speech benefits from a lower threshold (0.35)
# because pitch variation triggers false silence detections at 0.5.
ASR_VAD_THRESHOLD = 0.35

# Minimum silence duration (ms) to mark a segment boundary.
# 500 ms ≈ natural pause between dialogue turns.
ASR_VAD_MIN_SILENCE_DURATION_MS = 500

# Minimum speech duration (ms).  Shorter segments are treated as noise.
ASR_VAD_MIN_SPEECH_DURATION_MS = 150

# Padding (ms) added before/after each detected speech segment.
ASR_VAD_SPEECH_PAD_MS = 200

# ── ASR segmentation (v2) ─────────────────────────────────────
# Whisper emits *contiguous* segments inside a VAD chunk (each segment's
# ``start`` equals the previous segment's ``end``), so the old "merge when
# gap <= 0.05 s" rule chained a whole chunk into one blob — 44 s segments
# containing both speakers were routine.  v2 instead re-splits the word
# stream on punctuation / pauses / speaker turns with a hard duration cap.

# Ask Whisper for word-level timestamps (needed by the sentence splitter).
# Falls back automatically to segment-level splitting if the ROCm DTW pass
# fails.
ASR_WORD_TIMESTAMPS = True

# Whisper's "condition on previous text" causes runaway hallucination loops
# on Japanese; each VAD chunk is independent anyway.
ASR_CONDITION_ON_PREVIOUS = False

# Hard caps for one subtitle/dubbing segment.
ASR_MAX_SEGMENT_DURATION = 8.0     # seconds
ASR_MAX_SEGMENT_CHARS = 24         # characters (CJK)
ASR_MIN_SEGMENT_DURATION = 0.4     # shorter fragments get absorbed

# Inter-word silence (s) that forces a sentence break.
ASR_PAUSE_SPLIT_SEC = 0.45

# Characters that end a sentence / allow a soft break.
ASR_SENTENCE_END = "。！？!?…♪"
ASR_SOFT_BREAK = "、，,"

# Which audio Whisper hears.  Measured against the burned-in subtitles of
# output_test (same code, only this switch changed): mix median 0.909 with 14
# lines < 0.70; UVR vocals median 0.857 with 19 — separation artefacts cost
# Whisper more than the removed music helps.  "vocals" stays selectable.
ASR_AUDIO_SOURCE = "mix"             # "mix" | "vocals"
# Where pitch/timbre gender is measured.  Separation shifts the spectrum up:
# the same male speaker measured 132 Hz on the vocals and 107 Hz on the mix.
DIARIZE_GENDER_SOURCE = "mix"        # "mix" | "vocals"

# A segment whose separated-vocal level is below this is digital silence:
# Whisper invented it (v3.0.0 test_2 had 「ありがとうございました」 at -78 dBFS).
# Film-independent by construction — it measures energy, not words.
ASR_SILENCE_DBFS = -60.0

# Sentence units (ai_movie/units.py): adjacent segments are translated as
# one utterance, then split back, when the first has no sentence-final mark
# and either there was no pause at all (< UNIT_CONTINUOUS_GAP: the character
# cap cut running speech) or a short pause follows a clause connective.
# Values chosen by a Sakura A/B on v3.0.0 output_test — see
# Documentation/v3.1-asr-translation.md.
UNIT_CONTINUOUS_GAP = 0.05
UNIT_MAX_GAP = 0.30
UNIT_MAX_DUR = 12.0
UNIT_MAX_CHARS = 60

# Domain hint / proper-noun spellings fed to Whisper as ``initial_prompt``.
#
# EMPTY BY DEFAULT — measured, not assumed.  A hint of
# "以下は日本語のインタビュー音声です。話者は複数います。" made large-v3 emit
# "話者は複数います。" verbatim as the transcript of 8 different low-energy
# chunks on the reference video.  Whisper treats initial_prompt as decoded
# context, so on a quiet chunk the likeliest continuation is simply more of
# the prompt.  Proper nouns are handled downstream by the glossary instead
# (see ai_movie/glossary.py), which cannot corrupt timings.
#
# If you do set one, keep it to a bare comma-separated noun list (no
# sentences) and re-check the transcript for echoes.
ASR_INITIAL_PROMPT: dict[str, str] = {}

# ── Speaker diarization ───────────────────────────────────────

# Run speaker diarization as part of 转换文字.
ASR_DIARIZE = True

# "ecapa" — local models/speechbrain-ecapa (no download, no HF token).
# "pyannote": pyannote segmentation + clustering decides *who* (worker in the
# OSD venv, see ai_movie/diar_worker.py), our pitch decides the voice type;
# "ecapa": the original pitch-gender + ECAPA-within-gender path.  pyannote is
# the trial default from v3.2: it found the interviewer on test_1 that the
# ECAPA path never did.  Flip back here if it proves worse.
DIARIZE_BACKEND = "pyannote"
# Report lines whose confidently-classified on-screen face disagrees with the
# voice's gender label while the acoustic evidence is weak.  Recorded at the
# faces step (04_face_gender_conflicts.csv + state) for the review tools to
# apply; not applied automatically because faces run after TTS.
DIARIZE_FACE_FEEDBACK = True

# ECAPA embedding device.  CPU is plenty (192-d embeddings on 1.5 s windows)
# and avoids ROCm/MIOpen JIT stalls on gfx1151.
DIARIZE_DEVICE = "cpu"

# Sliding-window embedding geometry (seconds).
DIARIZE_WINDOW = 1.5
DIARIZE_PERIOD = 0.75

# Agglomerative-clustering cosine distance threshold when the speaker count
# is not known (num_speakers=None → auto).
DIARIZE_AHC_THRESHOLD = 0.55

# Upper bound for automatic speaker-count estimation.
DIARIZE_MAX_SPEAKERS = 6

# Absolute F0 threshold (Hz) separating male from female.
DIARIZE_GENDER_HZ = 165.0

# Relative fallback: when every cluster lands on the same side of the
# absolute threshold but their medians differ by at least this much, label
# the lowest cluster male and the highest female.  This is what rescues
# recordings where the absolute threshold is simply wrong (the baseline run
# tagged all 35 segments "female").
DIARIZE_GENDER_REL_MIN_HZ = 25.0

# A window further than this cosine distance from every centroid is treated
# as overlapped/uncertain speech (speaker_conf < 0.5).
DIARIZE_UNCERTAIN_DIST = 0.35

# ── TTS (Text-to-Speech) settings ──────────────────────────────

# CosyVoice3-0.5B local path (best quality, ~1-2 GB VRAM FP16).
# Download:
#   git clone https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 models/CosyVoice3-0.5B
# Mirror:
#   git clone https://hf-mirror.com/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 models/CosyVoice3-0.5B
COSYVOICE3_MODEL_DIR = str(ROOT_DIR / "models" / "CosyVoice3-0.5B")

# CosyVoice2-0.5B local path (lightweight fallback, ~1 GB VRAM).
# Download:
#   git clone https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B models/CosyVoice2-0.5B
# Mirror:
#   git clone https://hf-mirror.com/FunAudioLLM/CosyVoice2-0.5B models/CosyVoice2-0.5B
COSYVOICE2_MODEL_DIR = str(ROOT_DIR / "models" / "CosyVoice2-0.5B")

# CosyVoice-300M-SFT local path (SFT model with built-in speakers, ~600 MB).
# Download:
#   git clone https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT models/CosyVoice-300M-SFT
# Mirror:
#   git clone https://hf-mirror.com/FunAudioLLM/CosyVoice-300M-SFT models/CosyVoice-300M-SFT
COSYVOICE_SFT_MODEL_DIR = str(ROOT_DIR / "models" / "CosyVoice-300M-SFT")

# TTS model priority: "cosyvoice3" (best), "cosyvoice_sft" (gender speakers),
# "cosyvoice2" (lightweight).  Auto-detected from available models.
TTS_PREFERRED_MODEL = "cosyvoice3"

# ── Voice cloning (per-speaker) ────────────────────────────────

# Reference-clip length bounds (seconds) for zero-shot cloning.
#
# Measured on the reference interview (same target sentence, four prompts):
#   prompt 9.0 s, density 0.62 → output 0.80 s/char, speaker similarity 0.66
#   prompt 8.2 s, density 0.68 → output 0.65 s/char, similarity 0.69
#   prompt 5.0 s, density 0.83 → output 0.31 s/char, similarity 0.50
#   bundled Chinese prompt     → output 0.21 s/char, similarity 0.10
# Natural Mandarin is ~0.22 s/char.  So prompt *density* controls pacing (a
# gappy prompt makes the dub drawl) and prompt *length* controls timbre
# similarity.  5–7 s of dense speech is the usable middle.
TTS_REF_MIN_DURATION = 4.0
TTS_REF_MAX_DURATION = 10.0
# Shorter is better than longer here: a long prompt tends to span several
# utterances with pauses between them, and zero-shot cloning copies that
# pacing into every dubbed line.
TTS_REF_TARGET_DURATION = 6.0

# Minimum fraction of the reference clip that must be speech rather than
# internal pause (see diarize.speech_density).
TTS_REF_MIN_DENSITY = 0.55

# A reference span must be at least this voiced (librosa.pyin voiced_flag
# ratio) — rejects laughter / breath-only spans, which otherwise make every
# synthesized line breathy.
TTS_REF_MIN_VOICED_RATIO = 0.6

# Reject the separated-vocals track for a span whose RMS collapsed to below
# this fraction of the original audio's RMS (Demucs/UVR male suppression).
TTS_VOCALS_RMS_MIN_RATIO = 0.25

# Minimum ECAPA cosine similarity between a cloned segment and its speaker
# reference before the clone is accepted.  Calibrated against measurement:
# a correct clone scores 0.50–0.69 on this material while a *wrong* voice
# (the bundled prompt) scores 0.10, so 0.40 separates them with margin
# without rejecting usable clones.
TTS_CLONE_MIN_SIMILARITY = 0.40

# F0 gate (v3): ECAPA similarity cannot see octave collapse (the shipped
# female reference halved every line's pitch, 232→117 Hz, while scoring
# *higher* similarity than working clips).  A candidate reference is only
# eligible if a probe synthesized from it lands in the speaker's gender band
# and within this output/reference F0 ratio (see ai_movie/pitch.py).
TTS_F0_GATE = True
TTS_F0_RATIO_RANGE = (0.8, 1.25)
TTS_GENDER_HZ = {"female": (165.0, 320.0), "male": (70.0, 175.0)}

# ── Duration fitting (TTS → time slot) ─────────────────────────

# Fit each synthesized segment into its timeline slot.  Without this the
# Chinese dub simply overruns and additively mixes into the next line.
TTS_FIT_TO_SLOT = True

# Speed-up is capped: above ~1.3x Mandarin sounds rushed and MuseTalk's
# mouth turns mushy.  We never slow speech down (sounds drunk).
TTS_FIT_MAX_SPEEDUP = 1.25
TTS_FIT_MIN_SPEEDUP = 0.85

# Escalated cap, used only when staying at TTS_FIT_MAX_SPEEDUP would force us
# to cut more than TTS_FIT_MAX_TRUNCATE seconds off the end of a line.  The
# per-speaker rate correction is a *median*, so the slow tail of a speaker's
# output still overruns; losing words is worse than a slightly faster
# delivery, so those lines are allowed to speed up further.
TTS_FIT_MAX_SPEEDUP_HARD = 1.60
TTS_FIT_MAX_TRUNCATE = 0.30

# Overrun (as a fraction of the slot) tolerated without any stretching —
# a short tail into following silence sounds natural.
TTS_FIT_TAIL_TOLERANCE = 0.15

# Extra seconds a segment may borrow from the following silence.
TTS_FIT_MAX_TAIL = 0.6

# Guard gap (s) kept before the next segment's start.
TTS_FIT_MIN_GAP = 0.12

# "rubberband" (better quality) with automatic "atempo" fallback.
TTS_FIT_BACKEND = "rubberband"

# ── Speaking-rate normalisation ────────────────────────────────
#
# Zero-shot cloning copies the prompt's *pace* as well as its timbre, and a
# Japanese prompt makes CosyVoice3 deliver Chinese slowly: measured on the
# reference interview the clones came out at a median 0.62 s per character
# against a natural ~0.22.  Capping the per-segment fit at 1.25x can never
# absorb a 3x rate error, so we first correct the rate globally per speaker —
# uniformly speeding up uniformly-slow speech restores a normal delivery
# rather than making it sound rushed — and only then fit each segment.
TTS_NATURAL_SEC_PER_CHAR = 0.22

# Only correct when the speaker is at least this much slower than natural.
TTS_RATE_MIN_CORRECTION = 1.20

# Upper bound on the global correction (beyond this something else is wrong).
TTS_RATE_MAX_CORRECTION = 2.60

# ── Duration-constrained rewrite ("compact" stage, v3) ──────────
#
# The translator never knew how long a slot was; a Chinese line that could
# not physically fit was sped up to 1.60× and then truncated.  After TTS the
# real duration of every line is known, so lines whose natural duration
# exceeds COMPACT_TRIGGER_RATIO × slot are rewritten shorter (meaning kept,
# glossary names kept) and re-synthesized.  Measured, not predicted: SFT
# voices run 0.17–0.20 s/char, clones up to 0.6, so a prediction-only tier
# would rewrite the wrong lines.
COMPACT_ENABLED = True
COMPACT_TRIGGER_RATIO = 1.30      # natural duration / slot above which we rewrite
COMPACT_TARGET_RATIO = 1.15       # budget the rewrite for this ratio (fit absorbs it)
COMPACT_MAX_ROUNDS = 2            # rewrite → re-synth → re-measure, at most twice
COMPACT_MIN_CHARS = 4             # never ask for fewer visible characters than this
COMPACT_MODEL = "dolphin-mixtral:8x7b"   # same instruct model enforce_glossary uses
TTS_COMPACT_SEC_PER_CHAR_DEFAULT = 0.24  # only when a speaker has no measurable lines

# ── Vocal Separation settings ──────────────────────────────────

# Active backend: "demucs" (GPU, reliable) or "uvr" (Mel-Band RoiFormer,
# higher quality but requires model download via proxy).
VOCAL_SEPARATION_BACKEND = "uvr"

# Demucs model name.
DEMUCS_MODEL = "htdemucs"

# UVR model name (MelBand Roformer | Vocals by Kimberley Jensen).
# Requires audio-separator package + first-time model download.
UVR_MODEL_NAME = "vocals_mel_band_roformer.ckpt"
UVR_MODEL_FILE_DIR = str(ROOT_DIR / "models" / "uvr")

# ── Production bed + mixing (v3) ───────────────────────────────
#
# Until v3 the demuxer extracted a 16 kHz mono wav, separation ran on that,
# and mix_audio adopted the background's sample rate — so every final dub
# shipped with a 16 kHz mono soundtrack regardless of the source (48 kHz
# stereo).  The analysis chain (ASR, diarization, reference clips) still
# wants 16 kHz mono; the *production* bed is now separated from the
# full-rate stereo audio and the analysis stems are derived from it.
SEPARATE_FULL_RATE_BED = True
SEPARATE_ANALYSIS_FROM_FULL = True

# Envelope ducking: the bed drops MIX_DUCK_DB under speech with a smooth
# attack/release instead of a hard per-sample gate.
MIX_DUCK_DB = -10.0
MIX_DUCK_ATTACK_MS = 50.0
MIX_DUCK_RELEASE_MS = 300.0

# Each dubbed line is matched to the loudness of the original dialogue in
# the same slot (bounded, so a mis-measured slot cannot blow a line up).
MIX_MATCH_LOUDNESS = True
MIX_MATCH_CLAMP_DB = 3.0

# Final two-pass ffmpeg loudnorm target.
MIX_TARGET_LUFS = -16.0
# -2 dBTP, not -1: the WAV mix lands exactly on target, but AAC encoding of
# the delivered MP4 overshoots by ~1 dB (v3.0.0 measured -1.0 dBTP in the WAV,
# -0.2 in the v1 MP4 and 0.0 in the v2 MP4).
MIX_TRUE_PEAK_DB = -2.0

# ── Lip Sync settings ──────────────────────────────────────────

# MuseTalk model directory (clone from GitHub + download weights).
# git clone https://github.com/Tencent/MuseTalk models/musetalk
# Then download weights to models/musetalk/checkpoints/
MUSETALK_MODEL_DIR = str(ROOT_DIR / "models" / "musetalk")

# MuseTalk face crop size (256 = HQ, 128 = fast).
MUSETALK_FACE_SIZE = 256

# Batch size for MuseTalk inference (lower if OOM).
MUSETALK_BATCH_SIZE = 4

# Temporal smoothing window (frames) for MuseTalk's per-frame crop box, and
# an optional unsharp-mask amount on the generated face before it is pasted
# back (0 = off).  See patches/README.md (musetalk_quality.patch).
MUSETALK_BOX_SMOOTH = 5
MUSETALK_SHARPEN = 0.4

# ── Face tracking / person anchoring ───────────────────────────

# Drive lip-sync from a per-frame face plan (which face belongs to which
# speaker) instead of "whatever S3FD ranked first in this frame".
LIPSYNC_USE_FACE_PLAN = True

# Face detection runs on CPU: S3FD's conv shapes trigger a multi-minute,
# silent, uninterruptible MIOpen JIT compile on gfx1151 (see face_restore).
FACE_DET_DEVICE = "cpu"
FACE_DET_EVERY = 5          # detect every Nth frame, interpolate between
FACE_DET_MAX_WIDTH = 640    # downscale before detection
FACE_DET_CONF = 0.8         # S3FD score threshold
FACE_TRACK_IOU = 0.3        # IoU to continue a track
FACE_TRACK_MIN_FRAMES = 8   # discard shorter tracks
FACE_TRACK_MAX_GAP = 3      # keyframes a track may go unmatched

# insightface genderage.onnx (1.3 MB) — bbox-only attribute model, run via
# onnxruntime.  "heuristic" disables face gender (all tracks "unknown").
# Long films are mostly silence: chunk 2 of a 172-min film had 18 s of speech in 8 min, yet face
# detection scanned all 14,500 frames (27 of the chunk's 65 minutes).  Only the frames within
# FACE_SCAN_MARGIN_S of a line that will be lip-synced are scanned; interjection-only lines
# (units.is_nonlexical) are dubbed but not lip-synced, so they are not scanned or enhanced either.
FACE_SCAN_SPEECH_ONLY = True
FACE_SCAN_MARGIN_S = 2.0
FACE_SKIP_NONLEXICAL = True

FACE_GENDER_BACKEND = "insightface"
FACE_GENDER_MODEL = str(ROOT_DIR / "models" / "insightface" / "genderage.onnx")
FACE_GENDER_SAMPLES = 24    # frames voted per track
FACE_GENDER_MIN_CONF = 0.65 # below this the track is "unknown"

# Minimum speaker↔track binding score; below it the speaker gets no face
# (their segments pass through as original video).
FACE_BIND_MIN_SCORE = 0.35

# ── Head-pose gate (侧脸直通) ──────────────────────────────────
# MuseTalk is a frontal-face model: measured on test_2 (540 s drama), the
# mouth it paints on a head turned 60–75° keeps only ~14% of the source's
# sharpness (a smear), while 0–45° keeps ~50%.  Frames beyond FACE_YAW_MAX
# therefore keep the original footage — an unsynced profile mouth is far
# less visible than a blurred blob.  Yaw comes from insightface's 1k3d68
# (3-D 68-pt landmarks, onnxruntime on CPU) at every detection keyframe.
FACE_POSE_MODEL = str(ROOT_DIR / "models" / "insightface" / "1k3d68.onnx")
FACE_YAW_MAX = 55.0         # deg; |yaw| above this → pass through
FACE_MIN_WIDTH = 80         # px; faces narrower than this → pass through
                            # (measured: <80 px stays at 0.45 sharpness even
                            # after CodeFormer; ≥80 px reaches 0.85–1.0)
FACE_GATE_SMOOTH = 15       # frames (odd); median filter on the gate so the
                            # mouth doesn't flip synced/original every few frames

# ── Face enhancement (CodeFormer) defaults ────────────────────
# Measured on test_2: with the lips protected CodeFormer changes almost
# nothing (mouth sharpness 44→48); unprotected at w=0.7 it nearly doubles
# it (44→82, 64→80) while keeping the generated mouth shape.
FACE_ENHANCE_FIDELITY = 0.7
FACE_ENHANCE_PROTECT_LIPS = False

# ── Translation settings ──────────────────────────────────────

# Hy-MT1.5-1.8B local path (previous generation, lightweight).
# Download: git clone https://huggingface.co/tencent/Hy-MT1.5-1.8B models/Hy-MT1.5-1.8B
# Mirror:  git clone https://hf-mirror.com/tencent/Hy-MT1.5-1.8B models/Hy-MT1.5-1.8B
TRANSLATION_MODEL_PATH = str(ROOT_DIR / "models" / "Hy-MT1.5-1.8B")

# Hy-MT2-30B-A3B-FP8 local path (FP8 quantised, ~8 GB VRAM).
# Download: git clone https://huggingface.co/tencent/Hy-MT2-30B-A3B-FP8 models/Hy-MT2-30B-A3B-FP8
# Mirror:  git clone https://hf-mirror.com/tencent/Hy-MT2-30B-A3B-FP8 models/Hy-MT2-30B-A3B-FP8
HYMT2_FP8_MODEL_PATH = str(ROOT_DIR / "models" / "Hy-MT2-30B-A3B-FP8")

# Hy-MT2-30B-A3B local path (BF16, ~18 GB VRAM — best quality).
# Download:
#   export HF_ENDPOINT=https://hf-mirror.com
#   huggingface-cli download tencent/Hy-MT2-30B-A3B --local-dir models/Hy-MT2-30B-A3B --local-dir-use-symlinks False
HYMT2_MODEL_PATH = str(ROOT_DIR / "models" / "Hy-MT2-30B-A3B")

# Active Hy-MT model path — set to HYMT2_MODEL_PATH for the strongest
# local translation, or TRANSLATION_MODEL_PATH for the lightweight fallback.
TRANSLATION_ACTIVE_MODEL = HYMT2_MODEL_PATH

# Batch size: number of segments per GPU inference call.
TRANSLATION_BATCH_SIZE = 8

# Max tokens to generate per segment.
TRANSLATION_MAX_NEW_TOKENS = 256

# Number of preceding segments to include as translation context (0 = none).
# The output is guarded by _extract_translation's echo/explanation detectors
# plus a context-leak check that retries the segment context-free.
TRANSLATION_CONTEXT_SEGMENTS = 2

# ── Context-aware / colloquial translation (v2) ────────────────

# Preceding / following segments shown to the LLM engines.
TRANSLATION_CTX_BEFORE = 4
TRANSLATION_CTX_AFTER = 2

# Scene framing prepended to every LLM translation prompt.
TRANSLATION_SCENE_HINT = (
    "这是一段日语访谈/对话的字幕。请翻译成自然、口语化的简体中文，"
    "该用俚语、俗语、语气词的地方就用，不要翻译腔，不要书面语。"
)

# Ollama models used by the new engines.
OLLAMA_GPTOSS_MODEL = "huihui_ai/gpt-oss-abliterated:120b"
OLLAMA_SAKURA_MODEL = "quantumcookie/Sakura-qwen2.5-v1.0:14b"

# Context polish of flagged lines only (translator._polish_flagged).  The
# env override lets the release runner fall back automatically when the
# model fails its smoke test.
import os as _os
# Finished videos stay on this machine.  The uplink here is ~0.7 MB/s and metered, so the release
# scripts' ``--upload`` (VPS preview site, Google Drive) is a no-op unless AI_MOVIE_UPLOAD=1.
PUBLISH_UPLOAD = _os.environ.get("AI_MOVIE_UPLOAD", "0") == "1"

OLLAMA_POLISH_MODEL = _os.environ.get(
    "AI_MOVIE_POLISH_MODEL",
    "ttempvnn/HauhauCS-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4-K-M:latest")
POLISH_CTX_BEFORE = 4
POLISH_CTX_AFTER = 2
POLISH_TIMEOUT = 300

# Approximate resident size (GB) per ollama model — used to decide whether
# other models must be evicted first.  Hy-MT2-30B (57 GB) and gpt-oss-120b
# (88 GB) cannot coexist in 122 GB.
OLLAMA_MODEL_SIZE_GB = {
    "huihui_ai/gpt-oss-abliterated:120b": 88.0,
    "dolphin-mixtral:8x22b": 80.0,
    "dolphin-mixtral:8x7b": 27.0,
    "quantumcookie/Sakura-qwen2.5-v1.0:14b": 13.0,
    "ttempvnn/HauhauCS-Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4-K-M:latest": 24.0,
}

# Evict other loaded ollama models before running one bigger than this.
OLLAMA_EXCLUSIVE_ABOVE_GB = 40.0

# ── Glossary / terminology ─────────────────────────────────────

# User-editable seed glossary, merged over auto-extracted terms
# (user entries always win).
GLOSSARY_PATH = str(ROOT_DIR / "asset" / "glossary.json")

# Model used to normalise pinned terms in finished translations.
#
# Deliberately NOT the translation model: Sakura is a completion-style
# translator and cannot follow a correction instruction — asked to fix a name
# it returned 小卡娜 / 小勘 / 小勘娜.  A general instruct model handles the
# one-sentence rewrite reliably (4/4 on the reference cases).  Bulk structured
# output from the same model is unreliable, which is why enforcement is
# per-sentence.
GLOSSARY_ENFORCE_MODEL = "dolphin-mixtral:8x7b"

# Auto-extract proper nouns / slang with the LLM before translating.
GLOSSARY_AUTO_EXTRACT = True
GLOSSARY_MAX_TERMS = 40
GLOSSARY_MIN_COUNT = 2

# Supported target languages {label: language-name-for-model}
TRANSLATION_TARGET_LANGS = {
    "汉语 (中文)": "Chinese",
    "English":     "English",
    "한국어":       "Korean",
}

# ── Ollama translation settings ─────────────────────────────────

# Ollama server base URL
OLLAMA_BASE_URL = "http://localhost:11434"

# Model to use for NSFW / colloquial translation
OLLAMA_MODEL = "dolphin-mixtral:8x22b"

# Number of segments per Ollama batch request.
OLLAMA_BATCH_SIZE = 15

# Maximum retries when JSON output validation fails.
OLLAMA_MAX_RETRIES = 5

# Number of preceding-batch sentences to include as translation
# context.  0 = no context; 2-3 helps dialogue coherence.
OLLAMA_CONTEXT_SENTENCES = 3

# System prompt for Ollama direct translation (translate_ollama).
# Output MUST be a single JSON array — one string per input sentence.
OLLAMA_NSFW_SYSTEM_PROMPT = (
    "你是一个专业的成人电影字幕翻译助手。"
    "请将以下台词翻译成口语化、通俗化、俚语化的中文。"
    "注意：这是成人电影内容的翻译，请保留原文中的脏话、俚语、性暗示等，"
    "用最地道、最粗俗的中文表达来翻译，不要用书面语美化。\n\n"
    "重要：你必须严格按照 JSON 数组格式输出，每句话对应一个数组元素。"
    "不要输出任何解释、注释、说明文字。\n"
    '输出示例：["你真棒，再深一点！", "要去了…", "舒服吗？"]'
)

# System prompt for polishing Hy-MT output (polish_ollama).
# Input is a JSON array of {{"original": "...", "translated": "..."}}
# objects.  Output MUST be a JSON array of polished strings.
OLLAMA_POLISH_SYSTEM_PROMPT = (
    "你是一个成人电影中文字幕润色助手。\n"
    "我会给你一个 JSON 数组，每个元素包含：\n"
    '  "original": 日语原文\n'
    '  "translated": 机器翻译的中文（可能过于书面化，部分俚语未被翻译）\n\n'
    "对每句，请做两件事：\n"
    "1. 如果「translated」中有残留的日语俚语/脏话没有被翻译，先翻译成中文。"
    "可以参考「original」来理解这些残留日语的意思。\n"
    "2. 把整句改写为口语化、俚语化、粗俗的地道中文，保留脏话和性暗示。\n\n"
    "重要：你必须严格输出一个 JSON 字符串数组，"
    "与输入数组一一对应，不要有任何其他内容。\n"
    '输出示例：["你的屌真大，再往里顶！", "要去了…别，我还不想去…", "爽吗？想更猛一点？"]'
)

# Japanese keywords that indicate a segment contains NSFW content
# and should be polished by Ollama.  Segments without any match
# keep the Hy-MT translation unchanged.
OLLAMA_NSFW_KEYWORDS = [
    # ── body parts ──
    "ちんちん", "おちんちん", "ちんぽ", "チンポ",
    "デカちん", "おっぱい", "まんこ", "クリトリス",
    "アナル", "パイパン", "ぼっき", "勃起",
    # ── sexual acts ──
    "エッチ", "セックス",
    "イく", "イきそう", "イった", "イって",
    "フェラ", "手コキ", "中出し", "なか出し",
    "生ハメ", "クンニ", "顔射",
    # ── NSFW adjectives / phrases ──
    "エロい", "エロ",
    "おっきいおちんちん",
]

# ── SakuraLLM (Sakura-Qwen) specialised prompts ──────────────────
# Auto-detected when model name contains both "sakura" and "qwen"
# (case-insensitive).  SakuraLLM is fine-tuned on Galgame / light-novel
# JP→ZH translation and uses Qwen2.5 ChatML format internally.
#
# Key differences from the generic Ollama prompts:
# - Uses Sakura's native instruction style ("将以下日文翻译成中文")
# - Explicit Galgame domain framing (matches the training distribution)
# - Stricter JSON-only instruction (Sakura tends to add explanations)

OLLAMA_SAKURA_TRANSLATE_PROMPT = (
    "你是一个专精日文→中文翻译的轻小说/Galgame翻译模型。"
    "将输入的日文台词翻译为口语化、地道的中文，保留原文的成人向表达、脏话和性暗示，"
    "使用粗俗直白的中文，不要美化或回避。\n\n"
    "重要规则：\n"
    "1. 只输出翻译结果，一行一句，不要编号、不要前缀。\n"
    "2. 不要输出任何解释、注释、翻译思路、示例、参考信息。\n"
    "3. 不要重复输入原文，不要输出「翻译：」「译文：」等前缀。\n"
    "4. 不要输出「参考前文」「上下文」等元信息。"
)

OLLAMA_SAKURA_POLISH_PROMPT = (
    "你是一个专精日文→中文翻译的轻小说/Galgame翻译模型。\n"
    "我会给你一句机器翻译结果和对应的日语原文，请将其润色为口语化中文：\n"
    "1. 如果机翻中有残留的日语俚语/脏话未翻译，参考原文补译\n"
    "2. 把整句改写为口语化、粗俗的地道中文，保留脏话和性暗示\n\n"
    "重要规则：\n"
    "1. 只输出润色后的中文，一行即可，不要编号、不要前缀。\n"
    "2. 不要输出任何解释、注释、翻译思路、示例。\n"
    "3. 不要输出「润色后：」「翻译：」「译文：」等前缀。\n"
    "4. 不要输出「参考前文」「上下文」「示例」等元信息。"
)

# ── SakuraLLM single-segment tuning ────────────────────────────────
# When Sakura models are detected, JSON batch mode is bypassed in
# favour of per-segment plain-text translation with concurrent requests.
OLLAMA_SAKURA_CONCURRENCY = 4   # parallel Ollama requests
OLLAMA_SAKURA_TIMEOUT = 900     # 15 min — 14B+ models need time for cold-start

# ── Window defaults ─────────────────────────────────────────

WINDOW_TITLE = "AI Movie - 视频配音"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
# ── Overlapped-speech detection (v3, pyannote/segmentation-3.0 on CPU) ──
OSD_ENABLED = True
OSD_MODEL = "pyannote/segmentation-3.0"
OSD_DEVICE = "cpu"
OSD_VENV = str(ROOT_DIR / "vendor" / "osd_venv")
# Diarization units with more overlap than this are not used as seeds for
# the channel classifier (their pitch/mel evidence is a mixture).
OSD_SEED_EXCLUDE = 0.5
# Reference-clip windows with more overlap than this are rejected.
OSD_REF_MAX_OVERLAP = 0.2

# ── Shot detection (v3) ────────────────────────────────────────
# ffmpeg scdet threshold (0–100; 10 catches hard cuts without firing on
# fast motion).  Track interpolation and the occlusion gate never bridge a
# detected cut.
SHOT_DETECT = True
SHOT_SCDET_THRESHOLD = 10.0

# ── Small-face upscale route (v3) ──────────────────────────────
# Faces narrower than FACE_MIN_WIDTH used to pass through untouched (the
# 256² model produces mush below ~80 px).  Faces in [FACE_MIN_WIDTH_SR,
# FACE_MIN_WIDTH) are now rendered on a 2× upscaled clip (the face is then
# 80–160 px, inside the model's comfort zone) and scaled back.  A clip takes
# the route when at least LIPSYNC_SR_MIN_FRAC of its anchored frames are
# small faces.
FACE_MIN_WIDTH_SR = 40
LIPSYNC_SMALL_FACE_UPSCALE = True
LIPSYNC_SR_MIN_FRAC = 0.5
LIPSYNC_SR_MAX_CLIP_SEC = 6.0      # 2× frames cost 4× RAM; keep clips short

# ── A/V offset knob (v3, experiments) ──────────────────────────
# Milliseconds the driving audio is delayed relative to the picture when
# cutting MuseTalk's audio clips.  Positive = audio later.  Measured lag is
# 0 frames (Documentation/v2-quality-upgrade.md), so the default is 0; the
# knob exists so scripts/ab_offset.py can sweep it.
LIPSYNC_AUDIO_OFFSET_MS = 0

# ── Occlusion handling (v3) ────────────────────────────────────
# "frame":  revert the whole frame to the original when the mouth is
#           occluded for ≥ 6 consecutive frames (v2 behaviour).
# "region": paste the original back only where an occluder (hair, hat,
#           clothing, background — anything BiSeNet does not call face) sits
#           in the lower face; the whole frame is reverted only when there
#           are essentially no lip pixels at all (OCCLUSION_FULL_LIP_THRESH).
#           Hands are labelled skin by BiSeNet and are NOT caught by this.
OCCLUSION_MODE = "frame"
OCCLUSION_FULL_LIP_THRESH = 0.0005

# ── MuseTalk paste fusion (v3, patches/musetalk_fusion.patch) ──
# "alpha":     single feathered jaw mask (upstream behaviour, 8 % feather).
# "laplacian": three-layer mask (generated mouth interior + lips, original
#              outer face) blended through a Laplacian pyramid.
MUSETALK_FUSION = "alpha"

# ── Per-segment QC thresholds (v3, ai_movie/qc.py) ─────────────
QC_ASR_CONF_WARN = 0.5
QC_SPEAKER_CONF_WARN = 0.6
QC_FIT_WARN = 1.25
QC_FIT_FAIL = 1.60
QC_OVERRUN_WARN = 0.15
QC_OVERRUN_FAIL = 0.30
QC_GATED_FRAC_WARN = 0.5
QC_OVERLAP_WARN = 0.3
QC_OVERLAP_FAIL = 0.6
QC_CLONE_SIM_WARN = TTS_CLONE_MIN_SIMILARITY
QC_MIX_GAIN_WARN = 2.9
QC_OCCLUSION_FRAC_WARN = 0.25   # clip-level occlusion fallback fraction that flags its segments
