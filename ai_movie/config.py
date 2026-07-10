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

# ── Lip Sync settings ──────────────────────────────────────────

# MuseTalk model directory (clone from GitHub + download weights).
# git clone https://github.com/Tencent/MuseTalk models/musetalk
# Then download weights to models/musetalk/checkpoints/
MUSETALK_MODEL_DIR = str(ROOT_DIR / "models" / "musetalk")

# MuseTalk face crop size (256 = HQ, 128 = fast).
MUSETALK_FACE_SIZE = 256

# Batch size for MuseTalk inference (lower if OOM).
MUSETALK_BATCH_SIZE = 4

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
# NOTE: context can leak into output with some models — test before enabling.
TRANSLATION_CONTEXT_SEGMENTS = 0

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