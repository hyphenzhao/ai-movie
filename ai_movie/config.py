"""Centralized configuration management."""

import sys
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent

# Workspace for intermediate files
WORKSPACE_DIR = ROOT_DIR / "workspace"

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

# ── Translation settings ──────────────────────────────────────

# Hy-MT1.5-1.8B local path.
# Download: git clone https://huggingface.co/tencent/Hy-MT1.5-1.8B models/Hy-MT1.5-1.8B
TRANSLATION_MODEL_PATH = str(ROOT_DIR / "models" / "Hy-MT1.5-1.8B")

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
OLLAMA_MODEL = "dolphin-mixtral:8x7b"

# System prompt for adult film translation — instructs the model to use
# colloquial, slang-heavy, vulgar language instead of formal/written style.
OLLAMA_NSFW_SYSTEM_PROMPT = (
    "你是一个专业的成人电影字幕翻译助手。"
    "请将以下台词翻译成口语化、通俗化、俚语化的中文。"
    "注意：这是成人电影内容的翻译，请保留原文中的脏话、俚语、性暗示等，"
    "用最地道、最粗俗的中文表达来翻译，不要用书面语美化。"
    "只输出一行翻译结果，不要输出任何解释、注释、标注、原文或备选翻译。"
    "不要以'翻译：'或'Translation：'开头，直接给纯翻译文本。"
)

# System prompt for polishing Hy-MT output — the input is already a
# Chinese translation that may be too formal.  Some slang / NSFW terms
# may have been left untranslated (still in Japanese) by Hy-MT.
# The model should first translate those remaining terms, then rewrite
# the whole sentence in colloquial, slangy, vulgar Chinese.
OLLAMA_POLISH_SYSTEM_PROMPT = (
    "你是一个成人电影中文字幕润色助手。\n"
    "我会给你多段字幕，每段用 <<<SEG_N>>> 分隔，包含：\n"
    "  原文：日语原文\n"
    "  译文：机器翻译的中文（可能过于书面化，部分俚语未被翻译）\n\n"
    "对每段字幕，请做两件事：\n"
    "1. 如果「译文」中有残留的日语俚语/脏话没有被翻译，先把它们翻译成中文。"
    "可以参考「原文」来理解这些残留日语的意思。\n"
    "2. 把整句改写为口语化、俚语化、粗俗的地道中文，保留脏话和性暗示。\n\n"
    "输出格式：用同样的 <<<SEG_N>>> 标记每段，只输出改写后的中文，"
    "不要加「原文」「译文」标签，不要解释，不要括号注释。\n\n"
    "示例输出：\n"
    "<<<SEG_0>>>\n你的屌真大，太棒了，再往里顶！\n"
    "<<<SEG_1>>>\n要去了…别，我还不想去…\n"
    "<<<SEG_2>>>\n爽吗？想更猛一点？"
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

# ── Window defaults ─────────────────────────────────────────

WINDOW_TITLE = "AI Movie - 视频配音"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720