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

# ── Window defaults ─────────────────────────────────────────

WINDOW_TITLE = "AI Movie - 视频配音"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720