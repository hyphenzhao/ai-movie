"""Centralized configuration management."""

from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent

# Workspace for intermediate files
WORKSPACE_DIR = ROOT_DIR / "workspace"

# Supported video formats
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".flv", ".m4v"}

# ── ASR settings ─────────────────────────────────────────────

# CPU fallback: faster-whisper / CTranslate2 model.
# Can be a HuggingFace model name ("large-v3") or a local path.
ASR_MODEL_SIZE = "C:/models/faster-whisper-large-v3"

# GPU backends (DirectML, WSL+ROCm): openai-whisper model name or .pt path.
# Use "large-v3" for auto-download from OpenAI CDN.
ASR_OPENAI_WHISPER_MODEL = "large-v3"

# ── Window defaults ─────────────────────────────────────────

WINDOW_TITLE = "AI Movie - 视频配音"
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 640