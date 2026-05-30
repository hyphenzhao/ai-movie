"""Centralized configuration management."""

from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent

# Workspace for intermediate files
WORKSPACE_DIR = ROOT_DIR / "workspace"

# Supported video formats
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".flv", ".m4v"}

# ASR settings
# Change to a local path if you downloaded the model manually, e.g.:
#   ASR_MODEL_PATH = "C:/models/faster-whisper-large-v3"
# Use "large-v3" for auto-download from HuggingFace (requires internet).
ASR_MODEL_SIZE = "large-v3"

# Window defaults
WINDOW_TITLE = "AI Movie - 视频配音"
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 640

ASR_MODEL_SIZE = "C:/models/faster-whisper-large-v3"