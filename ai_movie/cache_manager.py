"""Workspace cache manager — safely finds and clears cache files.

Cache items are files/directories that can be regenerated automatically and
are NOT referenced by project save files (``.aimovie.json``).  Clearing them
frees disk space without breaking the ability to load a saved project.
"""

import shutil
from pathlib import Path

from ai_movie.config import WORKSPACE_DIR


class CacheManager:
    """Workspace cache scanner and cleaner.

    All methods are static — no state is kept between calls.
    """

    # ── patterns that are safe to delete ─────────────────────────

    # Glob patterns matched against WORKSPACE_DIR.
    # Order matters: broader patterns first, then more specific.
    CACHE_GLOBS: list[str] = [
        "**/players",           # VLC dual-channel player render caches
        "**/__pycache__",       # Python bytecode cache
        "**/.uvr_tmp",          # UVR separation temp directories
        "**/.locks",            # HuggingFace Hub lock files
        "**/models--*",         # HuggingFace Hub model cache (duplicate of models/)
        "**/temp",              # Wav2Lip temp files
        "**/results",           # Wav2Lip intermediate results (not project-referenced)
        "**/filelists",         # Wav2Lip temporary file lists
        "*.log",                # Debug log files
    ]

    # ── public API ───────────────────────────────────────────────

    @staticmethod
    def find_cache_items(workspace: Path | None = None) -> list[tuple[Path, int]]:
        """Scan the workspace and return ``(path, size_bytes)`` for every cache item.

        Parameters
        ----------
        workspace:
            Root directory to scan.  Defaults to ``WORKSPACE_DIR``.

        Returns
        -------
        List of (absolute_path, size_in_bytes), sorted by size descending.
        """
        if workspace is None:
            workspace = WORKSPACE_DIR
        if not workspace.exists():
            return []

        found: dict[Path, int] = {}  # path → size (dedup)

        for pattern in CacheManager.CACHE_GLOBS:
            for match in workspace.glob(pattern):
                if not match.exists():
                    continue
                # Use resolved path as key to deduplicate
                resolved = match.resolve()
                if resolved in found:
                    continue
                try:
                    size = CacheManager._get_size(resolved)
                except (OSError, PermissionError):
                    size = 0
                found[resolved] = size

        # Sort by size descending
        return sorted(found.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def clear_cache(workspace: Path | None = None) -> tuple[int, int]:
        """Delete all cache items.  Returns ``(deleted_count, freed_bytes)``.

        Parameters
        ----------
        workspace:
            Root directory to scan.  Defaults to ``WORKSPACE_DIR``.
        """
        items = CacheManager.find_cache_items(workspace)
        deleted = 0
        freed = 0

        for path, size in items:
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
                deleted += 1
                freed += size
            except (OSError, PermissionError):
                pass

        return deleted, freed

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Return a human-readable size string (e.g. ``"1.5 GB"``)."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 ** 2:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 ** 3:
            return f"{size_bytes / (1024 ** 2):.1f} MB"
        else:
            return f"{size_bytes / (1024 ** 3):.2f} GB"

    # ── internal helpers ─────────────────────────────────────────

    @staticmethod
    def _get_size(path: Path) -> int:
        """Recursively compute total size of a file or directory."""
        if path.is_file() and not path.is_symlink():
            return path.stat().st_size
        total = 0
        try:
            for child in path.rglob("*"):
                if child.is_file() and not child.is_symlink():
                    try:
                        total += child.stat().st_size
                    except OSError:
                        pass
        except (OSError, PermissionError):
            return 0
        return total
