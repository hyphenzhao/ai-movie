"""Video segment cutter using FFmpeg."""

import math
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable

from ai_movie.config import WORKSPACE_DIR
from ai_movie.utils import ensure_dir


def get_duration_seconds(video_path: Path) -> float:
    """Return video duration in seconds (float)."""
    result = subprocess.run([
        "ffprobe", "-v", "quiet", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ], capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def cut_video(
    video_path: Path,
    segment_duration: float = 180.0,
    progress_cb: Callable[[int, int, float], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[dict]:
    """Cut *video_path* into segments of *segment_duration* seconds.

    Parameters
    ----------
    progress_cb:
        Called as ``progress_cb(segment_index, total_segments, segment_fraction)``
        where *segment_fraction* is 0→1 for the current segment's encoding.

    cancel_check:
        Polled between segments.  Return ``True`` to abort.

    Returns
    -------
    list[dict] with keys ``path``, ``thumb``, ``index``, ``duration``, ``start``.
    ``start`` is the offset in seconds from the beginning of the original video.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ensure_dir(WORKSPACE_DIR / video_path.stem / f"cuts_{ts}")

    total_dur = get_duration_seconds(video_path)
    num_segments = math.ceil(total_dur / segment_duration)

    segments: list[dict] = []
    for i in range(num_segments):
        if cancel_check and cancel_check():
            break

        start = i * segment_duration
        dur = min(segment_duration, total_dur - start)
        seg_path = out_dir / f"seg_{i + 1:03d}.mp4"

        _run_ffmpeg_cut(video_path, start, dur, seg_path,
                        lambda frac: progress_cb(i, num_segments, frac)
                        if progress_cb else None)

        thumb_path = _extract_thumbnail(seg_path, out_dir / f"thumb_{i + 1:03d}.jpg")

        segments.append({
            "index": i + 1,
            "path": str(seg_path),
            "thumb": str(thumb_path),
            "duration": dur,
            "start": start,
        })

        if progress_cb:
            progress_cb(i + 1, num_segments, 1.0)

    return segments


def _run_ffmpeg_cut(
    src: Path, start: float, duration: float,
    dst: Path, progress_cb: Callable[[float], None] | None = None,
):
    """Cut a single segment.  Parse FFmpeg stderr for progress."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", str(src),
        "-t", str(duration),
        "-c", "copy",           # stream copy: fast, no re-encode
        "-progress", "pipe:1", "-nostats",
        str(dst),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1)

    # FFmpeg writes progress lines to stdout when -progress pipe:1 is used
    time_pattern = re.compile(r"^out_time_ms=(\d+)")
    total_ms = int(duration * 1_000_000)

    for line in proc.stdout:
        m = time_pattern.match(line)
        if m and total_ms > 0 and progress_cb:
            out_ms = int(m.group(1))
            frac = min(out_ms / total_ms, 1.0)
            progress_cb(frac)

    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read()
        raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=stderr)


def _extract_thumbnail(video_path: Path, thumb_path: Path) -> Path:
    """Extract first meaningful frame as JPEG thumbnail."""
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", "0.5", "-i", str(video_path),
        "-vframes", "1", "-q:v", "3",
        str(thumb_path),
    ], check=True, capture_output=True)
    return thumb_path
