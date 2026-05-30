"""Audio / video demuxer using FFmpeg."""

import subprocess
from pathlib import Path
from typing import Callable

from ai_movie.cutter import get_duration_seconds
from ai_movie.utils import ensure_dir


def demux_video(video_path: Path, out_dir: Path) -> dict:
    """Split a single video into silent video + audio WAV.

    Returns
    -------
    dict with keys ``video``, ``audio``, ``duration``.
    """
    out_dir = ensure_dir(out_dir)

    silent_video = out_dir / "video_silent.mp4"
    audio = out_dir / "audio.wav"

    # Silent video (stream copy — fast)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-an", "-c:v", "copy",
        str(silent_video),
    ], check=True, capture_output=True)

    # Audio track: 16 kHz mono WAV
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ar", "16000", "-ac", "1",
        str(audio),
    ], check=True, capture_output=True)

    duration = get_duration_seconds(video_path)
    return {
        "video": str(silent_video),
        "audio": str(audio),
        "duration": duration,
    }


def demux_all(
    original_video: Path,
    cut_segments: list[dict] | None,
    output_base: Path,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[dict]:
    """Demux the original video or every cut segment.

    If *cut_segments* is non-empty each segment is processed individually;
    otherwise the original video is demuxed as a whole.
    """
    results: list[dict] = []

    if cut_segments:
        targets = [
            (Path(s["path"]), f"seg_{s['index']:03d}")
            for s in cut_segments
        ]
    else:
        targets = [(original_video, "original")]

    for i, (src, label) in enumerate(targets):
        if cancel_check and cancel_check():
            break
        try:
            info = demux_video(src, output_base / label)
        except subprocess.CalledProcessError as e:
            info = {"error": str(e), "label": label, "source": str(src)}

        info["label"] = label
        info["source"] = str(src)
        results.append(info)

        if progress_cb:
            progress_cb(i + 1, len(targets))

    return results
