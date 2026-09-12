"""Audio / video demuxer using FFmpeg."""

import subprocess
from pathlib import Path
from typing import Callable

from ai_movie.cutter import get_duration_seconds
from ai_movie.utils import ensure_dir


def probe_audio_stream(video_path: Path) -> dict:
    """Sample rate / channel count of the first audio stream (0/0 if none)."""
    try:
        out = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "csv=p=0", str(video_path),
        ], check=True, capture_output=True, text=True).stdout.strip()
        sr, ch = out.split(",")[:2]
        return {"sample_rate": int(sr), "channels": int(ch)}
    except Exception:                                   # noqa: BLE001
        return {"sample_rate": 0, "channels": 0}


def extract_full_audio(video_path: Path, dst: Path) -> dict:
    """Full-rate stereo PCM for the production bed (see composer.mix_audio).

    Keeps the source sample rate; anything wider than stereo is downmixed
    to two channels (no 5.1 material in the test corpus, and the mixer
    places speech identically on every channel anyway).
    """
    info = probe_audio_stream(video_path)
    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-c:a", "pcm_s24le"]
    if info["channels"] > 2:
        cmd += ["-ac", "2"]
    cmd.append(str(dst))
    subprocess.run(cmd, check=True, capture_output=True)
    if info["channels"] > 2:
        info["channels"] = 2
    return {"audio_full": str(dst), **info}


def demux_video(video_path: Path, out_dir: Path) -> dict:
    """Split a single video into silent video + audio WAVs.

    Returns
    -------
    dict with keys ``video``, ``audio`` (16 kHz mono — the analysis track),
    ``audio_full`` (source rate, stereo — the production bed source),
    ``sample_rate``, ``channels``, ``duration``.
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

    # Audio track: 16 kHz mono WAV (ASR / diarization / reference clips).
    # This command is deliberately unchanged from v2 so cached analysis
    # results stay byte-identical.
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ar", "16000", "-ac", "1",
        str(audio),
    ], check=True, capture_output=True)

    full = extract_full_audio(video_path, out_dir / "audio_full.wav")

    duration = get_duration_seconds(video_path)
    return {
        "video": str(silent_video),
        "audio": str(audio),
        "duration": duration,
        **full,
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
