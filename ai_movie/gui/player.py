"""VLC-based video player widget for tkinter embedding."""

import subprocess
import sys
import tempfile
import tkinter as tk
from pathlib import Path

from ai_movie.config import WORKSPACE_DIR, CJK_FONT
from ai_movie.utils import ensure_dir


class VideoPlayer:
    """A VLC media player embedded in a tkinter Frame.

    Parameters
    ----------
    parent : tk.Widget
        Parent widget for the player frame.
    channel : "left" or "right"
        Which audio channel this player outputs to.
        "left" → only left speaker; "right" → only right speaker.
    placeholder : str
        Text to show when no video is loaded.
    """

    def __init__(self, parent: tk.Widget, channel: str, placeholder: str):
        self.parent = parent
        self.channel = channel
        self._loaded = False
        self._vlc_instance = None
        self._media_player = None
        self._current_file: Path | None = None
        self._muted = False

        # Frame that holds the VLC video output.
        # Note: the caller is responsible for placing this frame in its parent
        # (pack / grid / place) — do NOT call a geometry manager here.
        self.frame = tk.Frame(parent, bg="#1a1a1a", highlightthickness=1,
                              highlightbackground="#333")

        # Placeholder label (hidden once video is loaded)
        self._placeholder = tk.Label(
            self.frame, text=placeholder,
            bg="#1a1a1a", fg="#666",
            font=(CJK_FONT, 14),
        )
        self._placeholder.place(relx=0.5, rely=0.5, anchor="center")

        # Bind resize to reposition placeholder
        self.frame.bind("<Configure>", self._on_resize)

        self._init_vlc()

    def _init_vlc(self):
        try:
            import vlc
            # Use different audio output modules so each player gets an
            # independent Windows audio session (volume/mute are isolated).
            if sys.platform == "win32":
                aout = "directsound" if self.channel == "left" else "mmdevice"
                self._vlc_instance = vlc.Instance(f"--aout={aout} --quiet")
            else:
                # Linux/macOS: VLC auto-detects the best audio backend
                self._vlc_instance = vlc.Instance("--quiet")
            self._media_player = self._vlc_instance.media_player_new()

            # Embed into tkinter frame
            win_id = self.frame.winfo_id()
            if sys.platform == "win32":
                self._media_player.set_hwnd(win_id)
            elif sys.platform == "darwin":
                self._media_player.set_nsobject(win_id)
            else:
                self._media_player.set_xwindow(win_id)
        except Exception as e:
            self._placeholder.config(
                text="VLC 初始化失败\n请确认已安装 VLC media player",
                fg="#b00",
            )

    def _on_resize(self, event=None):
        self._placeholder.place(relx=0.5, rely=0.5, anchor="center")

    # ── public API ────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, filepath: Path):
        """Load and start playing a video file.

        Audio is pre-processed so that only the assigned channel is audible.
        """
        if self._media_player is None:
            return

        self.stop()
        self._current_file = filepath
        processed = self._prepare_channel_audio(filepath)

        media = self._vlc_instance.media_new(str(processed))
        self._media_player.set_media(media)
        media.release()  # player holds its own ref

        self._loaded = True
        self._placeholder.place_forget()

    def play(self):
        if self._media_player and self._loaded:
            self._media_player.play()

    def pause(self):
        if self._media_player:
            self._media_player.pause()

    def stop(self):
        """Stop playback (media stays loaded, play() restarts from beginning)."""
        if self._media_player:
            self._media_player.stop()

    def unload(self):
        """Stop playback and release the current media."""
        self.stop()
        self._loaded = False
        self._current_file = None

    def toggle_play_pause(self) -> bool:
        """Return True if now playing, False if paused."""
        if self.is_playing:
            self.pause()
            return False
        else:
            self.play()
            return True

    @property
    def is_playing(self) -> bool:
        if self._media_player is None:
            return False
        return self._media_player.is_playing() == 1

    def seek_relative(self, seconds: float):
        """Jump forward (+) or backward (-) by *seconds*."""
        if self._media_player and self._loaded:
            cur = self._media_player.get_time()
            new_ms = max(0, cur + int(seconds * 1000))
            self._media_player.set_time(new_ms)

    def seek_absolute(self, position: float):
        """Seek to *position* (0.0 – 1.0)."""
        if self._media_player and self._loaded:
            self._media_player.set_position(position)

    def get_position(self) -> float:
        """Playback position as 0.0 – 1.0."""
        if self._media_player and self._loaded:
            pos = self._media_player.get_position()
            return pos if pos >= 0 else 0.0
        return 0.0

    def get_duration_ms(self) -> int:
        """Total duration in milliseconds."""
        if self._media_player and self._loaded:
            dur = self._media_player.get_length()
            return dur if dur > 0 else 0
        return 0

    def get_time_ms(self) -> int:
        """Current playback time in milliseconds."""
        if self._media_player and self._loaded:
            t = self._media_player.get_time()
            return t if t > 0 else 0
        return 0

    @property
    def muted(self) -> bool:
        return self._muted

    @muted.setter
    def muted(self, state: bool):
        self._muted = state
        if self._media_player:
            if state:
                vol = self._media_player.audio_get_volume()
                self._stored_volume = vol if vol > 0 else 100
                self._media_player.audio_set_volume(0)
            else:
                self._media_player.audio_set_volume(
                    getattr(self, "_stored_volume", 100)
                )

    def toggle_mute(self) -> bool:
        """Return the new mute state."""
        self.muted = not self._muted
        return self._muted

    # ── audio channel pre-processing ───────────────────────────

    def _prepare_channel_audio(self, filepath: Path) -> Path:
        """Create a temp copy with one audio channel silenced.

        Uses FFmpeg to pan audio:
        - "left" player → right channel silent (c0=c0, c1=0)
        - "right" player → left channel silent (c0=0, c1=c1)
        """
        cache_dir = ensure_dir(
            WORKSPACE_DIR / filepath.stem / "players"
        )
        out_path = cache_dir / f"{self.channel}.mp4"

        if out_path.exists():
            return out_path  # cached

        if self.channel == "left":
            pan_filter = "pan=stereo|c0=c0|c1=0*c0"
        else:
            pan_filter = "pan=stereo|c0=0*c0|c1=c0"

        try:
            result = subprocess.run([
                "ffmpeg", "-y", "-i", str(filepath),
                "-af", pan_filter,
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-map_metadata", "0",
                str(out_path),
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"FFmpeg audio channel pre-process failed.\n"
                f"STDERR:\n{e.stderr}"
            ) from e

        return out_path

    def destroy(self):
        if self._media_player:
            self._media_player.stop()
            self._media_player.release()
        self.frame.destroy()
