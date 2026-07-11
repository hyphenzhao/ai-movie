"""Main application window."""

import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from ai_movie import config
from ai_movie.cache_manager import CacheManager
from ai_movie.config import PROJECTS_DIR, WORKSPACE_DIR
from ai_movie.cutter import cut_video
from ai_movie.asr import LANGUAGES, LANG_LABELS, transcribe_all
from ai_movie.demuxer import demux_all
from ai_movie.gui.player import VideoPlayer
from ai_movie.project_log import ProjectLog, STEP_NAMES
from ai_movie.task_manager import task_manager
from ai_movie.utils import ensure_dir

VIDEO_FILETYPES = [
    ("视频文件", " ".join(f"*{ext}" for ext in sorted(config.VIDEO_EXTENSIONS))),
    ("所有文件", "*.*"),
]
PROJECT_FILETYPES = [
    ("AI Movie 项目", "*.aimovie.json"),
    ("所有文件", "*.*"),
]
THUMB_SIZE = (280, 158)  # 16:9 thumbnail


def _fmt_time(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"


class App:
    _PIPELINE_STEPS: list[tuple[str, str]] = [
        ("切割视频",   "_on_cut_video"),
        ("拆分音轨",   "_on_split_audio"),
        ("转换文字",   "_on_transcribe"),
        ("文本翻译",   "_on_translate"),
        ("人声分离",   "_on_separate_vocals"),
        ("人声生成",   "_on_generate_speech"),
        ("重新混音",   "_on_remix_audio"),
        ("合成音轨",   "_on_synthesize_audio"),
        ("人物锚定",   "_on_anchor_person"),
        ("口型匹配",   "_on_lip_sync"),
        ("人脸增强",   "_on_face_enhance"),
        ("合成视频",   "_on_compose_video"),
    ]

    TOOLBAR_COLORS = {
        "locked":   ("#d0d0d0", "#999999", "sunken",   "disabled"),
        "ready":    ("#ffffff", "#000000", "raised",   "normal"),
        "running":  ("#fff3cd", "#856404", "raised",   "disabled"),
        "done":     ("#d4edda", "#155724", "raised",   "normal"),
        "failed":   ("#f8d7da", "#721c24", "raised",   "normal"),
    }

    # ═══ tab navigation ═══════════════════════════════════════

    def _switch_to_tab(self, step_name: str):
        """Select the notebook tab for the given step name."""
        for idx in range(self._notebook.index("end")):
            if self._notebook.tab(idx, "text") == step_name:
                self._notebook.select(idx)
                return

    def _switch_to_next_tab(self, current_step: str):
        """After a step completes, switch to the next step's tab in the pipeline.

        Skips "合成音轨" (no dedicated tab) and stops at the last step.
        """
        step_names = [s[0] for s in self._PIPELINE_STEPS]
        try:
            idx = step_names.index(current_step)
            for next_name in step_names[idx + 1:]:
                if next_name != "合成音轨" and next_name in self._tab_frames:
                    self._switch_to_tab(next_name)
                    return
        except ValueError:
            pass

    # ═══ init ══════════════════════════════════════════════════

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(config.WINDOW_TITLE)
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.root.minsize(1100, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        config.init_fonts(self.root)

        self._is_playing = False
        self._seeking = False
        self._ui_updating = False
        self._cancel_requested = False
        self._photo_refs: list[ImageTk.PhotoImage] = []  # prevent GC

        self.log = ProjectLog()
        self._project_path: Path | None = None

        self._tb_btns: dict[str, tk.Button] = {}
        self._tab_frames: dict[str, ttk.Frame] = {}

        self._build_menu()
        self._build_toolbar()
        self._build_main_area()
        self._bind_shortcuts()
        self._start_sync_timer()

    # ═══ menu bar ══════════════════════════════════════════════

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="加载视频文件...", command=self._on_load_video)
        file_menu.add_separator()
        file_menu.add_command(label="保存项目", command=self._on_save_project)
        file_menu.add_command(label="加载项目...", command=self._on_load_project)
        file_menu.add_separator()
        file_menu.add_command(label="清除缓存...", command=self._on_clear_cache)
        menubar.add_cascade(label="文件", menu=file_menu)
        menubar.add_command(label="退出", command=self._on_exit)

    def _bind_shortcuts(self):
        self.root.bind_all("<Control-s>", lambda _e: self._on_save_project())
        self.root.bind_all("<Control-o>", lambda _e: self._on_load_video())

    # ═══ process toolbar ═══════════════════════════════════════

    def _build_toolbar(self):
        BG = "#e8e8e8"
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", side="top")
        inner = tk.Frame(bar, bg=BG)
        inner.pack(pady=4, padx=8)

        col = 0

        def _arrow():
            nonlocal col
            tk.Label(inner, text="→", font=(config.CJK_FONT, 10),
                     fg="#999", bg=BG).grid(row=0, column=col, padx=2)
            col += 1

        def _btn(label, method):
            nonlocal col
            b = tk.Button(inner, text=label, font=(config.CJK_FONT, 9),
                          width=8, height=2, command=getattr(self, method))
            b.grid(row=0, column=col, padx=1, pady=2)
            self._tb_btns[label] = b
            c = col
            col += 1
            return b, c

        # ── steps before the audio group ──────────────────────────
        _btn("切割视频", "_on_cut_video");  _arrow()
        _btn("拆分音轨", "_on_split_audio"); _arrow()
        _btn("转换文字", "_on_transcribe");  _arrow()
        _btn("文本翻译", "_on_translate");   _arrow()

        # ── audio group (three individual steps in row 0) ─────────
        group_start = col
        _btn("人声分离", "_on_separate_vocals"); _arrow()
        _btn("人声生成", "_on_generate_speech"); _arrow()
        _, group_end = _btn("重新混音", "_on_remix_audio")
        group_span = group_end - group_start + 1  # 5 cols: btn → btn → btn

        # Row 1: U-bracket (two verticals + bottom horizontal)
        bracket = tk.Canvas(inner, height=10, bg=BG, highlightthickness=0)
        bracket.grid(row=1, column=group_start, columnspan=group_span,
                     sticky="ew", padx=1)

        def _draw_bracket(event=None):
            bracket.delete("all")
            w = bracket.winfo_width()
            h = bracket.winfo_height()
            if w <= 1:
                return
            m, color = 4, "#888888"
            bracket.create_line(m, 0, m, h - 1, fill=color, width=1)         # left side
            bracket.create_line(m, h - 1, w - m, h - 1, fill=color, width=1) # bottom
            bracket.create_line(w - m, 0, w - m, h - 1, fill=color, width=1) # right side

        bracket.bind("<Configure>", lambda e: _draw_bracket())
        inner.after(60, _draw_bracket)

        # Row 2: centered ▼ arrow
        tk.Label(inner, text="▼", font=(config.SYMBOL_FONT, 8),
                 fg="#888", bg=BG).grid(row=2, column=group_start,
                                        columnspan=group_span, pady=1)

        # Row 3: shortcut button spanning the full group width
        sh_btn = tk.Button(inner, text="合成音轨", font=(config.CJK_FONT, 9),
                           height=2, command=self._on_synthesize_audio)
        sh_btn.grid(row=3, column=group_start, columnspan=group_span,
                    sticky="ew", padx=1, pady=(0, 4))
        self._tb_btns["合成音轨"] = sh_btn

        # ── steps after the audio group ───────────────────────────
        _arrow()
        _btn("人物锚定", "_on_anchor_person"); _arrow()
        _btn("口型匹配", "_on_lip_sync");       _arrow()
        _btn("人脸增强", "_on_face_enhance");    _arrow()
        _btn("合成视频", "_on_compose_video")

        # ── one-click button: large, left-aligned, below 切割视频 ──
        one_click_btn = tk.Button(
            inner, text="🚀  一键生成", font=(config.CJK_FONT, 10, "bold"),
            width=14, height=3, bg="#0078d4", fg="white",
            activebackground="#005a9e", activeforeground="white",
            command=self._on_one_click)
        one_click_btn.grid(row=4, column=0, columnspan=2, padx=1, pady=(10, 4), sticky="w")

        self._refresh_toolbar()

    def _refresh_toolbar(self):
        for name, method_name in self._PIPELINE_STEPS:
            btn = self._tb_btns[name]
            status = self.log.step_status(name)
            bg, fg, relief, tk_state = self.TOOLBAR_COLORS[status]
            display = name
            if status == "running":
                display = f"{name}\n⏳"
            elif status == "done":
                display = f"✓ {name}"
            btn.configure(text=display, bg=bg, fg=fg, relief=relief,
                          state=tk_state)
            if status in ("ready", "done", "failed"):
                btn.configure(command=getattr(self, method_name))
            else:
                btn.configure(command=None)

    # ═══ toolbar: 切割视频 ═════════════════════════════════════

    def _on_cut_video(self):
        if self.log.video_path is None:
            messagebox.showwarning("提示", "请先加载视频文件。")
            return

        src = Path(self.log.video_path)
        if not src.exists():
            messagebox.showerror("错误", f"视频文件不存在:\n{src}")
            return

        self.log.mark_step("切割视频", "running")
        self.log.add_entry("切割视频", "start")
        self._refresh_toolbar()
        self._cancel_requested = False

        self._cut_dlg = tk.Toplevel(self.root)
        self._cut_dlg.title("切割视频")
        self._cut_dlg.geometry("420x200")
        self._cut_dlg.transient(self.root)
        self._cut_dlg.grab_set()
        self._cut_dlg.protocol("WM_DELETE_WINDOW", self._on_cancel_cut)
        self._cut_dlg.resizable(False, False)

        ttk.Label(self._cut_dlg, text="正在切割视频…",
                  font=(config.CJK_FONT, 11)).pack(pady=(12, 8))

        # Overall progress
        f1 = ttk.Frame(self._cut_dlg); f1.pack(fill="x", padx=20, pady=(4, 2))
        ttk.Label(f1, text="整体进度：").pack(side="left")
        self._lbl_overall = ttk.Label(f1, text="0 / 0")
        self._lbl_overall.pack(side="right")
        self._bar_overall = ttk.Progressbar(self._cut_dlg, length=380, mode="determinate")
        self._bar_overall.pack(padx=20)

        # Segment progress
        f2 = ttk.Frame(self._cut_dlg); f2.pack(fill="x", padx=20, pady=(10, 2))
        ttk.Label(f2, text="当前片段：").pack(side="left")
        self._lbl_segment = ttk.Label(f2, text="0%")
        self._lbl_segment.pack(side="right")
        self._bar_segment = ttk.Progressbar(self._cut_dlg, length=380, mode="determinate")
        self._bar_segment.pack(padx=20)

        ttk.Button(self._cut_dlg, text="取消", command=self._on_cancel_cut).pack(pady=12)

        threading.Thread(target=self._run_cut, args=(src,), daemon=True).start()

    def _on_cancel_cut(self):
        self._cancel_requested = True

    def _run_cut(self, src: Path):
        try:
            segments = cut_video(
                src,
                segment_duration=180.0,
                progress_cb=self._on_cut_progress,
                cancel_check=lambda: self._cancel_requested,
            )
        except Exception as exc:
            _err = str(exc)
            self.root.after(0, lambda: self._on_cut_error(_err))
            return

        self.root.after(0, lambda: self._on_cut_done(src, segments))

    def _on_cut_progress(self, seg_idx: int, total: int, frac: float):
        """Called from worker thread — schedule UI update on main thread."""
        self.root.after(0, lambda: self._update_cut_dialog(seg_idx, total, frac))

    def _update_cut_dialog(self, seg_idx: int, total: int, frac: float):
        if not hasattr(self, "_cut_dlg") or not self._cut_dlg.winfo_exists():
            return
        self._bar_overall.configure(maximum=total,
                                     value=seg_idx if frac < 1.0 else seg_idx)
        self._lbl_overall.configure(text=f"{seg_idx} / {total}")
        self._bar_segment.configure(value=int(frac * 100))
        self._lbl_segment.configure(text=f"{int(frac * 100)}%")

    def _on_cut_done(self, src: Path, segments: list[dict]):
        if hasattr(self, "_cut_dlg") and self._cut_dlg.winfo_exists():
            self._cut_dlg.destroy()

        seg_dir = str(Path(segments[0]["path"]).parent) if segments else ""

        self.log.mark_step("切割视频", "done")
        self.log.set_step_data("切割视频", {
            "segments": segments,
            "output_dir": seg_dir,
        })
        self.log.add_entry("切割视频", "done",
                           f"{len(segments)} 段 → {seg_dir}")
        self._refresh_toolbar()
        self._populate_cut_tab(segments)
        self._switch_to_next_tab("切割视频")

    def _on_cut_error(self, error_msg: str):
        if hasattr(self, "_cut_dlg") and self._cut_dlg.winfo_exists():
            self._cut_dlg.destroy()
        self.log.mark_step("切割视频", "failed")
        self.log.add_entry("切割视频", "error", error_msg)
        self._refresh_toolbar()
        messagebox.showerror("切割失败", error_msg)

    # ═══ cut tab: thumbnails + play buttons ═══════════════════

    def _populate_cut_tab(self, segments: list[dict]):
        """Fill the 切割视频 tab with first-frame thumbnails and play overlays."""
        tab = self._tab_frames.get("切割视频")
        if tab is None:
            return

        # Clear previous content
        for w in tab.winfo_children():
            w.destroy()

        if not segments:
            tk.Label(tab, text="当前为空", fg="#aaa",
                     font=(config.CJK_FONT, 13)).place(relx=0.5, rely=0.5,
                                                         anchor="center")
            return

        # Scrollable canvas
        canvas = tk.Canvas(tab, bg="#f5f5f5", highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        # Unbind when tab is destroyed
        tab.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # Grid of thumbnails
        cols = max(1, tab.winfo_width() // (THUMB_SIZE[0] + 20)) if tab.winfo_width() > 1 else 2
        row = col = 0

        self._photo_refs.clear()

        for seg in segments:
            frame = ttk.Frame(scroll_frame, relief="solid", borderwidth=1)
            frame.grid(row=row, column=col, padx=8, pady=8, sticky="nsew")

            # Thumbnail image
            thumb_path = Path(seg["thumb"])
            if thumb_path.exists():
                img = Image.open(thumb_path)
                img.thumbnail(THUMB_SIZE, Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._photo_refs.append(photo)
                lbl = tk.Label(frame, image=photo, bg="#000")
                lbl.image = photo
            else:
                lbl = tk.Label(frame, text="无预览", bg="#333", fg="#666",
                               width=THUMB_SIZE[0] // 10, height=THUMB_SIZE[1] // 20)
            lbl.pack()

            # Play button overlay (centered on thumbnail)
            play_btn = tk.Button(
                frame, text="▶", font=(config.SYMBOL_FONT, 18, "bold"),
                fg="#fff", bg="#555555", activebackground="#333333",
                relief="flat", bd=0, width=3,
                command=lambda p=Path(seg["path"]): self._play_segment(p),
            )
            play_btn.place(relx=0.5, rely=0.5, anchor="center", width=50, height=50)

            # Segment label
            dur_str = _fmt_time(int(seg.get("duration", 0) * 1000))
            tk.Label(frame, text=f"片段 {seg['index']}  ({dur_str})",
                     font=(config.CJK_FONT, 8), fg="#555").pack(pady=(2, 4))

            col += 1
            if col >= cols:
                col = 0
                row += 1

        scroll_frame.columnconfigure(tuple(range(cols)), weight=1)

    def _play_segment(self, seg_path: Path):
        """Open a popup window with a VLC player + seekable progress bar."""
        win = tk.Toplevel(self.root)
        win.title(f"播放: {seg_path.name}")
        win.geometry("720x520")
        win.transient(self.root)
        win.minsize(400, 320)

        player = VideoPlayer(win, channel="left", placeholder="")
        player.frame.pack(expand=True, fill="both")
        player.load(seg_path)
        is_playing = True

        # ── Progress bar ──
        prog_frame = ttk.Frame(win, padding=(8, 4))
        prog_frame.pack(fill="x", side="bottom")

        lbl_cur = tk.Label(prog_frame, text="0:00", font=(config.MONO_FONT, 9),
                           fg="#666")
        lbl_cur.pack(side="left")

        slider = ttk.Scale(prog_frame, from_=0, to=1000,
                           orient="horizontal",
                           command=lambda v: player.seek_absolute(float(v) / 1000))
        slider.pack(side="left", fill="x", expand=True, padx=6)

        lbl_total = tk.Label(prog_frame, text="0:00", font=(config.MONO_FONT, 9),
                             fg="#666")
        lbl_total.pack(side="right")

        # ── Buttons ──
        btn_frame = ttk.Frame(win, padding=(8, 0))
        btn_frame.pack(fill="x", side="bottom", pady=(0, 6))

        def _toggle():
            nonlocal is_playing
            if is_playing:
                player.pause()
                is_playing = False
                btn_play.configure(text="▶")
            else:
                player.play()
                is_playing = True
                btn_play.configure(text="⏸")

        btn_play = tk.Button(btn_frame, text="⏸", font=(config.SYMBOL_FONT, 12),
                             width=3, command=_toggle)
        btn_play.pack(side="left", padx=4)

        tk.Button(btn_frame, text="⏹", font=(config.SYMBOL_FONT, 12), width=3,
                  command=player.stop).pack(side="left", padx=4)

        # ── Periodic sync ──
        _closed = False
        _seeking = False

        def _sync():
            nonlocal _seeking, is_playing
            if _closed:
                return
            _seeking = True
            dur = player.get_duration_ms()
            cur = player.get_time_ms()
            is_playing_now = player.is_playing
            if dur > 0:
                slider.configure(to=dur)
                slider.set(cur)
            lbl_total.configure(text=_fmt_time(dur))
            lbl_cur.configure(text=_fmt_time(cur))
            if is_playing_now != is_playing:
                is_playing = is_playing_now
                btn_play.configure(text="⏸" if is_playing else "▶")
            _seeking = False
            win.after(150, _sync)

        def _on_slider_drag(value):
            nonlocal _seeking
            if _seeking:
                return
            ms = int(float(value))
            dur = player.get_duration_ms()
            player.seek_absolute(ms / max(1, dur))
            lbl_cur.configure(text=_fmt_time(ms))

        slider.configure(command=_on_slider_drag)

        def _on_close():
            nonlocal _closed
            _closed = True
            player.destroy()
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close)

        win.after(200, _sync)

    # ═══ toolbar: 拆分音轨 ═════════════════════════════════════

    def _on_split_audio(self):
        if self.log.video_path is None:
            messagebox.showwarning("提示", "请先加载视频文件。")
            return

        src = Path(self.log.video_path)
        if not src.exists():
            messagebox.showerror("错误", f"视频文件不存在:\n{src}")
            return

        # Determine targets: cut segments or original video
        cut_data = self.log.step_data.get("切割视频", {})
        segments = cut_data.get("segments", [])
        # Validate segment files still exist
        segments = [s for s in segments if Path(s["path"]).exists()]

        self.log.mark_step("拆分音轨", "running")
        self.log.add_entry("拆分音轨", "start",
                           f"{len(segments)} segments" if segments else "original video")
        self._refresh_toolbar()
        self._cancel_requested = False

        # Progress dialog
        total = len(segments) if segments else 1
        self._split_dlg = tk.Toplevel(self.root)
        self._split_dlg.title("拆分音轨")
        self._split_dlg.geometry("400x160")
        self._split_dlg.transient(self.root)
        self._split_dlg.grab_set()
        self._split_dlg.protocol("WM_DELETE_WINDOW", self._on_cancel_split)
        self._split_dlg.resizable(False, False)

        ttk.Label(self._split_dlg, text="正在拆分音轨…",
                  font=(config.CJK_FONT, 11)).pack(pady=(12, 8))
        f1 = ttk.Frame(self._split_dlg); f1.pack(fill="x", padx=20, pady=4)
        ttk.Label(f1, text="处理进度：").pack(side="left")
        self._lbl_split_prog = ttk.Label(f1, text=f"0 / {total}")
        self._lbl_split_prog.pack(side="right")
        self._bar_split = ttk.Progressbar(self._split_dlg, length=360,
                                          mode="determinate", maximum=total)
        self._bar_split.pack(padx=20)
        ttk.Button(self._split_dlg, text="取消",
                   command=self._on_cancel_split).pack(pady=12)

        # Determine output base
        if segments:
            cut_dir = Path(segments[0]["path"]).parent
            out_base = cut_dir.parent / "demuxed"
        else:
            out_base = src.parent.parent / "workspace" / src.stem / "demuxed"

        threading.Thread(
            target=self._run_split,
            args=(src, segments, out_base),
            daemon=True,
        ).start()

    def _on_cancel_split(self):
        self._cancel_requested = True

    def _run_split(self, src: Path, segments: list[dict], out_base: Path):
        try:
            results = demux_all(
                src, segments or None, out_base,
                progress_cb=self._on_split_progress,
                cancel_check=lambda: self._cancel_requested,
            )
        except Exception as exc:
            _err = str(exc)
            self.root.after(0, lambda: self._on_split_error(_err))
            return
        self.root.after(0, lambda: self._on_split_done(results))

    def _on_split_progress(self, current: int, total: int):
        self.root.after(0, lambda: self._update_split_dialog(current, total))

    def _update_split_dialog(self, current: int, total: int):
        if not hasattr(self, "_split_dlg") or not self._split_dlg.winfo_exists():
            return
        self._bar_split.configure(value=current)
        self._lbl_split_prog.configure(text=f"{current} / {total}")

    def _on_split_done(self, results: list[dict]):
        if hasattr(self, "_split_dlg") and self._split_dlg.winfo_exists():
            self._split_dlg.destroy()

        errors = [r for r in results if "error" in r]
        ok = [r for r in results if "error" not in r]

        self.log.mark_step("拆分音轨", "done" if not errors else "done")
        self.log.set_step_data("拆分音轨", {
            "results": results,
            "error_count": len(errors),
        })
        self.log.add_entry("拆分音轨", "done",
                           f"{len(ok)} ok, {len(errors)} errors")
        self._refresh_toolbar()
        self._populate_split_tab(results)
        self._switch_to_next_tab("拆分音轨")

    def _on_split_error(self, error_msg: str):
        if hasattr(self, "_split_dlg") and self._split_dlg.winfo_exists():
            self._split_dlg.destroy()
        self.log.mark_step("拆分音轨", "failed")
        self.log.add_entry("拆分音轨", "error", error_msg)
        self._refresh_toolbar()
        messagebox.showerror("拆分失败", error_msg)

    # ═══ split tab display ════════════════════════════════════

    def _populate_split_tab(self, results: list[dict]):
        tab = self._tab_frames.get("拆分音轨")
        if tab is None:
            return
        for w in tab.winfo_children():
            w.destroy()

        if not results:
            tk.Label(tab, text="当前为空", fg="#aaa",
                     font=(config.CJK_FONT, 13)).place(relx=0.5, rely=0.5,
                                                         anchor="center")
            return

        canvas = tk.Canvas(tab, bg="#f5f5f5", highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        tab.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))

        self._photo_refs.clear()

        for i, r in enumerate(results):
            card = ttk.Frame(scroll_frame, relief="solid", borderwidth=1)
            card.pack(fill="x", padx=12, pady=6)

            # Header
            hdr = ttk.Frame(card); hdr.pack(fill="x", padx=10, pady=(8, 4))
            label = r.get("label", f"#{i}")
            if "error" in r:
                tk.Label(hdr, text=f"✗ {label} — 失败",
                         font=(config.CJK_FONT, 10, "bold"),
                         fg="#721c24").pack(side="left")
                tk.Label(hdr, text=r["error"], fg="#999",
                         font=(config.CJK_FONT, 8),
                         wraplength=500).pack(side="left", padx=8)
                continue

            dur = _fmt_time(int(r.get("duration", 0) * 1000))
            tk.Label(hdr, text=f"▸ {label}",
                     font=(config.CJK_FONT, 10, "bold")).pack(side="left")
            tk.Label(hdr, text=f"时长 {dur}",
                     font=(config.CJK_FONT, 8), fg="#888").pack(side="right")

            # Body: silent video + audio side by side
            body = ttk.Frame(card); body.pack(fill="x", padx=10, pady=(2, 8))

            # -- silent video --
            vid_frame = ttk.Frame(body, relief="groove", borderwidth=1)
            vid_frame.pack(side="left", padx=(0, 12))

            video_path = Path(r["video"])
            if video_path.exists():
                size_mb = video_path.stat().st_size / (1024 * 1024)
                tk.Label(vid_frame, text="无声视频",
                         font=(config.CJK_FONT, 9, "bold")).pack(pady=(4, 2))
                tk.Label(vid_frame, text=f"{video_path.name}\n{size_mb:.1f} MB",
                         font=(config.CJK_FONT, 7), fg="#888").pack()

                play_vid = tk.Button(
                    vid_frame, text="▶ 播放",
                    font=(config.CJK_FONT, 9),
                    command=lambda p=video_path: self._play_segment(p),
                )
                play_vid.pack(pady=(2, 6))
            else:
                tk.Label(vid_frame, text="无声视频\n(文件缺失)",
                         font=(config.CJK_FONT, 8), fg="#999").pack(pady=8)

            # -- audio track --
            aud_frame = ttk.Frame(body, relief="groove", borderwidth=1)
            aud_frame.pack(side="left")

            audio_path = Path(r["audio"])
            if audio_path.exists():
                size_kb = audio_path.stat().st_size / 1024
                tk.Label(aud_frame, text="音频轨",
                         font=(config.CJK_FONT, 9, "bold")).pack(pady=(4, 2))
                tk.Label(aud_frame, text=f"{audio_path.name}\n{size_kb:.0f} KB  16kHz mono",
                         font=(config.CJK_FONT, 7), fg="#888").pack()

                play_aud = tk.Button(
                    aud_frame, text="▶ 播放",
                    font=(config.CJK_FONT, 9),
                    command=lambda p=audio_path: self._play_segment(p),
                )
                play_aud.pack(pady=(2, 6))
            else:
                tk.Label(aud_frame, text="音频轨\n(文件缺失)",
                         font=(config.CJK_FONT, 8), fg="#999").pack(pady=8)

    # ═══ toolbar: 转换文字 ═════════════════════════════════════

    def _on_transcribe(self):
        # Collect audio sources
        audio_paths = self._collect_audio_paths()
        if not audio_paths:
            messagebox.showwarning("提示", "请先完成「拆分音轨」步骤。")
            return

        lang_label = self._trans_lang_var.get()
        lang_code = LANGUAGES.get(lang_label, "ja")

        self.log.mark_step("转换文字", "running")
        self.log.add_entry("转换文字", "start", f"lang={lang_label} files={len(audio_paths)}")
        self._refresh_toolbar()
        self._cancel_requested = False

        # Progress dialog
        total = len(audio_paths)
        self._asr_seg_count = 0
        self._asr_current_file = -1

        self._asr_dlg = tk.Toplevel(self.root)
        self._asr_dlg.title("转换文字")
        self._asr_dlg.geometry("420x260")
        self._asr_dlg.transient(self.root)
        self._asr_dlg.grab_set()
        self._asr_dlg.protocol("WM_DELETE_WINDOW", self._on_cancel_asr)
        self._asr_dlg.resizable(False, False)

        ttk.Label(self._asr_dlg, text="正在语音识别…",
                  font=(config.CJK_FONT, 11)).pack(pady=(12, 8))

        # ── 文件进度（上） ──
        f1 = ttk.Frame(self._asr_dlg); f1.pack(fill="x", padx=20, pady=4)
        ttk.Label(f1, text="文件进度：").pack(side="left")
        self._lbl_asr_prog = ttk.Label(f1, text=f"0 / {total}")
        self._lbl_asr_prog.pack(side="right")
        self._bar_asr = ttk.Progressbar(self._asr_dlg, length=380,
                                        mode="determinate", maximum=total)
        self._bar_asr.pack(padx=20)

        # ── 当前文件（下） ──
        f2 = ttk.Frame(self._asr_dlg); f2.pack(fill="x", padx=20, pady=(10, 2))
        ttk.Label(f2, text="当前文件：").pack(side="left")
        self._lbl_asr_file = ttk.Label(f2, text="等待中…")
        self._lbl_asr_file.pack(side="right")
        self._bar_asr_file = ttk.Progressbar(self._asr_dlg, length=380,
                                             mode="determinate", maximum=100)
        self._bar_asr_file.pack(padx=20)

        self._lbl_asr_seg = ttk.Label(self._asr_dlg, text="已识别: 0 句",
                                      font=(config.CJK_FONT, 9))
        self._lbl_asr_seg.pack(pady=(6, 0))

        ttk.Button(self._asr_dlg, text="取消",
                   command=self._on_cancel_asr).pack(pady=12)

        # Swith to the 转换文字 tab
        for idx in range(self._notebook.index("end")):
            if self._notebook.tab(idx, "text") == "转换文字":
                self._notebook.select(idx)
                break

        # Prep result area — write header, segments will stream in
        self._trans_text.configure(state="normal")
        self._trans_text.delete("1.0", "end")
        self._trans_text.insert("1.0", f"识别中… 语言: {lang_label}\n\n")
        self._trans_text.see("1.0")
        self._trans_text.configure(state="disabled")

        threading.Thread(
            target=self._run_asr,
            args=(audio_paths, lang_code),
            daemon=True,
        ).start()

    def _collect_audio_paths(self) -> list[Path]:
        """Collect audio files from split results, or extract from original."""
        split_data = self.log.step_data.get("拆分音轨", {})
        results = split_data.get("results", [])

        audio_paths: list[Path] = []
        if results:
            for r in results:
                if "error" not in r:
                    p = Path(r["audio"])
                    if p.exists():
                        audio_paths.append(p)
        else:
            # Fallback: extract audio from original video on the fly
            import tempfile, subprocess
            tmp = Path(tempfile.mktemp(suffix=".wav"))
            subprocess.run([
                "ffmpeg", "-y", "-i", self.log.video_path,
                "-vn", "-ar", "16000", "-ac", "1", str(tmp),
            ], check=True, capture_output=True)
            audio_paths.append(tmp)

        return audio_paths

    def _on_cancel_asr(self):
        self._cancel_requested = True

    def _run_asr(self, audio_paths: list[Path], lang_code: str):
        backend_label = self._trans_backend_var.get()
        backend = self._backend_map.get(backend_label, "auto")
        try:
            results = transcribe_all(
                audio_paths, language=lang_code,
                backend=backend,
                segment_cb=self._on_asr_segment,
                progress_cb=self._on_asr_progress,
                file_start_cb=self._on_asr_file_start,
                file_progress_cb=self._on_asr_file_progress,
                cancel_check=lambda: self._cancel_requested,
            )
        except Exception as e:
            err_msg = str(e)
            if "openai-whisper" in err_msg:
                err_msg += "\n\n请尝试切换到「faster-whisper (CPU)」引擎。"
            self.root.after(0, lambda: self._on_asr_error(err_msg))
            return
        self.root.after(0, lambda: self._on_asr_done(results, lang_code))

    def _on_asr_progress(self, current: int, total: int):
        self.root.after(0, lambda: self._update_asr_dialog(current, total))

    def _update_asr_dialog(self, current: int, total: int):
        if not hasattr(self, "_asr_dlg") or not self._asr_dlg.winfo_exists():
            return
        self._bar_asr.configure(value=current)
        self._lbl_asr_prog.configure(text=f"{current} / {total}")

    def _on_asr_file_start(self, file_idx: int, filename: str):
        """Called from worker thread when a new file starts processing."""
        self.root.after(0, lambda: self._update_asr_file_start(file_idx, filename))

    def _update_asr_file_start(self, file_idx: int, filename: str):
        if not hasattr(self, "_asr_dlg") or not self._asr_dlg.winfo_exists():
            return
        self._bar_asr_file.configure(value=0)
        self._lbl_asr_file.configure(text=f"{filename}  (0%)")

    def _on_asr_file_progress(self, file_idx: int, pct: int):
        """Called from worker thread with 0–100 % within the current file."""
        self.root.after(0, lambda: self._update_asr_file_progress(file_idx, pct))

    def _update_asr_file_progress(self, file_idx: int, pct: int):
        if not hasattr(self, "_asr_dlg") or not self._asr_dlg.winfo_exists():
            return
        self._bar_asr_file.configure(value=pct)
        # Update the filename label with percentage
        current_text = self._lbl_asr_file.cget("text")
        name = current_text.rsplit("  (", 1)[0] if "  (" in current_text else current_text
        self._lbl_asr_file.configure(text=f"{name}  ({pct}%)")

    def _on_asr_segment(self, file_idx: int, segment: dict):
        """Called from worker thread — schedule UI append."""
        self.root.after(0, lambda: self._append_segment(file_idx, segment))

    def _append_segment(self, file_idx: int, seg: dict):
        """Append one transcribed segment to the text widget (main thread)."""
        # File header on first segment of a new file
        if not hasattr(self, "_asr_current_file"):
            self._asr_current_file = -1
        if file_idx != self._asr_current_file:
            self._asr_current_file = file_idx
            src_name = Path(seg.get("source", "")).name
            self._trans_text.configure(state="normal")
            self._trans_text.insert("end",
                f"\n── {src_name} ──────────\n")
            self._trans_text.configure(state="disabled")

        start = seg["start"]
        ts = f"{int(start // 60)}:{start % 60:04.1f}"
        line = f"  [{ts}]  {seg['text']}\n"

        self._trans_text.configure(state="normal")
        self._trans_text.insert("end", line)
        self._trans_text.see("end")
        self._trans_text.configure(state="disabled")

        # Update live counter in progress dialog
        self._asr_seg_count += 1
        if hasattr(self, "_asr_dlg") and self._asr_dlg.winfo_exists():
            self._lbl_asr_seg.configure(text=f"已识别: {self._asr_seg_count} 句")

    def _on_asr_done(self, results: list[dict], lang_code: str):
        if hasattr(self, "_asr_dlg") and self._asr_dlg.winfo_exists():
            self._asr_dlg.destroy()

        errors = [r for r in results if "error" in r]
        ok = [r for r in results if "error" not in r]

        self.log.mark_step("转换文字", "done" if not errors else "done")
        self.log.set_step_data("转换文字", {
            "language": lang_code,
            "results": results,
        })
        self.log.add_entry("转换文字", "done",
                           f"{len(ok)} ok, {len(errors)} errors")
        self._refresh_toolbar()
        self._populate_transcribe_tab(results)
        self._switch_to_next_tab("转换文字")

        # Check if there are errors to report
        if errors:
            messagebox.showwarning(
                "识别完成",
                f"{len(ok)} 个文件识别成功，{len(errors)} 个失败。\n\n"
                + "\n".join(f"• {Path(e['source']).name}: {e['error']}"
                           for e in errors[:3]),
            )

    def _on_asr_error(self, error_msg: str):
        if hasattr(self, "_asr_dlg") and self._asr_dlg.winfo_exists():
            self._asr_dlg.destroy()
        self.log.mark_step("转换文字", "failed")
        self.log.add_entry("转换文字", "error", error_msg)
        self._refresh_toolbar()
        messagebox.showerror("转换失败", error_msg)

    def _populate_transcribe_tab(self, results: list[dict]):
        """Rewrite the text widget with nicely formatted transcription."""
        self._trans_text.configure(state="normal")
        self._trans_text.delete("1.0", "end")

        lang_label = self._trans_lang_var.get()
        total_segs = sum(len(r.get("segments", [])) for r in results
                         if "error" not in r)
        self._trans_text.insert("1.0",
            f"语言: {lang_label}  —  共 {total_segs} 句\n\n")

        for r in results:
            source = Path(r["source"]).name
            if "error" in r:
                self._trans_text.insert("end",
                    f"── {source} ── 失败 ──────────\n"
                    f"  {r['error']}\n\n")
                continue

            segs = r.get("segments", [])
            self._trans_text.insert("end",
                f"── {source}  ({len(segs)} 句) ──────────\n")

            for seg in segs:
                start = seg["start"]
                ts = f"{int(start // 60):d}:{start % 60:04.1f}"
                self._trans_text.insert("end", f"  [{ts}]  {seg['text']}\n")

            self._trans_text.insert("end", "\n")

        if not results:
            self._trans_text.insert("1.0", "（无识别结果）")

        self._trans_text.configure(state="disabled")

    # ═══ toolbar: 文本翻译 ═════════════════════════════════════

    def _on_translate(self):
        # Collect segments from ASR step
        asr_data = self.log.step_data.get("转换文字", {})
        asr_results = asr_data.get("results", [])
        if not asr_results:
            messagebox.showwarning("提示", "请先完成「转换文字」步骤。")
            return

        # Build audio_path → offset map from cutter + split data
        # When the video is cut into segments, ASR timestamps are relative
        # to each cut segment's audio file (starting at 0).  The cutter
        # stores the original-video offset in ``start``.
        cut_data = self.log.step_data.get("切割视频", {})
        cut_segs = cut_data.get("segments", [])
        audio_offset_map: dict[str, float] = {}
        if cut_segs:
            index_to_offset = {
                s["index"]: s.get("start", (s["index"] - 1) * 180.0)
                for s in cut_segs
            }
            split_data = self.log.step_data.get("拆分音轨", {})
            split_results = split_data.get("results", [])
            for r in split_results:
                label = r.get("label", "")
                if label.startswith("seg_"):
                    try:
                        idx = int(label.rsplit("_", 1)[-1])
                        offset = index_to_offset.get(idx, 0.0)
                        audio = r.get("audio", "")
                        if audio:
                            audio_offset_map[audio] = offset
                    except (ValueError, IndexError):
                        pass

        # Flatten segments (applying cut-segment offset where needed)
        segments = []
        for r in asr_results:
            if "error" in r:
                continue
            src = r.get("source", "")
            offset = audio_offset_map.get(src, 0.0)
            for seg in r.get("segments", []):
                segments.append({
                    "text": seg["text"],
                    "start": seg["start"] + offset,
                    "end": seg["end"] + offset,
                    "source": src,
                })

        if not segments:
            messagebox.showwarning("提示", "没有可翻译的文字片段。")
            return

        tl_label = self._tl_lang_var.get()
        target_lang = self._tl_lang_map.get(tl_label, "Chinese")
        src_lang_code = asr_data.get("language", "ja")
        src_name_map = {"ja": "Japanese", "en": "English", "zh": "Chinese", "ko": "Korean"}
        src_lang = src_name_map.get(src_lang_code, "Japanese")

        # ── read engine selection ────────────────────────────────
        engine = getattr(self, "_tl_engine_var", tk.StringVar(value="hy-mt2")).get()
        _engine_labels = {
            "hy-mt2": "Hy-MT2", "hy-mt2+polish": "Hy-MT2 + 润色",
            "hy-mt": "Hy-MT1.5", "ollama": "Ollama 直翻", "hy-mt+polish": "Hy-MT + 润色",
        }
        engine_label = _engine_labels.get(engine, engine)

        # ── determine Hy-MT model path ──────────────────────────
        _hymt_path = config.HYMT2_MODEL_PATH
        if engine in ("hy-mt", "hy-mt+polish"):
            _hymt_path = config.TRANSLATION_MODEL_PATH  # Hy-MT1.5
        # For engine == "ollama" we don't need Hy-MT at all

        # ── read Ollama model (user-selected) ───────────────────
        ollama_model = getattr(self, "_tl_ollama_model_var", None)
        ollama_model = ollama_model.get() if ollama_model else config.OLLAMA_MODEL

        self.log.mark_step("文本翻译", "running")
        self.log.add_entry("文本翻译", "start",
                           f"{src_lang}→{target_lang} engine={engine}"
                           f"{' model=' + ollama_model if engine != 'hy-mt' and engine != 'hy-mt2' else ''}"
                           f" segs={len(segments)}")
        self._refresh_toolbar()
        self._cancel_requested = False

        total = len(segments)

        # Progress dialog
        self._tl_dlg = tk.Toplevel(self.root)
        self._tl_dlg.title(f"文本翻译 - {engine_label}")
        self._tl_dlg.geometry("420x200")
        self._tl_dlg.transient(self.root)
        self._tl_dlg.grab_set()
        self._tl_dlg.protocol("WM_DELETE_WINDOW", self._on_cancel_translate)
        self._tl_dlg.resizable(False, False)

        ttk.Label(self._tl_dlg, text=f"正在翻译 ({engine_label})…",
                  font=(config.CJK_FONT, 11)).pack(pady=(12, 8))

        f1 = ttk.Frame(self._tl_dlg); f1.pack(fill="x", padx=20, pady=4)
        ttk.Label(f1, text="片段进度：").pack(side="left")
        self._lbl_tl_prog = ttk.Label(f1, text=f"0 / {total}")
        self._lbl_tl_prog.pack(side="right")
        self._bar_tl = ttk.Progressbar(self._tl_dlg, length=380,
                                        mode="determinate", maximum=total)
        self._bar_tl.pack(padx=20)

        self._lbl_tl_seg = ttk.Label(self._tl_dlg, text="已翻译: 0 句",
                                      font=(config.CJK_FONT, 9))
        self._lbl_tl_seg.pack(pady=(6, 0))

        ttk.Button(self._tl_dlg, text="取消",
                   command=self._on_cancel_translate).pack(pady=12)

        # Switch to translate tab
        for idx in range(self._notebook.index("end")):
            if self._notebook.tab(idx, "text") == "文本翻译":
                self._notebook.select(idx)
                break

        # Update source language display
        src_label_map = {"ja": "日本語", "en": "English", "zh": "中文", "ko": "한국어"}
        self._tl_src_label.configure(text=src_label_map.get(src_lang_code, src_lang_code))

        # Prep result area
        self._tl_text.configure(state="normal")
        self._tl_text.delete("1.0", "end")
        self._tl_text.insert("1.0",
            f"翻译中…  {src_lang} → {target_lang}  (共 {total} 句)\n\n")
        self._tl_text.see("1.0")
        self._tl_text.configure(state="disabled")

        threading.Thread(
            target=self._run_translate,
            args=(segments, target_lang, src_lang, total, engine, ollama_model, _hymt_path),
            daemon=True,
        ).start()

    def _on_cancel_translate(self):
        self._cancel_requested = True

    def _run_translate(self, segments, target_lang, src_lang, total, engine="hy-mt2+polish", ollama_model=None, hymt_path=None):
        # ── Hy-MT2 + Ollama polish (two-stage) ──────────────────
        if engine in ("hy-mt+polish", "hy-mt2+polish"):
            from ai_movie.translator import translate, polish_ollama
            try:
                # Stage 1: accurate translation via Hy-MT
                results = translate(
                    segments,
                    target_lang=target_lang,
                    src_lang=src_lang,
                    model_path=hymt_path,
                    progress_cb=self._on_tl_progress,
                    cancel_check=lambda: self._cancel_requested,
                )
                if self._cancel_requested:
                    return
                # Stage 2: colloquial polish via Ollama
                results = polish_ollama(
                    results,
                    model=ollama_model,
                    progress_cb=self._on_tl_progress,
                    segment_cb=lambda idx, text: self._on_tl_segment(segments, idx, text),
                    cancel_check=lambda: self._cancel_requested,
                )
            except Exception as exc:
                _err = str(exc)
                self.root.after(0, lambda: self._on_translate_error(_err))
                return
            self.root.after(0, lambda: self._on_translate_done(results, target_lang, engine, ollama_model))

        # ── Ollama direct translation ──────────────────────────
        elif engine == "ollama":
            from ai_movie.translator import translate_ollama
            try:
                results = translate_ollama(
                    segments,
                    target_lang=target_lang,
                    src_lang=src_lang,
                    model=ollama_model,
                    progress_cb=self._on_tl_progress,
                    segment_cb=lambda idx, text: self._on_tl_segment(segments, idx, text),
                    cancel_check=lambda: self._cancel_requested,
                )
            except Exception as exc:
                _err = str(exc)
                self.root.after(0, lambda: self._on_translate_error(_err))
                return
            self.root.after(0, lambda: self._on_translate_done(results, target_lang, engine, ollama_model))

        # ── Hy-MT only (Hy-MT2 or Hy-MT1.5) ────────────────────
        else:
            from ai_movie.translator import translate
            try:
                results = translate(
                    segments,
                    target_lang=target_lang,
                    src_lang=src_lang,
                    model_path=hymt_path,
                    progress_cb=self._on_tl_progress,
                    cancel_check=lambda: self._cancel_requested,
                )
            except Exception as exc:
                _err = str(exc)
                self.root.after(0, lambda: self._on_translate_error(_err))
                return
            self.root.after(0, lambda: self._on_translate_done(results, target_lang, engine, ollama_model))

    def _on_tl_progress(self, current: int, total: int):
        self.root.after(0, lambda: self._update_tl_dialog(current, total))

    def _update_tl_dialog(self, current: int, total: int):
        if not hasattr(self, "_tl_dlg") or not self._tl_dlg.winfo_exists():
            return
        self._bar_tl.configure(value=current)
        self._lbl_tl_prog.configure(text=f"{current} / {total}")
        self._lbl_tl_seg.configure(text=f"已翻译: {current} 句")

    def _on_tl_segment(self, segments, idx: int, translated: str):
        """Called from background thread — schedule UI update."""
        seg = segments[idx] if idx < len(segments) else {}
        self.root.after(0, lambda: self._append_tl_segment(seg, translated))

    def _append_tl_segment(self, seg: dict, translated: str):
        """Append one translated segment to the result text area (main thread)."""
        try:
            self._tl_text.configure(state="normal")
            start = seg.get("start", 0)
            ts = f"{int(start // 60)}:{start % 60:04.1f}"
            original = seg.get("text", "")
            self._tl_text.insert("end",
                f"  [{ts}]  {original}\n"
                f"         → {translated}\n\n")
            self._tl_text.see("end")
            self._tl_text.configure(state="disabled")
        except Exception:
            pass

    def _on_translate_done(self, results, target_lang, engine="hy-mt", ollama_model=None):
        if hasattr(self, "_tl_dlg") and self._tl_dlg.winfo_exists():
            self._tl_dlg.destroy()

        self.log.mark_step("文本翻译", "done")
        log_data = {
            "target_lang": target_lang,
            "engine": engine,
            "segments": results,
            "count": len(results),
        }
        if ollama_model:
            log_data["ollama_model"] = ollama_model
        self.log.set_step_data("文本翻译", log_data)
        self.log.add_entry("文本翻译", "done",
                           f"{len(results)} segments → {target_lang}")
        self._refresh_toolbar()
        self._populate_translate_tab(results, target_lang)
        self._switch_to_next_tab("文本翻译")

    def _on_translate_error(self, error_msg: str):
        if hasattr(self, "_tl_dlg") and self._tl_dlg.winfo_exists():
            self._tl_dlg.destroy()
        self.log.mark_step("文本翻译", "failed")
        self.log.add_entry("文本翻译", "error", error_msg)
        self._refresh_toolbar()
        messagebox.showerror("翻译失败", error_msg)

    def _populate_translate_tab(self, segments, target_lang):
        """Display original + translated text side by side, grouped by source file."""
        self._tl_text.configure(state="normal")
        self._tl_text.delete("1.0", "end")

        tl_label = self._tl_lang_var.get()
        engine = getattr(self, "_tl_engine_var", None)
        engine_label = engine.get() if engine else "hy-mt+polish"
        ollama_model = getattr(self, "_tl_ollama_model_var", None)
        ollama_model_str = ollama_model.get() if ollama_model else ""

        header = f"目标语言: {tl_label}  |  引擎: {engine_label}"
        if engine_label not in ("hy-mt", "hy-mt2") and ollama_model_str:
            header += f"  |  Ollama: {ollama_model_str}"
        header += f"  |  共 {len(segments)} 句\n\n"
        self._tl_text.insert("1.0", header)

        from collections import defaultdict
        by_source = defaultdict(list)
        for seg in segments:
            src_name = Path(seg.get("source", "?")).name
            by_source[src_name].append(seg)

        for src_name, segs in by_source.items():
            self._tl_text.insert("end",
                f"── {src_name}  ({len(segs)} 句) ──────────\n")

            for seg in segs:
                start = seg["start"]
                ts = f"{int(start // 60)}:{start % 60:04.1f}"
                original = seg["text"]
                translated = seg.get("text_translated", "（翻译失败）")
                self._tl_text.insert("end",
                    f"  [{ts}]  {original}\n"
                    f"         → {translated}\n\n")

        self._tl_text.configure(state="disabled")

    def _build_separate_tab(self, tab: ttk.Frame):
        """Tab showing separated vocals + background audio with play buttons."""
        # ── backend selector ─────────────────────────────────────
        bar = ttk.Frame(tab, padding=(10, 8))
        bar.pack(fill="x")

        ttk.Label(bar, text="分离引擎：",
                  font=(config.CJK_FONT, 10)).pack(side="left", padx=(0, 8))

        self._sep_backend_var = tk.StringVar(value="uvr")
        ttk.Radiobutton(
            bar, text="Demucs (GPU，稳定)",
            variable=self._sep_backend_var, value="demucs",
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            bar, text="UVR Mel-Band RoiFormer (GPU，需下载模型)",
            variable=self._sep_backend_var, value="uvr",
        ).pack(side="left")

        # ── result area ──────────────────────────────────────────
        self._sep_result_frame = ttk.Frame(tab)
        self._sep_result_frame.pack(expand=True, fill="both")

        placeholder = tk.Label(self._sep_result_frame, text="选择引擎后点击工具栏「人声分离」执行。\n默认：Demucs (GPU)",
                               fg="#aaa", font=(config.CJK_FONT, 13))
        placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def _build_generate_tab(self, tab: ttk.Frame):
        """Tab showing generated speech segments with play buttons."""
        # ── voice mode selector ────────────────────────────────────
        # ── Row 1: voice mode ─────────────────────────────────────
        mode_bar = ttk.Frame(tab, padding=(10, 6, 10, 2))
        mode_bar.pack(fill="x", side="top")
        ttk.Label(mode_bar, text="声音选择：",
                  font=(config.CJK_FONT, 10)).pack(side="left")
        self._tts_mode_var = tk.StringVar(value="gender")
        for text, val in [
            ("自动检测性别", "gender"),
            ("全部女声（中文女）", "female"),
            ("全部男声（中文男）", "male"),
            ("台湾柔和女声（妩媚）", "style"),
        ]:
            ttk.Radiobutton(mode_bar, text=text,
                            variable=self._tts_mode_var,
                            value=val).pack(side="left", padx=(8, 2))

        # ── Row 2: detection algorithm (only for "自动检测") ──────
        algo_bar = ttk.Frame(tab, padding=(10, 0, 10, 4))
        algo_bar.pack(fill="x", side="top")
        ttk.Label(algo_bar, text="检测算法：",
                  font=(config.CJK_FONT, 9), foreground="#555").pack(side="left")
        self._tts_algo_var = tk.StringVar(value="f0_per_seg")
        _algo_opts = [
            ("F0 逐片段（快速，推荐）",  "f0_per_seg"),
            ("F0 全局统计（稳定）",       "f0_global"),
            ("ECAPA 说话人日志（最准，较慢）", "ecapa"),
        ]
        for text, val in _algo_opts:
            ttk.Radiobutton(algo_bar, text=text,
                            variable=self._tts_algo_var,
                            value=val).pack(side="left", padx=(6, 2))

        # Grey out algorithm row when voice is fixed (not auto-detect)
        def _update_algo_state(*_):
            state = "normal" if self._tts_mode_var.get() == "gender" else "disabled"
            for w in algo_bar.winfo_children():
                try: w.configure(state=state)
                except Exception: pass
        self._tts_mode_var.trace_add("write", _update_algo_state)
        _update_algo_state()

        ttk.Separator(tab, orient="horizontal").pack(fill="x", padx=10)

        # Scrollable canvas for segment list
        self._gen_result_canvas = tk.Canvas(tab, bg="#f5f5f5", highlightthickness=0)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=self._gen_result_canvas.yview)
        self._gen_scroll_frame = ttk.Frame(self._gen_result_canvas)

        self._gen_scroll_frame.bind(
            "<Configure>",
            lambda e: self._gen_result_canvas.configure(
                scrollregion=self._gen_result_canvas.bbox("all")))
        self._gen_result_canvas.create_window(
            (0, 0), window=self._gen_scroll_frame, anchor="nw")
        self._gen_result_canvas.configure(yscrollcommand=scrollbar.set)

        self._gen_result_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mousewheel binding
        def _mw(event):
            self._gen_result_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self._gen_result_canvas.bind_all("<MouseWheel>", _mw)
        tab.bind("<Destroy>", lambda e: self._gen_result_canvas.unbind_all("<MouseWheel>"))

        placeholder = tk.Label(self._gen_scroll_frame, text="点击工具栏「人声生成」执行。",
                               fg="#aaa", font=(config.CJK_FONT, 13))
        placeholder.pack(expand=True, pady=80)

    def _populate_generate_tab(self, results):
        """Show generated speech segments — with voice selector, regen button, sync play."""
        # Store full results for per-segment regeneration
        self._gen_all_results = results

        for w in self._gen_scroll_frame.winfo_children():
            w.destroy()

        n_ok = sum(1 for r in results if r.get("audio"))
        if n_ok == 0:
            tk.Label(self._gen_scroll_frame, text="无生成结果",
                     fg="#aaa", font=(config.CJK_FONT, 13)).pack(expand=True, pady=80)
            return

        hdr = ttk.Frame(self._gen_scroll_frame)
        hdr.pack(fill="x", padx=16, pady=(12, 8))
        tk.Label(hdr, text=f"✓ 共生成 {n_ok}/{len(results)} 个片段",
                 font=(config.CJK_FONT, 12, "bold"), fg="#155724").pack(side="left")

        for seg_idx, seg in enumerate(results):
            has_audio = bool(seg.get("audio"))
            card = ttk.Frame(self._gen_scroll_frame, relief="solid", borderwidth=1)
            card.pack(fill="x", padx=16, pady=4, ipady=4)
            card.configure(style="TFrame")

            # ── header row ──────────────────────────────────────
            ch = ttk.Frame(card); ch.pack(fill="x", padx=12, pady=(6, 2))
            start = seg.get("start", 0); end = seg.get("end", 0)
            ts = f"{int(start//60)}:{start%60:04.1f} — {int(end//60)}:{end%60:04.1f}"
            status = "✓" if has_audio else ("✗" if seg.get("tts_error") else "–")
            tk.Label(ch, text=f"{status} 片段 {seg_idx+1}  [{ts}]",
                     font=(config.CJK_FONT, 9, "bold"),
                     fg="#155724" if has_audio else "#721c24").pack(side="left")
            tk.Label(ch, text=f"{end-start:.1f}s",
                     font=(config.CJK_FONT, 8), fg="#888").pack(side="right")

            # ── text rows ────────────────────────────────────────
            tf = ttk.Frame(card); tf.pack(fill="x", padx=12, pady=(2, 0))
            if seg.get("text"):
                tk.Label(tf, text=f"原文: {seg['text']}",
                         font=(config.CJK_FONT, 8), fg="#666",
                         wraplength=680, anchor="w", justify="left").pack(anchor="w")
            if seg.get("text_translated"):
                tk.Label(tf, text=f"译文: {seg['text_translated']}",
                         font=(config.CJK_FONT, 9), fg="#333",
                         wraplength=680, anchor="w", justify="left").pack(anchor="w")
            if seg.get("tts_error"):
                tk.Label(tf, text=f"错误: {seg['tts_error']}",
                         font=(config.CJK_FONT, 8), fg="#721c24").pack(anchor="w")

            # ── action row ───────────────────────────────────────
            af = ttk.Frame(card); af.pack(fill="x", padx=12, pady=(4, 6))

            # Voice selector (shows current voice used, editable before regen)
            detected = seg.get("tts_gender", "female")
            import ai_movie.tts as _t
            init_spk = _t._SFT_FEMALE_SPK if detected == "female" else _t._SFT_MALE_SPK
            voice_var = tk.StringVar(value=init_spk)
            ttk.Combobox(af, textvariable=voice_var,
                         values=[_t._SFT_FEMALE_SPK, _t._SFT_MALE_SPK],
                         width=8, state="readonly").pack(side="left", padx=(0, 6))

            # Regen button
            def _regen(idx=seg_idx, vv=voice_var):
                self._regenerate_one_segment(idx, vv.get())
            tk.Button(af, text="🔄 重新生成", font=(config.CJK_FONT, 9),
                      command=_regen).pack(side="left", padx=(0, 8))

            # Play + sync button (only if audio exists)
            if has_audio:
                audio_path = Path(seg["audio"])
                def _play_sync(p=audio_path, s=seg):
                    self._play_with_left_sync(p, s)
                tk.Button(af, text="▶ 播放（同步原视频）", font=(config.CJK_FONT, 9),
                          command=_play_sync).pack(side="right")

    def _play_with_left_sync(self, audio_path: Path, seg: dict):
        """Play a generated audio segment and seek the left video to its start time."""
        # Seek and play left video in sync
        try:
            dur_ms = self.left_player.get_duration_ms()
            if dur_ms and dur_ms > 0:
                start_ms = seg.get("start", 0) * 1000
                self.left_player.seek_absolute(start_ms / dur_ms)
                self.left_player.play()
                self._is_playing = True
                self._btn_play.configure(text="⏸")
        except Exception:
            pass
        # Open audio playback popup
        self._play_segment(audio_path)

    def _regenerate_one_segment(self, seg_idx: int, spk: str):
        """Regenerate a single segment with the chosen SFT speaker on the main thread."""
        import ai_movie.tts as tts_mod
        tts_mod._load_model()

        results = getattr(self, "_gen_all_results", None)
        if not results or seg_idx >= len(results):
            messagebox.showerror("错误", "找不到片段数据，请重新生成全部片段。")
            return

        seg = results[seg_idx]
        text = seg.get("text_translated", "").strip()
        if not text:
            messagebox.showwarning("提示", "此片段无译文，跳过。")
            return

        # Determine output path
        existing = seg.get("audio")
        if existing:
            out_path = Path(existing)
        else:
            out_dir = ensure_dir(WORKSPACE_DIR / "synthesized")
            out_path = out_dir / f"seg_{seg_idx + 1:04d}.wav"

        try:
            audio_np = tts_mod.call_tts(tts_mod._model, text, spk, None, "sft")
            import soundfile as sf
            sf.write(str(out_path), audio_np, tts_mod._model.sample_rate)
        except Exception as e:
            messagebox.showerror("重新生成失败", str(e))
            return

        # Update results
        results[seg_idx]["audio"] = str(out_path)
        results[seg_idx]["tts_gender"] = "female" if spk == tts_mod._SFT_FEMALE_SPK else "male"
        results[seg_idx].pop("tts_error", None)
        n_ok = sum(1 for r in results if r.get("audio"))
        self.log.set_step_data("人声生成", {"results": results, "ok": n_ok})

        # Refresh tab
        self._populate_generate_tab(results)
        messagebox.showinfo("重新生成完成",
            f"片段 {seg_idx + 1} 已用「{spk}」重新合成。")

    def _populate_separate_tab(self, vocals_path, bg_path):
        """Show separated audio files with play buttons."""
        for w in self._sep_result_frame.winfo_children():
            w.destroy()

        card = ttk.Frame(self._sep_result_frame, relief="solid", borderwidth=1)
        card.pack(fill="x", padx=24, pady=24, ipady=12)

        header = ttk.Frame(card); header.pack(fill="x", padx=16, pady=(12, 8))
        tk.Label(header, text="✓ 人声分离完成",
                 font=(config.CJK_FONT, 12, "bold"), fg="#155724").pack(side="left")

        # Vocals
        vf = ttk.Frame(card, relief="groove", borderwidth=1)
        vf.pack(side="left", padx=16, pady=(8, 16), ipadx=12, ipady=8)
        vp = Path(vocals_path)
        if vp.exists():
            tk.Label(vf, text="人声", font=(config.CJK_FONT, 10, "bold")).pack(pady=(8, 2))
            size_kb = vp.stat().st_size / 1024
            tk.Label(vf, text=f"{vp.name}\n{size_kb:.0f} KB",
                     font=(config.CJK_FONT, 8), fg="#888").pack()
            tk.Button(vf, text="▶ 播放", font=(config.CJK_FONT, 9),
                      command=lambda p=vp: self._play_segment(p)).pack(pady=(4, 8))

        # Background
        bf = ttk.Frame(card, relief="groove", borderwidth=1)
        bf.pack(side="left", padx=16, pady=(8, 16), ipadx=12, ipady=8)
        bp = Path(bg_path)
        if bp.exists():
            tk.Label(bf, text="背景音", font=(config.CJK_FONT, 10, "bold")).pack(pady=(8, 2))
            size_kb = bp.stat().st_size / 1024
            tk.Label(bf, text=f"{bp.name}\n{size_kb:.0f} KB",
                     font=(config.CJK_FONT, 8), fg="#888").pack()
            tk.Button(bf, text="▶ 播放", font=(config.CJK_FONT, 9),
                      command=lambda p=bp: self._play_segment(p)).pack(pady=(4, 8))

    # ═══ toolbar: 人声分离 ═════════════════════════════════════

    def _on_separate_vocals(self):
        ref_audio = self._get_full_audio()
        if ref_audio is None:
            messagebox.showwarning("提示", "请先加载视频文件。")
            return

        backend = self._sep_backend_var.get()
        backend_label = {"demucs": "Demucs (GPU)", "uvr": "UVR Mel-Band RoiFormer"}.get(backend, backend)

        self.log.mark_step("人声分离", "running")
        self.log.add_entry("人声分离", "start", f"backend={backend}")
        self._refresh_toolbar()
        self._cancel_requested = False

        self._sep_dlg = tk.Toplevel(self.root)
        self._sep_dlg.title(f"人声分离 — {backend_label}")
        self._sep_dlg.geometry("400x150")
        self._sep_dlg.transient(self.root)
        self._sep_dlg.grab_set()
        self._sep_dlg.resizable(False, False)

        ttk.Label(self._sep_dlg, text=f"正在分离人声与背景音…\n引擎：{backend_label}",
                  font=(config.CJK_FONT, 11)).pack(pady=(12, 8))
        self._bar_sep = ttk.Progressbar(self._sep_dlg, length=360,
                                         mode="indeterminate")
        self._bar_sep.pack(padx=20, pady=10)
        self._bar_sep.start(10)

        threading.Thread(target=self._run_separate, args=(ref_audio, backend), daemon=True).start()

    def _run_separate(self, ref_audio, backend="demucs"):
        import ai_movie.composer as composer
        try:
            sep = composer.separate_vocals(ref_audio, backend=backend)
        except Exception as exc:
            _err = str(exc)
            self.root.after(0, lambda: self._on_separate_error(_err))
            return
        self.root.after(0, lambda: self._on_separate_done(sep, backend))

    def _on_separate_done(self, sep, backend="demucs"):
        if hasattr(self, "_sep_dlg") and self._sep_dlg.winfo_exists():
            self._sep_dlg.destroy()
        self.log.mark_step("人声分离", "done")
        self.log.set_step_data("人声分离", {
            "vocals": str(sep["vocals"]),
            "background": str(sep["background"]),
        })
        self.log.add_entry("人声分离", "done",
                           f"backend={backend} vocals={sep['vocals']}, bg={sep['background']}")
        self._refresh_toolbar()
        self._populate_separate_tab(sep["vocals"], sep["background"])
        self._switch_to_next_tab("人声分离")
        messagebox.showinfo("分离完成", f"人声：{sep['vocals']}\n背景音：{sep['background']}")

    def _on_separate_error(self, error_msg):
        if hasattr(self, "_sep_dlg") and self._sep_dlg.winfo_exists():
            self._sep_dlg.destroy()
        self.log.mark_step("人声分离", "failed")
        self.log.add_entry("人声分离", "error", error_msg)
        self._refresh_toolbar()
        messagebox.showerror("分离失败", error_msg)

    # ═══ toolbar: 人声生成 ═════════════════════════════════════

    def _on_generate_speech(self):
        segments = self._get_translated_segments()
        vocals_path = self._get_vocals_path()
        if segments is None or vocals_path is None:
            return

        self.log.mark_step("人声生成", "running")
        self.log.add_entry("人声生成", "start", f"segs={len(segments)}")
        self._refresh_toolbar()
        self._cancel_requested = False

        total = len(segments)
        self._gen_dlg = tk.Toplevel(self.root)
        self._gen_dlg.title("人声生成")
        self._gen_dlg.geometry("420x200")
        self._gen_dlg.transient(self.root)
        self._gen_dlg.grab_set()
        self._gen_dlg.protocol("WM_DELETE_WINDOW", self._on_cancel_generate)
        self._gen_dlg.resizable(False, False)

        ttk.Label(self._gen_dlg, text="正在合成语音…",
                  font=(config.CJK_FONT, 11)).pack(pady=(12, 8))
        f1 = ttk.Frame(self._gen_dlg); f1.pack(fill="x", padx=20, pady=4)
        ttk.Label(f1, text="片段进度：").pack(side="left")
        self._lbl_gen_prog = ttk.Label(f1, text=f"0 / {total}")
        self._lbl_gen_prog.pack(side="right")
        self._bar_gen = ttk.Progressbar(self._gen_dlg, length=380,
                                         mode="determinate", maximum=total)
        self._bar_gen.pack(padx=20)
        ttk.Button(self._gen_dlg, text="取消",
                   command=self._on_cancel_generate).pack(pady=12)

        out_dir = ensure_dir(WORKSPACE_DIR / "synthesized")
        voice_mode = getattr(self, "_tts_mode_var", None)
        voice_mode = voice_mode.get() if voice_mode else "gender"
        self._gen_voice_mode = voice_mode

        # Resolve which model will be used.  SFT (CosyVoice-300M) runs
        # in-process on the main thread; Qwen models (CosyVoice2/3) run in an
        # isolated subprocess (transformers 4.51.3) — see ai_movie/tts_worker.py.
        # "台湾柔和女声" needs an instruct2-capable model (CosyVoice3/2), not SFT.
        import ai_movie.tts as tts_mod
        # Gender-auto also prefers CosyVoice3 so the female voice can use the
        # soft/Taiwanese style reference (male stays Mandarin). Falls back to
        # SFT (中文女/中文男) automatically when CosyVoice3 isn't installed.
        prefer = "cosyvoice3" if voice_mode in ("style", "gender") else None
        self._gen_choice = tts_mod.resolve_model_choice(prefer)
        if self._gen_choice == "sft":
            tts_mod._load_model(prefer=prefer)   # main-thread in-process load
        algo = getattr(self, "_tts_algo_var", None)
        self._gen_algo = algo.get() if algo else "f0_per_seg"
        self._gen_ecapa_result = None   # filled below for ecapa mode

        if voice_mode == "female":
            ref_audio, ref_text, ref_method = tts_mod._SFT_FEMALE_SPK, None, "sft"
            self._lbl_gen_prog.configure(text="固定使用：中文女")
        elif voice_mode == "male":
            ref_audio, ref_text, ref_method = tts_mod._SFT_MALE_SPK, None, "sft"
            self._lbl_gen_prog.configure(text="固定使用：中文男")
        elif voice_mode == "style":
            # Taiwanese / soft female via CosyVoice3 instruct2 (fixed for all segs)
            ref_audio, ref_text, ref_method = tts_mod.prepare_reference(
                vocals_path, mode="style", cache_dir=out_dir)
            if self._gen_choice == "sft":
                self.root.after(0, lambda: messagebox.showwarning(
                    "提示", "台湾柔和女声需要 CosyVoice3/2 模型，"
                    "当前仅有 SFT，已退回音色克隆模式。"))
            self._lbl_gen_prog.configure(text="台湾柔和女声（instruct2）…")
        else:
            ref_audio, ref_text, ref_method = tts_mod.prepare_reference(
                vocals_path, mode="gender", cache_dir=out_dir)
            algo_label = {"f0_per_seg": "F0逐片段", "f0_global": "F0全局",
                          "ecapa": "ECAPA日志"}.get(self._gen_algo, self._gen_algo)
            self._lbl_gen_prog.configure(text=f"自动检测（{algo_label}）…")

        # For ECAPA mode, pre-compute diarization in background then start TTS
        self._gen_segments = segments
        self._gen_vocals_path = vocals_path
        self._gen_ref_audio = ref_audio
        self._gen_ref_text = ref_text
        self._gen_ref_method = ref_method
        self._gen_output_dir = out_dir
        self._gen_idx = 0
        self._gen_results: list[dict] = []
        self._gen_tts_mod = tts_mod

        # Qwen models (CosyVoice2/3, any non-SFT method) synthesize in an
        # isolated subprocess; batch the whole run there instead of the
        # in-process per-segment root.after() loop.
        if ref_method != "sft":
            self._run_isolated_generate()
            return

        if voice_mode == "gender" and self._gen_algo == "ecapa":
            self._lbl_gen_prog.configure(text="ECAPA：正在分析说话人（约30-60秒）…")
            def _run_ecapa():
                try:
                    result = tts_mod.build_ecapa_gender_map(
                        vocals_path,
                        progress_cb=lambda msg: self.root.after(
                            0, lambda m=msg: self._lbl_gen_prog.configure(text=m)))
                    self.root.after(0, lambda r=result: self._on_ecapa_done(r))
                except Exception as e:
                    self.root.after(0, lambda: self._lbl_gen_prog.configure(
                        text=f"ECAPA 失败，改用F0逐片段: {e}"))
                    self.root.after(0, self._start_tts_loop)
            threading.Thread(target=_run_ecapa, daemon=True).start()
        else:
            self.root.after(100, self._start_tts_loop)

    def _on_ecapa_done(self, result):
        self._gen_ecapa_result = result
        n_spk = len(result.get("gender", {}))
        gmap = result.get("gender", {})
        summary = "、".join(f"说话人{k}={v}" for k, v in gmap.items())
        self._lbl_gen_prog.configure(text=f"ECAPA完成（{n_spk}人：{summary}）")
        self.root.after(100, self._start_tts_loop)

    def _start_tts_loop(self):
        """Kick off the per-segment TTS root.after() loop."""
        if not self._cancel_requested:
            self.root.after(0, self._process_next_generate)

    def _run_isolated_generate(self):
        """Synthesize all segments via the isolated CosyVoice2/3 subprocess.

        Runs in a background thread (the subprocess owns the main-thread
        requirement); progress and completion are marshalled back onto the
        Tk main thread with ``root.after``.
        """
        tts_mod = self._gen_tts_mod
        segments = self._gen_segments
        seg_texts = [(i, s.get("text_translated", "").strip())
                     for i, s in enumerate(segments)]
        total = len(segments)

        def _progress(done, _total):
            self.root.after(0, lambda d=done: (
                self._bar_gen.configure(value=d),
                self._lbl_gen_prog.configure(text=f"{d} / {total}（隔离进程合成）"))
                if hasattr(self, "_gen_dlg") and self._gen_dlg.winfo_exists() else None)

        def _worker():
            try:
                # Gender-auto with CosyVoice3: route each segment's reference by
                # detected gender (female → soft/Taiwanese style ref, male →
                # Mandarin) and record the gender for person-anchoring.
                seg_refs = None
                seg_genders: dict[int, str] = {}
                if getattr(self, "_gen_voice_mode", "gender") == "gender":
                    fem_ref = tts_mod.prepare_reference(
                        self._gen_vocals_path, mode="style", cache_dir=self._gen_output_dir)
                    if tts_mod._MALE_REF_WAV.exists():
                        male_ref = (str(tts_mod._MALE_REF_WAV), None, "cross_lingual")
                    else:
                        male_ref = (tts_mod.trim_reference_audio(self._gen_vocals_path), None, "cross_lingual")
                    algo = getattr(self, "_gen_algo", "f0_per_seg")
                    last = "female"
                    seg_refs = {}
                    for i, s in enumerate(segments):
                        if algo == "ecapa" and getattr(self, "_gen_ecapa_result", None):
                            g = tts_mod.lookup_ecapa_gender(s, self._gen_ecapa_result, fallback=last)
                        elif algo == "f0_global":
                            g = tts_mod.detect_gender_global_f0(s, fallback=last)
                        else:
                            g = tts_mod.detect_gender_from_segment(s, fallback=last)
                        last = g
                        seg_genders[i] = g
                        seg_refs[i] = fem_ref if g == "female" else male_ref

                items = tts_mod.run_isolated_synthesis(
                    seg_texts, self._gen_choice,
                    self._gen_ref_audio, self._gen_ref_text, self._gen_ref_method,
                    self._gen_output_dir,
                    progress_cb=_progress,
                    cancel_check=lambda: self._cancel_requested,
                    seg_refs=seg_refs,
                )
                results = []
                for i, seg in enumerate(segments):
                    it = items.get(i, {})
                    r = {**seg, "audio": it.get("audio")}
                    if i in seg_genders:
                        r["tts_gender"] = seg_genders[i]
                    if it.get("tts_error") or it.get("error"):
                        r["tts_error"] = it.get("tts_error") or it.get("error")
                    results.append(r)
                self.root.after(0, lambda: self._on_generate_done(results))
            except Exception as exc:
                self.root.after(0, lambda e=exc: self._on_generate_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_cancel_generate(self):
        self._cancel_requested = True

    def _process_next_generate(self):
        if self._cancel_requested or not hasattr(self, "_gen_dlg") or not self._gen_dlg.winfo_exists():
            return

        mod = self._gen_tts_mod
        idx = self._gen_idx
        segments = self._gen_segments
        total = len(segments)

        if idx >= total:
            self._on_generate_done(self._gen_results)
            return

        # Update progress
        self._bar_gen.configure(value=idx)

        seg = segments[idx]
        text = seg.get("text_translated", "").strip()
        if not text:
            self._gen_results.append({**seg, "audio": None})
            self._gen_idx = idx + 1
            self.root.after(50, self._process_next_generate)
            return

        # Per-segment speaker selection
        if self._gen_ref_method == "sft":
            if getattr(self, "_gen_voice_mode", "gender") == "gender":
                last = getattr(self, "_gen_last_gender", "female")
                algo = getattr(self, "_gen_algo", "f0_per_seg")
                if algo == "ecapa" and getattr(self, "_gen_ecapa_result", None):
                    seg_gender = mod.lookup_ecapa_gender(
                        seg, self._gen_ecapa_result, fallback=last)
                elif algo == "f0_global":
                    seg_gender = mod.detect_gender_global_f0(seg, fallback=last)
                else:  # f0_per_seg (default)
                    seg_gender = mod.detect_gender_from_segment(seg, fallback=last)
                self._gen_last_gender = seg_gender
                spk = mod._SFT_FEMALE_SPK if seg_gender == "female" else mod._SFT_MALE_SPK
            else:
                spk = self._gen_ref_audio
                seg_gender = "female" if spk == mod._SFT_FEMALE_SPK else "male"
            ref, ref_text, ref_method = spk, None, "sft"
        else:
            ref, ref_text, ref_method = self._gen_ref_audio, self._gen_ref_text, self._gen_ref_method
            seg_gender = "female"

        self._lbl_gen_prog.configure(
            text=f"{idx + 1} / {total}  ({'女声' if seg_gender == 'female' else '男声'})")

        # Run inference on main thread
        try:
            import soundfile as sf
            audio_np = mod.call_tts(
                mod._model, text, ref, ref_text, ref_method,
            )
            out_path = str(self._gen_output_dir / f"seg_{idx + 1:04d}.wav")
            sf.write(out_path, audio_np, mod._model.sample_rate)
            self._gen_results.append({**seg, "audio": out_path})
        except Exception as exc:
            self._gen_results.append({**seg, "audio": None, "tts_error": str(exc)})

        self._gen_idx = idx + 1
        # Schedule next segment
        self.root.after(50, self._process_next_generate)

    def _on_generate_done(self, results):
        if hasattr(self, "_gen_dlg") and self._gen_dlg.winfo_exists():
            self._gen_dlg.destroy()
        ok = sum(1 for r in results if r.get("audio"))
        self.log.mark_step("人声生成", "done")
        self.log.set_step_data("人声生成", {"results": results, "ok": ok})
        self.log.add_entry("人声生成", "done", f"{ok}/{len(results)} segments")
        self._refresh_toolbar()
        self._populate_generate_tab(results)
        self._switch_to_next_tab("人声生成")
        messagebox.showinfo("生成完成", f"语音生成完成：{ok}/{len(results)} 个片段")

    def _on_generate_error(self, error_msg):
        if hasattr(self, "_gen_dlg") and self._gen_dlg.winfo_exists():
            self._gen_dlg.destroy()
        self.log.mark_step("人声生成", "failed")
        self.log.add_entry("人声生成", "error", error_msg)
        self._refresh_toolbar()
        messagebox.showerror("生成失败", error_msg)

    # ═══ toolbar: 重新混音 ═════════════════════════════════════

    def _on_remix_audio(self):
        bg_path = self._get_background_path()
        gen_data = self.log.step_data.get("人声生成", {})
        gen_results = gen_data.get("results", [])
        if bg_path is None or not gen_results:
            messagebox.showwarning("提示", "请先完成「人声分离」和「人声生成」。")
            return

        self.log.mark_step("重新混音", "running")
        self.log.add_entry("重新混音", "start")
        self._refresh_toolbar()

        self._mix_dlg = tk.Toplevel(self.root)
        self._mix_dlg.title("重新混音")
        self._mix_dlg.geometry("360x140")
        self._mix_dlg.transient(self.root)
        self._mix_dlg.grab_set()
        self._mix_dlg.resizable(False, False)

        ttk.Label(self._mix_dlg, text="正在混音…",
                  font=(config.CJK_FONT, 11)).pack(pady=(12, 8))
        self._bar_mix = ttk.Progressbar(self._mix_dlg, length=320,
                                         mode="indeterminate")
        self._bar_mix.pack(padx=20, pady=10)
        self._bar_mix.start(10)

        threading.Thread(target=self._run_remix,
                         args=(gen_results, bg_path), daemon=True).start()

    def _run_remix(self, gen_results, bg_path):
        import ai_movie.composer as composer
        try:
            out_dir = ensure_dir(WORKSPACE_DIR / "synthesized")
            mixed_path = out_dir / "final_audio.wav"
            # Pass full segment dicts — mix_audio uses start/end for time alignment
            composer.mix_audio(gen_results, bg_path, mixed_path)
        except Exception as exc:
            _err = str(exc)
            self.root.after(0, lambda: self._on_remix_error(_err))
            return
        self.root.after(0, lambda: self._on_remix_done(mixed_path))

    def _on_remix_done(self, mixed_path):
        if hasattr(self, "_mix_dlg") and self._mix_dlg.winfo_exists():
            self._mix_dlg.destroy()
        self.log.mark_step("重新混音", "done")
        self.log.set_step_data("重新混音", {"mixed_audio": str(mixed_path)})
        self.log.add_entry("重新混音", "done", str(mixed_path))
        self._refresh_toolbar()
        self._populate_remix_tab(mixed_path)
        self._switch_to_next_tab("重新混音")
        messagebox.showinfo("混音完成", f"混音完成：\n{mixed_path}")

    def _on_remix_error(self, error_msg):
        if hasattr(self, "_mix_dlg") and self._mix_dlg.winfo_exists():
            self._mix_dlg.destroy()
        self.log.mark_step("重新混音", "failed")
        self.log.add_entry("重新混音", "error", error_msg)
        self._refresh_toolbar()
        messagebox.showerror("混音失败", error_msg)

    def _populate_remix_tab(self, mixed_path: Path):
        """Show mixed audio result in the 重新混音 tab."""
        tab = self._tab_frames.get("重新混音")
        if tab is None:
            return
        for w in tab.winfo_children():
            w.destroy()

        f = ttk.Frame(tab, padding=20)
        f.pack(expand=True)

        tk.Label(f, text="✓ 混音完成", font=(config.CJK_FONT, 14, "bold"),
                 fg="#155724").pack(pady=(0, 12))

        # File info
        info_frame = ttk.LabelFrame(f, text="输出文件", padding=10)
        info_frame.pack(fill="x", pady=(0, 12))
        p = Path(mixed_path)
        size_mb = p.stat().st_size / 1e6 if p.exists() else 0
        import soundfile as sf
        try:
            info = sf.info(str(p))
            dur = f"{info.duration:.1f}s  {info.samplerate}Hz  {'单声道' if info.channels==1 else '立体声'}"
        except Exception:
            dur = "—"
        tk.Label(info_frame, text=str(p), font=(config.MONO_FONT, 8),
                 fg="#555", wraplength=500).pack(anchor="w")
        tk.Label(info_frame, text=f"大小：{size_mb:.1f} MB    时长：{dur}",
                 font=(config.CJK_FONT, 9), fg="#666").pack(anchor="w", pady=(4, 0))

        tk.Button(f, text="▶  播放混音", font=(config.CJK_FONT, 11),
                  command=lambda: self._play_segment(p)).pack(pady=8)

    # ═══ helper methods for audio pipeline ═══════════════════════

    def _get_reference_audio(self):
        """Get original audio from split step or video."""
        split_data = self.log.step_data.get("拆分音轨", {})
        split_results = split_data.get("results", [])
        if split_results:
            for r in split_results:
                if "error" not in r:
                    p = Path(r["audio"])
                    if p.exists():
                        return p
        if self.log.video_path:
            import tempfile
            tmp = Path(tempfile.mktemp(suffix=".wav"))
            subprocess.run([
                "ffmpeg", "-y", "-i", self.log.video_path,
                "-vn", "-ar", "24000", "-ac", "1", str(tmp),
            ], check=True, capture_output=True)
            return tmp
        messagebox.showwarning("提示", "未找到参考音频，请先完成「拆分音轨」。")
        return None

    def _get_vocals_path(self):
        sep_data = self.log.step_data.get("人声分离", {})
        path = sep_data.get("vocals")
        if path and Path(path).exists():
            return Path(path)
        messagebox.showwarning("提示", "请先完成「人声分离」。")
        return None

    def _get_background_path(self):
        sep_data = self.log.step_data.get("人声分离", {})
        path = sep_data.get("background")
        if path and Path(path).exists():
            return Path(path)
        messagebox.showwarning("提示", "请先完成「人声分离」。")
        return None

    def _get_full_audio(self) -> Path | None:
        """Return a WAV of the **full** original video audio.

        Unlike ``_get_reference_audio``, which may return a single cut
        segment's audio when the video was split, this always extracts
        audio from the original source file.  Use this for Demucs
        background separation so the background length matches the
        full video.

        The audio is saved to a stable workspace path so Demucs outputs
        (vocals.wav / background.wav) land in a known location and are
        not overwritten by other projects.
        """
        if self.log.video_path:
            out_dir = ensure_dir(WORKSPACE_DIR / "separated")
            # Use video hash to avoid cross-project contamination
            import hashlib
            vid_hash = hashlib.md5(str(self.log.video_path).encode()).hexdigest()[:8]
            tmp = out_dir / f"full_audio_{vid_hash}.wav"

            # Only re-extract if the file doesn't exist or is stale
            if not tmp.exists():
                # Keep native sample rate for Demucs quality — don't downsample!
                # Demucs needs 44100 Hz for best separation; 16000 Hz loses
                # critical high-frequency content that distinguishes music.
                subprocess.run([
                    "ffmpeg", "-y", "-i", self.log.video_path,
                    "-vn", "-ac", "1", str(tmp),
                ], check=True, capture_output=True)
            return tmp
        return None

    def _get_translated_segments(self):
        tl_data = self.log.step_data.get("文本翻译", {})
        segments = tl_data.get("segments", [])
        if not segments:
            messagebox.showwarning("提示", "请先完成「文本翻译」。")
            return None
        return segments

    # ═══ toolbar: 合成音轨 (一键完成上面三步) ═══════════════════

    def _on_synthesize_audio(self):
        """One-click: runs 人声分离 → 人声生成 → 重新混音 in sequence."""
        segments = self._get_translated_segments()
        ref_audio = self._get_full_audio()  # full audio for Demucs background
        if segments is None or ref_audio is None:
            return

        self.log.mark_step("合成音轨", "running")
        self.log.add_entry("合成音轨", "start", f"segs={len(segments)}")
        self._refresh_toolbar()
        self._cancel_requested = False

        self._syn_dlg = tk.Toplevel(self.root)
        self._syn_dlg.title("合成音轨（一键）")
        self._syn_dlg.geometry("420x180")
        self._syn_dlg.transient(self.root)
        self._syn_dlg.grab_set()
        self._syn_dlg.protocol("WM_DELETE_WINDOW", self._on_cancel_synthesize)
        self._syn_dlg.resizable(False, False)

        ttk.Label(self._syn_dlg, text="正在合成配音（一键完成三步）…",
                  font=(config.CJK_FONT, 11)).pack(pady=(12, 8))
        self._lbl_syn_stage = ttk.Label(self._syn_dlg, text="准备中…",
                                         font=(config.CJK_FONT, 9))
        self._lbl_syn_stage.pack(pady=4)
        self._bar_syn = ttk.Progressbar(self._syn_dlg, length=380,
                                         mode="indeterminate")
        self._bar_syn.pack(padx=20, pady=10)
        self._bar_syn.start(10)

        ttk.Button(self._syn_dlg, text="取消",
                   command=self._on_cancel_synthesize).pack(pady=8)

        threading.Thread(target=self._run_synthesize,
                         args=(segments, ref_audio), daemon=True).start()

    def _on_cancel_synthesize(self):
        self._cancel_requested = True

    def _update_syn_stage(self, text: str):
        self.root.after(0, lambda t=text: (
            self._lbl_syn_stage.configure(text=t)
            if hasattr(self, "_syn_dlg") and self._syn_dlg.winfo_exists() else None
        ))

    def _run_synthesize(self, segments, ref_audio):
        """Stage 1 only (background thread): Demucs vocal separation.
        Hands off to main thread for Stage 2 (TTS requires main thread).
        """
        import ai_movie.composer as composer
        try:
            self._update_syn_stage("Step 1/3: 分离人声与背景音…")
            sep = composer.separate_vocals(ref_audio, backend=self._sep_backend_var.get())
            if self._cancel_requested:
                return
            self.log.mark_step("人声分离", "done")
            self.log.set_step_data("人声分离", {
                "vocals": str(sep["vocals"]), "background": str(sep["background"])})
            self.root.after(0, lambda: self._populate_separate_tab(
                str(sep["vocals"]), str(sep["background"])))
        except Exception as exc:
            _err = str(exc)
            self.root.after(0, lambda: self._on_synthesize_error(_err))
            return
        # Stage 2 must run on main thread
        self.root.after(0, lambda: self._start_syn_tts(segments, sep))

    def _start_syn_tts(self, segments, sep):
        """Called on main thread: initialize TTS loop for 合成音轨."""
        if self._cancel_requested:
            return
        import ai_movie.tts as tts_mod
        self._update_syn_stage("Step 2/3: 合成语音…")
        mode = getattr(self, "_tts_mode_var", None)
        mode = mode.get() if mode else "gender"
        self._gen_voice_mode = mode  # per-segment loop reads this
        # 台湾柔和女声 uses instruct2 (CosyVoice3/2 only), not SFT.
        # SFT loads in-process; Qwen models synthesize in an isolated subprocess.
        prefer = "cosyvoice3" if mode == "style" else None
        self._gen_choice = tts_mod.resolve_model_choice(prefer)
        if self._gen_choice == "sft":
            tts_mod._load_model(prefer=prefer)
        out_dir = ensure_dir(WORKSPACE_DIR / "synthesized")
        ref_audio, ref_text, ref_method = tts_mod.prepare_reference(
            sep["vocals"], mode=mode, cache_dir=out_dir
        )
        self._syn_segments = segments
        self._syn_sep = sep
        self._syn_vocals_path = sep["vocals"]  # kept for per-segment gender detection
        self._syn_vocals_ref = ref_audio
        self._syn_ref_text = ref_text
        self._syn_ref_method = ref_method
        self._syn_output_dir = out_dir
        self._syn_idx = 0
        self._syn_results: list[dict] = []
        self._syn_tts_mod = tts_mod
        if ref_method != "sft":
            self._run_isolated_syn_tts()
            return
        self.root.after(50, self._process_next_syn_segment)

    def _run_isolated_syn_tts(self):
        """合成音轨: batch Qwen (CosyVoice2/3) synthesis via the subprocess."""
        tts_mod = self._syn_tts_mod
        segments = self._syn_segments
        seg_texts = [(i, s.get("text_translated", "").strip())
                     for i, s in enumerate(segments)]
        total = len(segments)

        def _progress(done, _total):
            self.root.after(0, lambda d=done: self._update_syn_stage(
                f"Step 2/3: 合成语音…（隔离进程 {d}/{total}）"))

        def _worker():
            try:
                items = tts_mod.run_isolated_synthesis(
                    seg_texts, self._gen_choice,
                    self._syn_vocals_ref, self._syn_ref_text, self._syn_ref_method,
                    self._syn_output_dir,
                    progress_cb=_progress,
                    cancel_check=lambda: self._cancel_requested,
                )
                results = []
                for i, seg in enumerate(segments):
                    it = items.get(i, {})
                    r = {**seg, "audio": it.get("audio")}
                    if it.get("tts_error"):
                        r["tts_error"] = it["tts_error"]
                    results.append(r)
                self._syn_results = results
                self.root.after(0, self._finish_syn_tts)
            except Exception as exc:
                self.root.after(0, lambda e=exc: self._on_synthesize_error(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _process_next_syn_segment(self):
        """Main-thread TTS loop for 合成音轨 (mirrors _process_next_generate)."""
        if self._cancel_requested or not hasattr(self, "_syn_dlg") or not self._syn_dlg.winfo_exists():
            return
        idx = self._syn_idx
        segments = self._syn_segments
        total = len(segments)
        if idx >= total:
            self._finish_syn_tts()
            return
        seg = segments[idx]
        text = seg.get("text_translated", "").strip()
        if not text:
            self._syn_results.append({**seg, "audio": None})
            self._syn_idx = idx + 1
            self.root.after(50, self._process_next_syn_segment)
            return
        mod = self._syn_tts_mod

        # Per-segment speaker selection
        if self._syn_ref_method == "sft":
            if getattr(self, "_gen_voice_mode", "gender") == "gender":
                last = getattr(self, "_syn_last_gender", "female")
                seg_gender = mod.detect_gender_from_segment(seg, fallback=last)
                self._syn_last_gender = seg_gender
                spk = mod._SFT_FEMALE_SPK if seg_gender == "female" else mod._SFT_MALE_SPK
            else:
                spk = self._syn_vocals_ref
                seg_gender = "female" if spk == mod._SFT_FEMALE_SPK else "male"
            ref, ref_text, ref_method = spk, None, "sft"
        else:
            ref, ref_text, ref_method = self._syn_vocals_ref, self._syn_ref_text, self._syn_ref_method
            seg_gender = "female"

        self._update_syn_stage(
            f"Step 2/3: 合成语音… ({idx + 1}/{total})  {'女声' if seg_gender == 'female' else '男声'}")

        try:
            import soundfile as sf
            audio_np = mod.call_tts(
                mod._model, text, ref, ref_text, ref_method,
            )
            out_path = str(self._syn_output_dir / f"seg_{idx + 1:04d}.wav")
            sf.write(out_path, audio_np, mod._model.sample_rate)
            self._syn_results.append({**seg, "audio": out_path})
        except Exception as exc:
            self._syn_results.append({**seg, "audio": None, "tts_error": str(exc)})
        self._syn_idx = idx + 1
        self.root.after(50, self._process_next_syn_segment)

    def _finish_syn_tts(self):
        """Called on main thread after all TTS segments; kicks off Stage 3 mixing."""
        results = self._syn_results
        sep = self._syn_sep
        ok = sum(1 for r in results if r.get("audio"))
        self.log.mark_step("人声生成", "done")
        self.log.set_step_data("人声生成", {"results": results, "ok": ok})
        self._populate_generate_tab(results)
        self._update_syn_stage("Step 3/3: 混音…")
        threading.Thread(target=self._run_syn_mix, args=(results, sep), daemon=True).start()

    def _run_syn_mix(self, results, sep):
        """Stage 3 (background thread): mix synthesized audio with background."""
        import ai_movie.composer as composer
        try:
            out_dir = ensure_dir(WORKSPACE_DIR / "synthesized")
            mixed_path = out_dir / "final_audio.wav"
            composer.mix_audio(results, sep["background"], mixed_path)
            if self._cancel_requested:
                return
            self.log.mark_step("重新混音", "done")
            self.log.set_step_data("重新混音", {"mixed_audio": str(mixed_path)})
        except Exception as exc:
            _err = str(exc)
            self.root.after(0, lambda: self._on_synthesize_error(_err))
            return
        self.root.after(0, lambda: self._on_synthesize_done(results, mixed_path))

    def _on_synthesize_done(self, results, mixed_path):
        if hasattr(self, "_syn_dlg") and self._syn_dlg.winfo_exists():
            self._syn_dlg.destroy()
        ok = sum(1 for r in results if r.get("audio"))
        self.log.mark_step("合成音轨", "done")
        self.log.set_step_data("合成音轨", {"mixed_audio": str(mixed_path), "ok": ok})
        self.log.add_entry("合成音轨", "done", f"{ok} segments → {mixed_path}")
        self._refresh_toolbar()
        self._switch_to_next_tab("合成音轨")
        messagebox.showinfo("合成完成",
            f"配音合成完成：{ok}/{len(results)} 个片段\n输出：{mixed_path}")

    def _on_synthesize_error(self, error_msg: str):
        if hasattr(self, "_syn_dlg") and self._syn_dlg.winfo_exists():
            self._syn_dlg.destroy()
        self.log.mark_step("合成音轨", "failed")
        self.log.add_entry("合成音轨", "error", error_msg)
        self._refresh_toolbar()
        messagebox.showerror("合成失败", error_msg)

    # ═══ toolbar: 合成视频 ═════════════════════════════════════

    def _on_compose_video(self):
        # Get mixed audio — written by either 重新混音 or 合成音轨 shortcut
        remix_data = self.log.step_data.get("重新混音", {})
        audio_path = remix_data.get("mixed_audio")
        if not audio_path or not Path(audio_path).exists():
            messagebox.showwarning("提示", "请先完成「重新混音」或「合成音轨」步骤。")
            return

        # Prefer face-enhanced video, then lip-synced video, if available
        enh_output = self.log.step_data.get("人脸增强", {}).get("output_video")
        ls_output = self.log.step_data.get("口型匹配", {}).get("output_video")
        if enh_output and Path(enh_output).exists():
            video_path = Path(enh_output)
        elif ls_output and Path(ls_output).exists():
            video_path = Path(ls_output)
        else:
            # Use the original full video when it was cut into segments;
            # otherwise use the silent video from the demux step.
            cut_data = self.log.step_data.get("切割视频", {})
            has_cuts = bool(cut_data.get("segments"))
            if has_cuts and self.log.video_path:
                video_path = Path(self.log.video_path)
            else:
                split_data = self.log.step_data.get("拆分音轨", {})
                split_results = split_data.get("results", [])
                video_path = None
                if split_results:
                    for r in split_results:
                        if "error" not in r:
                            p = Path(r.get("video", ""))
                            if p.exists() and p.suffix in (".mp4", ".mkv", ".mov"):
                                video_path = p
                                break
        if video_path is None or not video_path.exists():
            messagebox.showwarning("提示", "未找到视频文件。")
            return

        self.log.mark_step("合成视频", "running")
        self.log.add_entry("合成视频", "start")
        self._refresh_toolbar()
        self._cancel_requested = False

        # Progress dialog
        self._cmp_dlg = tk.Toplevel(self.root)
        self._cmp_dlg.title("合成视频")
        self._cmp_dlg.geometry("360x140")
        self._cmp_dlg.transient(self.root)
        self._cmp_dlg.grab_set()
        self._cmp_dlg.resizable(False, False)

        ttk.Label(self._cmp_dlg, text="正在合成最终视频…",
                  font=(config.CJK_FONT, 11)).pack(pady=(12, 4))
        self._lbl_cmp_status = ttk.Label(self._cmp_dlg, text="准备中…",
                                          font=(config.CJK_FONT, 9), foreground="#666")
        self._lbl_cmp_status.pack()
        self._bar_cmp = ttk.Progressbar(self._cmp_dlg, length=320,
                                         mode="indeterminate")
        self._bar_cmp.pack(padx=20, pady=8)
        self._bar_cmp.start(10)

        threading.Thread(
            target=self._run_compose,
            args=(video_path, Path(audio_path)),
            daemon=True,
        ).start()

    def _run_compose(self, video_path, audio_path):
        import ai_movie.composer as composer
        def _upd(msg):
            self.root.after(0, lambda m=msg: (
                hasattr(self, "_lbl_cmp_status") and
                self._lbl_cmp_status.configure(text=m)))
        try:
            out_dir = ensure_dir(WORKSPACE_DIR / "output")
            out_path = out_dir / f"{Path(video_path).stem}_dubbed.mp4"
            composer.compose_video(video_path, audio_path, out_path,
                                   progress_cb=_upd)
        except Exception as exc:
            _err = str(exc)
            self.root.after(0, lambda: self._on_compose_error(_err))
            return
        self.root.after(0, lambda: self._on_compose_done(out_path))

    def _on_compose_done(self, out_path):
        if hasattr(self, "_cmp_dlg") and self._cmp_dlg.winfo_exists():
            self._cmp_dlg.destroy()
        self.log.mark_step("合成视频", "done")
        self.log.set_step_data("合成视频", {"output_video": str(out_path)})
        self.log.add_entry("合成视频", "done", str(out_path))
        self._refresh_toolbar()
        self._populate_compose_tab(out_path)
        self._switch_to_next_tab("合成视频")
        messagebox.showinfo("合成完成", f"配音视频已生成：\n{out_path}")

    def _populate_compose_tab(self, out_path: Path):
        """Show finished video in the 合成视频 tab and auto-load in left player."""
        tab = self._tab_frames.get("合成视频")
        if tab is None:
            return
        for w in tab.winfo_children():
            w.destroy()

        f = ttk.Frame(tab, padding=20)
        f.pack(expand=True)

        tk.Label(f, text="✓ 配音视频合成完成", font=(config.CJK_FONT, 14, "bold"),
                 fg="#155724").pack(pady=(0, 12))

        info_frame = ttk.LabelFrame(f, text="输出文件", padding=10)
        info_frame.pack(fill="x", pady=(0, 16))
        p = Path(out_path)
        size_mb = p.stat().st_size / 1e6 if p.exists() else 0
        tk.Label(info_frame, text=str(p), font=(config.MONO_FONT, 8),
                 fg="#555", wraplength=500).pack(anchor="w")
        tk.Label(info_frame, text=f"大小：{size_mb:.1f} MB",
                 font=(config.CJK_FONT, 9), fg="#666").pack(anchor="w", pady=(4, 0))

        btn_frame = ttk.Frame(f)
        btn_frame.pack()
        tk.Button(btn_frame, text="▶  在左侧播放", font=(config.CJK_FONT, 11),
                  command=lambda: self._load_composed_video(p)).pack(
                      side="left", padx=8)
        tk.Button(btn_frame, text="📂  打开所在目录", font=(config.CJK_FONT, 10),
                  command=lambda: subprocess.Popen(
                      ["xdg-open", str(p.parent)])).pack(side="left", padx=8)

        # Auto-load in left player
        self._load_composed_video(p)

    def _populate_lipsync_tab(self, output_path: Path):
        """Show finished lip-sync result WITHOUT destroying the option bar."""
        frame = getattr(self, "_ls_result_frame", None)
        if frame is None or not frame.winfo_exists():
            return
        for w in frame.winfo_children():
            w.destroy()

        f = ttk.Frame(frame, padding=20)
        f.pack(expand=True)

        tk.Label(f, text="✓ 口型匹配完成", font=(config.CJK_FONT, 14, "bold"),
                 fg="#155724").pack(pady=(0, 12))

        info_frame = ttk.LabelFrame(f, text="输出文件", padding=10)
        info_frame.pack(fill="x", pady=(0, 16))
        p = Path(output_path)
        size_mb = p.stat().st_size / 1e6 if p.exists() else 0

        # Get video duration
        try:
            import cv2
            cap = cv2.VideoCapture(str(p))
            if cap.isOpened():
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                dur_sec = frames / fps if fps > 0 else 0
                dur_str = f"{int(dur_sec // 60)}:{int(dur_sec % 60):02d}  ({fps:.1f} fps)"
                cap.release()
            else:
                dur_str = "—"
        except Exception:
            dur_str = "—"

        tk.Label(info_frame, text=str(p), font=(config.MONO_FONT, 8),
                 fg="#555", wraplength=500).pack(anchor="w")
        tk.Label(info_frame, text=f"大小：{size_mb:.1f} MB    时长：{dur_str}",
                 font=(config.CJK_FONT, 9), fg="#666").pack(anchor="w", pady=(4, 0))

        # Driving audio source info
        ls_data = self.log.step_data.get("口型匹配", {})
        audio_src = ls_data.get("driving_audio_source", "—")
        tk.Label(info_frame, text=f"驱动音频：{audio_src}",
                 font=(config.CJK_FONT, 9), fg="#666").pack(anchor="w", pady=(2, 0))

        btn_frame = ttk.Frame(f)
        btn_frame.pack()
        tk.Button(btn_frame, text="▶  在左侧播放", font=(config.CJK_FONT, 11),
                  command=lambda: self._load_lipsync_video(p)).pack(
                      side="left", padx=8)
        tk.Button(btn_frame, text="📂  打开所在目录", font=(config.CJK_FONT, 10),
                  command=lambda: subprocess.Popen(
                      ["xdg-open", str(p.parent)])).pack(side="left", padx=8)

        # Auto-load in left player
        self._load_lipsync_video(p)

    def _load_lipsync_video(self, path: Path):
        """Load a lip-synced video into the left player."""
        try:
            self._on_stop()
            self.left_player.load(path)
            self._is_playing = True
            self._btn_play.configure(text="⏸")
        except Exception as e:
            messagebox.showwarning("播放失败", str(e))

    def _load_composed_video(self, path: Path):
        """Load a composed video into the left player."""
        try:
            self._on_stop()
            self.left_player.load(path)
            self._is_playing = True
            self._btn_play.configure(text="⏸")
        except Exception as e:
            messagebox.showwarning("播放失败", str(e))

    def _on_compose_error(self, error_msg: str):
        if hasattr(self, "_cmp_dlg") and self._cmp_dlg.winfo_exists():
            self._cmp_dlg.destroy()
        self.log.mark_step("合成视频", "failed")
        self.log.add_entry("合成视频", "error", error_msg)
        self._refresh_toolbar()
        messagebox.showerror("合成失败", error_msg)

    # ═══ one-click pipeline ═════════════════════════════════════

    def _on_one_click(self):
        """Show the one-click pipeline dialog with all step options."""
        if self.log.video_path is None:
            messagebox.showwarning("提示", "请先加载视频文件。")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("一键生成 — 流水线配置")
        dlg.geometry("600x680")
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(True, True)

        # ── scrollable canvas ───────────────────────────────────
        canvas = tk.Canvas(dlg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(dlg, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _mw(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _mw)
        dlg.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # ── variables ───────────────────────────────────────────
        vars_dict = {}

        # Pre-compute completed steps so we can uncheck them by default
        completed_steps = {name for name, status in self.log.steps.items()
                           if status == "done"}

        def _add_section(title: str, key: str, step_name: str = "", **kw):
            """Add a collapsible-like section with checkbox."""
            frm = ttk.LabelFrame(scroll_frame, text="", padding=(10, 6))
            frm.pack(fill="x", padx=12, pady=4)

            # Auto-uncheck if this step is already done
            default_enabled = kw.pop("enabled", True)
            if step_name and step_name in completed_steps:
                default_enabled = False
            enabled = tk.BooleanVar(value=default_enabled)
            vars_dict[f"{key}_enabled"] = enabled

            hdr = ttk.Frame(frm); hdr.pack(fill="x")
            cb = ttk.Checkbutton(hdr, text=title, variable=enabled)
            cb.pack(side="left")

            content = ttk.Frame(frm)
            content.pack(fill="x", padx=(24, 0), pady=(4, 0))

            def _toggle():
                state = "normal" if enabled.get() else "disabled"
                for child in content.winfo_children():
                    try: child.configure(state=state)
                    except Exception: pass
            enabled.trace_add("write", lambda *_: _toggle())
            _toggle()

            return content

        # ── 1. 切割视频 ─────────────────────────────────────────
        sec1 = _add_section("1. 切割视频（将视频按静音分片）", "cut", step_name="切割视频",
                            enabled=False)   # default: 一键生成不切割
        ttk.Label(sec1, text="片段时长上限：180 秒", foreground="#888").pack(anchor="w")

        # ── 2. 拆分音轨 ─────────────────────────────────────────
        sec2 = _add_section("2. 拆分音轨（分离视频轨和音频轨）", "split", step_name="拆分音轨")

        # ── 3. 转换文字 (ASR) ───────────────────────────────────
        sec3 = _add_section("3. 转换文字（语音识别）", "asr", step_name="转换文字")
        asr_bar = ttk.Frame(sec3); asr_bar.pack(fill="x", pady=(2, 2))
        ttk.Label(asr_bar, text="语言：").pack(side="left")
        asr_lang_var = tk.StringVar(value=LANG_LABELS[0])
        vars_dict["asr_lang"] = asr_lang_var
        ttk.Combobox(asr_bar, textvariable=asr_lang_var,
                     values=LANG_LABELS, state="readonly", width=10).pack(side="left", padx=(4, 12))
        ttk.Label(asr_bar, text="引擎：").pack(side="left")
        asr_eng_labels = ["faster-whisper (CPU+VAD)", "openai-whisper (GPU+VAD)", "自动选择"]
        asr_eng_vals = ["faster-whisper", "openai-whisper", "auto"]
        asr_eng_var = tk.StringVar(value=asr_eng_labels[1])
        vars_dict["asr_engine"] = asr_eng_var
        vars_dict["_asr_eng_map"] = dict(zip(asr_eng_labels, asr_eng_vals))
        ttk.Combobox(asr_bar, textvariable=asr_eng_var,
                     values=asr_eng_labels, state="readonly", width=24).pack(side="left")

        # ── 4. 文本翻译 ─────────────────────────────────────────
        sec4 = _add_section("4. 文本翻译", "translate", step_name="文本翻译")
        tl_bar1 = ttk.Frame(sec4); tl_bar1.pack(fill="x", pady=(2, 2))
        ttk.Label(tl_bar1, text="目标语言：").pack(side="left")
        from ai_movie.translator import TARGET_LANG_LABELS
        tl_lang_var = tk.StringVar(value=TARGET_LANG_LABELS[0])
        vars_dict["tl_lang"] = tl_lang_var
        ttk.Combobox(tl_bar1, textvariable=tl_lang_var,
                     values=TARGET_LANG_LABELS, state="readonly", width=10).pack(side="left", padx=(4, 12))
        ttk.Label(tl_bar1, text="引擎：").pack(side="left")
        tl_eng_labels = ["Hy-MT2 + 润色", "Hy-MT2", "Hy-MT1.5", "Hy-MT1.5 + 润色", "Ollama 直翻"]
        tl_eng_vals = ["hy-mt2+polish", "hy-mt2", "hy-mt", "hy-mt+polish", "ollama"]
        tl_eng_var = tk.StringVar(value=tl_eng_labels[1])  # default: 仅 Hy-MT2
        vars_dict["tl_engine"] = tl_eng_var
        vars_dict["_tl_eng_map"] = dict(zip(tl_eng_labels, tl_eng_vals))
        ttk.Combobox(tl_bar1, textvariable=tl_eng_var,
                     values=tl_eng_labels, state="readonly", width=14).pack(side="left")
        tl_bar2 = ttk.Frame(sec4); tl_bar2.pack(fill="x", pady=(2, 0))
        ttk.Label(tl_bar2, text="Ollama 模型（润色/直翻时使用）：").pack(side="left")
        ollama_model_var = tk.StringVar(value=config.OLLAMA_MODEL)
        vars_dict["tl_ollama"] = ollama_model_var
        ttk.Entry(tl_bar2, textvariable=ollama_model_var, width=30).pack(side="left", padx=(4, 0))

        # ── 5. 人声分离 ─────────────────────────────────────────
        sec5 = _add_section("5. 人声分离（分离人声和背景音）", "separate", step_name="人声分离")
        sep_bar = ttk.Frame(sec5); sep_bar.pack(fill="x")
        sep_backend_var = tk.StringVar(value="uvr")
        vars_dict["sep_backend"] = sep_backend_var
        ttk.Radiobutton(sep_bar, text="Demucs (GPU，稳定)", variable=sep_backend_var, value="demucs").pack(side="left", padx=(0, 12))
        ttk.Radiobutton(sep_bar, text="UVR Mel-Band RoiFormer (GPU，高质量)", variable=sep_backend_var, value="uvr").pack(side="left")

        # ── 6. 人声生成 (TTS) ───────────────────────────────────
        sec6 = _add_section("6. 人声生成（TTS 语音合成）", "tts", step_name="人声生成")
        tts_mode_bar = ttk.Frame(sec6); tts_mode_bar.pack(fill="x")
        tts_mode_var = tk.StringVar(value="gender")
        vars_dict["tts_mode"] = tts_mode_var
        ttk.Label(tts_mode_bar, text="声音：").pack(side="left")
        for text, val in [("自动检测性别", "gender"), ("全部女声", "female"),
                          ("全部男声", "male"), ("台湾柔和女声", "style")]:
            ttk.Radiobutton(tts_mode_bar, text=text, variable=tts_mode_var, value=val).pack(side="left", padx=(6, 2))
        # Engine is implied by the voice choice — no separate selector:
        #   自动检测性别 / 台湾柔和女声 → CosyVoice3; 全部女声 / 全部男声 → SFT.
        ttk.Label(sec6, text="（引擎自动：自动/台湾=CosyVoice3，全部女/男=SFT内置音色）",
                  foreground="#999").pack(anchor="w", pady=(2, 0))

        tts_algo_bar = ttk.Frame(sec6); tts_algo_bar.pack(fill="x", pady=(2, 0))
        tts_algo_var = tk.StringVar(value="f0_per_seg")
        vars_dict["tts_algo"] = tts_algo_var
        ttk.Label(tts_algo_bar, text="算法：", foreground="#555").pack(side="left")
        for text, val in [("F0逐片段", "f0_per_seg"), ("F0全局", "f0_global"), ("ECAPA日志", "ecapa")]:
            ttk.Radiobutton(tts_algo_bar, text=text, variable=tts_algo_var, value=val).pack(side="left", padx=(6, 2))

        # Detection algorithm only applies to "自动检测性别" — grey it out otherwise.
        def _oc_update_algo_state(*_):
            state = "normal" if tts_mode_var.get() == "gender" else "disabled"
            for w in tts_algo_bar.winfo_children():
                try: w.configure(state=state)
                except Exception: pass
        tts_mode_var.trace_add("write", _oc_update_algo_state)
        _oc_update_algo_state()

        # ── 7. 重新混音 ─────────────────────────────────────────
        _add_section("7. 重新混音（合成人声 + 背景音）", "remix", step_name="重新混音")

        # ── 人物锚定（可选）：只对指定性别做口型匹配 ──────────────
        sec_anchor = _add_section("人物锚定（只对指定性别匹配口型）", "anchor",
                                  step_name="人物锚定")
        abar = ttk.Frame(sec_anchor); abar.pack(fill="x")
        ttk.Label(abar, text="口型匹配范围：").pack(side="left")
        oc_anchor_var = tk.StringVar(value="female")   # default: 仅女声
        vars_dict["oc_anchor_gender"] = oc_anchor_var
        for _lbl, _val in [("仅女声", "female"), ("仅男声", "male"), ("全部", "")]:
            ttk.Radiobutton(abar, text=_lbl, value=_val,
                            variable=oc_anchor_var).pack(side="left", padx=4)
        oc_anchor_occ_var = tk.BooleanVar(value=True)  # default: 遮挡检测开
        vars_dict["oc_anchor_occ"] = oc_anchor_occ_var
        ttk.Checkbutton(abar, text="遮挡/误检检测（嘴被挡时不匹配口型）",
                        variable=oc_anchor_occ_var).pack(side="left", padx=(12, 0))

        # ── 8. 口型匹配 ─────────────────────────────────────────
        sec8 = _add_section("8. 口型匹配（唇形同步）", "lipsync", step_name="口型匹配")
        ls_bar = ttk.Frame(sec8); ls_bar.pack(fill="x")
        ttk.Label(ls_bar, text="引擎：").pack(side="left")
        ls_eng_var = tk.StringVar(value="MuseTalk (256x256)")   # default: MuseTalk
        vars_dict["ls_engine"] = ls_eng_var
        ttk.Combobox(ls_bar, textvariable=ls_eng_var,
                     values=["自动检测", "MuseTalk (256x256)", "Wav2Lip (96x96)"],
                     state="readonly", width=22).pack(side="left", padx=(4, 0))

        # ── 9. 人脸增强（可选） ──────────────────────────────────
        from ai_movie.face_restore import codeformer_available, face_parser_available
        sec9 = _add_section("9. 人脸增强（CodeFormer 修复嘴部，可选）", "face_enhance",
                            step_name="人脸增强", enabled=False)
        fe_bar = ttk.Frame(sec9); fe_bar.pack(fill="x")
        ttk.Label(fe_bar, text="保真度：").pack(side="left")
        fe_fidelity_var = tk.DoubleVar(value=0.85)
        vars_dict["fe_fidelity"] = fe_fidelity_var
        ttk.Scale(fe_bar, from_=0.0, to=1.0, orient="horizontal", length=110,
                  variable=fe_fidelity_var).pack(side="left", padx=(4, 0))
        ttk.Label(fe_bar, text="  每N帧：").pack(side="left", padx=(8, 0))
        fe_detevery_var = tk.IntVar(value=4)
        vars_dict["fe_det_every"] = fe_detevery_var
        ttk.Spinbox(fe_bar, from_=1, to=15, width=3,
                    textvariable=fe_detevery_var).pack(side="left", padx=(4, 0))
        fe_protect_var = tk.BooleanVar(value=True)
        vars_dict["fe_protect_lips"] = fe_protect_var
        ttk.Checkbutton(fe_bar, text="保护嘴唇", variable=fe_protect_var).pack(
            side="left", padx=(8, 0))
        fe_occ_var = tk.BooleanVar(value=face_parser_available())
        vars_dict["fe_occlusion"] = fe_occ_var
        ttk.Checkbutton(fe_bar, text="遮挡/误检修正", variable=fe_occ_var).pack(
            side="left", padx=(10, 0))
        if not codeformer_available():
            vars_dict["face_enhance_enabled"].set(False)

        # ── 10. 合成视频 ────────────────────────────────────────
        _add_section("10. 合成视频（输出最终配音视频）", "compose", step_name="合成视频")

        # ── buttons ─────────────────────────────────────────────
        btn_frame = ttk.Frame(scroll_frame, padding=(12, 12))
        btn_frame.pack(fill="x")

        def _start():
            dlg.destroy()
            self._start_one_click_pipeline(vars_dict)

        def _cancel():
            dlg.destroy()

        ttk.Button(btn_frame, text="🚀 开始一键生成", command=_start).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="取消", command=_cancel).pack(side="right", padx=4)

        # Center dialog
        dlg.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 600) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 680) // 2
        dlg.geometry(f"+{x}+{y}")

    def _start_one_click_pipeline(self, vars_dict: dict):
        """Kick off the one-click pipeline in a background thread."""
        self._oc_opts = vars_dict
        self._oc_cancelled = False

        # Build ordered list of enabled steps
        steps = []
        step_names = [
            ("切割视频", "cut"),
            ("拆分音轨", "split"),
            ("转换文字", "asr"),
            ("文本翻译", "translate"),
            ("人声分离", "separate"),
            ("人声生成", "tts"),
            ("重新混音", "remix"),
            ("人物锚定", "anchor"),
            ("口型匹配", "lipsync"),
            ("人脸增强", "face_enhance"),
            ("合成视频", "compose"),
        ]
        enabled_steps = []
        for name, key in step_names:
            if vars_dict.get(f"{key}_enabled", tk.BooleanVar(value=True)).get():
                enabled_steps.append((name, key))

        if not enabled_steps:
            messagebox.showwarning("提示", "请至少启用一个步骤。")
            return

        total = len(enabled_steps)

        # ── progress dialog ─────────────────────────────────────
        self._oc_dlg = tk.Toplevel(self.root)
        self._oc_dlg.title("一键生成 — 执行中")
        self._oc_dlg.geometry("480x220")
        self._oc_dlg.transient(self.root)
        self._oc_dlg.grab_set()
        self._oc_dlg.protocol("WM_DELETE_WINDOW", self._on_cancel_one_click)
        self._oc_dlg.resizable(False, False)

        ttk.Label(self._oc_dlg, text="🚀 正在自动执行流水线…",
                  font=(config.CJK_FONT, 12, "bold")).pack(pady=(16, 8))
        self._lbl_oc_step = ttk.Label(self._oc_dlg, text=f"步骤 0 / {total}：准备中…",
                                       font=(config.CJK_FONT, 10))
        self._lbl_oc_step.pack()
        self._bar_oc_step = ttk.Progressbar(self._oc_dlg, length=420,
                                            mode="determinate", maximum=total)
        self._bar_oc_step.pack(pady=(8, 4))
        self._lbl_oc_detail = ttk.Label(self._oc_dlg, text="",
                                         font=(config.CJK_FONT, 9), foreground="#666")
        self._lbl_oc_detail.pack()
        self._bar_oc_detail = ttk.Progressbar(self._oc_dlg, length=420,
                                              mode="indeterminate")
        self._bar_oc_detail.pack(pady=(4, 12))
        ttk.Button(self._oc_dlg, text="取消",
                   command=self._on_cancel_one_click).pack()

        threading.Thread(
            target=self._run_one_click_pipeline,
            args=(enabled_steps,),
            daemon=True,
        ).start()

    def _on_cancel_one_click(self):
        self._oc_cancelled = True

    def _run_one_click_pipeline(self, enabled_steps: list[tuple[str, str]]):
        """Run each enabled step in sequence (background thread)."""
        total = len(enabled_steps)
        opts = self._oc_opts

        for i, (step_name, step_key) in enumerate(enabled_steps):
            if self._oc_cancelled:
                self.root.after(0, lambda: self._finish_one_click(cancelled=True))
                return

            self.root.after(0, lambda n=step_name, c=i: (
                self._lbl_oc_step.configure(text=f"步骤 {c + 1} / {total}：{n}"),
                self._bar_oc_step.configure(value=c),
                self._lbl_oc_detail.configure(text="正在执行…"),
                self._bar_oc_detail.configure(mode="indeterminate"),
                self._bar_oc_detail.start(10)
            ))

            try:
                success = self._execute_one_step(step_name, step_key, opts)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.root.after(0, lambda n=step_name, err=str(e), t=tb: (
                    self._oc_dlg.destroy() if hasattr(self, "_oc_dlg") and self._oc_dlg.winfo_exists() else None,
                    self.log.mark_step(n, "failed"),
                    self.log.add_entry(n, "error", f"{err}\n{t}"),
                    self._refresh_toolbar(),
                    messagebox.showerror("一键生成失败", f"步骤「{n}」出错：\n{err}")
                ))
                return

            if not success:
                # Step was skipped or has unmet dependencies
                continue

            # Auto-save project after each successful step
            self.root.after(0, lambda: self._auto_save_project())

            # Switch to the completed step's tab
            _TABBED_STEPS = {"切割视频", "拆分音轨", "转换文字", "文本翻译",
                             "人声分离", "人声生成", "重新混音",
                             "口型匹配", "人脸增强", "合成视频"}
            if step_name in _TABBED_STEPS:
                self.root.after(0, lambda n=step_name: self._switch_to_tab(n))

            if self._oc_cancelled:
                self.root.after(0, lambda: self._finish_one_click(cancelled=True))
                return

        self.root.after(0, lambda: self._finish_one_click(cancelled=False))

    def _execute_one_step(self, step_name: str, step_key: str, opts: dict) -> bool:
        """Execute a single pipeline step. Returns True on success."""
        src = Path(self.log.video_path) if self.log.video_path else None
        if src is None or not src.exists():
            return False

        # ── 切割视频 ────────────────────────────────────────────
        if step_key == "cut":
            self._update_step_status(step_name, "running")
            segments = cut_video(
                src, segment_duration=180.0,
                progress_cb=None,
                cancel_check=lambda: self._oc_cancelled,
            )
            if self._oc_cancelled:
                return False
            seg_dir = str(Path(segments[0]["path"]).parent) if segments else ""
            self.log.mark_step(step_name, "done")
            self.log.set_step_data(step_name, {"segments": segments, "output_dir": seg_dir})
            self.log.add_entry(step_name, "done", f"{len(segments)} 段 → {seg_dir}")
            self.root.after(0, self._refresh_toolbar)
            return True

        # ── 拆分音轨 ────────────────────────────────────────────
        if step_key == "split":
            cut_data = self.log.step_data.get("切割视频", {})
            segments = cut_data.get("segments", [])
            segments = [s for s in segments if Path(s["path"]).exists()]
            self._update_step_status(step_name, "running")
            if segments:
                out_base = Path(segments[0]["path"]).parent.parent / "demuxed"
            else:
                out_base = src.parent.parent / "workspace" / src.stem / "demuxed"
            results = demux_all(
                src, segments or None, out_base,
                progress_cb=None,
                cancel_check=lambda: self._oc_cancelled,
            )
            if self._oc_cancelled:
                return False
            errors = [r for r in results if "error" in r]
            ok = [r for r in results if "error" not in r]
            self.log.mark_step(step_name, "done")
            self.log.set_step_data(step_name, {"results": results, "error_count": len(errors)})
            self.log.add_entry(step_name, "done", f"{len(ok)} ok, {len(errors)} errors")
            self.root.after(0, self._refresh_toolbar)
            return True

        # ── 转换文字 (ASR) ──────────────────────────────────────
        if step_key == "asr":
            # Collect audio paths
            split_data = self.log.step_data.get("拆分音轨", {})
            results = split_data.get("results", [])
            audio_paths = []
            if results:
                for r in results:
                    if "error" not in r:
                        p = Path(r["audio"])
                        if p.exists():
                            audio_paths.append(p)
            if not audio_paths:
                # Fallback: extract audio from original
                import tempfile
                tmp = Path(tempfile.mktemp(suffix=".wav"))
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(src),
                    "-vn", "-ar", "16000", "-ac", "1", str(tmp),
                ], check=True, capture_output=True)
                audio_paths.append(tmp)

            lang_label = opts.get("asr_lang", tk.StringVar(value=LANG_LABELS[0])).get()
            lang_code = LANGUAGES.get(lang_label, "ja")
            eng_label = opts.get("asr_engine", tk.StringVar()).get()
            eng_map = opts.get("_asr_eng_map", {"openai-whisper (GPU+VAD)": "openai-whisper"})
            if isinstance(eng_map, dict):
                backend = eng_map.get(eng_label, "openai-whisper")
            else:
                backend = "openai-whisper"

            self._update_step_status(step_name, "running")

            def _asr_progress(current, total):
                self.root.after(0, lambda c=current, t=total: (
                    self._bar_oc_detail.configure(mode="determinate", maximum=t, value=c)
                    if hasattr(self, "_bar_oc_detail") else None,
                    self._lbl_oc_detail.configure(text=f"ASR: {c}/{t}")
                    if hasattr(self, "_lbl_oc_detail") else None,
                ))

            asr_results = transcribe_all(
                audio_paths, language=lang_code,
                backend=backend,
                segment_cb=None,
                progress_cb=_asr_progress,
                file_start_cb=None,
                file_progress_cb=None,
                cancel_check=lambda: self._oc_cancelled,
            )
            if self._oc_cancelled:
                return False
            errors = [r for r in asr_results if "error" in r]
            ok = [r for r in asr_results if "error" not in r]
            self.log.mark_step(step_name, "done")
            self.log.set_step_data(step_name, {"language": lang_code, "results": asr_results})
            self.log.add_entry(step_name, "done", f"{len(ok)} ok, {len(errors)} errors")
            self.root.after(0, self._refresh_toolbar)
            return True

        # ── 文本翻译 ────────────────────────────────────────────
        if step_key == "translate":
            asr_data = self.log.step_data.get("转换文字", {})
            asr_results = asr_data.get("results", [])
            if not asr_results:
                self.root.after(0, lambda: messagebox.showwarning("提示", "无 ASR 结果，跳过翻译。"))
                return False

            # Build audio offset map (same as _on_translate)
            cut_data = self.log.step_data.get("切割视频", {})
            cut_segs = cut_data.get("segments", [])
            audio_offset_map = {}
            if cut_segs:
                index_to_offset = {
                    s["index"]: s.get("start", (s["index"] - 1) * 180.0)
                    for s in cut_segs
                }
                split_data = self.log.step_data.get("拆分音轨", {})
                split_results = split_data.get("results", [])
                for r in split_results:
                    label = r.get("label", "")
                    if label.startswith("seg_"):
                        try:
                            idx = int(label.rsplit("_", 1)[-1])
                            offset = index_to_offset.get(idx, 0.0)
                            audio = r.get("audio", "")
                            if audio:
                                audio_offset_map[audio] = offset
                        except (ValueError, IndexError):
                            pass

            segments = []
            for r in asr_results:
                if "error" in r:
                    continue
                src_name = r.get("source", "")
                offset = audio_offset_map.get(src_name, 0.0)
                for seg in r.get("segments", []):
                    segments.append({
                        "text": seg["text"],
                        "start": seg["start"] + offset,
                        "end": seg["end"] + offset,
                        "source": src_name,
                    })

            if not segments:
                return False

            tl_label = opts.get("tl_lang", tk.StringVar(value="汉语 (中文)")).get()
            from ai_movie.translator import TRANSLATION_TARGET_LANGS
            target_lang = TRANSLATION_TARGET_LANGS.get(tl_label, "Chinese")
            src_lang_code = asr_data.get("language", "ja")
            src_name_map = {"ja": "Japanese", "en": "English", "zh": "Chinese", "ko": "Korean"}
            src_lang = src_name_map.get(src_lang_code, "Japanese")

            eng_label = opts.get("tl_engine", tk.StringVar(value="Hy-MT2 + 润色")).get()
            eng_map = opts.get("_tl_eng_map", {"Hy-MT2 + 润色": "hy-mt2+polish"})
            if isinstance(eng_map, dict):
                engine = eng_map.get(eng_label, "hy-mt2+polish")
            else:
                engine = "hy-mt2+polish"

            ollama_model = opts.get("tl_ollama", tk.StringVar(value=config.OLLAMA_MODEL)).get()

            # Determine Hy-MT path
            _hymt_path = config.HYMT2_MODEL_PATH
            if engine in ("hy-mt", "hy-mt+polish"):
                _hymt_path = config.TRANSLATION_MODEL_PATH

            self._update_step_status(step_name, "running")

            def _tl_progress(current, total):
                self.root.after(0, lambda c=current, t=total: (
                    self._bar_oc_detail.configure(mode="determinate", maximum=t, value=c)
                    if hasattr(self, "_bar_oc_detail") else None,
                    self._lbl_oc_detail.configure(text=f"翻译: {c}/{t}")
                    if hasattr(self, "_lbl_oc_detail") else None,
                ))

            from ai_movie.translator import translate, translate_ollama, polish_ollama

            if engine == "ollama":
                tl_results = translate_ollama(
                    segments, target_lang=target_lang, src_lang=src_lang,
                    model=ollama_model,
                    progress_cb=_tl_progress,
                    segment_cb=None,
                    cancel_check=lambda: self._oc_cancelled,
                )
            elif engine in ("hy-mt+polish", "hy-mt2+polish"):
                tl_results = translate(
                    segments, target_lang=target_lang, src_lang=src_lang,
                    model_path=_hymt_path,
                    progress_cb=_tl_progress,
                    cancel_check=lambda: self._oc_cancelled,
                )
                if not self._oc_cancelled:
                    tl_results = polish_ollama(
                        tl_results, model=ollama_model,
                        progress_cb=_tl_progress,
                        segment_cb=None,
                        cancel_check=lambda: self._oc_cancelled,
                    )
            else:
                tl_results = translate(
                    segments, target_lang=target_lang, src_lang=src_lang,
                    model_path=_hymt_path,
                    progress_cb=_tl_progress,
                    cancel_check=lambda: self._oc_cancelled,
                )

            if self._oc_cancelled:
                return False

            self.log.mark_step(step_name, "done")
            log_data = {"target_lang": target_lang, "engine": engine,
                        "segments": tl_results, "count": len(tl_results)}
            if ollama_model:
                log_data["ollama_model"] = ollama_model
            self.log.set_step_data(step_name, log_data)
            self.log.add_entry(step_name, "done", f"{len(tl_results)} segments → {target_lang}")
            self.root.after(0, self._refresh_toolbar)
            return True

        # ── 人声分离 ────────────────────────────────────────────
        if step_key == "separate":
            if not self.log.video_path:
                return False
            import tempfile
            tmp = Path(tempfile.mktemp(suffix=".wav"))
            subprocess.run([
                "ffmpeg", "-y", "-i", self.log.video_path,
                "-vn", "-ar", "16000", "-ac", "1", str(tmp),
            ], check=True, capture_output=True)
            ref_audio = tmp

            backend = opts.get("sep_backend", tk.StringVar(value="uvr")).get()
            self._update_step_status(step_name, "running")

            import ai_movie.composer as composer
            sep = composer.separate_vocals(ref_audio, backend=backend)

            if self._oc_cancelled:
                return False

            self.log.mark_step(step_name, "done")
            self.log.set_step_data(step_name, {"vocals": str(sep["vocals"]), "background": str(sep["background"])})
            self.log.add_entry(step_name, "done", f"backend={backend}")
            self.root.after(0, self._refresh_toolbar)
            return True

        # ── 人声生成 (TTS) ──────────────────────────────────────
        if step_key == "tts":
            tl_data = self.log.step_data.get("文本翻译", {})
            segments = tl_data.get("segments", [])
            sep_data = self.log.step_data.get("人声分离", {})
            vocals_path_str = sep_data.get("vocals")
            if not segments or not vocals_path_str:
                self.root.after(0, lambda: messagebox.showwarning("提示", "缺少翻译或人声分离结果，跳过 TTS。"))
                return False
            vocals_path = Path(vocals_path_str)

            import ai_movie.tts as tts_mod
            voice_mode = opts.get("tts_mode", tk.StringVar(value="gender")).get()
            algo = opts.get("tts_algo", tk.StringVar(value="f0_per_seg")).get()

            # Engine is implied by the voice choice — no separate selector:
            #   自动检测性别 / 台湾柔和女声 → CosyVoice3 (soft/Taiwanese female,
            #   Mandarin male); 全部女声 / 全部男声 → SFT built-in speaker.
            # resolve_model_choice() falls back if the preferred model is absent.
            tts_model = "cosyvoice3" if voice_mode in ("style", "gender") else "sft"
            # SFT loads in-process; Qwen models synthesize in the isolated subprocess.
            choice = tts_mod.resolve_model_choice(tts_model)
            is_sft = (choice == "sft")
            if is_sft:
                tts_mod._load_model(prefer=choice)

            out_dir = ensure_dir(WORKSPACE_DIR / "synthesized")

            self._update_step_status(step_name, "running")

            # Resolve reference audio / method — handle non-SFT fallback
            if voice_mode == "female":
                if is_sft:
                    ref_audio, ref_text, ref_method = tts_mod._SFT_FEMALE_SPK, None, "sft"
                else:
                    ref_audio = str(tts_mod._FEMALE_REF_WAV)
                    ref_text, ref_method = tts_mod._FEMALE_REF_TEXT, "zero_shot"
            elif voice_mode == "male":
                if is_sft:
                    ref_audio, ref_text, ref_method = tts_mod._SFT_MALE_SPK, None, "sft"
                else:
                    ref_audio = str(tts_mod._MALE_REF_WAV)
                    ref_text, ref_method = None, "cross_lingual"
            elif voice_mode == "style":
                ref_audio, ref_text, ref_method = tts_mod.prepare_reference(
                    vocals_path, mode="style", cache_dir=out_dir)
                if is_sft:
                    self.root.after(0, lambda: self._lbl_oc_detail.configure(
                        text="⚠ 台湾柔和女声需 CosyVoice3，已退回克隆") if hasattr(self, "_lbl_oc_detail") else None)
            else:
                ref_audio, ref_text, ref_method = tts_mod.prepare_reference(
                    vocals_path, mode="gender", cache_dir=out_dir)

            # Qwen models (non-SFT): synthesize the whole batch in the isolated
            # subprocess (this step already runs in a background thread).
            if ref_method != "sft":
                seg_texts = [(i, seg.get("text_translated", "").strip())
                             for i, seg in enumerate(segments)]

                # Gender-auto: per-segment reference routing + record gender.
                oc_seg_refs = None
                oc_genders: dict[int, str] = {}
                if voice_mode == "gender":
                    fem_ref = tts_mod.prepare_reference(vocals_path, mode="style", cache_dir=out_dir)
                    if tts_mod._MALE_REF_WAV.exists():
                        male_ref = (str(tts_mod._MALE_REF_WAV), None, "cross_lingual")
                    else:
                        male_ref = (tts_mod.trim_reference_audio(str(vocals_path)), None, "cross_lingual")
                    oc_seg_refs = {}
                    last = "female"
                    for i, seg in enumerate(segments):
                        if algo == "f0_global":
                            g = tts_mod.detect_gender_global_f0(seg, fallback=last)
                        else:
                            g = tts_mod.detect_gender_from_segment(seg, fallback=last)
                        last = g
                        oc_genders[i] = g
                        oc_seg_refs[i] = fem_ref if g == "female" else male_ref

                def _oc_prog(done, total):
                    self.root.after(0, lambda d=done, t=total: (
                        self._bar_oc_detail.configure(mode="determinate", maximum=t, value=d)
                        if hasattr(self, "_bar_oc_detail") else None,
                        self._lbl_oc_detail.configure(text=f"TTS(隔离): {d}/{t}")
                        if hasattr(self, "_lbl_oc_detail") else None,
                    ))

                items = tts_mod.run_isolated_synthesis(
                    seg_texts, choice, ref_audio, ref_text, ref_method, out_dir,
                    progress_cb=_oc_prog, cancel_check=lambda: self._oc_cancelled,
                    seg_refs=oc_seg_refs,
                )
                tts_results = []
                for i, seg in enumerate(segments):
                    it = items.get(i, {})
                    r = {**seg, "audio": it.get("audio")}
                    if i in oc_genders:
                        r["tts_gender"] = oc_genders[i]
                    if it.get("tts_error"):
                        r["tts_error"] = it["tts_error"]
                    tts_results.append(r)
                ok = sum(1 for r in tts_results if r.get("audio"))
                self.log.mark_step(step_name, "done")
                self.log.set_step_data(step_name, {"results": tts_results, "ok": ok})
                self.log.add_entry(step_name, "done", f"{ok}/{len(tts_results)} segments")
                self.root.after(0, self._refresh_toolbar)
                return True

            # ECAPA pre-compute
            ecapa_result = None
            if voice_mode == "gender" and algo == "ecapa":
                self.root.after(0, lambda: self._lbl_oc_detail.configure(text="ECAPA：分析说话人中…"))
                ecapa_result = tts_mod.build_ecapa_gender_map(vocals_path)

            # Per-segment TTS loop
            tts_results = []
            last_gender = "female"
            total_tts = len(segments)
            import soundfile as sf

            for seg_idx, seg in enumerate(segments):
                if self._oc_cancelled:
                    return False

                self.root.after(0, lambda i=seg_idx, t=total_tts: (
                    self._bar_oc_detail.configure(mode="determinate", maximum=t, value=i)
                    if hasattr(self, "_bar_oc_detail") else None,
                    self._lbl_oc_detail.configure(text=f"TTS: {i + 1}/{t}")
                    if hasattr(self, "_lbl_oc_detail") else None,
                ))

                text = seg.get("text_translated", "").strip()
                if not text:
                    tts_results.append({**seg, "audio": None})
                    continue

                if ref_method == "sft":
                    if voice_mode == "gender":
                        if algo == "ecapa" and ecapa_result:
                            seg_gender = tts_mod.lookup_ecapa_gender(seg, ecapa_result, fallback=last_gender)
                        elif algo == "f0_global":
                            seg_gender = tts_mod.detect_gender_global_f0(seg, fallback=last_gender)
                        else:
                            seg_gender = tts_mod.detect_gender_from_segment(seg, fallback=last_gender)
                        last_gender = seg_gender
                        spk = tts_mod._SFT_FEMALE_SPK if seg_gender == "female" else tts_mod._SFT_MALE_SPK
                    else:
                        spk = ref_audio
                        seg_gender = "female" if spk == tts_mod._SFT_FEMALE_SPK else "male"
                    seg_ref, seg_ref_text, seg_ref_method = spk, None, "sft"
                else:
                    seg_ref, seg_ref_text, seg_ref_method = ref_audio, ref_text, ref_method
                    seg_gender = "female"

                try:
                    audio_np = tts_mod.call_tts(tts_mod._model, text, seg_ref, seg_ref_text, seg_ref_method)
                    out_path = str(out_dir / f"seg_{seg_idx + 1:04d}.wav")
                    sf.write(out_path, audio_np, tts_mod._model.sample_rate)
                    tts_results.append({**seg, "audio": out_path, "tts_gender": seg_gender})
                except Exception as exc:
                    tts_results.append({**seg, "audio": None, "tts_error": str(exc)})

            ok = sum(1 for r in tts_results if r.get("audio"))
            self.log.mark_step(step_name, "done")
            self.log.set_step_data(step_name, {"results": tts_results, "ok": ok})
            self.log.add_entry(step_name, "done", f"{ok}/{len(tts_results)} segments")
            self.root.after(0, self._refresh_toolbar)
            return True

        # ── 重新混音 ────────────────────────────────────────────
        if step_key == "remix":
            gen_data = self.log.step_data.get("人声生成", {})
            gen_results = gen_data.get("results", [])
            sep_data = self.log.step_data.get("人声分离", {})
            bg_path = sep_data.get("background")
            if not gen_results or not bg_path:
                self.root.after(0, lambda: messagebox.showwarning("提示", "缺少人声或背景音，跳过混音。"))
                return False

            self._update_step_status(step_name, "running")
            import ai_movie.composer as composer
            out_dir = ensure_dir(WORKSPACE_DIR / "synthesized")
            mixed_path = out_dir / "final_audio.wav"
            composer.mix_audio(gen_results, Path(bg_path), mixed_path)

            self.log.mark_step(step_name, "done")
            self.log.set_step_data(step_name, {"mixed_audio": str(mixed_path)})
            self.log.add_entry(step_name, "done", str(mixed_path))
            self.root.after(0, self._refresh_toolbar)
            return True

        # ── 人物锚定 ────────────────────────────────────────────
        if step_key == "anchor":
            gen_data = self.log.step_data.get("人声生成", {})
            results = gen_data.get("results", [])
            anchor_gender = opts.get("oc_anchor_gender", tk.StringVar(value="female")).get() or None
            occlusion_gate = bool(opts.get("oc_anchor_occ", tk.BooleanVar(value=True)).get())
            female = sum(1 for r in results if r.get("tts_gender") == "female" and r.get("audio"))
            male = sum(1 for r in results if r.get("tts_gender") == "male" and r.get("audio"))
            self.log.mark_step(step_name, "done")
            self.log.set_step_data(step_name, {
                "anchor_gender": anchor_gender,
                "occlusion_gate": occlusion_gate,
                "female_segments": female, "male_segments": male,
            })
            self.log.add_entry(step_name, "done",
                               f"anchor={anchor_gender or '全部'} 遮挡检测={occlusion_gate} (♀{female}/♂{male})")
            self.root.after(0, self._refresh_toolbar)
            return True

        # ── 口型匹配 ────────────────────────────────────────────
        if step_key == "lipsync":
            gen_data = self.log.step_data.get("人声生成", {})
            tts_results = gen_data.get("results", [])
            if not tts_results or not self.log.video_path:
                self.root.after(0, lambda: messagebox.showwarning("提示", "缺少 TTS 结果，跳过口型匹配。"))
                return False

            self._update_step_status(step_name, "running")

            ls_choice = opts.get("ls_engine", tk.StringVar(value="自动检测")).get()
            from ai_movie.lip_sync import musetalk_available, segment_based_lip_sync

            use_musetalk = musetalk_available()
            if ls_choice == "Wav2Lip (96x96)":
                use_musetalk = False
            elif ls_choice == "MuseTalk (256x256)":
                use_musetalk = True

            backend = "musetalk" if use_musetalk else "wav2lip"
            out_path = Path(WORKSPACE_DIR) / "lipsync_output.mp4"

            def _ls_progress(cur, total):
                self.root.after(0, lambda c=cur, t=total: (
                    self._bar_oc_detail.configure(mode="determinate", maximum=t, value=c)
                    if hasattr(self, "_bar_oc_detail") else None,
                    self._lbl_oc_detail.configure(text=f"口型匹配: {c}/{t}")
                    if hasattr(self, "_lbl_oc_detail") else None,
                ))

            anchor_data = self.log.step_data.get("人物锚定", {})
            anchor_gender = anchor_data.get("anchor_gender")
            occlusion_gate = bool(anchor_data.get("occlusion_gate", False))
            result = segment_based_lip_sync(
                video_path=self.log.video_path,
                tts_results=tts_results,
                output_path=out_path,
                backend=backend,
                anchor_gender=anchor_gender,
                occlusion_gate=occlusion_gate,
                progress_cb=_ls_progress,
                cancel_check=lambda: self._oc_cancelled,
            )

            if self._oc_cancelled or result is None:
                return False

            self.log.mark_step(step_name, "done")
            self.log.set_step_data(step_name, {"output_video": str(result), "driving_audio_source": "tts_speech_track"})
            self.log.add_entry(step_name, "done", f"→ {result}")
            self.root.after(0, self._refresh_toolbar)
            self.root.after(0, lambda: self._populate_lipsync_tab(result))
            return True

        # ── 人脸增强（可选） ────────────────────────────────────
        if step_key == "face_enhance":
            from ai_movie.face_restore import codeformer_available, restore_video
            ls_output = self.log.step_data.get("口型匹配", {}).get("output_video")
            if not codeformer_available() or not ls_output or not Path(ls_output).exists():
                self.root.after(0, lambda: messagebox.showwarning(
                    "提示", "缺少 CodeFormer 模型或口型匹配结果，跳过人脸增强。"))
                return False

            self._update_step_status(step_name, "running")
            fidelity = float(opts.get("fe_fidelity", tk.DoubleVar(value=0.85)).get())
            det_every = int(opts.get("fe_det_every", tk.IntVar(value=4)).get())
            occlusion = bool(opts.get("fe_occlusion", tk.BooleanVar(value=True)).get())
            protect_lips = bool(opts.get("fe_protect_lips", tk.BooleanVar(value=True)).get())
            out_path = Path(WORKSPACE_DIR) / "lipsync_enhanced.mp4"

            def _fe_progress(done, total):
                if done == 1 or done == total or done % 15 == 0:
                    self.root.after(0, lambda c=done, t=total: (
                        self._bar_oc_detail.configure(mode="determinate", maximum=t, value=c)
                        if hasattr(self, "_bar_oc_detail") else None,
                        self._lbl_oc_detail.configure(text=f"人脸增强: {c}/{t}")
                        if hasattr(self, "_lbl_oc_detail") else None,
                    ))

            result = restore_video(
                Path(ls_output), out_path,
                fidelity_weight=fidelity, det_every=det_every,
                occlusion_aware=occlusion, protect_lips=protect_lips,
                progress_cb=_fe_progress,
                cancel_check=lambda: self._oc_cancelled,
            )
            if self._oc_cancelled or result is None:
                return False

            self.log.mark_step(step_name, "done")
            self.log.set_step_data(step_name, {"output_video": str(result)})
            self.log.add_entry(step_name, "done", f"→ {result}")
            self.root.after(0, self._refresh_toolbar)
            self.root.after(0, lambda: self._populate_face_enhance_tab(result))
            return True

        # ── 合成视频 ────────────────────────────────────────────
        if step_key == "compose":
            remix_data = self.log.step_data.get("重新混音", {})
            audio_path = remix_data.get("mixed_audio")
            if not audio_path or not Path(audio_path).exists():
                self.root.after(0, lambda: messagebox.showwarning("提示", "缺少混音结果，跳过合成视频。"))
                return False

            # Prefer face-enhanced video, then lip-synced video
            enh_output = self.log.step_data.get("人脸增强", {}).get("output_video")
            ls_output = self.log.step_data.get("口型匹配", {}).get("output_video")
            if enh_output and Path(enh_output).exists():
                video_path = Path(enh_output)
            elif ls_output and Path(ls_output).exists():
                video_path = Path(ls_output)
            else:
                cut_data = self.log.step_data.get("切割视频", {})
                has_cuts = bool(cut_data.get("segments"))
                if has_cuts and self.log.video_path:
                    video_path = Path(self.log.video_path)
                else:
                    split_data = self.log.step_data.get("拆分音轨", {})
                    split_results = split_data.get("results", [])
                    video_path = None
                    if split_results:
                        for r in split_results:
                            if "error" not in r:
                                p = Path(r.get("video", ""))
                                if p.exists() and p.suffix in (".mp4", ".mkv", ".mov"):
                                    video_path = p
                                    break
            if video_path is None or not video_path.exists():
                self.root.after(0, lambda: messagebox.showwarning("提示", "未找到视频文件，跳过合成。"))
                return False

            self._update_step_status(step_name, "running")

            import ai_movie.composer as composer
            out_dir = ensure_dir(WORKSPACE_DIR / "output")
            out_path = out_dir / f"{Path(video_path).stem}_dubbed.mp4"

            def _cmp_progress(msg):
                self.root.after(0, lambda m=msg: (
                    self._lbl_oc_detail.configure(text=m)
                    if hasattr(self, "_lbl_oc_detail") else None,
                ))

            composer.compose_video(video_path, Path(audio_path), out_path, progress_cb=_cmp_progress)

            self.log.mark_step(step_name, "done")
            self.log.set_step_data(step_name, {"output_video": str(out_path)})
            self.log.add_entry(step_name, "done", str(out_path))
            self.root.after(0, self._refresh_toolbar)
            return True

        return False

    def _update_step_status(self, step_name: str, status: str):
        """Update step status from background thread — must be called via root.after."""
        def _inner():
            self.log.mark_step(step_name, status)
            self.log.add_entry(step_name, "start" if status == "running" else status)
            self._refresh_toolbar()
        self.root.after(0, _inner)

    def _finish_one_click(self, cancelled: bool):
        """Clean up after the one-click pipeline finishes."""
        if hasattr(self, "_oc_dlg") and self._oc_dlg.winfo_exists():
            self._oc_dlg.destroy()
        # Populate every step tab with its result (one-click only refreshed
        # lipsync/face_enhance before — other tabs stayed on the placeholder).
        try:
            self._refresh_all_step_tabs()
        except Exception as _e:
            print(f"[one-click] tab refresh failed: {_e}", file=sys.stderr)
        if cancelled:
            self._auto_save_project()
            messagebox.showinfo("已取消", "一键生成已取消。\n已完成步骤的状态已保存。")
        else:
            self._auto_save_project()
            done_count = sum(1 for v in self.log.steps.values() if v == "done")
            messagebox.showinfo("一键生成完成",
                f"🎉 流水线执行完成！\n已完成 {done_count}/{len(STEP_NAMES)} 个步骤。\n"
                f"请在「合成视频」标签查看输出。")
            # Switch to compose tab
            self._switch_to_tab("合成视频")

    # ═══ remaining toolbar stubs ════════════════════════════════

    def _on_anchor_person(self):
        """Person anchoring: choose which speakers get lip-synced.

        v1: female-only — only segments whose speaker is female are lip-synced
        (their video mouth is driven); male segments keep the original video
        (their dubbed male audio still plays). Reads per-segment gender computed
        during 人声生成 (``tts_gender``).
        """
        gen_data = self.log.step_data.get("人声生成", {})
        results = gen_data.get("results", [])
        if not results:
            messagebox.showwarning("提示", "请先完成「人声生成」步骤。")
            return

        female = sum(1 for r in results if r.get("tts_gender") == "female"
                     and r.get("audio"))
        male = sum(1 for r in results if r.get("tts_gender") == "male"
                   and r.get("audio"))
        # Respect existing toggles if set; defaults: female-only + occlusion gate on.
        anchor = getattr(self, "_anchor_gender_var", None)
        anchor_gender = anchor.get() if anchor is not None else "female"
        anchor_gender = anchor_gender or None  # "" → 全部
        occ = getattr(self, "_anchor_occ_var", None)
        occlusion_gate = bool(occ.get()) if occ is not None else True

        self.log.mark_step("人物锚定", "done")
        self.log.set_step_data("人物锚定", {
            "anchor_gender": anchor_gender,
            "occlusion_gate": occlusion_gate,
            "female_segments": female,
            "male_segments": male,
        })
        self.log.add_entry("人物锚定", "done",
                           f"anchor={anchor_gender or '全部'} 遮挡检测={occlusion_gate} (♀{female}/♂{male})")
        self._refresh_toolbar()
        self._populate_anchor_tab(female, male, anchor_gender, occlusion_gate)
        self._switch_to_next_tab("人物锚定")

    def _populate_anchor_tab(self, female: int, male: int, anchor_gender, occlusion_gate=True):
        tab = self._tab_frames.get("人物锚定")
        if tab is None:
            return
        for w in tab.winfo_children():
            w.destroy()
        f = ttk.Frame(tab, padding=20)
        f.pack(expand=True)
        tk.Label(f, text="✓ 人物锚定完成", font=(config.CJK_FONT, 14, "bold"),
                 fg="#155724").pack(pady=(0, 12))
        info = ttk.LabelFrame(f, text="按性别检测的语音段", padding=10)
        info.pack(fill="x", pady=(0, 12))
        tk.Label(info, text=f"女声段：{female}    男声段：{male}",
                 font=(config.CJK_FONT, 10)).pack(anchor="w")
        tk.Label(f, text=f"口型匹配范围：{'仅女声' if anchor_gender=='female' else ('仅男声' if anchor_gender=='male' else '全部')}"
                        f"    遮挡检测：{'开' if occlusion_gate else '关'}",
                 font=(config.CJK_FONT, 10), fg="#555").pack(anchor="w")
        bar = ttk.Frame(f); bar.pack(anchor="w", pady=(10, 0))
        ttk.Label(bar, text="修改范围：").pack(side="left")
        self._anchor_gender_var = tk.StringVar(value=anchor_gender or "")
        for label, val in [("仅女声", "female"), ("仅男声", "male"), ("全部", "")]:
            ttk.Radiobutton(bar, text=label, value=val,
                            variable=self._anchor_gender_var).pack(side="left", padx=4)
        self._anchor_occ_var = tk.BooleanVar(value=occlusion_gate)
        ttk.Checkbutton(bar, text="遮挡/误检检测（嘴被挡时不匹配口型）",
                        variable=self._anchor_occ_var).pack(side="left", padx=(12, 0))
        ttk.Button(bar, text="重新锚定", command=self._on_anchor_person).pack(side="left", padx=(8, 0))

    def _on_lip_sync(self):
        """Run lip-sync on speech segments only (segment-based).

        Only video portions that have TTS speech are processed.
        Silent segments and face-less segments pass through unchanged.
        """
        # ── Get TTS speech segments ──────────────────────────────
        gen_data = self.log.step_data.get("人声生成", {})
        tts_results = gen_data.get("results", [])
        if not tts_results:
            messagebox.showwarning("提示", "请先完成「人声生成」步骤。")
            return

        # ── Use the original video ─────────────────────────────────
        video_path = Path(self.log.video_path) if self.log.video_path else None
        if video_path is None or not video_path.exists():
            messagebox.showwarning("提示", "未找到原始视频文件。")
            return

        self.log.mark_step("口型匹配", "running")
        self.log.add_entry("口型匹配", "start")
        self._refresh_toolbar()
        self._cancel_requested = False

        # ── Read backend from tab selector ─────────────────────────
        ls_choice = getattr(self, "_ls_backend_var", tk.StringVar(value="自动检测")).get()
        if ls_choice == "Wav2Lip (96x96)":
            backend = "wav2lip"
        elif ls_choice == "MuseTalk (256x256)":
            backend = "musetalk"
        else:
            from ai_movie.lip_sync import musetalk_available
            backend = "musetalk" if musetalk_available() else "wav2lip"
        backend_label = "MuseTalk (256×256)" if backend == "musetalk" else "Wav2Lip (96×96)"

        # Count speech segments
        speech_count = sum(
            1 for s in tts_results
            if s.get("audio") and Path(str(s["audio"])).exists()
        )

        # ── Two-level progress dialog ──────────────────────────────
        self._ls_dlg = tk.Toplevel(self.root)
        self._ls_dlg.title(f"口型匹配 ({backend_label})")
        self._ls_dlg.geometry("450x280")
        self._ls_dlg.transient(self.root)
        self._ls_dlg.grab_set()
        self._ls_dlg.protocol("WM_DELETE_WINDOW", self._on_cancel_lip_sync)
        self._ls_dlg.resizable(False, False)

        ttk.Label(self._ls_dlg, text=f"正在运行口型匹配 ({backend_label})…\n"
                  f"仅处理有语音的片段（{speech_count} 个语音段）",
                  font=(config.CJK_FONT, 11)).pack(pady=(12, 8))

        # Level 1: segment-level (determinate from start)
        f1 = ttk.Frame(self._ls_dlg); f1.pack(fill="x", padx=20, pady=(4, 2))
        ttk.Label(f1, text="片段进度：").pack(side="left")
        self._lbl_ls_prog = ttk.Label(f1, text=f"0 / {max(speech_count, 1)}")
        self._lbl_ls_prog.pack(side="right")
        self._bar_ls = ttk.Progressbar(
            self._ls_dlg, length=360,
            mode="determinate", maximum=max(speech_count, 1),
        )
        self._bar_ls.pack(padx=20)

        # Level 2: intra-segment batch progress (indeterminate → determinate)
        f2 = ttk.Frame(self._ls_dlg); f2.pack(fill="x", padx=20, pady=(10, 2))
        ttk.Label(f2, text="处理进度：").pack(side="left")
        self._lbl_ls_inner_prog = ttk.Label(f2, text="等待中…")
        self._lbl_ls_inner_prog.pack(side="right")
        self._bar_ls_inner = ttk.Progressbar(
            self._ls_dlg, length=360, mode="indeterminate",
        )
        self._bar_ls_inner.pack(padx=20)
        self._bar_ls_inner.start(10)

        ttk.Button(self._ls_dlg, text="取消",
                   command=self._on_cancel_lip_sync).pack(pady=12)

        # Switch to lip-sync tab
        for idx in range(self._notebook.index("end")):
            if self._notebook.tab(idx, "text") == "口型匹配":
                self._notebook.select(idx)
                break

        # ── Run segment-based lip sync (background thread) ────────
        # Face enhancement is handled by the separate «人脸增强» step.
        # Person anchoring (人物锚定): restrict lip-sync to a given gender.
        anchor_data = self.log.step_data.get("人物锚定", {})
        anchor_gender = anchor_data.get("anchor_gender")
        occlusion_gate = bool(anchor_data.get("occlusion_gate", False))
        threading.Thread(
            target=self._run_segment_lip_sync,
            args=(str(video_path), tts_results, backend, anchor_gender, occlusion_gate),
            daemon=True,
        ).start()

    def _run_segment_lip_sync(self, video_path: str, tts_results: list[dict], backend: str,
                              anchor_gender: str | None = None, occlusion_gate: bool = False):
        import traceback as _tb
        from ai_movie.lip_sync import segment_based_lip_sync
        out_path = Path(WORKSPACE_DIR) / "lipsync_output.mp4"
        try:
            result = segment_based_lip_sync(
                video_path=video_path,
                tts_results=tts_results,
                output_path=out_path,
                backend=backend,
                anchor_gender=anchor_gender,
                occlusion_gate=occlusion_gate,
                progress_cb=lambda cur, total: self.root.after(
                    0, lambda c=cur, t=total: self._update_lip_sync_progress(c, t)),
                detail_progress_cb=lambda cur, total: self.root.after(
                    0, lambda c=cur, t=total: self._update_lip_sync_detail_progress(c, t)),
                cancel_check=lambda: self._cancel_requested,
            )
        except Exception as e:
            tb = _tb.format_exc()
            print(f"[LipSync ERROR] {tb}", file=sys.stderr)
            err_msg = f"{type(e).__name__}: {e}"
            self.root.after(0, lambda msg=err_msg: self._on_lip_sync_error(msg))
            return
        if result is None:
            self.root.after(0, lambda: self._on_lip_sync_cancelled())
            return
        self.root.after(0, lambda: self._on_lip_sync_done(result))

    def _on_cancel_lip_sync(self):
        self._cancel_requested = True

    def _run_lip_sync(self, video_path: str, audio_path: str):
        import traceback as _tb
        from ai_movie.lip_sync import wav2lip_sync
        out_path = Path(WORKSPACE_DIR) / "lipsync_output.mp4"
        try:
            result = wav2lip_sync(
                video_path=video_path,
                audio_path=audio_path,
                output_path=out_path,
                resize_factor=2,
                progress_cb=lambda cur, total: self.root.after(
                    0, lambda c=cur, t=total: self._update_lip_sync_progress(c, t)),
                cancel_check=lambda: self._cancel_requested,
            )
        except Exception as e:
            tb = _tb.format_exc()
            print(f"[LipSync ERROR] {tb}", file=sys.stderr)
            err_msg = f"{type(e).__name__}: {e}\n\n详细日志已写入:\n{out_path.parent / 'wav2lip_debug.log'}"
            self.root.after(0, lambda msg=err_msg: self._on_lip_sync_error(msg))
            return
        if result is None:
            # Cancelled
            self.root.after(0, lambda: self._on_lip_sync_cancelled())
            return
        self.root.after(0, lambda: self._on_lip_sync_done(result))

    def _run_musetalk(self, video_path: str, audio_path: str):
        import traceback as _tb
        from ai_movie.lip_sync import musetalk_sync, _downscale_video as _ds_vid
        out_path = Path(WORKSPACE_DIR) / "lipsync_output.mp4"

        # Downscale video first to bound frame memory (MuseTalk subprocess
        # loads ALL frames at native resolution — 540p vs 1080p saves 4× RAM)
        import tempfile as _tf
        tmp_dir = Path(_tf.mkdtemp(prefix="musetalk_gui_"))
        try:
            scaled_video = tmp_dir / "video_scaled.mp4"
            _ds_vid(Path(video_path), scaled_video, factor=2)
        except Exception:
            # If downscale fails, fall back to original
            scaled_video = Path(video_path)

        try:
            result = musetalk_sync(
                video_path=scaled_video,
                audio_path=audio_path,
                output_path=out_path,
                progress_cb=lambda cur, total: self.root.after(
                    0, lambda c=cur, t=total: self._update_lip_sync_progress(c, t)),
                cancel_check=lambda: self._cancel_requested,
            )
        except Exception as e:
            tb = _tb.format_exc()
            print(f"[MuseTalk ERROR] {tb}", file=sys.stderr)
            err_msg = f"{type(e).__name__}: {e}"
            self.root.after(0, lambda msg=err_msg: self._on_lip_sync_error(msg))
            return
        finally:
            import shutil as _sh
            _sh.rmtree(str(tmp_dir), ignore_errors=True)
        if result is None:
            self.root.after(0, lambda: self._on_lip_sync_cancelled())
            return
        self.root.after(0, lambda: self._on_lip_sync_done(result))

    def _update_lip_sync_progress(self, cur: int, total: int):
        """Update the segment-level progress bar from worker thread."""
        if not hasattr(self, "_ls_dlg") or not self._ls_dlg.winfo_exists():
            return
        self._bar_ls.configure(maximum=total, value=cur)
        self._lbl_ls_prog.configure(text=f"{cur} / {total}")

    def _update_lip_sync_detail_progress(self, cur: int, total: int):
        """Update the inner batch-level progress bar from worker thread.

        The bar starts in 'indeterminate' mode and is switched to
        'determinate' on the first callback, after calling stop() to
        kill the animation (this is what prevents flickering).
        """
        if not hasattr(self, "_ls_dlg") or not self._ls_dlg.winfo_exists():
            return
        if not hasattr(self, "_bar_ls_inner"):
            return

        # Stop indeterminate animation on first callback
        if self._bar_ls_inner.cget("mode") == "indeterminate":
            try:
                self._bar_ls_inner.stop()
            except Exception:
                pass

        self._bar_ls_inner.configure(
            mode="determinate", maximum=total, value=cur,
        )
        self._lbl_ls_inner_prog.configure(text=f"{cur} / {total}")

    def _on_lip_sync_done(self, result: Path):
        if hasattr(self, "_ls_dlg") and self._ls_dlg.winfo_exists():
            self._ls_dlg.destroy()
        self.log.mark_step("口型匹配", "done")
        self.log.set_step_data("口型匹配", {
            "output_video": str(result),
            "driving_audio_source": "tts_speech_track",
        })
        self.log.add_entry("口型匹配", "done",
                           f"driven by clean TTS vocals → {result}")
        self._refresh_toolbar()
        self._populate_lipsync_tab(result)
        self._switch_to_next_tab("口型匹配")
        messagebox.showinfo("口型匹配完成",
            f"口型匹配完成。\n"
            f"驱动音频: 干净 TTS 人声\n"
            f"输出: {result}")

    def _on_lip_sync_error(self, error_msg: str):
        if hasattr(self, "_ls_dlg") and self._ls_dlg.winfo_exists():
            self._ls_dlg.destroy()
        self.log.mark_step("口型匹配", "failed")
        self.log.add_entry("口型匹配", "error", error_msg)
        self._refresh_toolbar()
        messagebox.showerror("口型匹配失败", error_msg)

    def _on_lip_sync_cancelled(self):
        if hasattr(self, "_ls_dlg") and self._ls_dlg.winfo_exists():
            self._ls_dlg.destroy()
        self.log.mark_step("口型匹配", "ready")
        self.log.add_entry("口型匹配", "cancelled")
        self._refresh_toolbar()

    # ═══ 人脸增强 (standalone CodeFormer pass) ══════════════════

    def _on_face_enhance(self):
        """Run CodeFormer face enhancement on the lip-sync output.

        Reads the «口型匹配» output video and produces an enhanced copy — so it
        can be re-run / re-tuned without re-running lip-sync.  The «合成视频»
        step prefers this enhanced output when present.
        """
        from ai_movie.face_restore import codeformer_available
        if not codeformer_available():
            messagebox.showwarning(
                "提示", "未找到 CodeFormer 模型（models/codeformer/codeformer.pth），"
                "无法进行人脸增强。")
            return

        ls_data = self.log.step_data.get("口型匹配", {})
        in_video = ls_data.get("output_video")
        if not in_video or not Path(in_video).exists():
            messagebox.showwarning("提示", "请先完成「口型匹配」步骤。")
            return
        in_video = Path(in_video)

        fidelity = float(getattr(self, "_fe_fidelity_var", tk.DoubleVar(value=0.85)).get())
        det_every = int(getattr(self, "_fe_detevery_var", tk.IntVar(value=4)).get())
        occlusion = bool(getattr(self, "_fe_occlusion_var", tk.BooleanVar(value=True)).get())
        protect_lips = bool(getattr(self, "_fe_protect_lips_var", tk.BooleanVar(value=True)).get())

        self.log.mark_step("人脸增强", "running")
        self.log.add_entry("人脸增强", "start",
                           f"fidelity={fidelity:.2f}, det_every={det_every}, "
                           f"occlusion={occlusion}, protect_lips={protect_lips}")
        self._refresh_toolbar()
        self._cancel_requested = False

        # ── progress dialog ─────────────────────────────────────
        self._fe_dlg = tk.Toplevel(self.root)
        self._fe_dlg.title("人脸增强 (CodeFormer)")
        self._fe_dlg.geometry("440x180")
        self._fe_dlg.transient(self.root)
        self._fe_dlg.grab_set()
        self._fe_dlg.resizable(False, False)
        self._fe_dlg.protocol("WM_DELETE_WINDOW", self._on_cancel_face_enhance)

        ttk.Label(self._fe_dlg, text="正在进行人脸增强…",
                  font=(config.CJK_FONT, 11)).pack(pady=(14, 4))
        self._lbl_fe_status = ttk.Label(self._fe_dlg, text="准备中…",
                                        font=(config.CJK_FONT, 9), foreground="#666")
        self._lbl_fe_status.pack()
        self._bar_fe = ttk.Progressbar(self._fe_dlg, length=360, mode="determinate")
        self._bar_fe.pack(padx=20, pady=10)
        ttk.Button(self._fe_dlg, text="取消",
                   command=self._on_cancel_face_enhance).pack(pady=(0, 8))

        self._switch_to_tab("人脸增强")

        threading.Thread(
            target=self._run_face_enhance,
            args=(str(in_video), fidelity, det_every, occlusion, protect_lips),
            daemon=True,
        ).start()

    def _on_cancel_face_enhance(self):
        self._cancel_requested = True

    def _run_face_enhance(self, in_video: str, fidelity: float, det_every: int,
                          occlusion: bool = True, protect_lips: bool = True):
        import traceback as _tb
        from ai_movie import face_restore
        out_path = Path(WORKSPACE_DIR) / "lipsync_enhanced.mp4"

        # Throttle per-frame progress → the Tk event loop (restore_video calls
        # progress_cb once per frame; updating on every frame floods `after`).
        def _progress(done, total):
            if done == 1 or done == total or done % 15 == 0:
                self.root.after(0, lambda d=done, t=total: self._update_face_enhance_progress(d, t))

        def _log(msg):
            self.root.after(0, lambda m=msg: (
                hasattr(self, "_lbl_fe_status") and self._lbl_fe_status.winfo_exists()
                and self._lbl_fe_status.configure(text=m)))

        try:
            result = face_restore.restore_video(
                in_video, out_path,
                fidelity_weight=fidelity,
                det_every=det_every,
                occlusion_aware=occlusion,
                protect_lips=protect_lips,
                progress_cb=_progress,
                cancel_check=lambda: self._cancel_requested,
                log_cb=_log,
            )
        except Exception as e:
            tb = _tb.format_exc()
            print(f"[FaceEnhance ERROR] {tb}", file=sys.stderr)
            err_msg = f"{type(e).__name__}: {e}"
            self.root.after(0, lambda msg=err_msg: self._on_face_enhance_error(msg))
            return
        if result is None:
            self.root.after(0, self._on_face_enhance_cancelled)
            return
        self.root.after(0, lambda: self._on_face_enhance_done(Path(result)))

    def _update_face_enhance_progress(self, cur: int, total: int):
        if not hasattr(self, "_fe_dlg") or not self._fe_dlg.winfo_exists():
            return
        self._bar_fe.configure(maximum=max(total, 1), value=cur)
        self._lbl_fe_status.configure(text=f"修复帧 {cur} / {total}")

    def _on_face_enhance_done(self, result: Path):
        if hasattr(self, "_fe_dlg") and self._fe_dlg.winfo_exists():
            self._fe_dlg.destroy()
        self.log.mark_step("人脸增强", "done")
        self.log.set_step_data("人脸增强", {"output_video": str(result)})
        self.log.add_entry("人脸增强", "done", f"→ {result}")
        self._refresh_toolbar()
        self._populate_face_enhance_tab(result)
        self._switch_to_next_tab("人脸增强")
        messagebox.showinfo("人脸增强完成",
                            f"人脸增强完成。\n输出: {result}\n"
                            f"「合成视频」将优先使用此增强结果。")

    def _on_face_enhance_error(self, error_msg: str):
        if hasattr(self, "_fe_dlg") and self._fe_dlg.winfo_exists():
            self._fe_dlg.destroy()
        self.log.mark_step("人脸增强", "failed")
        self.log.add_entry("人脸增强", "error", error_msg)
        self._refresh_toolbar()
        messagebox.showerror("人脸增强失败", error_msg)

    def _on_face_enhance_cancelled(self):
        if hasattr(self, "_fe_dlg") and self._fe_dlg.winfo_exists():
            self._fe_dlg.destroy()
        self.log.mark_step("人脸增强", "ready")
        self.log.add_entry("人脸增强", "cancelled")
        self._refresh_toolbar()

    # ═══ project save / load ═══════════════════════════════════

    def _auto_save_project(self):
        """Silently save the project (no messagebox). Auto-generates path if needed."""
        if self._project_path is None:
            name = self.log.name or "project"
            path = PROJECTS_DIR / f"{name}.aimovie.json"
            self._project_path = path
        self.log.name = self._project_path.stem
        self.log.save(self._project_path)

    def _on_clear_cache(self):
        """Scan for and clear workspace cache files."""
        items = CacheManager.find_cache_items(WORKSPACE_DIR)
        if not items:
            messagebox.showinfo("清除缓存", "没有可清除的缓存文件。")
            return

        total_size = sum(s for _, s in items)
        preview_lines = [f"  {p}" for p, _ in items[:5]]
        preview = "\n".join(preview_lines)
        if len(items) > 5:
            preview += f"\n  … 以及其他 {len(items) - 5} 项"

        ok = messagebox.askyesno(
            "确认清除缓存",
            f"将删除 {len(items)} 个缓存项"
            f"（约 {CacheManager.format_size(total_size)}）：\n\n"
            f"{preview}\n\n"
            f"注意：播放器缓存会在下次播放时自动重建，\n"
            f"中间项目文件不会被删除。\n\n"
            f"确定清除？"
        )
        if not ok:
            return

        deleted, freed = CacheManager.clear_cache(WORKSPACE_DIR)
        messagebox.showinfo(
            "清除完成",
            f"已删除 {deleted} 个缓存项，"
            f"释放 {CacheManager.format_size(freed)} 空间。"
        )

    def _on_save_project(self):
        if self._project_path is None:
            PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
            path = filedialog.asksaveasfilename(
                title="保存项目", defaultextension=".aimovie.json",
                filetypes=PROJECT_FILETYPES,
                initialdir=str(PROJECTS_DIR),
                initialfile=f"{self.log.name or 'project'}.aimovie.json",
            )
            if not path:
                return
            self._project_path = Path(path)
        self.log.name = self._project_path.stem
        self.log.save(self._project_path)
        messagebox.showinfo("保存项目", f"项目已保存到:\n{self._project_path}")

    def _refresh_all_step_tabs(self):
        """Repopulate every step tab from ``self.log.step_data``.

        Used both when loading a project AND at the end of the one-click
        pipeline — otherwise one-click runs leave the step tabs (translation,
        etc.) showing placeholders until the project JSON is reloaded.
        """
        steps = self.log.steps

        cut_data = self.log.step_data.get("切割视频", {})
        if cut_data.get("segments") and steps.get("切割视频") == "done":
            self._populate_cut_tab(cut_data["segments"])

        split_data = self.log.step_data.get("拆分音轨", {})
        if split_data.get("results") and steps.get("拆分音轨") == "done":
            self._populate_split_tab(split_data["results"])

        trans_data = self.log.step_data.get("转换文字", {})
        if trans_data.get("language"):
            for label, code in LANGUAGES.items():
                if code == trans_data["language"]:
                    self._trans_lang_var.set(label)
                    break
        if trans_data.get("results") and steps.get("转换文字") == "done":
            self._populate_transcribe_tab(trans_data["results"])

        tl_data = self.log.step_data.get("文本翻译", {})
        if tl_data.get("target_lang"):
            from ai_movie.translator import TRANSLATION_TARGET_LANGS
            for label, name in TRANSLATION_TARGET_LANGS.items():
                if name == tl_data["target_lang"] and hasattr(self, "_tl_lang_var"):
                    self._tl_lang_var.set(label)
                    break
        if tl_data.get("segments") and steps.get("文本翻译") == "done":
            self._populate_translate_tab(tl_data["segments"], tl_data.get("target_lang", "Chinese"))

        sep_data = self.log.step_data.get("人声分离", {})
        if sep_data.get("vocals") and sep_data.get("background") and steps.get("人声分离") == "done":
            self._populate_separate_tab(sep_data["vocals"], sep_data["background"])

        gen_data = self.log.step_data.get("人声生成", {})
        if gen_data.get("results") and steps.get("人声生成") == "done":
            self._populate_generate_tab(gen_data["results"])

        remix_data = self.log.step_data.get("重新混音", {})
        if steps.get("重新混音") == "done" and remix_data.get("mixed_audio"):
            p = Path(remix_data["mixed_audio"])
            if p.exists():
                self._populate_remix_tab(p)

        anchor_data = self.log.step_data.get("人物锚定", {})
        if steps.get("人物锚定") == "done" and anchor_data:
            self._populate_anchor_tab(anchor_data.get("female_segments", 0),
                                      anchor_data.get("male_segments", 0),
                                      anchor_data.get("anchor_gender"))

        ls_data = self.log.step_data.get("口型匹配", {})
        if steps.get("口型匹配") == "done" and ls_data.get("output_video"):
            p = Path(ls_data["output_video"])
            if p.exists():
                self._populate_lipsync_tab(p)

        fe_data = self.log.step_data.get("人脸增强", {})
        if steps.get("人脸增强") == "done" and fe_data.get("output_video"):
            p = Path(fe_data["output_video"])
            if p.exists():
                self._populate_face_enhance_tab(p)

        compose_data = self.log.step_data.get("合成视频", {})
        if steps.get("合成视频") == "done" and compose_data.get("output_video"):
            p = Path(compose_data["output_video"])
            if p.exists():
                self._populate_compose_tab(p)

    def _on_load_project(self):
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.askopenfilename(
            title="加载项目", filetypes=PROJECT_FILETYPES,
            initialdir=str(PROJECTS_DIR),
        )
        if not path:
            return
        try:
            self.log = ProjectLog.load(Path(path))
            self._project_path = Path(path)
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载项目文件:\n{e}")
            return

        self._refresh_toolbar()

        # Restore video
        if self.log.video_path and Path(self.log.video_path).exists():
            self._on_stop()
            try:
                self.left_player.load(Path(self.log.video_path))
                self._is_playing = True
                self._btn_play.configure(text="⏸")
            except Exception as e:
                messagebox.showwarning("视频加载失败", f"项目关联的视频无法加载:\n{e}")

        self._refresh_all_step_tabs()

        self.root.title(f"{config.WINDOW_TITLE} - {self.log.name}")
        done_count = sum(1 for v in self.log.steps.values() if v == "done")
        messagebox.showinfo("加载项目",
                            f"项目「{self.log.name}」已加载\n已完成步骤: {done_count}/{len(STEP_NAMES)}")

    # ═══ main area: left player + right work tabs ══════════════

    def _build_main_area(self):
        main = ttk.Frame(self.root)
        main.pack(expand=True, fill="both", padx=6, pady=(6, 0))
        main.columnconfigure(0, weight=2)
        main.columnconfigure(1, weight=3)
        main.rowconfigure(0, weight=1)

        # ── left panel ────────────────────────────────────────
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left.rowconfigure(0, weight=1)
        left.rowconfigure(1, weight=0)

        self.left_player = VideoPlayer(left, channel="left", placeholder="原视频")
        self.left_player.frame.grid(row=0, column=0, sticky="nsew")
        self._build_player_controls(left)

        # ── right panel ───────────────────────────────────────
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self._notebook = ttk.Notebook(right)
        self._notebook.grid(row=0, column=0, sticky="nsew")

        for name in STEP_NAMES:
            if name == "合成音轨":
                continue  # no dedicated tab; results appear in the three sub-step tabs
            tab = ttk.Frame(self._notebook)
            self._notebook.add(tab, text=name)
            self._tab_frames[name] = tab
            if name == "转换文字":
                self._build_transcribe_tab(tab)
            elif name == "文本翻译":
                self._build_translate_tab(tab)
            elif name == "人声分离":
                self._build_separate_tab(tab)
            elif name == "人声生成":
                self._build_generate_tab(tab)
            elif name in ("重新混音", "合成视频", "切割视频", "人物锚定"):
                self._build_output_tab(tab, name)
            elif name == "口型匹配":
                self._build_lipsync_tab(tab)
            elif name == "人脸增强":
                self._build_face_enhance_tab(tab)
            else:
                tk.Label(tab, text="当前为空", fg="#aaa",
                         font=(config.CJK_FONT, 13)).place(relx=0.5, rely=0.5,
                                                             anchor="center")

    def _build_transcribe_tab(self, tab: ttk.Frame):
        """Add language selector bar + backend selector + scrollable results area."""
        # Top bar
        bar = ttk.Frame(tab, padding=(10, 8))
        bar.pack(fill="x")

        ttk.Label(bar, text="语言：",
                  font=(config.CJK_FONT, 10)).pack(side="left", padx=(0, 4))

        self._trans_lang_var = tk.StringVar(value=LANG_LABELS[0])  # 日本語
        lang_combo = ttk.Combobox(
            bar, textvariable=self._trans_lang_var,
            values=LANG_LABELS, state="readonly",
            font=(config.CJK_FONT, 10), width=10,
        )
        lang_combo.pack(side="left", padx=(0, 16))

        ttk.Label(bar, text="引擎：",
                  font=(config.CJK_FONT, 10)).pack(side="left", padx=(0, 4))

        BACKEND_LABELS = ["faster-whisper (CPU+VAD)", "openai-whisper (GPU+VAD)", "自动选择"]
        BACKEND_VALUES = ["faster-whisper", "openai-whisper", "auto"]
        self._trans_backend_var = tk.StringVar(value=BACKEND_LABELS[1])  # default: GPU+VAD
        backend_combo = ttk.Combobox(
            bar, textvariable=self._trans_backend_var,
            values=BACKEND_LABELS, state="readonly",
            font=(config.CJK_FONT, 10), width=24,
        )
        backend_combo.pack(side="left")

        # Store mapping for lookup
        self._backend_map = dict(zip(BACKEND_LABELS, BACKEND_VALUES))

        # Result area (scrollable text, populated after transcription)
        result_frame = ttk.Frame(tab)
        result_frame.pack(expand=True, fill="both", padx=4, pady=(4, 4))

        self._trans_text = tk.Text(
            result_frame, font=(config.CJK_FONT, 9),
            wrap="word", state="disabled",
            bg="#fafafa", relief="flat",
            padx=12, pady=10,
        )
        scroll = ttk.Scrollbar(result_frame, command=self._trans_text.yview)
        self._trans_text.configure(yscrollcommand=scroll.set)

        self._trans_text.pack(side="left", expand=True, fill="both")
        scroll.pack(side="right", fill="y")

        # Placeholder
        self._trans_text.configure(state="normal")
        self._trans_text.insert("1.0", "选择语言和引擎后点击工具栏「转换文字」开始识别。\n"
                                        "默认语言：日本語\n"
                                        "默认引擎：openai-whisper (GPU+VAD，精度更高)")
        self._trans_text.configure(state="disabled")

    # ═══ translate tab ════════════════════════════════════════════

    def _build_translate_tab(self, tab: ttk.Frame):
        """Target language selector + scrollable translation results."""
        from ai_movie.translator import TARGET_LANG_LABELS, TRANSLATION_TARGET_LANGS
        from ai_movie.config import OLLAMA_MODEL

        bar = ttk.Frame(tab, padding=(10, 8))
        bar.pack(fill="x")

        ttk.Label(bar, text="目标语言：",
                  font=(config.CJK_FONT, 10)).pack(side="left", padx=(0, 4))

        self._tl_lang_var = tk.StringVar(value=TARGET_LANG_LABELS[0])
        lang_combo = ttk.Combobox(
            bar, textvariable=self._tl_lang_var,
            values=TARGET_LANG_LABELS, state="readonly",
            font=(config.CJK_FONT, 10), width=14,
        )
        lang_combo.pack(side="left", padx=(0, 16))

        ttk.Label(bar, text="源语言：",
                  font=(config.CJK_FONT, 10)).pack(side="left", padx=(0, 4))
        self._tl_src_label = ttk.Label(bar, text="（自动检测）",
                                        font=(config.CJK_FONT, 10, "bold"))
        self._tl_src_label.pack(side="left")

        # ── translation engine selector ──────────────────────────
        bar2 = ttk.Frame(tab, padding=(10, 4))
        bar2.pack(fill="x")

        ttk.Label(bar2, text="翻译引擎：",
                  font=(config.CJK_FONT, 10)).pack(side="left", padx=(0, 8))

        self._tl_engine_var = tk.StringVar(value="hy-mt2")
        ttk.Radiobutton(
            bar2, text="Hy-MT2 (本地，推荐)",
            variable=self._tl_engine_var, value="hy-mt2",
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            bar2, text="Hy-MT2 + Ollama 润色",
            variable=self._tl_engine_var, value="hy-mt2+polish",
        ).pack(side="left", padx=(0, 12))
        ttk.Radiobutton(
            bar2, text="Ollama 直翻",
            variable=self._tl_engine_var, value="ollama",
        ).pack(side="left", padx=(0, 12))
        # ── legacy / fallback ─────────────────────────────────
        ttk.Radiobutton(
            bar2, text="Hy-MT1.5 (轻量)",
            variable=self._tl_engine_var, value="hy-mt",
        ).pack(side="left")

        # ── Ollama model selector (visible when Ollama is involved) ─
        self._tl_ollama_bar = ttk.Frame(tab, padding=(10, 4))

        ttk.Label(self._tl_ollama_bar, text="Ollama 模型：",
                  font=(config.CJK_FONT, 10)).pack(side="left", padx=(0, 6))

        self._tl_ollama_model_var = tk.StringVar(value=OLLAMA_MODEL)
        self._tl_ollama_combo = ttk.Combobox(
            self._tl_ollama_bar, textvariable=self._tl_ollama_model_var,
            values=[OLLAMA_MODEL], state="readonly",
            font=(config.CJK_FONT, 10), width=28,
        )
        self._tl_ollama_combo.pack(side="left", padx=(0, 8))

        self._tl_ollama_refresh_btn = tk.Button(
            self._tl_ollama_bar, text="🔄 刷新模型列表",
            font=(config.CJK_FONT, 9),
            command=self._on_refresh_ollama_models,
        )
        self._tl_ollama_refresh_btn.pack(side="left")

        self._tl_ollama_status = ttk.Label(
            self._tl_ollama_bar, text="",
            font=(config.CJK_FONT, 8), foreground="#888",
        )
        self._tl_ollama_status.pack(side="left", padx=(8, 0))

        # Show/hide Ollama bar based on engine selection
        def _update_ollama_bar(*_):
            engine = self._tl_engine_var.get()
            if engine in ("hy-mt", "hy-mt2"):
                self._tl_ollama_bar.pack_forget()
            else:
                self._tl_ollama_bar.pack(fill="x", after=bar2)

        self._tl_engine_var.trace_add("write", _update_ollama_bar)
        _update_ollama_bar()  # initial state: hy-mt2 → hide ollama bar

        # Scrollable result area
        result_frame = ttk.Frame(tab)
        result_frame.pack(expand=True, fill="both", padx=4, pady=(4, 4))

        self._tl_text = tk.Text(
            result_frame, font=(config.CJK_FONT, 9),
            wrap="word", state="disabled",
            bg="#fafafa", relief="flat",
            padx=12, pady=10,
        )
        scroll = ttk.Scrollbar(result_frame, command=self._tl_text.yview)
        self._tl_text.configure(yscrollcommand=scroll.set)
        self._tl_text.pack(side="left", expand=True, fill="both")
        scroll.pack(side="right", fill="y")

        self._tl_lang_map = TRANSLATION_TARGET_LANGS

        self._tl_text.configure(state="normal")
        self._tl_text.insert("1.0",
            "选择目标语言后点击工具栏「文本翻译」开始翻译。\n"
            "源语言自动从「转换文字」步骤获取。\n"
            "默认目标语言：汉语 (中文)")
        self._tl_text.configure(state="disabled")

    def _on_refresh_ollama_models(self):
        """Fetch models from Ollama and populate the combobox."""
        from ai_movie.translator import fetch_ollama_models

        self._tl_ollama_refresh_btn.configure(state="disabled", text="⏳ 获取中…")
        self._tl_ollama_status.configure(text="")

        def _fetch():
            models = fetch_ollama_models()
            self.root.after(0, lambda: self._update_ollama_models(models))

        threading.Thread(target=_fetch, daemon=True).start()

    def _update_ollama_models(self, models: list[str]):
        """Populate the Ollama model combobox with fetched models."""
        self._tl_ollama_refresh_btn.configure(state="normal", text="🔄 刷新模型列表")

        if not models:
            self._tl_ollama_status.configure(
                text="⚠ 无法连接 Ollama，请检查服务是否启动",
                foreground="#c00",
            )
            return

        current = self._tl_ollama_model_var.get()
        self._tl_ollama_combo.configure(values=models)

        # Keep current selection if it's in the new list, otherwise pick first
        if current in models:
            self._tl_ollama_model_var.set(current)
        else:
            self._tl_ollama_model_var.set(models[0])

        self._tl_ollama_status.configure(
            text=f"✓ 共 {len(models)} 个模型可用",
            foreground="#155724",
        )

    def _build_lipsync_tab(self, tab: ttk.Frame):
        """Backend selector bar + placeholder result area for lip-sync step.

        Mirrors the structure of _build_transcribe_tab.  The result area
        is replaced on completion by _populate_lipsync_tab.
        """
        # ── Backend selector ────────────────────────────────────────
        bar = ttk.Frame(tab, padding=(10, 8))
        bar.pack(fill="x")

        ttk.Label(bar, text="引擎：",
                  font=(config.CJK_FONT, 10)).pack(side="left", padx=(0, 4))

        LS_BACKEND_LABELS = ["自动检测", "MuseTalk (256x256)", "Wav2Lip (96x96)"]
        self._ls_backend_var = tk.StringVar(value=LS_BACKEND_LABELS[1])  # default: MuseTalk
        backend_combo = ttk.Combobox(
            bar, textvariable=self._ls_backend_var,
            values=LS_BACKEND_LABELS, state="readonly",
            font=(config.CJK_FONT, 10), width=22,
        )
        backend_combo.pack(side="left")

        ttk.Label(bar, text="（人脸增强/人物锚定为独立步骤）",
                  font=(config.CJK_FONT, 9), foreground="#999").pack(side="left", padx=(16, 0))

        # Re-run button persists alongside options after a result is shown.
        self._ls_rerun_btn = tk.Button(bar, text="▶ 运行/重新运行",
                                       font=(config.CJK_FONT, 9),
                                       command=self._on_lip_sync)
        self._ls_rerun_btn.pack(side="right")

        # ── Persistent result container (populate replaces ONLY this) ───
        self._ls_result_frame = ttk.Frame(tab)
        self._ls_result_frame.pack(expand=True, fill="both")
        self._ls_out_text = tk.Text(
            self._ls_result_frame, font=(config.CJK_FONT, 10),
            wrap="word", state="disabled",
            bg="#fafafa", relief="flat",
            padx=20, pady=20, height=8,
        )
        self._ls_out_text.pack(expand=True, fill="both", padx=10, pady=10)
        self._ls_out_text.configure(state="normal")
        self._ls_out_text.insert("1.0",
            "选择引擎后点击工具栏「口型匹配」或上方「运行」执行。\n"
            "默认：MuseTalk（256×256）")
        self._ls_out_text.configure(state="disabled")

    def _build_face_enhance_tab(self, tab: ttk.Frame):
        """Controls for the standalone CodeFormer face-enhancement step."""
        from ai_movie.face_restore import codeformer_available, face_parser_available
        available = codeformer_available()
        parser_ok = face_parser_available()

        bar = ttk.Frame(tab, padding=(10, 8))
        bar.pack(fill="x")

        # Fidelity weight (higher = stay closer to input / more faithful).
        ttk.Label(bar, text="保真度：",
                  font=(config.CJK_FONT, 10)).pack(side="left", padx=(0, 4))
        self._fe_fidelity_var = tk.DoubleVar(value=0.85)
        ttk.Scale(bar, from_=0.0, to=1.0, orient="horizontal", length=120,
                  variable=self._fe_fidelity_var).pack(side="left")
        self._lbl_fe_fidelity = ttk.Label(bar, text="0.85", width=5,
                                           font=(config.MONO_FONT, 9))
        self._lbl_fe_fidelity.pack(side="left", padx=(4, 0))
        self._fe_fidelity_var.trace_add(
            "write",
            lambda *_: self._lbl_fe_fidelity.configure(
                text=f"{self._fe_fidelity_var.get():.2f}"))

        # Detection subsample: detect every N frames (faces move little).
        ttk.Label(bar, text="  每N帧：",
                  font=(config.CJK_FONT, 10)).pack(side="left", padx=(10, 2))
        self._fe_detevery_var = tk.IntVar(value=4)
        ttk.Spinbox(bar, from_=1, to=15, width=3,
                    textvariable=self._fe_detevery_var).pack(side="left")

        # Protect the generated lip-sync mouth (exclude lips from restoration).
        self._fe_protect_lips_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="保护嘴唇",
                        variable=self._fe_protect_lips_var).pack(side="left", padx=(10, 0))

        # Occlusion / false-detection correction (BiSeNet face parsing).
        self._fe_occlusion_var = tk.BooleanVar(value=parser_ok)
        occ_chk = ttk.Checkbutton(bar, text="遮挡/误检修正",
                                  variable=self._fe_occlusion_var)
        occ_chk.pack(side="left", padx=(10, 0))

        # Re-run button (persists alongside options after a result is shown).
        self._fe_rerun_btn = tk.Button(bar, text="▶ 运行/重新运行",
                                       font=(config.CJK_FONT, 9),
                                       command=self._on_face_enhance)
        self._fe_rerun_btn.pack(side="right")

        if not available:
            for w in bar.winfo_children():
                try: w.configure(state="disabled")
                except Exception: pass
        if not parser_ok:
            self._fe_occlusion_var.set(False)
            occ_chk.configure(state="disabled")

        # Persistent result container — populate replaces ONLY this, not the bar.
        self._fe_result_frame = ttk.Frame(tab)
        self._fe_result_frame.pack(expand=True, fill="both")
        self._fe_out_text = tk.Text(
            self._fe_result_frame, font=(config.CJK_FONT, 10),
            wrap="word", state="disabled",
            bg="#fafafa", relief="flat",
            padx=20, pady=20, height=8,
        )
        self._fe_out_text.pack(expand=True, fill="both", padx=10, pady=10)
        self._fe_out_text.configure(state="normal")
        if available:
            self._fe_out_text.insert("1.0",
                "对「口型匹配」的输出做 CodeFormer 人脸增强（修复 MuseTalk 重绘的嘴部）。\n\n"
                "点击工具栏「人脸增强」或上方「运行」执行。\n"
                "· 保真度越高越贴近生成口型（默认 0.85）。\n"
                "· 保护嘴唇：只锐化皮肤/下巴，生成的口型完全不改。\n"
                "· 完成后「合成视频」会优先使用增强结果。")
        else:
            self._fe_out_text.insert("1.0",
                "未找到 CodeFormer 模型：models/codeformer/codeformer.pth\n"
                "放入模型后可启用人脸增强。")
        self._fe_out_text.configure(state="disabled")

    def _populate_face_enhance_tab(self, output_path: Path):
        """Show the finished result WITHOUT destroying the option bar."""
        frame = getattr(self, "_fe_result_frame", None)
        if frame is None or not frame.winfo_exists():
            return
        for w in frame.winfo_children():
            w.destroy()

        f = ttk.Frame(frame, padding=20)
        f.pack(expand=True)
        tk.Label(f, text="✓ 人脸增强完成", font=(config.CJK_FONT, 14, "bold"),
                 fg="#155724").pack(pady=(0, 12))

        info = ttk.LabelFrame(f, text="输出文件", padding=10)
        info.pack(fill="x", pady=(0, 16))
        p = Path(output_path)
        size_mb = p.stat().st_size / 1e6 if p.exists() else 0
        tk.Label(info, text=str(p), font=(config.MONO_FONT, 8),
                 fg="#555", wraplength=500).pack(anchor="w")
        tk.Label(info, text=f"大小：{size_mb:.1f} MB", font=(config.CJK_FONT, 9),
                 fg="#666").pack(anchor="w", pady=(4, 0))

        btn = ttk.Frame(f)
        btn.pack()
        tk.Button(btn, text="▶  在左侧播放", font=(config.CJK_FONT, 11),
                  command=lambda: self._load_lipsync_video(p)).pack(side="left", padx=8)
        tk.Button(btn, text="📂  打开所在目录", font=(config.CJK_FONT, 10),
                  command=lambda: subprocess.Popen(
                      ["xdg-open", str(p.parent)])).pack(side="left", padx=8)
        self._load_lipsync_video(p)

    def _build_output_tab(self, tab: ttk.Frame, name: str):
        """Simple tab showing the step's output path when done."""
        text = tk.Text(
            tab, font=(config.CJK_FONT, 10),
            wrap="word", state="disabled",
            bg="#fafafa", relief="flat",
            padx=20, pady=20, height=8,
        )
        text.pack(expand=True, fill="both", padx=10, pady=10)
        text.configure(state="normal")
        text.insert("1.0", f"点击工具栏「{name}」执行。")
        text.configure(state="disabled")

        # Store reference keyed by step name
        attr = {"合成音轨": "_syn_out_text", "合成视频": "_cmp_out_text"}.get(name, "_out_text")
        setattr(self, attr, text)

    # ═══ player controls ═══════════════════════════════════════

    def _build_player_controls(self, parent):
        bar = ttk.Frame(parent, padding=(8, 6))
        bar.grid(row=1, column=0, sticky="ew")

        btn_frame = ttk.Frame(bar)
        btn_frame.pack(fill="x")

        self._btn_mute = tk.Button(btn_frame, text="🔊", font=(config.SYMBOL_FONT, 12),
                                   width=3, relief="flat", bg="#f0f0f0",
                                   command=self._on_toggle_mute)
        self._btn_mute.pack(side="left", padx=(0, 8))

        transport = ttk.Frame(btn_frame)
        transport.pack(side="left", expand=True)

        self._btn_rew = tk.Button(transport, text="⏪ 10s",
                                  font=(config.CJK_FONT, 10),
                                  relief="flat", bg="#f0f0f0",
                                  command=self._on_rewind)
        self._btn_rew.pack(side="left", padx=2)

        self._btn_play = tk.Button(transport, text="▶",
                                   font=(config.SYMBOL_FONT, 14, "bold"),
                                   width=3, relief="flat", bg="#f0f0f0",
                                   command=self._on_toggle_play)
        self._btn_play.pack(side="left", padx=2)

        self._btn_stop = tk.Button(transport, text="⏹", font=(config.SYMBOL_FONT, 14),
                                   width=3, relief="flat", bg="#f0f0f0",
                                   command=self._on_stop)
        self._btn_stop.pack(side="left", padx=2)

        self._btn_fwd = tk.Button(transport, text="10s ⏩",
                                  font=(config.CJK_FONT, 10),
                                  relief="flat", bg="#f0f0f0",
                                  command=self._on_forward)
        self._btn_fwd.pack(side="left", padx=2)

        prog_frame = ttk.Frame(bar)
        prog_frame.pack(fill="x", pady=(6, 0))

        self._lbl_cur = tk.Label(prog_frame, text="0:00", font=(config.MONO_FONT, 9),
                                 fg="#666", bg=self.root.cget("bg"))
        self._lbl_cur.pack(side="left")

        self._slider = ttk.Scale(prog_frame, from_=0, to=1000,
                                 orient="horizontal",
                                 command=self._on_slider_drag)
        self._slider.pack(side="left", fill="x", expand=True, padx=6)

        self._lbl_total = tk.Label(prog_frame, text="0:00", font=(config.MONO_FONT, 9),
                                   fg="#666", bg=self.root.cget("bg"))
        self._lbl_total.pack(side="right")

    # ═══ periodic sync ═════════════════════════════════════════

    def _start_sync_timer(self):
        self._sync_ui()
        self.root.after(150, self._start_sync_timer)

    def _sync_ui(self):
        if self._seeking:
            return
        self._ui_updating = True
        dur = self.left_player.get_duration_ms()
        cur = self.left_player.get_time_ms()
        if dur > 0:
            self._slider.configure(to=dur)
            self._slider.set(cur)
        self._lbl_total.configure(text=_fmt_time(dur))
        self._lbl_cur.configure(text=_fmt_time(cur))
        self._ui_updating = False

        playing = self.left_player.is_playing
        if playing != self._is_playing:
            self._is_playing = playing
            self._btn_play.configure(text="⏸" if playing else "▶")

    # ═══ transport actions ═════════════════════════════════════

    def _on_toggle_play(self):
        if self._is_playing:
            self.left_player.pause()
        elif self.left_player.loaded:
            self.left_player.play()

    def _on_stop(self):
        self.left_player.stop()
        self._is_playing = False
        self._btn_play.configure(text="▶")
        self._slider.set(0)
        self._lbl_cur.configure(text="0:00")

    def _on_rewind(self):
        self.left_player.seek_relative(-10)

    def _on_forward(self):
        self.left_player.seek_relative(+10)

    def _on_toggle_mute(self):
        muted = self.left_player.toggle_mute()
        self._btn_mute.configure(text="🔇" if muted else "🔊")

    def _on_slider_drag(self, value: str):
        if self._ui_updating:
            return
        ms = int(float(value))
        self._seeking = True
        dur = self.left_player.get_duration_ms()
        self.left_player.seek_absolute(ms / max(1, dur))
        self._lbl_cur.configure(text=_fmt_time(ms))
        self._seeking = False

    # ═══ file load ═════════════════════════════════════════════

    def _on_load_video(self):
        filepath = filedialog.askopenfilename(title="选择视频文件",
                                              filetypes=VIDEO_FILETYPES)
        if not filepath:
            return
        filepath = Path(filepath)
        self._on_stop()
        try:
            self.left_player.load(filepath)
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载视频文件:\n{e}")
            return
        self.log.name = filepath.stem
        self.log.video_path = str(filepath)
        self.log.add_entry("项目", "加载视频", filepath.name)
        self._is_playing = True
        self._btn_play.configure(text="⏸")
        self.root.title(f"{config.WINDOW_TITLE} - {self.log.name}")

    # ═══ exit ══════════════════════════════════════════════════

    def _on_exit(self):
        active = task_manager.get_active_names()
        if not active:
            self._shutdown()
            return
        task_list = "\n".join(f"  • {name}" for name in active)
        confirm = messagebox.askyesno(
            title="确认退出",
            message=f"以下任务仍在运行中:\n\n{task_list}\n\n是否强制退出？",
            icon="warning",
        )
        if confirm:
            still_running = task_manager.shutdown(timeout=2.0)
            if still_running:
                messagebox.showwarning(
                    title="退出",
                    message="部分任务未能及时停止:\n\n"
                    + "\n".join(f"  • {n}" for n in still_running),
                )
            self._shutdown()

    def _on_close(self):
        self._on_exit()

    def _shutdown(self):
        self.left_player.destroy()
        self.root.destroy()

    # ═══ lifecycle ═════════════════════════════════════════════

    def run(self):
        self.root.mainloop()


def run_gui():
    App().run()
