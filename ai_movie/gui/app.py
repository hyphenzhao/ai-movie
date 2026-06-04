"""Main application window."""

import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from ai_movie import config
from ai_movie.config import WORKSPACE_DIR
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
        ("合成视频",   "_on_compose_video"),
    ]

    TOOLBAR_COLORS = {
        "locked":   ("#d0d0d0", "#999999", "sunken",   "disabled"),
        "ready":    ("#ffffff", "#000000", "raised",   "normal"),
        "running":  ("#fff3cd", "#856404", "raised",   "disabled"),
        "done":     ("#d4edda", "#155724", "raised",   "normal"),
        "failed":   ("#f8d7da", "#721c24", "raised",   "normal"),
    }

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
        _btn("合成视频", "_on_compose_video")

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
        except Exception as e:
            self.root.after(0, lambda: self._on_cut_error(str(e)))
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
        except Exception as e:
            self.root.after(0, lambda: self._on_split_error(str(e)))
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

        # Flatten segments
        segments = []
        for r in asr_results:
            if "error" in r:
                continue
            src = r.get("source", "")
            for seg in r.get("segments", []):
                segments.append({
                    "text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
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

        self.log.mark_step("文本翻译", "running")
        self.log.add_entry("文本翻译", "start",
                           f"{src_lang}→{target_lang} segs={len(segments)}")
        self._refresh_toolbar()
        self._cancel_requested = False

        total = len(segments)

        # Progress dialog
        self._tl_dlg = tk.Toplevel(self.root)
        self._tl_dlg.title("文本翻译")
        self._tl_dlg.geometry("420x200")
        self._tl_dlg.transient(self.root)
        self._tl_dlg.grab_set()
        self._tl_dlg.protocol("WM_DELETE_WINDOW", self._on_cancel_translate)
        self._tl_dlg.resizable(False, False)

        ttk.Label(self._tl_dlg, text="正在翻译…",
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
            args=(segments, target_lang, src_lang, total),
            daemon=True,
        ).start()

    def _on_cancel_translate(self):
        self._cancel_requested = True

    def _run_translate(self, segments, target_lang, src_lang, total):
        from ai_movie.translator import translate
        try:
            results = translate(
                segments,
                target_lang=target_lang,
                src_lang=src_lang,
                progress_cb=self._on_tl_progress,
                cancel_check=lambda: self._cancel_requested,
            )
        except Exception as e:
            self.root.after(0, lambda: self._on_translate_error(str(e)))
            return
        self.root.after(0, lambda: self._on_translate_done(results, target_lang))

    def _on_tl_progress(self, current: int, total: int):
        self.root.after(0, lambda: self._update_tl_dialog(current, total))

    def _update_tl_dialog(self, current: int, total: int):
        if not hasattr(self, "_tl_dlg") or not self._tl_dlg.winfo_exists():
            return
        self._bar_tl.configure(value=current)
        self._lbl_tl_prog.configure(text=f"{current} / {total}")
        self._lbl_tl_seg.configure(text=f"已翻译: {current} 句")

    def _on_translate_done(self, results, target_lang):
        if hasattr(self, "_tl_dlg") and self._tl_dlg.winfo_exists():
            self._tl_dlg.destroy()

        self.log.mark_step("文本翻译", "done")
        self.log.set_step_data("文本翻译", {
            "target_lang": target_lang,
            "segments": results,
            "count": len(results),
        })
        self.log.add_entry("文本翻译", "done",
                           f"{len(results)} segments → {target_lang}")
        self._refresh_toolbar()
        self._populate_translate_tab(results, target_lang)

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
        self._tl_text.insert("1.0",
            f"目标语言: {tl_label}  —  共 {len(segments)} 句\n\n")

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
        self._sep_result_frame = ttk.Frame(tab)
        self._sep_result_frame.pack(expand=True, fill="both")

        placeholder = tk.Label(self._sep_result_frame, text="点击工具栏「人声分离」执行。",
                               fg="#aaa", font=(config.CJK_FONT, 13))
        placeholder.place(relx=0.5, rely=0.5, anchor="center")

    def _build_generate_tab(self, tab: ttk.Frame):
        """Tab showing generated speech segments with play buttons."""
        # Mode selector bar
        mode_bar = ttk.Frame(tab, padding=(10, 6, 10, 2))
        mode_bar.pack(fill="x", side="top")
        ttk.Label(mode_bar, text="合成模式：",
                  font=(config.CJK_FONT, 10)).pack(side="left")
        self._tts_mode_var = tk.StringVar(value="gender")
        ttk.Radiobutton(mode_bar, text="性别匹配（清晰，推荐）",
                        variable=self._tts_mode_var,
                        value="gender").pack(side="left", padx=(8, 4))
        ttk.Radiobutton(mode_bar, text="声音克隆（近似原声）",
                        variable=self._tts_mode_var,
                        value="clone").pack(side="left", padx=4)

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
        """Show generated speech segments with play buttons + translated text."""
        for w in self._gen_scroll_frame.winfo_children():
            w.destroy()

        ok = [r for r in results if r.get("audio")]
        if not ok:
            tk.Label(self._gen_scroll_frame, text="无生成结果",
                     fg="#aaa", font=(config.CJK_FONT, 13)).pack(expand=True, pady=80)
            return

        # Header
        hdr = ttk.Frame(self._gen_scroll_frame)
        hdr.pack(fill="x", padx=16, pady=(12, 8))
        tk.Label(hdr, text=f"✓ 共生成 {len(ok)} 个语音片段",
                 font=(config.CJK_FONT, 12, "bold"), fg="#155724").pack(side="left")

        for i, seg in enumerate(ok):
            card = ttk.Frame(self._gen_scroll_frame, relief="solid", borderwidth=1)
            card.pack(fill="x", padx=16, pady=4, ipady=6)

            # Segment header
            ch = ttk.Frame(card); ch.pack(fill="x", padx=12, pady=(8, 2))
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            ts = f"{int(start // 60)}:{start % 60:04.1f} — {int(end // 60)}:{end % 60:04.1f}"
            tk.Label(ch, text=f"片段 {i+1}  [{ts}]",
                     font=(config.CJK_FONT, 9, "bold")).pack(side="left")

            dur_str = f"{end - start:.1f}s"
            tk.Label(ch, text=dur_str, font=(config.CJK_FONT, 8), fg="#888").pack(side="right")

            # Text: original
            txt_frame = ttk.Frame(card); txt_frame.pack(fill="x", padx=12, pady=(2, 0))
            orig = seg.get("text", "")
            trans = seg.get("text_translated", "")
            if orig:
                tk.Label(txt_frame, text=f"原文: {orig}",
                         font=(config.CJK_FONT, 8), fg="#666",
                         wraplength=600, anchor="w", justify="left").pack(anchor="w")
            if trans:
                tk.Label(txt_frame, text=f"译文: {trans}",
                         font=(config.CJK_FONT, 9), fg="#333",
                         wraplength=600, anchor="w", justify="left").pack(anchor="w")

            # Play button
            audio_path = Path(seg["audio"])
            btn_frame = ttk.Frame(card); btn_frame.pack(fill="x", padx=12, pady=(4, 8))
            size_kb = audio_path.stat().st_size / 1024 if audio_path.exists() else 0
            tk.Label(btn_frame, text=f"{audio_path.name}  ({size_kb:.0f} KB)",
                     font=(config.CJK_FONT, 7), fg="#aaa").pack(side="left")
            tk.Button(btn_frame, text="▶ 播放", font=(config.CJK_FONT, 9),
                      command=lambda p=audio_path: self._play_segment(p)).pack(side="right")

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
        ref_audio = self._get_reference_audio()
        if ref_audio is None:
            return

        self.log.mark_step("人声分离", "running")
        self.log.add_entry("人声分离", "start")
        self._refresh_toolbar()
        self._cancel_requested = False

        self._sep_dlg = tk.Toplevel(self.root)
        self._sep_dlg.title("人声分离")
        self._sep_dlg.geometry("360x140")
        self._sep_dlg.transient(self.root)
        self._sep_dlg.grab_set()
        self._sep_dlg.resizable(False, False)

        ttk.Label(self._sep_dlg, text="正在分离人声与背景音…",
                  font=(config.CJK_FONT, 11)).pack(pady=(12, 8))
        self._bar_sep = ttk.Progressbar(self._sep_dlg, length=320,
                                         mode="indeterminate")
        self._bar_sep.pack(padx=20, pady=10)
        self._bar_sep.start(10)

        threading.Thread(target=self._run_separate, args=(ref_audio,), daemon=True).start()

    def _run_separate(self, ref_audio):
        import ai_movie.composer as composer
        try:
            sep = composer.separate_vocals(ref_audio)
        except Exception as e:
            self.root.after(0, lambda: self._on_separate_error(str(e)))
            return
        self.root.after(0, lambda: self._on_separate_done(sep))

    def _on_separate_done(self, sep):
        if hasattr(self, "_sep_dlg") and self._sep_dlg.winfo_exists():
            self._sep_dlg.destroy()
        self.log.mark_step("人声分离", "done")
        self.log.set_step_data("人声分离", {
            "vocals": str(sep["vocals"]),
            "background": str(sep["background"]),
        })
        self.log.add_entry("人声分离", "done",
                           f"vocals={sep['vocals']}, bg={sep['background']}")
        self._refresh_toolbar()
        self._populate_separate_tab(sep["vocals"], sep["background"])
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

        # Pre-load CosyVoice model on main thread to avoid thread-safety issues
        import ai_movie.tts as tts_mod
        tts_mod._load_model()

        # Detect gender / extract reference segment (a few seconds, acceptable on main thread)
        mode = getattr(self, "_tts_mode_var", None)
        mode = mode.get() if mode else "gender"
        out_dir = ensure_dir(WORKSPACE_DIR / "synthesized")
        ref_audio, ref_text, ref_method = tts_mod.prepare_reference(
            vocals_path, mode=mode, cache_dir=out_dir
        )
        # Update dialog to show detected info
        gender_hint = ""
        if mode == "gender":
            gender_hint = "（女声）" if ref_text else "（男声）"
        self._lbl_gen_prog.configure(
            text=f"模式：{'性别匹配' if mode == 'gender' else '声音克隆'}{gender_hint}")

        # Process segments one at a time on main thread via root.after()
        self._gen_segments = segments
        self._gen_ref_audio = ref_audio
        self._gen_ref_text = ref_text
        self._gen_ref_method = ref_method
        self._gen_output_dir = out_dir
        self._gen_idx = 0
        self._gen_results: list[dict] = []
        self._gen_tts_mod = tts_mod
        self.root.after(100, self._process_next_generate)

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
        self._lbl_gen_prog.configure(text=f"{idx} / {total}")

        seg = segments[idx]
        text = seg.get("text_translated", "").strip()
        if not text:
            self._gen_results.append({**seg, "audio": None})
            self._gen_idx = idx + 1
            self.root.after(50, self._process_next_generate)
            return

        # Run inference on main thread
        try:
            import soundfile as sf
            audio_np = mod.call_tts(
                mod._model, text,
                self._gen_ref_audio, self._gen_ref_text, self._gen_ref_method,
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
            audio_paths = [
                Path(r["audio"]) if r.get("audio") else None
                for r in gen_results
            ]
            mixed_path = out_dir / "final_audio.wav"
            composer.mix_audio(
                [p for p in audio_paths if p is not None],
                bg_path, mixed_path,
            )
        except Exception as e:
            self.root.after(0, lambda: self._on_remix_error(str(e)))
            return
        self.root.after(0, lambda: self._on_remix_done(mixed_path))

    def _on_remix_done(self, mixed_path):
        if hasattr(self, "_mix_dlg") and self._mix_dlg.winfo_exists():
            self._mix_dlg.destroy()
        self.log.mark_step("重新混音", "done")
        self.log.set_step_data("重新混音", {"mixed_audio": str(mixed_path)})
        self.log.add_entry("重新混音", "done", str(mixed_path))
        self._refresh_toolbar()
        messagebox.showinfo("混音完成", f"混音完成：{mixed_path}")

    def _on_remix_error(self, error_msg):
        if hasattr(self, "_mix_dlg") and self._mix_dlg.winfo_exists():
            self._mix_dlg.destroy()
        self.log.mark_step("重新混音", "failed")
        self.log.add_entry("重新混音", "error", error_msg)
        self._refresh_toolbar()
        messagebox.showerror("混音失败", error_msg)

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
        ref_audio = self._get_reference_audio()
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
            sep = composer.separate_vocals(ref_audio)
            if self._cancel_requested:
                return
            self.log.mark_step("人声分离", "done")
            self.log.set_step_data("人声分离", {
                "vocals": str(sep["vocals"]), "background": str(sep["background"])})
            self.root.after(0, lambda: self._populate_separate_tab(
                str(sep["vocals"]), str(sep["background"])))
        except Exception as e:
            self.root.after(0, lambda: self._on_synthesize_error(str(e)))
            return
        # Stage 2 must run on main thread
        self.root.after(0, lambda: self._start_syn_tts(segments, sep))

    def _start_syn_tts(self, segments, sep):
        """Called on main thread: initialize TTS loop for 合成音轨."""
        if self._cancel_requested:
            return
        import ai_movie.tts as tts_mod
        self._update_syn_stage("Step 2/3: 合成语音…")
        tts_mod._load_model()
        out_dir = ensure_dir(WORKSPACE_DIR / "synthesized")
        mode = getattr(self, "_tts_mode_var", None)
        mode = mode.get() if mode else "gender"
        ref_audio, ref_text, ref_method = tts_mod.prepare_reference(
            sep["vocals"], mode=mode, cache_dir=out_dir
        )
        self._syn_segments = segments
        self._syn_sep = sep
        self._syn_vocals_ref = ref_audio
        self._syn_ref_text = ref_text
        self._syn_ref_method = ref_method
        self._syn_output_dir = out_dir
        self._syn_idx = 0
        self._syn_results: list[dict] = []
        self._syn_tts_mod = tts_mod
        self.root.after(50, self._process_next_syn_segment)

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
        self._update_syn_stage(f"Step 2/3: 合成语音… ({idx + 1}/{total})")
        seg = segments[idx]
        text = seg.get("text_translated", "").strip()
        if not text:
            self._syn_results.append({**seg, "audio": None})
            self._syn_idx = idx + 1
            self.root.after(50, self._process_next_syn_segment)
            return
        mod = self._syn_tts_mod
        try:
            import soundfile as sf
            audio_np = mod.call_tts(
                mod._model, text,
                self._syn_vocals_ref, self._syn_ref_text, self._syn_ref_method,
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
            audio_paths = [Path(r["audio"]) for r in results if r.get("audio")]
            mixed_path = out_dir / "final_audio.wav"
            composer.mix_audio(audio_paths, sep["background"], mixed_path)
            if self._cancel_requested:
                return
            self.log.mark_step("重新混音", "done")
            self.log.set_step_data("重新混音", {"mixed_audio": str(mixed_path)})
        except Exception as e:
            self.root.after(0, lambda: self._on_synthesize_error(str(e)))
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

        # Get silent video from demux step
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
        if video_path is None:
            messagebox.showwarning("提示", "未找到无声视频文件。")
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
                  font=(config.CJK_FONT, 11)).pack(pady=(12, 8))
        self._bar_cmp = ttk.Progressbar(self._cmp_dlg, length=320,
                                         mode="indeterminate")
        self._bar_cmp.pack(padx=20, pady=10)
        self._bar_cmp.start(10)

        threading.Thread(
            target=self._run_compose,
            args=(video_path, Path(audio_path)),
            daemon=True,
        ).start()

    def _run_compose(self, video_path, audio_path):
        import ai_movie.composer as composer
        try:
            out_dir = ensure_dir(WORKSPACE_DIR / "output")
            out_path = out_dir / f"{Path(video_path).stem}_dubbed.mp4"
            composer.compose_video(video_path, audio_path, out_path)
        except Exception as e:
            self.root.after(0, lambda: self._on_compose_error(str(e)))
            return
        self.root.after(0, lambda: self._on_compose_done(out_path))

    def _on_compose_done(self, out_path):
        if hasattr(self, "_cmp_dlg") and self._cmp_dlg.winfo_exists():
            self._cmp_dlg.destroy()
        self.log.mark_step("合成视频", "done")
        self.log.set_step_data("合成视频", {"output_video": str(out_path)})
        self.log.add_entry("合成视频", "done", str(out_path))
        self._refresh_toolbar()

        # Populate output tab
        if hasattr(self, "_cmp_out_text"):
            self._cmp_out_text.configure(state="normal")
            self._cmp_out_text.delete("1.0", "end")
            self._cmp_out_text.insert("1.0",
                f"✓ 合成完成\n\n输出视频：\n{out_path}")
            self._cmp_out_text.configure(state="disabled")

        messagebox.showinfo("合成完成",
            f"配音视频已生成：\n{out_path}")

    def _on_compose_error(self, error_msg: str):
        if hasattr(self, "_cmp_dlg") and self._cmp_dlg.winfo_exists():
            self._cmp_dlg.destroy()
        self.log.mark_step("合成视频", "failed")
        self.log.add_entry("合成视频", "error", error_msg)
        self._refresh_toolbar()
        messagebox.showerror("合成失败", error_msg)

    # ═══ remaining toolbar stubs ════════════════════════════════

    def _on_anchor_person(self):
        self.log.mark_step("人物锚定", "running"); self.log.add_entry("人物锚定", "start"); self._refresh_toolbar()
        self.log.mark_step("人物锚定", "done"); self.log.add_entry("人物锚定", "done", "placeholder"); self._refresh_toolbar()

    def _on_lip_sync(self):
        self.log.mark_step("口型匹配", "running"); self.log.add_entry("口型匹配", "start"); self._refresh_toolbar()
        self.log.mark_step("口型匹配", "done"); self.log.add_entry("口型匹配", "done", "placeholder"); self._refresh_toolbar()

    # ═══ project save / load ═══════════════════════════════════

    def _on_save_project(self):
        if self._project_path is None:
            path = filedialog.asksaveasfilename(
                title="保存项目", defaultextension=".aimovie.json",
                filetypes=PROJECT_FILETYPES,
                initialfile=f"{self.log.name or 'project'}.aimovie.json",
            )
            if not path:
                return
            self._project_path = Path(path)
        self.log.name = self._project_path.stem
        self.log.save(self._project_path)
        messagebox.showinfo("保存项目", f"项目已保存到:\n{self._project_path}")

    def _on_load_project(self):
        path = filedialog.askopenfilename(title="加载项目", filetypes=PROJECT_FILETYPES)
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

        # Restore cut tab if step was done
        cut_data = self.log.step_data.get("切割视频", {})
        segments = cut_data.get("segments", [])
        if segments and self.log.steps.get("切割视频") == "done":
            self._populate_cut_tab(segments)

        # Restore split tab if step was done
        split_data = self.log.step_data.get("拆分音轨", {})
        results = split_data.get("results", [])
        if results and self.log.steps.get("拆分音轨") == "done":
            self._populate_split_tab(results)

        # Restore transcribe language & tab
        trans_data = self.log.step_data.get("转换文字", {})
        if trans_data.get("language"):
            lang_code = trans_data["language"]
            for label, code in LANGUAGES.items():
                if code == lang_code:
                    self._trans_lang_var.set(label)
                    break
        trans_results = trans_data.get("results", [])
        if trans_results and self.log.steps.get("转换文字") == "done":
            self._populate_transcribe_tab(trans_results)

        # Restore translate tab
        tl_data = self.log.step_data.get("文本翻译", {})
        if tl_data.get("target_lang"):
            from ai_movie.translator import TARGET_LANG_LABELS, TRANSLATION_TARGET_LANGS
            tgt_lang = tl_data["target_lang"]
            for label, name in TRANSLATION_TARGET_LANGS.items():
                if name == tgt_lang:
                    if hasattr(self, "_tl_lang_var"):
                        self._tl_lang_var.set(label)
                    break
        tl_segments = tl_data.get("segments", [])
        if tl_segments and self.log.steps.get("文本翻译") == "done":
            self._populate_translate_tab(tl_segments, tl_data.get("target_lang", "Chinese"))

        # Restore separate tab (人声分离)
        sep_data = self.log.step_data.get("人声分离", {})
        if sep_data.get("vocals") and sep_data.get("background"):
            if self.log.steps.get("人声分离") == "done":
                self._populate_separate_tab(sep_data["vocals"], sep_data["background"])

        # Restore generate tab (人声生成)
        gen_data = self.log.step_data.get("人声生成", {})
        gen_results = gen_data.get("results", [])
        if gen_results and self.log.steps.get("人声生成") == "done":
            self._populate_generate_tab(gen_results)

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
            elif name in ("重新混音", "合成视频", "切割视频", "人物锚定", "口型匹配"):
                self._build_output_tab(tab, name)
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

        BACKEND_LABELS = ["faster-whisper (CPU)", "openai-whisper (GPU)", "自动选择"]
        BACKEND_VALUES = ["faster-whisper", "openai-whisper", "auto"]
        self._trans_backend_var = tk.StringVar(value=BACKEND_LABELS[0])
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
                                        "默认引擎：faster-whisper (CPU，断句更准)")
        self._trans_text.configure(state="disabled")

    # ═══ translate tab ════════════════════════════════════════════

    def _build_translate_tab(self, tab: ttk.Frame):
        """Target language selector + scrollable translation results."""
        from ai_movie.translator import TARGET_LANG_LABELS, TRANSLATION_TARGET_LANGS

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
