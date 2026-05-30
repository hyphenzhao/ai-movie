"""Main application window."""

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from ai_movie.config import VIDEO_EXTENSIONS, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT
from ai_movie.cutter import cut_video
from ai_movie.asr import LANGUAGES, LANG_LABELS, transcribe_all
from ai_movie.demuxer import demux_all
from ai_movie.gui.player import VideoPlayer
from ai_movie.project_log import ProjectLog, STEP_NAMES
from ai_movie.task_manager import task_manager

VIDEO_FILETYPES = [
    ("视频文件", " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS))),
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
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.minsize(960, 550)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

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
        bar = tk.Frame(self.root, bg="#e8e8e8", height=44)
        bar.pack(fill="x", side="top")
        inner = tk.Frame(bar, bg="#e8e8e8")
        inner.pack(pady=5)

        for i, (label, _method) in enumerate(self._PIPELINE_STEPS):
            if i > 0:
                tk.Label(inner, text="→", font=("Microsoft YaHei", 10),
                         fg="#999", bg="#e8e8e8").pack(side="left", padx=2)
            btn = tk.Button(inner, text=label,
                            font=("Microsoft YaHei", 9),
                            width=10, height=2,
                            command=getattr(self, _method))
            btn.pack(side="left", padx=1)
            self._tb_btns[label] = btn

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
                  font=("Microsoft YaHei", 11)).pack(pady=(12, 8))

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
                     font=("Microsoft YaHei", 13)).place(relx=0.5, rely=0.5,
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
                frame, text="▶", font=("Segoe UI", 18, "bold"),
                fg="#fff", bg="#555555", activebackground="#333333",
                relief="flat", bd=0, width=3,
                command=lambda p=Path(seg["path"]): self._play_segment(p),
            )
            play_btn.place(relx=0.5, rely=0.5, anchor="center", width=50, height=50)

            # Segment label
            dur_str = _fmt_time(int(seg.get("duration", 0) * 1000))
            tk.Label(frame, text=f"片段 {seg['index']}  ({dur_str})",
                     font=("Microsoft YaHei", 8), fg="#555").pack(pady=(2, 4))

            col += 1
            if col >= cols:
                col = 0
                row += 1

        scroll_frame.columnconfigure(tuple(range(cols)), weight=1)

    def _play_segment(self, seg_path: Path):
        """Open a popup window with a VLC player for a single segment."""
        win = tk.Toplevel(self.root)
        win.title(f"播放: {seg_path.name}")
        win.geometry("720x480")
        win.transient(self.root)

        player = VideoPlayer(win, channel="left", placeholder="")
        player.frame.pack(expand=True, fill="both")
        player.load(seg_path)

        # Simple controls
        ctrl = ttk.Frame(win, padding=(8, 4))
        ctrl.pack(fill="x", side="bottom")

        def _toggle():
            if player.is_playing:
                player.pause()
                btn_play.configure(text="▶")
            else:
                player.play()
                btn_play.configure(text="⏸")

        btn_play = tk.Button(ctrl, text="⏸", font=("Segoe UI", 12),
                             width=3, command=_toggle)
        btn_play.pack(side="left", padx=4)

        tk.Button(ctrl, text="⏹", font=("Segoe UI", 12), width=3,
                  command=player.stop).pack(side="left", padx=4)

        def _on_close():
            player.destroy()
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close)

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
                  font=("Microsoft YaHei", 11)).pack(pady=(12, 8))
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
                     font=("Microsoft YaHei", 13)).place(relx=0.5, rely=0.5,
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
                         font=("Microsoft YaHei", 10, "bold"),
                         fg="#721c24").pack(side="left")
                tk.Label(hdr, text=r["error"], fg="#999",
                         font=("Microsoft YaHei", 8),
                         wraplength=500).pack(side="left", padx=8)
                continue

            dur = _fmt_time(int(r.get("duration", 0) * 1000))
            tk.Label(hdr, text=f"▸ {label}",
                     font=("Microsoft YaHei", 10, "bold")).pack(side="left")
            tk.Label(hdr, text=f"时长 {dur}",
                     font=("Microsoft YaHei", 8), fg="#888").pack(side="right")

            # Body: silent video + audio side by side
            body = ttk.Frame(card); body.pack(fill="x", padx=10, pady=(2, 8))

            # -- silent video --
            vid_frame = ttk.Frame(body, relief="groove", borderwidth=1)
            vid_frame.pack(side="left", padx=(0, 12))

            video_path = Path(r["video"])
            if video_path.exists():
                size_mb = video_path.stat().st_size / (1024 * 1024)
                tk.Label(vid_frame, text="无声视频",
                         font=("Microsoft YaHei", 9, "bold")).pack(pady=(4, 2))
                tk.Label(vid_frame, text=f"{video_path.name}\n{size_mb:.1f} MB",
                         font=("Microsoft YaHei", 7), fg="#888").pack()

                play_vid = tk.Button(
                    vid_frame, text="▶ 播放",
                    font=("Microsoft YaHei", 9),
                    command=lambda p=video_path: self._play_segment(p),
                )
                play_vid.pack(pady=(2, 6))
            else:
                tk.Label(vid_frame, text="无声视频\n(文件缺失)",
                         font=("Microsoft YaHei", 8), fg="#999").pack(pady=8)

            # -- audio track --
            aud_frame = ttk.Frame(body, relief="groove", borderwidth=1)
            aud_frame.pack(side="left")

            audio_path = Path(r["audio"])
            if audio_path.exists():
                size_kb = audio_path.stat().st_size / 1024
                tk.Label(aud_frame, text="音频轨",
                         font=("Microsoft YaHei", 9, "bold")).pack(pady=(4, 2))
                tk.Label(aud_frame, text=f"{audio_path.name}\n{size_kb:.0f} KB  16kHz mono",
                         font=("Microsoft YaHei", 7), fg="#888").pack()

                play_aud = tk.Button(
                    aud_frame, text="▶ 播放",
                    font=("Microsoft YaHei", 9),
                    command=lambda p=audio_path: self._play_segment(p),
                )
                play_aud.pack(pady=(2, 6))
            else:
                tk.Label(aud_frame, text="音频轨\n(文件缺失)",
                         font=("Microsoft YaHei", 8), fg="#999").pack(pady=8)

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

        self._asr_dlg = tk.Toplevel(self.root)
        self._asr_dlg.title("转换文字")
        self._asr_dlg.geometry("420x180")
        self._asr_dlg.transient(self.root)
        self._asr_dlg.grab_set()
        self._asr_dlg.protocol("WM_DELETE_WINDOW", self._on_cancel_asr)
        self._asr_dlg.resizable(False, False)

        ttk.Label(self._asr_dlg, text="正在语音识别…",
                  font=("Microsoft YaHei", 11)).pack(pady=(12, 8))
        f1 = ttk.Frame(self._asr_dlg); f1.pack(fill="x", padx=20, pady=4)
        ttk.Label(f1, text="文件进度：").pack(side="left")
        self._lbl_asr_prog = ttk.Label(f1, text=f"0 / {total}")
        self._lbl_asr_prog.pack(side="right")
        self._bar_asr = ttk.Progressbar(self._asr_dlg, length=380,
                                        mode="determinate", maximum=total)
        self._bar_asr.pack(padx=20)

        self._lbl_asr_seg = ttk.Label(self._asr_dlg, text="已识别: 0 句",
                                      font=("Microsoft YaHei", 9))
        self._lbl_asr_seg.pack(pady=(6, 0))

        ttk.Button(self._asr_dlg, text="取消",
                   command=self._on_cancel_asr).pack(pady=12)

        # Swith to the 转换文字 tab
        for idx in range(self._notebook.index("end")):
            if self._notebook.tab(idx, "text") == "转换文字":
                self._notebook.select(idx)
                break

        # Prep result area — write header, segments will stream in
        self._asr_current_file = -1
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
        try:
            results = transcribe_all(
                audio_paths, language=lang_code,
                segment_cb=self._on_asr_segment,
                progress_cb=self._on_asr_progress,
                cancel_check=lambda: self._cancel_requested,
            )
        except Exception as e:
            self.root.after(0, lambda: self._on_asr_error(str(e)))
            return
        self.root.after(0, lambda: self._on_asr_done(results, lang_code))

    def _on_asr_progress(self, current: int, total: int):
        self.root.after(0, lambda: self._update_asr_dialog(current, total))

    def _update_asr_dialog(self, current: int, total: int):
        if not hasattr(self, "_asr_dlg") or not self._asr_dlg.winfo_exists():
            return
        self._bar_asr.configure(value=current)
        self._lbl_asr_prog.configure(text=f"{current} / {total}")

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

    # ═══ remaining toolbar stubs ═══════════════════════════════

    def _on_translate(self):
        self.log.mark_step("文本翻译", "running"); self.log.add_entry("文本翻译", "start"); self._refresh_toolbar()
        self.log.mark_step("文本翻译", "done"); self.log.add_entry("文本翻译", "done", "placeholder"); self._refresh_toolbar()

    def _on_synthesize_audio(self):
        self.log.mark_step("合成音轨", "running"); self.log.add_entry("合成音轨", "start"); self._refresh_toolbar()
        self.log.mark_step("合成音轨", "done"); self.log.add_entry("合成音轨", "done", "placeholder"); self._refresh_toolbar()

    def _on_anchor_person(self):
        self.log.mark_step("人物锚定", "running"); self.log.add_entry("人物锚定", "start"); self._refresh_toolbar()
        self.log.mark_step("人物锚定", "done"); self.log.add_entry("人物锚定", "done", "placeholder"); self._refresh_toolbar()

    def _on_lip_sync(self):
        self.log.mark_step("口型匹配", "running"); self.log.add_entry("口型匹配", "start"); self._refresh_toolbar()
        self.log.mark_step("口型匹配", "done"); self.log.add_entry("口型匹配", "done", "placeholder"); self._refresh_toolbar()

    def _on_compose_video(self):
        self.log.mark_step("合成视频", "running"); self.log.add_entry("合成视频", "start"); self._refresh_toolbar()
        self.log.mark_step("合成视频", "done"); self.log.add_entry("合成视频", "done", "placeholder"); self._refresh_toolbar()

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

        self.root.title(f"{WINDOW_TITLE} - {self.log.name}")
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
            tab = ttk.Frame(self._notebook)
            self._notebook.add(tab, text=name)
            self._tab_frames[name] = tab
            if name == "转换文字":
                self._build_transcribe_tab(tab)
            else:
                tk.Label(tab, text="当前为空", fg="#aaa",
                         font=("Microsoft YaHei", 13)).place(relx=0.5, rely=0.5,
                                                             anchor="center")

    def _build_transcribe_tab(self, tab: ttk.Frame):
        """Add language selector bar + scrollable results area."""
        # Top bar
        bar = ttk.Frame(tab, padding=(10, 8))
        bar.pack(fill="x")

        ttk.Label(bar, text="语言选择：",
                  font=("Microsoft YaHei", 10)).pack(side="left", padx=(0, 6))

        self._trans_lang_var = tk.StringVar(value=LANG_LABELS[0])  # 日本語
        lang_combo = ttk.Combobox(
            bar, textvariable=self._trans_lang_var,
            values=LANG_LABELS, state="readonly",
            font=("Microsoft YaHei", 10), width=10,
        )
        lang_combo.pack(side="left")

        # Result area (scrollable text, populated after transcription)
        result_frame = ttk.Frame(tab)
        result_frame.pack(expand=True, fill="both", padx=4, pady=(4, 4))

        self._trans_text = tk.Text(
            result_frame, font=("Microsoft YaHei", 9),
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
        self._trans_text.insert("1.0", "选择语言后点击工具栏「转换文字」开始识别。\n"
                                        "默认语言：日本語")
        self._trans_text.configure(state="disabled")

    # ═══ player controls ═══════════════════════════════════════

    def _build_player_controls(self, parent):
        bar = ttk.Frame(parent, padding=(8, 6))
        bar.grid(row=1, column=0, sticky="ew")

        btn_frame = ttk.Frame(bar)
        btn_frame.pack(fill="x")

        self._btn_mute = tk.Button(btn_frame, text="🔊", font=("Segoe UI", 12),
                                   width=3, relief="flat", bg="#f0f0f0",
                                   command=self._on_toggle_mute)
        self._btn_mute.pack(side="left", padx=(0, 8))

        transport = ttk.Frame(btn_frame)
        transport.pack(side="left", expand=True)

        self._btn_rew = tk.Button(transport, text="⏪ 10s",
                                  font=("Microsoft YaHei", 10),
                                  relief="flat", bg="#f0f0f0",
                                  command=self._on_rewind)
        self._btn_rew.pack(side="left", padx=2)

        self._btn_play = tk.Button(transport, text="▶",
                                   font=("Segoe UI", 14, "bold"),
                                   width=3, relief="flat", bg="#f0f0f0",
                                   command=self._on_toggle_play)
        self._btn_play.pack(side="left", padx=2)

        self._btn_stop = tk.Button(transport, text="⏹", font=("Segoe UI", 14),
                                   width=3, relief="flat", bg="#f0f0f0",
                                   command=self._on_stop)
        self._btn_stop.pack(side="left", padx=2)

        self._btn_fwd = tk.Button(transport, text="10s ⏩",
                                  font=("Microsoft YaHei", 10),
                                  relief="flat", bg="#f0f0f0",
                                  command=self._on_forward)
        self._btn_fwd.pack(side="left", padx=2)

        prog_frame = ttk.Frame(bar)
        prog_frame.pack(fill="x", pady=(6, 0))

        self._lbl_cur = tk.Label(prog_frame, text="0:00", font=("Consolas", 9),
                                 fg="#666", bg=self.root.cget("bg"))
        self._lbl_cur.pack(side="left")

        self._slider = ttk.Scale(prog_frame, from_=0, to=1000,
                                 orient="horizontal",
                                 command=self._on_slider_drag)
        self._slider.pack(side="left", fill="x", expand=True, padx=6)

        self._lbl_total = tk.Label(prog_frame, text="0:00", font=("Consolas", 9),
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
        self.root.title(f"{WINDOW_TITLE} - {self.log.name}")

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
