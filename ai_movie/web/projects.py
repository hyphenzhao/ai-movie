"""Project discovery, options, status and read-only views for the web UI."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ai_movie.config import VIDEO_EXTENSIONS, WORKSPACE_DIR   # noqa: E402
import run_pipeline as rp                                      # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")
INPUTS = ROOT / "inputs"
DELIVER = ROOT / "deliver"
WORKSPACE = Path(WORKSPACE_DIR)
UPLOADS = WORKSPACE / "_uploads"
MEDIA_ROOTS = (WORKSPACE, INPUTS, DELIVER)
MEDIA_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".wav", ".mp3", ".jpg", ".jpeg", ".png",
              ".json", ".csv", ".srt", ".md", ".txt", ".log"}
NAME_RE = re.compile(r"^[A-Za-z0-9_.\-一-鿿]{1,64}$")

STEP_LABELS = {
    "demux": "拆分音轨", "separate": "人声分离", "osd": "重叠检测", "asr": "转换文字",
    "glossary": "术语表", "translate": "文本翻译", "tts": "人声生成", "compact": "台词压缩",
    "fit": "时长适配", "mix": "重新混音", "faces": "人物锚定", "lipsync": "口型匹配",
    "enhance": "人脸增强", "compose": "合成视频", "qc": "质检",
    "v2": "原声音色(v2)", "deliver": "交付包",
}
ENGINES = ["sakura", "sakura+gptoss", "gptoss", "hy-mt2", "hy-mt2+gptoss", "hy-mt2+sakura"]

# Options the UI edits, in the order the panels show them.  Each maps 1:1 to
# a run_pipeline.py flag (see build_argv).
OPTION_SPEC = [
    ("language", "str"), ("asr_backend", "str"), ("num_speakers", "int?"),
    ("no_diarize", "bool"), ("dialogue_refine", "bool"), ("dialogue_model", "str?"),
    ("translate_helper", "str?"), ("engines", "str"), ("chosen_engine", "str?"),
    ("voice_mode", "str"), ("no_ref_probe", "bool"), ("no_osd", "bool"),
    ("no_compact", "bool"), ("lipsync_backend", "str"),
    ("lipsync_audio_offset_ms", "float?"), ("fusion", "str?"), ("occlusion_mode", "str?"),
    ("enhance_fidelity", "float?"), ("enhance_protect_lips", "int?"), ("faces_bind", "str?"),
]


def default_options() -> dict:
    ns = rp.build_parser().parse_args(["_"])
    opts = {k: getattr(ns, k, None) for k, _ in OPTION_SPEC}
    opts["voice_mode"] = "sft"          # the production recipe (run_v3.sh)
    return opts


# ── paths ───────────────────────────────────────────────────────────

def valid_name(name: str) -> bool:
    return bool(NAME_RE.match(name or "")) and not name.startswith("_") and name not in (".", "..")


def workdir(name: str) -> Path:
    return WORKSPACE / name


def webdir(name: str) -> Path:
    d = workdir(name) / "web"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_path(name: str) -> Path:
    return workdir(name) / "state.json"


def load_state(name: str) -> dict:
    p = state_path(name)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return {}


def find_video(name: str, state: dict | None = None) -> Path | None:
    st = state if state is not None else load_state(name)
    v = st.get("_video")
    if v and Path(v).exists():
        return Path(v)
    for ext in sorted(VIDEO_EXTENSIONS):
        p = INPUTS / f"{name}{ext}"
        if p.exists():
            return p
    return None


def safe_media_path(p: str) -> Path | None:
    """Absolute path under workspace/inputs/deliver with an allowed extension."""
    try:
        path = Path(p)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
    except Exception:                                   # noqa: BLE001
        return None
    if path.suffix.lower() not in MEDIA_EXTS or not path.is_file():
        return None
    for root in MEDIA_ROOTS:
        try:
            if path.is_relative_to(root.resolve()):
                return path
        except Exception:                               # noqa: BLE001
            continue
    return None


def media_url(p: str | Path | None) -> str | None:
    if not p:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    from urllib.parse import quote
    return "/api/media?p=" + quote(str(path))


def download_url(p: str | Path | None) -> str | None:
    u = media_url(p)
    return u.replace("/api/media?", "/api/download?") if u else None


def _ffprobe_duration(p: Path) -> float | None:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(p)], capture_output=True, text=True,
                             timeout=30).stdout.strip()
        return round(float(out), 2)
    except Exception:                                   # noqa: BLE001
        return None


# ── projects ────────────────────────────────────────────────────────

def list_projects() -> list[dict]:
    names: set[str] = set()
    if INPUTS.exists():
        for p in INPUTS.iterdir():
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and valid_name(p.stem):
                names.add(p.stem)
    if WORKSPACE.exists():
        for d in WORKSPACE.iterdir():
            if d.is_dir() and (d / "state.json").exists() and valid_name(d.name):
                names.add(d.name)
    out = []
    for n in sorted(names):
        st = load_state(n)
        video = find_video(n, st)
        done = [s for s in rp.ALL_STEPS if st.get(s)]
        dm = st.get("demux") or {}
        out.append({
            "name": n,
            "video": str(video) if video else None,
            "size": video.stat().st_size if video else None,
            "duration": dm.get("duration"),
            "steps_done": len(done),
            "has_state": bool(st),
            "has_vc": bool((st.get("vc") or {}).get("video")),
            "updated": state_path(n).stat().st_mtime if state_path(n).exists() else None,
        })
    return out


def list_inputs() -> list[dict]:
    known = {p["name"] for p in list_projects() if p["has_state"]}
    out = []
    if INPUTS.exists():
        for p in sorted(INPUTS.iterdir()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                out.append({"file": p.name, "name": p.stem, "size": p.stat().st_size,
                            "has_project": p.stem in known})
    return out


# ── options ─────────────────────────────────────────────────────────

def load_options(name: str) -> dict:
    opts = default_options()
    p = webdir(name) / "options.json"
    if p.exists():
        try:
            opts.update({k: v for k, v in json.loads(p.read_text(encoding="utf-8")).items()
                         if k in opts})
        except Exception:                               # noqa: BLE001
            pass
    return opts


def save_options(name: str, new: dict) -> dict:
    opts = load_options(name)
    for k, kind in OPTION_SPEC:
        if k not in new:
            continue
        v = new[k]
        if kind == "bool":
            v = bool(v)
        elif kind.startswith("int"):
            v = None if v in (None, "", "null") else int(v)
        elif kind.startswith("float"):
            v = None if v in (None, "", "null") else float(v)
        else:
            v = None if v in (None, "", "null") and kind.endswith("?") else str(v)
        opts[k] = v
    p = webdir(name) / "options.json"
    p.write_text(json.dumps(opts, ensure_ascii=False, indent=1), encoding="utf-8")
    invalidate_status(name)
    return opts


def option_flags(opts: dict) -> list[str]:
    """run_pipeline.py flags for the stored options (only non-default values)."""
    flags: list[str] = []
    defaults = default_options()
    defaults["voice_mode"] = "clone"     # the parser default; we always pass voice_mode
    for k, kind in OPTION_SPEC:
        v = opts.get(k)
        flag = "--" + k.replace("_", "-")
        if kind == "bool":
            if v:
                flags.append(flag)
        elif v is None:
            continue
        elif k == "voice_mode" or v != defaults.get(k):
            flags += [flag, str(v)]
    return flags


def build_argv(name: str, steps: list[str] | None, force: bool = False,
               extra: list[str] | None = None) -> list[str]:
    video = find_video(name)
    if video is None:
        raise FileNotFoundError(f"no source video for {name}")
    argv = [PY, "-u", str(SCRIPTS / "run_pipeline.py"), str(video), "--name", name]
    if steps:
        argv += ["--steps", ",".join(steps)]
    if force:
        argv.append("--force")
    argv += option_flags(load_options(name))
    argv += extra or []
    return argv


# ── status ──────────────────────────────────────────────────────────

_status_cache: dict[str, tuple[tuple, dict]] = {}
_status_lock = threading.Lock()


def _code_mtime() -> float:
    m = 0.0
    for d in (ROOT / "ai_movie", ROOT / "scripts", ROOT / "patches"):
        for p in d.glob("*.py"):
            m = max(m, p.stat().st_mtime)
        for p in d.glob("*.patch"):
            m = max(m, p.stat().st_mtime)
    return m


def invalidate_status(name: str | None = None) -> None:
    with _status_lock:
        if name is None:
            _status_cache.clear()
        else:
            _status_cache.pop(name, None)


def raw_status(name: str) -> dict:
    """``{step: {status, reasons}}`` from ``run_pipeline.py --status-json`` (cached)."""
    sp = state_path(name)
    key = (sp.stat().st_mtime_ns if sp.exists() else 0,
           (webdir(name) / "options.json").stat().st_mtime_ns
           if (webdir(name) / "options.json").exists() else 0,
           _code_mtime())
    with _status_lock:
        hit = _status_cache.get(name)
        if hit and hit[0] == key:
            return hit[1]
    if not sp.exists():
        raw = {s: {"status": "missing", "reasons": []} for s in rp.ALL_STEPS}
    else:
        try:
            argv = build_argv(name, None) + ["--status-json"]
            r = subprocess.run(argv, capture_output=True, text=True, timeout=120, cwd=str(ROOT))
            line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("{")), "{}")
            raw = json.loads(line)
        except Exception as exc:                        # noqa: BLE001
            raw = {s: {"status": "unknown", "reasons": [str(exc)]} for s in rp.ALL_STEPS}
    with _status_lock:
        _status_cache[name] = (key, raw)
    return raw


def derive_status(name: str, raw: dict, state: dict, running_step: str | None = None,
                  failed_step: str | None = None, running_kind: str | None = None) -> dict:
    """Topological status: locked / ready / done / stale / skipped / legacy / running / failed."""
    out: dict[str, dict] = {}
    for s in rp.ALL_STEPS:
        r = raw.get(s) or {"status": "missing", "reasons": []}
        st, why = r["status"], list(r.get("reasons") or [])
        deps = [d for d in rp.STEP_DEPS.get(s, []) if d in out]
        dep_bad = [d for d in deps if out[d]["status"] in ("missing", "locked", "stale", "failed")
                   and not (d == "compact" and out.get("tts", {}).get("status") == "done")
                   and not (d == "osd")]
        if st == "missing":
            status = "locked" if any(out[d]["status"] in ("missing", "locked") for d in deps
                                     if d not in ("osd", "compact")) else "ready"
        elif st in ("valid", "legacy", "skipped"):
            status = "done" if not dep_bad else "stale"
            if dep_bad:
                why = [f"上游 {STEP_LABELS.get(d, d)} 已变更" for d in dep_bad]
            if st == "legacy":
                why = why + ["无指纹（建议指纹采用）"]
            if st == "skipped":
                why = why + ["已跳过"]
        elif st == "stale":
            status = "stale"
        else:
            status = "unknown"
        if failed_step == s and status != "done":
            status = "failed"
        if running_step == s:
            status = "running"
        out[s] = {"status": status, "reasons": why, "label": STEP_LABELS.get(s, s)}

    vc = state.get("vc") or {}
    fps = state.get("_fp") or {}
    web = state.get("_web") or {}
    if vc.get("video") and Path(vc["video"] if Path(vc["video"]).is_absolute()
                                else ROOT / vc["video"]).exists():
        dep_now = {k: (fps.get(k) or {}).get("hash") for k in ("fit", "compose")}
        rec = web.get("vc_deps")
        ok = (rec == dep_now) if rec else True
        out["v2"] = {"status": "done" if ok else "stale", "label": STEP_LABELS["v2"],
                     "reasons": [] if ok else ["v1 已变更（需重做 v2）"]}
    else:
        out["v2"] = {"status": "ready" if out["compose"]["status"] == "done" else "locked",
                     "label": STEP_LABELS["v2"], "reasons": []}
    final = DELIVER / f"{name}_dubbed.mp4"
    src = vc.get("video") or (state.get("compose") or {}).get("video")
    if final.exists() and src and Path(src).exists() and final.stat().st_mtime >= Path(src).stat().st_mtime:
        out["deliver"] = {"status": "done", "label": STEP_LABELS["deliver"], "reasons": []}
    else:
        out["deliver"] = {"status": "ready" if src else "locked", "label": STEP_LABELS["deliver"],
                          "reasons": []}
    if running_kind in ("v2", "one_click") and running_step is None:
        pass
    return out


# ── views ───────────────────────────────────────────────────────────

def _tc(t) -> str:
    t = float(t or 0)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


def segments_view(name: str, state: dict | None = None) -> list[dict]:
    st = state if state is not None else load_state(name)
    stages = {k: ((st.get(k) or {}).get("segments") or []) for k in ("asr", "translate", "tts", "compact", "fit", "vc")}
    n = max((len(v) for v in stages.values()), default=0)
    comp = {r["idx"]: r for r in ((st.get("compact") or {}).get("report") or [])}
    rows = []
    for i in range(n):
        seg: dict = {}
        for k in ("asr", "translate", "tts", "compact", "fit"):
            if i < len(stages[k]):
                seg.update(stages[k][i])
        vcs = stages["vc"][i] if i < len(stages["vc"]) else {}
        rows.append({
            "idx": i, "start": seg.get("start"), "end": seg.get("end"), "tc": _tc(seg.get("start")),
            "speaker": seg.get("speaker"), "gender": seg.get("gender") or seg.get("tts_gender"),
            "speaker_conf": seg.get("speaker_conf"), "asr_conf": seg.get("asr_conf"),
            "overlap": seg.get("overlap"), "text": seg.get("text"),
            "text_translated": seg.get("text_translated"),
            "text_translated_full": seg.get("text_translated_full"),
            "audio_url": media_url(seg.get("audio")), "audio_fit_url": media_url(seg.get("audio_fit")),
            "vc_audio_url": media_url(vcs.get("audio_fit") or vcs.get("audio")) if vcs else None,
            "vc": bool(vcs.get("vc")) if vcs else None,
            "fit_ratio": seg.get("fit_ratio"), "rate_factor": seg.get("rate_factor"),
            "overrun": seg.get("overrun"), "mix_gain_db": seg.get("mix_gain_db"),
            "tts_fallback": bool(seg.get("tts_fallback")), "tts_error": seg.get("tts_error"),
            "compact": (comp.get(i) or {}).get("status"),
        })
    return rows


def speakers_view(name: str, state: dict | None = None) -> dict:
    st = state if state is not None else load_state(name)
    diar = (st.get("asr") or {}).get("diarization") or {}
    segs = (st.get("asr") or {}).get("segments") or []
    deliver = workdir(name) / "deliverables"
    spk = {}
    for k, v in (diar.get("speakers") or {}).items():
        demo = next(iter(deliver.glob(f"01_spk_{k}_*_demo.wav")), None)
        ab = deliver / f"03_ab_{k}.wav"
        spk[k] = {**v, "n_segments": sum(1 for s in segs if s.get("speaker") == k),
                  "demo_url": media_url(demo), "ab_url": media_url(ab) if ab.exists() else None}
    refs = (st.get("tts") or {}).get("refs") or {}
    vc_refs = (st.get("vc") or {}).get("refs") or {}
    quality = (st.get("tts") or {}).get("quality") or {}
    return {"speakers": spk, "refs": refs, "vc_refs": vc_refs, "quality": quality,
            "backend": diar.get("backend"), "overlap_total_s": (st.get("osd") or {}).get("total_overlap_s")}


def faces_view(name: str, state: dict | None = None) -> dict:
    st = state if state is not None else load_state(name)
    faces = st.get("faces") or {}
    plan = {}
    pp = faces.get("plan_path")
    if pp and Path(pp).exists():
        try:
            plan = json.loads(Path(pp).read_text(encoding="utf-8"))
        except Exception:                               # noqa: BLE001
            plan = {}
    deliver = workdir(name) / "deliverables"
    tracks = []
    for t in (plan.get("tracks") or faces.get("tracks") or []):
        tid = t.get("id")
        thumb = next(iter(deliver.glob(f"track_{tid}_*.jpg")), None)
        tracks.append({**{k: v for k, v in t.items() if k not in ("keyframes", "yaw")},
                       "thumb_url": media_url(thumb)})
    seg_track = plan.get("segment_track") or {}
    counts: dict[str, int] = {}
    for v in seg_track.values():
        counts[str(v)] = counts.get(str(v), 0) + 1
    return {"tracks": tracks, "speaker_track": plan.get("speaker_track") or faces.get("speaker_track") or {},
            "segment_track_counts": counts, "gate": plan.get("gate"), "cuts": len(plan.get("cuts") or []),
            "anchored_frames": len(plan.get("frames") or {}), "n_frames": plan.get("n_frames"),
            "override": load_options(name).get("faces_bind")}


def videos_view(name: str, state: dict | None = None) -> dict:
    st = state if state is not None else load_state(name)
    video = find_video(name, st)
    vc = st.get("vc") or {}
    vcv = vc.get("video")
    if vcv and not Path(vcv).is_absolute():
        vcv = str(ROOT / vcv)
    return {
        "original": media_url(video),
        "silent": media_url((st.get("demux") or {}).get("video")),
        "lipsync": media_url((st.get("lipsync") or {}).get("video")),
        "enhanced": media_url((st.get("enhance") or {}).get("video")),
        "final": media_url((st.get("compose") or {}).get("video")),
        "vc": media_url(vcv),
        "mix": media_url((st.get("mix") or {}).get("audio")),
        "vc_mix": media_url(workdir(name) / "synthesized_vc" / "final_audio.wav"),
        "deliver": media_url(DELIVER / f"{name}_dubbed.mp4"),
    }


def deliverables_view(name: str) -> list[dict]:
    out = []
    for base in (workdir(name) / "deliverables", DELIVER / f"{name}_dubbed"):
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in MEDIA_EXTS:
                out.append({"name": str(p.relative_to(base.parent)), "size": p.stat().st_size,
                            "mtime": p.stat().st_mtime, "url": media_url(p),
                            "download": download_url(p), "kind": p.suffix.lower().lstrip(".")})
    return out


def qc_view(name: str, key: str = "fit") -> dict:
    suffix = "_vc" if key == "vc" else ""
    p = workdir(name) / "deliverables" / f"06_qc{suffix}.json"
    if not p.exists():
        return {"available": False, "key": key}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        d["available"] = True
        return d
    except Exception as exc:                            # noqa: BLE001
        return {"available": False, "key": key, "error": str(exc)}


def review_view(name: str) -> dict:
    p = workdir(name) / "deliverables" / "review" / "manifest.json"
    if not p.exists():
        return {"available": False}
    try:
        items = json.loads(p.read_text(encoding="utf-8"))
        for it in items:
            clip = it.get("clip")
            it["clip_url"] = media_url(clip if clip and Path(clip).is_absolute() else (p.parent / clip if clip else None))
        return {"available": True, "items": items}
    except Exception as exc:                            # noqa: BLE001
        return {"available": False, "error": str(exc)}


def acceptance_view(name: str) -> str | None:
    p = workdir(name) / "deliverables" / "ACCEPTANCE.md"
    return p.read_text(encoding="utf-8") if p.exists() else None


# ── uploads (chunked) ───────────────────────────────────────────────

def upload_init(filename: str, size: int, name: str | None) -> dict:
    ext = Path(filename).suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise ValueError(f"unsupported extension {ext}")
    base = name or Path(filename).stem
    base = re.sub(r"[^A-Za-z0-9_.\-一-鿿]", "_", base)[:64] or "video"
    if not valid_name(base):
        raise ValueError("invalid project name")
    uid = uuid.uuid4().hex[:12]
    d = UPLOADS / uid
    d.mkdir(parents=True, exist_ok=True)
    meta = {"id": uid, "name": base, "ext": ext, "size": int(size), "chunks": [], "created": time.time()}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def upload_chunk(uid: str, index: int, data: bytes) -> dict:
    d = UPLOADS / uid
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    (d / f"{index:06d}.part").write_bytes(data)
    if index not in meta["chunks"]:
        meta["chunks"].append(index)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return {"id": uid, "received": sorted(meta["chunks"])}


def upload_finalize(uid: str) -> dict:
    d = UPLOADS / uid
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    parts = sorted(d.glob("*.part"))
    INPUTS.mkdir(parents=True, exist_ok=True)
    dst = INPUTS / f"{meta['name']}{meta['ext']}"
    tmp = dst.with_suffix(dst.suffix + ".part")
    with open(tmp, "wb") as out:
        for p in parts:
            with open(p, "rb") as fh:
                shutil.copyfileobj(fh, out, 1 << 20)
    if meta["size"] and tmp.stat().st_size != meta["size"]:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"size mismatch: got {tmp.stat().st_size if tmp.exists() else 0}, expected {meta['size']}")
    os.replace(tmp, dst)
    shutil.rmtree(d, ignore_errors=True)
    return {"name": meta["name"], "file": str(dst), "size": dst.stat().st_size,
            "duration": _ffprobe_duration(dst)}


def import_input(filename: str) -> dict:
    p = INPUTS / Path(filename).name
    if not p.exists() or p.suffix.lower() not in VIDEO_EXTENSIONS:
        raise FileNotFoundError(filename)
    if not valid_name(p.stem):
        raise ValueError("invalid project name")
    return {"name": p.stem, "file": str(p), "size": p.stat().st_size, "duration": _ffprobe_duration(p)}
