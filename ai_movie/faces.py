"""Face tracks, per-track gender, and speaker→face binding.

Why this module exists
----------------------
Nothing in the pipeline ever chose *which* face to lip-sync.  MuseTalk takes
DWPose's ``keypoints[0]`` and S3FD's ``d[0]`` — that is, whichever face
scored highest in that individual frame — so in a two-person shot the mouth
being driven can change from frame to frame.  The previous "人物锚定" step
did not address this at all: it filtered *audio segments* by
``tts_gender`` and left face selection untouched.

Here we build real per-frame face tracks, label each track male/female, bind
each diarized speaker to a track, and emit a per-frame target box that
MuseTalk is then forced to use (see ``patches/musetalk_target_face.patch``).
Frames with no bound face get ``null`` and pass through untouched.

Everything runs on CPU on purpose: S3FD's conv shapes trigger a multi-minute,
silent, uninterruptible MIOpen JIT compile on gfx1151 — the same trap that
``face_restore.restore_video`` documents and avoids.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ai_movie.config import (
    FACE_BIND_MIN_SCORE,
    FACE_DET_CONF,
    FACE_DET_DEVICE,
    FACE_DET_EVERY,
    FACE_DET_MAX_WIDTH,
    FACE_GENDER_BACKEND,
    FACE_GENDER_MIN_CONF,
    FACE_GENDER_MODEL,
    FACE_GENDER_SAMPLES,
    FACE_GATE_SMOOTH,
    FACE_MIN_WIDTH,
    FACE_POSE_MODEL,
    FACE_TRACK_IOU,
    FACE_TRACK_MAX_GAP,
    FACE_TRACK_MIN_FRAMES,
    FACE_YAW_MAX,
)

_MODELS_DIR = Path(__file__).parent.parent / "models" / "wav2lip"


def _log(msg: str) -> None:
    print(f"[faces] {msg}", file=sys.stderr, flush=True)


# ── video probing ──────────────────────────────────────────────────

def probe_video(path: str | Path) -> dict:
    out = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries", "format=duration", "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    data = json.loads(out.stdout)
    st = (data.get("streams") or [{}])[0]
    num, _, den = (st.get("r_frame_rate") or "25/1").partition("/")
    fps = float(num) / float(den or 1)
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    nb = st.get("nb_frames")
    n_frames = int(nb) if nb and str(nb).isdigit() else int(round(duration * fps))
    return {"fps": fps, "n_frames": n_frames, "duration": duration,
            "width": int(st.get("width") or 0), "height": int(st.get("height") or 0)}


# ── multi-face detection ───────────────────────────────────────────

_detector = None


def _load_detector(device: str = FACE_DET_DEVICE):
    """S3FD detector from the bundled Wav2Lip tree (singleton)."""
    global _detector
    if _detector is not None:
        return _detector
    s = str(_MODELS_DIR)
    if s not in sys.path:
        sys.path.insert(0, s)
    import face_detection
    _detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False, device=device)
    return _detector


def detect_all_faces(frames: list[np.ndarray],
                     device: str = FACE_DET_DEVICE,
                     conf: float = FACE_DET_CONF) -> list[list[list[float]]]:
    """Return **every** face per frame as ``[x1, y1, x2, y2, score]``.

    ``face_detection.api.get_detections_for_batch`` collapses each frame to
    ``d[0]``; the underlying ``detect_from_batch`` already returns all of
    them, so we go straight to it.
    """
    det = _load_detector(device)
    batch = np.asarray(frames)[..., ::-1]        # RGB → BGR for S3FD
    raw = det.face_detector.detect_from_batch(batch.copy())
    out: list[list[list[float]]] = []
    for d in raw:
        faces = []
        for f in (d if d is not None else []):
            f = np.clip(np.asarray(f, dtype=np.float32), 0, None)
            score = float(f[-1]) if len(f) > 4 else 1.0
            if score < conf:
                continue
            faces.append([float(f[0]), float(f[1]), float(f[2]), float(f[3]), score])
        faces.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
        out.append(faces)
    return out


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / max(ua, 1e-6))


# ── tracking ───────────────────────────────────────────────────────

def detect_face_tracks(
    video_path: str | Path,
    *,
    det_every: int = FACE_DET_EVERY,
    det_max_width: int = FACE_DET_MAX_WIDTH,
    device: str = FACE_DET_DEVICE,
    conf: float = FACE_DET_CONF,
    iou_thresh: float = FACE_TRACK_IOU,
    min_track_frames: int = FACE_TRACK_MIN_FRAMES,
    max_gap: int = FACE_TRACK_MAX_GAP,
    batch: int = 8,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    """Detect faces every *det_every* frames and link them into tracks.

    Boxes are stored in **full-resolution** coordinates even though detection
    runs downscaled.  Returns::

        {"fps", "n_frames", "size": (w, h), "det_every",
         "tracks": [{"id", "keyframes": {frame_idx: [x1,y1,x2,y2]},
                     "first", "last", "n", "mean_area"}]}
    """
    info = probe_video(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    scale = 1.0
    if info["width"] > det_max_width:
        scale = det_max_width / float(info["width"])

    tracks: list[dict] = []
    active: list[dict] = []           # {"track": ref, "last_kf": int, "box": [...]}
    pending: list[tuple[int, np.ndarray]] = []
    processed = 0
    idx = -1

    def _flush() -> None:
        nonlocal pending, active
        if not pending:
            return
        frames = [f for _, f in pending]
        dets = detect_all_faces(frames, device=device, conf=conf)
        for (fi, _), faces in zip(pending, dets):
            faces_full = [[b[0] / scale, b[1] / scale, b[2] / scale,
                           b[3] / scale, b[4]] for b in faces]
            _associate(fi, faces_full)
        pending = []

    def _associate(frame_idx: int, faces: list[list[float]]) -> None:
        used = set()
        for a in list(active):
            if frame_idx - a["last_kf"] > max_gap * det_every:
                active.remove(a)
                continue
            best_j, best_iou = -1, 0.0
            for j, f in enumerate(faces):
                if j in used:
                    continue
                v = _iou(a["box"], f)
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_j >= 0 and best_iou >= iou_thresh:
                used.add(best_j)
                box = faces[best_j][:4]
                a["box"] = box
                a["last_kf"] = frame_idx
                a["track"]["keyframes"][frame_idx] = [round(x, 1) for x in box]
        for j, f in enumerate(faces):
            if j in used:
                continue
            t = {"id": len(tracks), "keyframes": {frame_idx: [round(x, 1) for x in f[:4]]}}
            tracks.append(t)
            active.append({"track": t, "last_kf": frame_idx, "box": f[:4]})

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % det_every:
            continue
        if cancel_check and cancel_check():
            break
        small = frame if scale == 1.0 else cv2.resize(
            frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
            interpolation=cv2.INTER_AREA)
        pending.append((idx, cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        if len(pending) >= batch:
            _flush()
            processed += batch
            if progress_cb:
                progress_cb(idx, info["n_frames"])
    _flush()
    cap.release()

    kept = []
    for t in tracks:
        kfs = t["keyframes"]
        if len(kfs) < min_track_frames:
            continue
        keys = sorted(kfs)
        areas = [(kfs[k][2] - kfs[k][0]) * (kfs[k][3] - kfs[k][1]) for k in keys]
        kept.append({
            "id": len(kept),
            "keyframes": {int(k): kfs[k] for k in keys},
            "first": int(keys[0]), "last": int(keys[-1]), "n": len(keys),
            "mean_area": float(np.mean(areas)),
        })

    _log(f"{len(kept)} tracks from {len(tracks)} raw "
         f"(det_every={det_every}, scale={scale:.2f})")
    return {"fps": info["fps"], "n_frames": info["n_frames"],
            "size": (info["width"], info["height"]),
            "det_every": det_every, "tracks": kept}


def interpolate_track(track: dict, n_frames: int,
                      *, det_every: int = FACE_DET_EVERY) -> dict[int, list[float]]:
    """Fill in every frame of a track by interpolating between keyframes.

    Refuses to interpolate across an apparent scene cut — the same guard
    ``face_restore._resolve_boxes`` uses — so a track never smears a box
    across a hard shot change.
    """
    kfs = {int(k): v for k, v in track["keyframes"].items()}
    keys = sorted(kfs)
    out: dict[int, list[float]] = {}
    for a, b in zip(keys, keys[1:]):
        box_a, box_b = kfs[a], kfs[b]
        out[a] = box_a
        ca = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2)
        cb = ((box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2)
        sa = max(box_a[2] - box_a[0], box_a[3] - box_a[1])
        sb = max(box_b[2] - box_b[0], box_b[3] - box_b[1])
        dist = float(np.hypot(cb[0] - ca[0], cb[1] - ca[1]))
        cut = dist > 0.6 * max(sa, sb) or max(sa, sb) > 1.8 * max(min(sa, sb), 1e-6)
        if cut or (b - a) > det_every * (FACE_TRACK_MAX_GAP + 1):
            continue
        for f in range(a + 1, b):
            w = (f - a) / float(b - a)
            out[f] = [box_a[i] * (1 - w) + box_b[i] * w for i in range(4)]
    if keys:
        out[keys[-1]] = kfs[keys[-1]]
        # Hold the first/last keyframe box over the surrounding det_every window
        # so a range boundary doesn't land on a missing frame.
        for f in range(max(0, keys[0] - det_every), keys[0]):
            out[f] = kfs[keys[0]]
        for f in range(keys[-1] + 1, min(n_frames, keys[-1] + det_every + 1)):
            out[f] = kfs[keys[-1]]
    return out


# ── gender ─────────────────────────────────────────────────────────

_gender_sess = None


def gender_model_available() -> bool:
    return Path(FACE_GENDER_MODEL).exists()


def _load_gender_model():
    """insightface ``genderage.onnx`` via onnxruntime (no insightface package).

    The upstream ``Attribute`` head is bbox-only — it does not need the 5-point
    landmarks — so an S3FD box is enough input.  That is why we can avoid
    pip-installing insightface (a cython build) into this Python 3.14 /
    torch-rocm environment.
    """
    global _gender_sess
    if _gender_sess is not None:
        return _gender_sess
    import onnxruntime as ort
    if not gender_model_available():
        raise FileNotFoundError(
            f"genderage model not found at {FACE_GENDER_MODEL}. "
            f"Run: bash scripts/download_face_models.sh")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    _gender_sess = ort.InferenceSession(str(FACE_GENDER_MODEL), so,
                                        providers=["CPUExecutionProvider"])
    return _gender_sess


def _predict_gender(frame_bgr: np.ndarray, box: list[float]) -> tuple[str, float]:
    """Return ``(gender, confidence)`` for one face box."""
    sess = _load_gender_model()
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if w <= 2 or h <= 2:
        return "unknown", 0.0
    s = 96.0 / (max(w, h) * 1.5)
    M = np.array([[s, 0, 48 - s * cx], [0, s, 48 - s * cy]], np.float32)
    aimg = cv2.warpAffine(frame_bgr, M, (96, 96), borderValue=0.0)
    blob = cv2.dnn.blobFromImage(aimg, 1.0, (96, 96), (0, 0, 0), swapRB=True)
    pred = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0]
    logits = np.asarray(pred[:2], dtype=np.float64)
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    female_p = float(p[0])
    return ("female" if female_p >= 0.5 else "male",
            float(max(female_p, 1.0 - female_p)))


def classify_track_gender(
    video_path: str | Path,
    plan: dict,
    *,
    samples_per_track: int = FACE_GENDER_SAMPLES,
    min_conf: float = FACE_GENDER_MIN_CONF,
    backend: str = FACE_GENDER_BACKEND,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[int, dict]:
    """Area-weighted majority vote of per-frame gender over each track.

    Voting (rather than trusting any single frame) matters because the
    attribute model degrades badly on profile and partially-occluded faces,
    which are common in an interview.
    """
    tracks = plan["tracks"]
    if backend != "insightface" or not gender_model_available():
        _log("gender backend unavailable — tracks marked unknown")
        return {t["id"]: {"gender": "unknown", "conf": 0.0, "votes": 0}
                for t in tracks}

    wanted: dict[int, list[tuple[int, list[float]]]] = {}
    for t in tracks:
        keys = sorted(int(k) for k in t["keyframes"])
        if not keys:
            continue
        step = max(1, len(keys) // samples_per_track)
        for k in keys[::step][:samples_per_track]:
            wanted.setdefault(k, []).append((t["id"], t["keyframes"][k]))

    votes: dict[int, list[tuple[str, float, float]]] = {}
    cap = cv2.VideoCapture(str(video_path))
    idx = -1
    todo = set(wanted)
    while todo:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx not in todo:
            continue
        todo.discard(idx)
        for tid, box in wanted[idx]:
            g, c = _predict_gender(frame, box)
            area = (box[2] - box[0]) * (box[3] - box[1])
            votes.setdefault(tid, []).append((g, c, area))
        if progress_cb and len(todo) % 25 == 0:
            progress_cb(f"人脸性别识别，剩余 {len(todo)} 帧")
    cap.release()

    out: dict[int, dict] = {}
    for t in tracks:
        v = votes.get(t["id"], [])
        if not v:
            out[t["id"]] = {"gender": "unknown", "conf": 0.0, "votes": 0}
            continue
        weight: dict[str, float] = {}
        for g, c, area in v:
            weight[g] = weight.get(g, 0.0) + c * float(np.sqrt(area))
        total = sum(weight.values()) or 1.0
        best = max(weight, key=weight.get)
        conf = weight[best] / total
        out[t["id"]] = {
            "gender": best if conf >= min_conf else "unknown",
            "conf": round(float(conf), 3), "votes": len(v),
            "raw_gender": best,
        }
    return out


# ── head pose (yaw) ────────────────────────────────────────────────

_pose_sess = None


def pose_model_available() -> bool:
    return Path(FACE_POSE_MODEL).exists()


def _load_pose_model():
    """insightface ``1k3d68.onnx`` (3-D 68-point landmarks) via onnxruntime."""
    global _pose_sess
    if _pose_sess is not None:
        return _pose_sess
    import onnxruntime as ort
    if not pose_model_available():
        raise FileNotFoundError(f"pose model not found at {FACE_POSE_MODEL}")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    _pose_sess = ort.InferenceSession(str(FACE_POSE_MODEL), so,
                                      providers=["CPUExecutionProvider"])
    return _pose_sess


def _predict_pose(frame_bgr: np.ndarray, box: list[float]) -> tuple[float, float]:
    """Return ``(yaw_deg, roll_deg)`` for one face box.

    Yaw is the angle of the jaw vector (landmark 0 → 16) in the x/z plane of
    the aligned 192² crop, where x and z share one scale — no mean-shape
    fitting needed.  0 = frontal, ±90 = full profile.  It saturates a little
    on hard profiles (a true 90° reads ~75°), which FACE_YAW_MAX accounts for.
    """
    sess = _load_pose_model()
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    if w <= 2 or h <= 2:
        return 0.0, 0.0
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    s = 192.0 / (max(w, h) * 1.5)
    M = np.array([[s, 0, 96 - s * cx], [0, s, 96 - s * cy]], np.float32)
    aimg = cv2.warpAffine(frame_bgr, M, (192, 192), borderValue=0.0)
    blob = cv2.dnn.blobFromImage(aimg, 1.0, (192, 192), (0, 0, 0), swapRB=True)
    pred = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0]
    pts = pred.reshape(-1, 3)[-68:].astype(np.float64)
    dx, dz = pts[16, 0] - pts[0, 0], pts[16, 2] - pts[0, 2]
    yaw = float(np.degrees(np.arctan2(dz, dx)))
    re_, le_ = pts[36:42].mean(0), pts[42:48].mean(0)
    roll = float(np.degrees(np.arctan2(le_[1] - re_[1], le_[0] - re_[0])))
    return yaw, roll


def estimate_track_pose(
    video_path: str | Path,
    plan: dict,
    *,
    progress_cb: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """Fill ``track["yaw"] = {keyframe: yaw_deg}`` for every track, in place.

    One sequential pass over the video at full resolution (the detector's
    640-px copy is too coarse for a 60-px face); ~5 ms per face on CPU.
    """
    tracks = plan["tracks"]
    if not pose_model_available():
        _log("pose model unavailable — no yaw gate")
        for t in tracks:
            t["yaw"] = {}
        return
    wanted: dict[int, list[tuple[int, list[float]]]] = {}
    for t in tracks:
        t["yaw"] = {}
        for k, box in t["keyframes"].items():
            wanted.setdefault(int(k), []).append((t["id"], box))
    by_id = {t["id"]: t for t in tracks}
    cap = cv2.VideoCapture(str(video_path))
    idx = -1
    todo = set(wanted)
    n_total = len(todo)
    while todo:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx not in todo:
            continue
        todo.discard(idx)
        if cancel_check and cancel_check():
            break
        for tid, box in wanted[idx]:
            try:
                yaw, _roll = _predict_pose(frame, box)
            except Exception as exc:                    # noqa: BLE001
                _log(f"pose failed at frame {idx}: {exc}")
                continue
            by_id[tid]["yaw"][idx] = round(yaw, 1)
        if progress_cb and len(todo) % 100 == 0:
            progress_cb(f"头部姿态估计，剩余 {len(todo)}/{n_total} 帧")
    cap.release()


def interpolate_scalar(values: dict[int, float], n_frames: int,
                       *, det_every: int = FACE_DET_EVERY) -> dict[int, float]:
    """Per-frame linear interpolation of a per-keyframe scalar (e.g. yaw).

    Mirrors ``interpolate_track``'s hold-at-the-ends behaviour so every frame
    that has a box also has a value.
    """
    kfs = {int(k): float(v) for k, v in values.items()}
    keys = sorted(kfs)
    out: dict[int, float] = {}
    for a, b in zip(keys, keys[1:]):
        out[a] = kfs[a]
        if (b - a) > det_every * (FACE_TRACK_MAX_GAP + 1):
            continue
        for f in range(a + 1, b):
            w = (f - a) / float(b - a)
            out[f] = kfs[a] * (1 - w) + kfs[b] * w
    if keys:
        out[keys[-1]] = kfs[keys[-1]]
        for f in range(max(0, keys[0] - det_every), keys[0]):
            out[f] = kfs[keys[0]]
        for f in range(keys[-1] + 1, min(n_frames, keys[-1] + det_every + 1)):
            out[f] = kfs[keys[-1]]
    return out


def gate_frames(
    frame_ids: list[int],
    boxes: dict[int, list[float]],
    yaw: dict[int, float],
    *,
    yaw_max: float = FACE_YAW_MAX,
    min_width: float = FACE_MIN_WIDTH,
    smooth: int = FACE_GATE_SMOOTH,
) -> set[int]:
    """Frames MuseTalk should NOT paint: hard profiles and tiny faces.

    Decided per contiguous run of frames and median-filtered over *smooth*
    frames, so the mouth never flips between synced and original for a
    handful of frames.  A frame with no yaw estimate is never gated by yaw.
    """
    if not frame_ids:
        return set()
    ids = sorted(frame_ids)
    gated: set[int] = set()
    runs: list[list[int]] = [[ids[0]]]
    for f in ids[1:]:
        if f == runs[-1][-1] + 1:
            runs[-1].append(f)
        else:
            runs.append([f])
    k = max(1, int(smooth) | 1)
    for run in runs:
        bad = np.zeros(len(run), dtype=bool)
        for i, f in enumerate(run):
            b = boxes.get(f)
            if b is None:
                continue
            w = b[2] - b[0]
            y = yaw.get(f)
            bad[i] = (w < min_width) or (y is not None and abs(y) > yaw_max)
        if k > 1 and len(run) >= k:
            pad = k // 2
            padded = np.concatenate([np.repeat(bad[:1], pad), bad, np.repeat(bad[-1:], pad)])
            bad = np.array([np.median(padded[i:i + k]) > 0.5 for i in range(len(run))])
        elif k > 1:
            bad[:] = bad.mean() > 0.5
        gated.update(f for f, g in zip(run, bad) if g)
    return gated


# ── active-speaker heuristic ───────────────────────────────────────

def mouth_motion(
    video_path: str | Path,
    plan: dict,
    spans: list[tuple[float, float]],
    *,
    sample_fps: float = 8.0,
    max_frames: int = 400,
) -> dict[int, float]:
    """Mean inter-frame change in each track's mouth region during *spans*.

    A model-free active-speaker proxy: normalised by the track's overall box
    motion so head movement and camera shake largely cancel out.  Used only
    to break ties between two same-gender faces.
    """
    fps = plan["fps"]
    want: set[int] = set()
    for s, e in spans:
        step = max(1, int(round(fps / sample_fps)))
        for f in range(int(s * fps), int(e * fps) + 1, step):
            want.add(f)
            want.add(f + 1)
    if not want:
        return {}
    want = set(sorted(want)[:max_frames * 2])

    boxes = {t["id"]: interpolate_track(t, plan["n_frames"],
                                        det_every=plan.get("det_every", 5))
             for t in plan["tracks"]}

    prev: dict[int, np.ndarray] = {}
    diffs: dict[int, list[float]] = {}
    cap = cv2.VideoCapture(str(video_path))
    idx = -1
    todo = set(want)
    while todo:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx not in todo:
            continue
        todo.discard(idx)
        for tid, per_frame in boxes.items():
            b = per_frame.get(idx)
            if b is None:
                prev.pop(tid, None)
                continue
            x1, y1, x2, y2 = [int(v) for v in b]
            h = y2 - y1
            my1 = y1 + int(h * 0.60)
            crop = frame[max(0, my1):max(0, y2), max(0, x1):max(0, x2)]
            if crop.size == 0:
                prev.pop(tid, None)
                continue
            g = cv2.cvtColor(cv2.resize(crop, (64, 32)), cv2.COLOR_BGR2GRAY)
            g = g.astype(np.float32) / 255.0
            p = prev.get(tid)
            if p is not None and p.shape == g.shape:
                diffs.setdefault(tid, []).append(float(np.abs(g - p).mean()))
            prev[tid] = g
    cap.release()
    return {tid: float(np.mean(v)) for tid, v in diffs.items() if v}


# ── speaker ↔ track binding ────────────────────────────────────────

def bind_speakers_to_tracks(
    plan: dict,
    track_gender: dict[int, dict],
    segments: list[dict],
    *,
    video_path: str | Path | None = None,
    require_gender_match: bool = True,
    min_score: float = FACE_BIND_MIN_SCORE,
) -> dict[str, int | None]:
    """Decide which face track (if any) belongs to each diarized speaker.

    Returns ``{speaker: track_id | None}``.  ``None`` is a legitimate and
    important answer: on the reference interview the male interviewer is
    off-camera, so his lines must leave the picture untouched rather than
    move the woman's mouth.
    """
    fps = plan["fps"]
    per_track = {t["id"]: interpolate_track(t, plan["n_frames"],
                                            det_every=plan.get("det_every", 5))
                 for t in plan["tracks"]}

    spans_by_spk: dict[str, list[tuple[float, float]]] = {}
    gender_by_spk: dict[str, str] = {}
    for seg in segments:
        spk = seg.get("speaker") or ""
        if not spk:
            continue
        spans_by_spk.setdefault(spk, []).append(
            (float(seg["start"]), float(seg["end"])))
        gender_by_spk.setdefault(
            spk, seg.get("gender") or seg.get("tts_gender") or "female")

    # Presence: how much of each speaker's speech has this face on screen.
    presence: dict[str, dict[int, float]] = {}
    for spk, spans in spans_by_spk.items():
        presence[spk] = {}
        for tid, frames in per_track.items():
            covered = 0.0
            for s, e in spans:
                a, b = int(s * fps), int(e * fps)
                if b <= a:
                    continue
                hit = sum(1 for f in range(a, b + 1) if f in frames)
                covered += hit / fps
            total = sum(e - s for s, e in spans) or 1e-6
            presence[spk][tid] = covered / total

    # Motion contrast, computed only when two same-gender candidates compete.
    motion: dict[str, dict[int, float]] = {}
    contested = False
    for spk, spans in spans_by_spk.items():
        cands = [t for t in per_track
                 if not require_gender_match
                 or track_gender.get(t, {}).get("gender") in
                 (gender_by_spk[spk], "unknown")]
        if len(cands) > 1:
            contested = True
    if contested and video_path:
        for spk, spans in spans_by_spk.items():
            try:
                motion[spk] = mouth_motion(video_path, plan, spans)
            except Exception as exc:                    # noqa: BLE001
                _log(f"mouth motion failed for {spk}: {exc}")
                motion[spk] = {}

    bindings: dict[str, int | None] = {}
    for spk in spans_by_spk:
        want = gender_by_spk[spk]
        best_tid, best_score = None, -1.0
        for tid in per_track:
            tg = track_gender.get(tid, {}).get("gender", "unknown")
            if require_gender_match and tg != "unknown" and tg != want:
                continue
            score = 0.6 * presence[spk].get(tid, 0.0)
            if tg == want:
                score += 0.25
            m = motion.get(spk, {})
            if m:
                others = [v for k, v in m.items() if k != tid]
                mine = m.get(tid, 0.0)
                ref = max(others) if others else 0.0
                score += 0.15 * float(np.clip(
                    (mine - ref) / max(mine + ref, 1e-6), -1, 1))
            if score > best_score:
                best_score, best_tid = score, tid
        bindings[spk] = best_tid if best_score >= min_score else None
        _log(f"speaker {spk} ({want}) → track {bindings[spk]} "
             f"(score {best_score:.2f})")
    return bindings


def bind_segments_to_tracks(
    plan: dict,
    track_gender: dict[int, dict],
    segments: list[dict],
    per_track: dict[int, dict[int, list[float]]],
    speakers: set[str],
    *,
    video_path: str | Path | None = None,
    min_presence: float = 0.5,
) -> dict[int, int]:
    """Per-SEGMENT face choice for speakers with no global track.

    One-track-per-speaker is an interview assumption: a single continuous
    shot in which each person is one long track.  Cut-heavy footage breaks
    it — the same actor is a fresh track in every shot, so no single track
    covers enough of a speaker's speech to clear the binding threshold and
    the whole film silently passes through with no lip-sync at all
    (measured: 72 tracks, best global score 0.34, 0 anchored frames).

    So for those speakers, choose a track per segment instead: candidates
    must match the speaker's gender and be on screen for at least
    *min_presence* of the segment.  A single candidate wins outright; when
    several same-gender faces share the shot, mouth motion during that
    segment decides (the speaker's mouth moves with the audio; a listener's
    doesn't), falling back to the larger face (drama frames the person
    speaking) when motion can't be measured.

    Returns ``{segment_index: track_id}`` — segments with no qualifying face
    are simply absent, which downstream means pass-through.
    """
    fps = plan["fps"]
    out: dict[int, int] = {}
    n_motion = 0
    for i, seg in enumerate(segments):
        spk = seg.get("speaker") or ""
        if spk not in speakers:
            continue
        want = seg.get("gender") or seg.get("tts_gender") or "female"
        a = int(float(seg["start"]) * fps)
        b = max(a + 1, int(float(seg["end"]) * fps))
        cands = []
        for tid, frames in per_track.items():
            tg = track_gender.get(tid, {}).get("gender", "unknown")
            if tg != want:          # unknown is NOT enough to animate a face
                continue
            hits = [frames[f] for f in range(a, b + 1) if f in frames]
            pres = len(hits) / (b - a + 1)
            if pres < min_presence:
                continue
            area = float(np.mean([(x2 - x1) * (y2 - y1)
                                  for x1, y1, x2, y2 in hits]))
            cands.append((tid, pres, area))
        if not cands:
            continue
        if len(cands) == 1:
            out[i] = cands[0][0]
            continue
        # Contested shot: two same-gender faces on screen.  Mouth motion over
        # exactly this segment separates speaker from listener.
        chosen = None
        if video_path is not None:
            try:
                span = [(float(seg["start"]), float(seg["end"]))]
                m = mouth_motion(video_path, plan, span)
                n_motion += 1
                ranked = sorted(((m.get(t, 0.0), t) for t, _, _ in cands),
                                reverse=True)
                if ranked and ranked[0][0] > 0:
                    chosen = ranked[0][1]
            except Exception as exc:                    # noqa: BLE001
                _log(f"segment {i}: mouth motion failed ({exc})")
        if chosen is None:
            chosen = max(cands, key=lambda c: c[2])[0]   # larger face
        out[i] = chosen
    if out:
        _log(f"per-segment binding: {len(out)} segments anchored "
             f"({n_motion} decided by mouth motion)")
    return out


# ── the plan ───────────────────────────────────────────────────────

def _tracks_cache_key(video_path: str | Path, det_kw: dict) -> str:
    """Fingerprint the inputs that detection + gender voting depend on."""
    p = Path(video_path)
    try:
        stamp = f"{p.stat().st_size}:{int(p.stat().st_mtime)}"
    except OSError:
        stamp = "0:0"
    params = json.dumps(sorted(det_kw.items()), sort_keys=True, default=str)
    return hashlib.sha1(f"{p.resolve()}|{stamp}|{params}".encode()).hexdigest()[:16]


def _load_tracks_cache(path: Path, key: str) -> dict | None:
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if blob.get("key") != key:
        return None
    plan = blob["plan"]
    plan["size"] = tuple(plan["size"])
    for t in plan["tracks"]:
        t["keyframes"] = {int(k): v for k, v in t["keyframes"].items()}
        if "yaw" in t:
            t["yaw"] = {int(k): v for k, v in t["yaw"].items()}
    return plan


def _save_tracks_cache(path: Path, key: str, plan: dict) -> None:
    ser = dict(plan)
    ser["size"] = list(plan["size"])
    ser["tracks"] = [{**t, "keyframes": {str(k): v for k, v in t["keyframes"].items()},
                      "yaw": {str(k): v for k, v in (t.get("yaw") or {}).items()}}
                     for t in plan["tracks"]]
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"key": key, "plan": ser}), encoding="utf-8")
    except OSError:
        pass


def build_face_plan(
    video_path: str | Path,
    segments: list[dict],
    *,
    out_json: str | Path | None = None,
    require_gender_match: bool = True,
    pad_seconds: float = 0.7,
    tracks_cache: str | Path | None = None,
    yaw_max: float = FACE_YAW_MAX,
    min_width: float = FACE_MIN_WIDTH,
    gate_smooth: int = FACE_GATE_SMOOTH,
    progress_cb: Callable[[str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    **det_kw,
) -> dict:
    """End-to-end: tracks → gender → binding → per-frame target boxes.

    ``frames`` maps a frame index to the box MuseTalk must drive, or omits it
    entirely when that frame should pass through unmodified.

    Detection and gender voting depend only on the *video*, not on the
    segments, but together they cost ~24 minutes on a 390 s clip — so every
    re-run triggered by an edited speaker label paid for them again.
    ``tracks_cache`` stores that half of the work, keyed on the video's
    identity plus the detection parameters, and only the binding downstream of
    it is recomputed.
    """
    def _say(m: str) -> None:
        if progress_cb:
            progress_cb(m)

    cache_path = Path(tracks_cache) if tracks_cache else None
    key = _tracks_cache_key(video_path, det_kw)
    plan = _load_tracks_cache(cache_path, key) if cache_path else None

    if plan is not None:
        _say(f"复用已缓存的人脸轨迹（{len(plan['tracks'])} 条）")
    else:
        _say("人脸检测与跟踪中…")
        plan = detect_face_tracks(
            video_path,
            progress_cb=(lambda i, n: _say(f"人脸检测 {i}/{n} 帧")) if progress_cb else None,
            cancel_check=cancel_check, **det_kw)

        _say("人脸性别识别中…")
        tg = classify_track_gender(video_path, plan, progress_cb=progress_cb)
        for t in plan["tracks"]:
            t.update(tg.get(t["id"], {}))
        if cache_path:
            _save_tracks_cache(cache_path, key, plan)

    # Head pose is video-only too, so it lives in the same cache; a cache
    # written before the yaw gate existed simply gets it filled in here.
    if any("yaw" not in t for t in plan["tracks"]):
        _say("头部姿态估计中…")
        estimate_track_pose(video_path, plan, progress_cb=progress_cb,
                            cancel_check=cancel_check)
        if cache_path:
            _save_tracks_cache(cache_path, key, plan)

    # Rebuilt from the tracks either way, so a cache hit and a fresh run feed
    # bind_speakers_to_tracks the same structure.
    tg = {t["id"]: {"gender": t.get("gender", "unknown"),
                    "conf": t.get("conf", 0.0),
                    "votes": t.get("votes", 0)}
          for t in plan["tracks"]}

    _say("说话人与人脸绑定中…")
    bindings = bind_speakers_to_tracks(
        plan, tg, segments, video_path=video_path,
        require_gender_match=require_gender_match)

    per_track = {t["id"]: interpolate_track(t, plan["n_frames"],
                                            det_every=plan.get("det_every", 5))
                 for t in plan["tracks"]}

    # Speakers no single track could cover (cut-heavy footage fragments one
    # actor into a track per shot) get a per-segment choice instead of
    # silently passing the whole film through.
    unbound = {spk for spk, tid in bindings.items() if tid is None}
    seg_bindings: dict[int, int] = {}
    if unbound:
        _say("全局绑定失败的说话人改为逐段绑定…")
        seg_bindings = bind_segments_to_tracks(
            plan, tg, segments, per_track, unbound, video_path=video_path)

    per_track_yaw = {t["id"]: interpolate_scalar(t.get("yaw") or {}, plan["n_frames"],
                                                 det_every=plan.get("det_every", 5))
                     for t in plan["tracks"]}

    fps = plan["fps"]
    frames: dict[int, list[float]] = {}
    frame_yaw: dict[int, float] = {}
    ranges: list[tuple[float, float]] = []
    segment_gated: dict[int, int] = {}
    n_gated = 0
    # lip_sync pads each speech range (0.5 s before / 0.3 s after) for a
    # smooth transition; the plan has to cover that padding too, otherwise the
    # padded frames pass through unmodified and leave a visible seam.
    pad = int(round(pad_seconds * fps))
    for i, seg in enumerate(segments):
        spk = seg.get("speaker") or ""
        tid = seg_bindings.get(i, bindings.get(spk))
        if tid is None:
            continue
        a = max(0, int(float(seg["start"]) * fps) - pad)
        b = min(plan["n_frames"], int(float(seg["end"]) * fps) + pad)
        cand: dict[int, list[float]] = {}
        for f in range(a, b + 1):
            box = per_track[tid].get(f)
            if box is not None:
                cand[f] = [round(x, 1) for x in box]
        if not cand:
            continue
        # Hard profiles / tiny faces keep the original footage (see config).
        gated = gate_frames(list(cand), cand, per_track_yaw[tid],
                            yaw_max=yaw_max, min_width=min_width,
                            smooth=gate_smooth)
        if gated:
            segment_gated[i] = len(gated)
            n_gated += len(gated)
        got = 0
        for f, box in cand.items():
            if f in gated:
                continue
            frames[f] = box
            y = per_track_yaw[tid].get(f)
            if y is not None:
                frame_yaw[f] = round(y, 1)
            got += 1
        if got:
            ranges.append((float(seg["start"]), float(seg["end"])))
    if n_gated:
        _log(f"yaw/size gate: {n_gated} frames pass through "
             f"(|yaw|>{yaw_max:.0f}° or width<{min_width}px) "
             f"across {len(segment_gated)} segments")

    out = {
        "video": str(video_path),
        "fps": fps,
        "n_frames": plan["n_frames"],
        "size": plan["size"],
        "det_every": plan.get("det_every"),
        "tracks": [{k: v for k, v in t.items() if k != "keyframes"}
                   for t in plan["tracks"]],
        "speaker_track": bindings,
        "segment_track": {str(k): v for k, v in sorted(seg_bindings.items())},
        "frames": {str(k): v for k, v in sorted(frames.items())},
        "frame_yaw": {str(k): v for k, v in sorted(frame_yaw.items())},
        "gate": {"yaw_max": yaw_max, "min_width": min_width,
                 "smooth": gate_smooth, "gated_frames": n_gated},
        "segment_gated": {str(k): v for k, v in sorted(segment_gated.items())},
        "sync_ranges": ranges,
        "_tracks_full": plan["tracks"],
    }
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(
            json.dumps({k: v for k, v in out.items() if k != "_tracks_full"},
                       ensure_ascii=False), encoding="utf-8")
    _say(f"人脸方案完成：{len(plan['tracks'])} 条轨迹，"
         f"{len(frames)} 帧有目标脸，{n_gated} 帧因侧脸/过小直通")
    return out


def save_track_thumbnails(video_path: str | Path, plan: dict,
                          out_dir: str | Path) -> dict[int, str]:
    """Write one representative crop per track so a human can eyeball gender."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks = plan.get("_tracks_full") or plan.get("tracks") or []
    want: dict[int, tuple[int, list[float]]] = {}
    for t in tracks:
        kfs = t.get("keyframes") or {}
        if not kfs:
            continue
        keys = sorted(int(k) for k in kfs)
        mid = keys[len(keys) // 2]
        want[t["id"]] = (mid, kfs[mid] if mid in kfs else kfs[str(mid)])

    written: dict[int, str] = {}
    cap = cv2.VideoCapture(str(video_path))
    idx = -1
    todo = {v[0] for v in want.values()}
    while todo:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx not in todo:
            continue
        todo.discard(idx)
        for tid, (fi, box) in want.items():
            if fi != idx:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            pad = int(0.25 * max(x2 - x1, y2 - y1))
            crop = frame[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
            if crop.size == 0:
                continue
            info = next((t for t in tracks if t["id"] == tid), {})
            dst = out_dir / (f"track_{tid}_{info.get('gender', 'unknown')}"
                             f"_{info.get('conf', 0)}.jpg")
            cv2.imwrite(str(dst), crop)
            written[tid] = str(dst)
    cap.release()
    return written
