"""CodeFormer face restoration for lip-sync output.

MuseTalk regenerates only a 256×256 face region, so on close-up (high-res)
faces the re-rendered mouth looks soft / slightly distorted next to the crisp
original skin.  This module runs a CodeFormer super-resolution / restoration
pass over the lip-synced video and blends the sharpened **lower face** (mouth
and jaw — the part MuseTalk actually changed) back onto each frame.  The upper
face (eyes/identity) is left untouched to avoid drift and temporal flicker.

Self-contained: uses the vendored CodeFormer arch under ``vendor/codeformer``
(no basicsr / facexlib / numba) and the S3FD face detector already bundled
with Wav2Lip for per-frame face boxes.
"""

import gc
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ai_movie.config import FACE_ENHANCE_FIDELITY, FACE_ENHANCE_PROTECT_LIPS
import torch

_ROOT = Path(__file__).parent.parent
_VENDOR = _ROOT / "vendor"
_CF_WEIGHTS = _ROOT / "models" / "codeformer" / "codeformer.pth"

# BiSeNet face-parsing model, bundled with MuseTalk — reused here to constrain
# the paste-back mask to real facial pixels and to detect mouth occlusion.
_MUSETALK_DIR = _ROOT / "models" / "musetalk"
_PARSE_DIR = _MUSETALK_DIR / "models" / "face-parse-bisent"
_PARSE_RESNET = _PARSE_DIR / "resnet18-5c106cde.pth"
_PARSE_WEIGHTS = _PARSE_DIR / "79999_iter.pth"

# CelebAMask-HQ BiSeNet class ids we care about.
_SKIN = 1
_LIP_CLASSES = (11, 12, 13)        # mouth interior + upper/lower lip
_FACE_CLASSES = (1, 11, 12, 13)    # skin + mouth + lips (lower-face material)
# CelebAMask-HQ ids: 0 background, 1 skin, 2/3 brows, 4/5 eyes, 6 glasses,
# 7/8 ears, 9 earring, 10 nose, 11 mouth, 12 u_lip, 13 l_lip, 14 neck,
# 15 necklace, 16 cloth, 17 hair, 18 hat.  Things that can sit in front of
# a mouth and are not face material (hands are "skin" — not catchable):
_OCCLUDER_CLASSES = (0, 15, 16, 17, 18)

_net = None
_parser = None
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def codeformer_available() -> bool:
    """True if the vendored arch and weights are present."""
    return (_VENDOR / "codeformer" / "codeformer_arch.py").exists() and _CF_WEIGHTS.exists()


def face_parser_available() -> bool:
    """True if the BiSeNet face-parsing weights (from MuseTalk) are present."""
    return _PARSE_RESNET.exists() and _PARSE_WEIGHTS.exists()


def _load_face_parser(device: str):
    """Lazy-load the BiSeNet face-parsing network (singleton)."""
    global _parser
    if _parser is not None:
        return _parser
    if str(_MUSETALK_DIR) not in sys.path:
        sys.path.insert(0, str(_MUSETALK_DIR))
    from musetalk.utils.face_parsing.model import BiSeNet

    net = BiSeNet(str(_PARSE_RESNET), n_classes=19)
    net.load_state_dict(torch.load(str(_PARSE_WEIGHTS), map_location="cpu", weights_only=False))
    net.to(device).eval()
    _parser = net
    return net


@torch.no_grad()
def _parse_crop(crop_bgr: np.ndarray, parser, device: str) -> np.ndarray:
    """Return the BiSeNet class map (uint8, 512×512) for a BGR face crop."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0)
    t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
    out = parser(t.to(device))[0]
    return out.squeeze(0).argmax(0).to("cpu").numpy().astype(np.uint8)


def _load_codeformer(device: str):
    """Lazy-load the CodeFormer network (singleton)."""
    global _net
    if _net is not None:
        return _net
    if str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))
    from codeformer.codeformer_arch import CodeFormer

    net = CodeFormer(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(device)
    ckpt = torch.load(str(_CF_WEIGHTS), map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["params_ema"])
    net.eval()
    _net = net
    return net


@torch.no_grad()
def _restore_crop(face_bgr: np.ndarray, net, device: str, w: float) -> np.ndarray:
    """Restore a (roughly square) BGR face crop; returns a 512×512 BGR image."""
    inp = cv2.resize(face_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    t = (t - 0.5) / 0.5
    t = t.to(device)
    out = net(t, w=w, adain=True)[0]
    out = out.squeeze(0).permute(1, 2, 0).clamp(-1, 1).cpu().numpy()
    out = ((out * 0.5 + 0.5) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _resolve_boxes(key_idx: list[int], key_boxes: list, key_valid: list[bool],
                   n: int, det_every: int, cuts=None) -> list:
    """Resolve a per-frame face box (or ``None``) from sparse keyframe detections.

    Faces move little between adjacent frames, so we only detect on a subset of
    keyframes (``key_idx``) and interpolate. But two things must be handled to
    avoid "restoring" the wrong pixels:

    * **Invalid keyframes** (``key_valid[k]`` False — no face / low confidence):
      frames near them get ``None`` → the caller leaves those frames untouched
      instead of restoring a non-face region.
    * **Scene cuts**: if two *valid* keyframes' boxes jump far apart, we do NOT
      interpolate across them (that would sweep a box through the cut); each
      keyframe's box is instead applied only to frames within ``det_every`` of it.

    Returns a list of length *n*; each element is an ``(y1,y2,x1,x2)`` int tuple
    or ``None``.
    """
    from ai_movie.faces import boxes_disjoint
    from ai_movie.shots import crosses_cut

    boxes: list = [None] * n
    half = max(1, det_every)

    def _xyxy(b):
        y1, y2, x1, x2 = b
        return [x1, y1, x2, y2]

    for k in range(len(key_idx)):
        if not key_valid[k]:
            continue
        i0, b0 = key_idx[k], key_boxes[k]
        # Try to interpolate forward to the next VALID, non-cut keyframe.
        # ``cuts`` (clip-local scdet cut indices) is the explicit signal;
        # ``boxes_disjoint`` is the geometric fallback shared with faces.py.
        if k + 1 < len(key_idx) and key_valid[k + 1]:
            i1, b1 = key_idx[k + 1], key_boxes[k + 1]
            cut = boxes_disjoint(_xyxy(b0), _xyxy(b1)) or crosses_cut(i0, i1, cuts)
            if not cut:
                span = max(1, i1 - i0)
                for i in range(i0, i1):
                    t = (i - i0) / span
                    boxes[i] = tuple(int(round(b0[j] + (b1[j] - b0[j]) * t)) for j in range(4))
                continue
        # No forward interpolation (last/invalid-next/cut): apply this box to a
        # small neighborhood so a lone valid detection isn't wasted.
        for i in range(max(0, i0 - half + 1), min(n, i0 + half)):
            if boxes[i] is None:
                boxes[i] = tuple(int(round(v)) for v in b0)
    return boxes


def _lower_face_mask(size: int) -> np.ndarray:
    """Feathered mask that is 1 over the lower-central face, 0 elsewhere."""
    m = np.zeros((size, size), np.float32)
    m[int(0.46 * size):int(0.99 * size), int(0.06 * size):int(0.94 * size)] = 1.0
    sigma = max(1.0, size * 0.04)
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return m


def _mouth_band_keep(size: int) -> np.ndarray:
    """Feathered mask that is 0 over the central mouth band, 1 elsewhere.

    Fallback used to protect the lip-sync mouth when the BiSeNet parser is
    unavailable (no per-pixel lip classes) — restoration then sharpens the
    surrounding skin/jaw but leaves the generated mouth region untouched.
    """
    m = np.ones((size, size), np.float32)
    m[int(0.60 * size):int(0.90 * size), int(0.24 * size):int(0.76 * size)] = 0.0
    sigma = max(1.0, size * 0.03)
    return cv2.GaussianBlur(m, (0, 0), sigmaX=sigma, sigmaY=sigma)


def restore_video(
    in_video: str | Path,
    out_video: str | Path,
    *,
    fidelity_weight: float = FACE_ENHANCE_FIDELITY,
    face_det_batch_size: int = 8,
    det_max_width: int = 640,
    det_device: str = "cpu",
    det_every: int = 4,
    conf_thresh: float = 0.9,
    occlusion_aware: bool = True,
    occlusion_lip_thresh: float = 0.004,
    protect_lips: bool = FACE_ENHANCE_PROTECT_LIPS,
    parse_device: str | None = None,
    chunk_size: int = 600,
    frame_filter: Callable[[int], bool] | None = None,
    cuts=None,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> Path | None:
    """Restore faces in *in_video*, writing the result (with original audio) to *out_video*.

    ``cuts``: global scdet cut frame indices (see ``ai_movie.shots``); face
    boxes are never interpolated across one.

    ``frame_filter(global_frame_index) -> bool`` restricts the pass to the
    frames it returns True for (typically the lip-synced ones from the face
    plan, see ``frame_filter_from_plan``); every other frame is written through
    untouched and never detected on — on a 540 s film that is the difference
    between enhancing 2 800 frames and "restoring" all 16 200, most of them
    original footage nobody asked to change.

    Parameters
    ----------
    fidelity_weight:
        CodeFormer ``w`` in [0, 1].  Higher = stay closer to the input
        (more identity fidelity, less aggressive restoration); lower = more
        restoration (sharper but can drift).  Default from config
        (0.7): measured on test_2 it keeps the generated mouth shape while
        nearly doubling mouth sharpness; 0.85 was too timid to matter.
    protect_lips:
        When True the mouth/lip pixels are **excluded** from the
        restoration mask, so the generated lip-sync mouth shape is left
        completely untouched and only the surrounding skin/jaw is sharpened.
        Default from config (False): with the lips protected the pass
        changed mouth sharpness 44→48 on test_2, i.e. it did not address
        the soft mouth the user actually sees.
        Requires the BiSeNet parser for accurate lip pixels; without it, a
        central mouth band of the geometric mask is zeroed as a fallback.
    det_max_width:
        Face detection is run on a copy of each frame downscaled so its
        width ≤ this many pixels; the resulting boxes are scaled back to
        full-resolution coordinates.  S3FD does not need full 1080p to
        locate a face, and running it at native 1080p is pathologically
        slow on ROCm (MIOpen re-tunes large convolutions).  Restoration
        still operates on the full-resolution crop.
    det_device:
        Device for the S3FD face detector — ``"cpu"`` by default.  On ROCm
        APUs (e.g. gfx1151 / Radeon 8060S) the detector's convolution shapes
        trigger a ~2-minute one-time MIOpen kernel JIT-compile that is silent
        and cannot be interrupted (Ctrl-C is stuck in a native HIP call), so
        the terminal *looks* frozen — and the on-disk kernel cache is not
        reused across processes, so it recurs on essentially every run.
        Detection is cheap enough on CPU (~0.2 s/frame at 640 px) that we run
        it there instead; CodeFormer restoration itself stays on the GPU,
        where its standard conv shapes compile in well under a second.
    det_every:
        Only run the detector on every N-th frame and linearly interpolate
        the face box for the frames in between (faces move little between
        adjacent frames).  Keeps CPU detection fast and smooths the crop.
    conf_thresh:
        Minimum S3FD detection confidence in [0, 1].  Keyframes below this are
        treated as "no face" — the restore is **skipped** for the affected
        frames (written through unchanged) instead of "restoring" a whole
        non-face / scene region.  Fixes spurious changes on scene shots.
    occlusion_aware:
        When True (and the BiSeNet parser is available), the paste-back mask is
        intersected with a face-parsing mask so only real facial-skin/mouth
        pixels are touched, and a **lip-presence gate** skips frames whose mouth
        is occluded (e.g. a hand over the mouth) — otherwise CodeFormer would
        hallucinate a mouth floating on top of the hand.
    occlusion_lip_thresh:
        Fraction of the crop area that must be lip/mouth pixels (BiSeNet classes
        11/12/13) inside the lower-face region for the frame to be considered
        non-occluded.  Below it, the mouth is deemed covered and restore is
        skipped for that frame.  (Empirically ~0.5 % cleanly separates a visible
        mouth from a hand-covered one.)
    parse_device:
        Device for the BiSeNet parser.  Defaults to the restoration device
        (GPU) — BiSeNet is a standard ResNet-18 with a fixed 512² input, so it
        does **not** trigger the S3FD MIOpen stall (verified ~0.1 s/forward).
    chunk_size:
        Frames are streamed and processed in windows of this size instead of
        loading the whole video into RAM at once — a 1080p, 10-minute video
        held as a single frame list would need ~70+ GB.  600 frames caps the
        working set to a few GB regardless of video length.
    """
    in_video = Path(in_video)
    out_video = Path(out_video)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from ai_movie.lip_sync import _detect_faces

    cap = cv2.VideoCapture(str(in_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {in_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"No frames read from {in_video}")

    net = _load_codeformer(device)

    # Face-parsing for occlusion-aware masking (optional / graceful fallback).
    parse_dev = parse_device or device
    use_parser = occlusion_aware and face_parser_available()
    parser = _load_face_parser(parse_dev) if use_parser else None
    if occlusion_aware and not use_parser and log_cb:
        log_cb("CodeFormer: BiSeNet parser unavailable — geometric mask only")
    skipped_occluded = 0

    # ── stream frames through in bounded windows, write as we go ──
    tmp_dir = Path(tempfile.mkdtemp(prefix="cf_restore_"))
    tmp_avi = tmp_dir / "restored.avi"
    writer = None
    processed = 0
    try:
        while processed < total:
            if cancel_check and cancel_check():
                cap.release()
                if writer is not None:
                    writer.release()
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return None

            window = []
            for _ in range(min(chunk_size, total - processed)):
                ok, f = cap.read()
                if not ok:
                    break
                window.append(f)
            if not window:
                break
            H, W = window[0].shape[:2]
            base = processed
            keep = ([bool(frame_filter(base + i)) for i in range(len(window))]
                    if frame_filter is not None else [True] * len(window))

            if writer is None:
                writer = cv2.VideoWriter(str(tmp_avi), cv2.VideoWriter_fourcc(*"FFV1"), fps, (W, H))
                if not writer.isOpened():   # FFV1 (lossless) may be unavailable — fall back
                    writer = cv2.VideoWriter(str(tmp_avi), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))

            # Detect faces on CPU (det_device) — the S3FD conv shapes trigger a
            # multi-minute, silent, uninterruptible MIOpen JIT-compile on ROCm
            # APUs that makes the terminal look frozen (see det_device docstring).
            # Detection runs on a downscaled copy for speed, only on every
            # det_every-th keyframe; boxes are interpolated back over the window
            # and scaled to full-resolution coordinates.
            det_scale = det_max_width / W if W > det_max_width else 1.0
            n = len(window)
            if not any(keep):
                # Nothing to enhance in this window: pass it through untouched.
                for frame in window:
                    writer.write(frame)
                    processed += 1
                if progress_cb:
                    progress_cb(processed, total)
                continue
            step = max(1, det_every)
            key_idx = [i for i in range(0, n, step)
                       if any(keep[max(0, i - step): i + step + 1])]
            if not key_idx:
                key_idx = [next(i for i in range(n) if keep[i])]
            if key_idx[-1] != n - 1 and keep[n - 1]:
                key_idx.append(n - 1)

            def _prep(f):
                if det_scale == 1.0:
                    return f
                dw, dh = int(round(W * det_scale)), int(round(H * det_scale))
                return cv2.resize(f, (dw, dh), interpolation=cv2.INTER_AREA)

            det_frames = [_prep(window[i]) for i in key_idx]
            if log_cb:
                log_cb(f"CodeFormer: detecting faces on {len(det_frames)}/{n} "
                       f"frames ({det_device}), {sum(keep)} to enhance…")
            t_det = time.time()
            face_results = _detect_faces(
                det_frames, device=det_device, batch_size=face_det_batch_size,
                pads=(0, 0, 0, 0), nosmooth=False, return_scores=True,
            )
            inv = 1.0 / det_scale
            key_boxes = [(b[0] * inv, b[1] * inv, b[2] * inv, b[3] * inv)
                         for _, b, _sc in face_results]              # (y1, y2, x1, x2)
            key_valid = [sc >= conf_thresh for _, _b, sc in face_results]
            from ai_movie.shots import local_cuts as _local_cuts
            boxes = _resolve_boxes(key_idx, key_boxes, key_valid, n, det_every,
                                   cuts=_local_cuts(cuts, base, n))
            boxes = [b if keep[i] else None for i, b in enumerate(boxes)]
            if log_cb:
                nvalid = sum(key_valid)
                log_cb(f"CodeFormer: detection done ({time.time() - t_det:.1f}s), "
                       f"{nvalid}/{len(key_valid)} keyframes confident, restoring…")
            del det_frames, face_results

            for i, frame in enumerate(window):
                if cancel_check and cancel_check():
                    cap.release()
                    if writer is not None:
                        writer.release()
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    return None
                box = boxes[i]   # None on low-confidence / no-face / cut frames
                if box is not None:
                    y1, y2, x1, x2 = box
                    bw, bh = x2 - x1, y2 - y1
                    if bw > 8 and bh > 8:
                        # Square crop centred on the face box, with margin for FFHQ-like framing.
                        S = int(max(bw, bh) * 1.4)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        sx = max(0, min(cx - S // 2, W - 1))
                        sy = max(0, min(cy - S // 2, H - 1))
                        S = min(S, W - sx, H - sy)
                        if S > 8:
                            crop = frame[sy:sy + S, sx:sx + S]
                            mask2d = _lower_face_mask(S)
                            do_blend = True
                            if parser is not None:
                                par = _parse_crop(crop, parser, parse_dev)
                                par = cv2.resize(par, (S, S), interpolation=cv2.INTER_NEAREST)
                                region_sel = mask2d > 0.5
                                lip_px = int((np.isin(par, _LIP_CLASSES) & region_sel).sum())
                                if lip_px < occlusion_lip_thresh * S * S:
                                    # No visible mouth in the lower face → occluded
                                    # (e.g. a hand). Skip so we don't paint a mouth
                                    # onto the occluder.
                                    do_blend = False
                                    skipped_occluded += 1
                                else:
                                    face = np.isin(par, _FACE_CLASSES).astype(np.float32)
                                    if protect_lips:
                                        # Exclude the generated lips so CodeFormer
                                        # can't regularize the lip-sync mouth shape.
                                        lips = np.isin(par, _LIP_CLASSES).astype(np.uint8)
                                        k = max(3, S // 40)
                                        lips = cv2.dilate(lips, np.ones((k, k), np.uint8), 1)
                                        face[lips > 0] = 0.0
                                    face = cv2.GaussianBlur(face, (0, 0), sigmaX=max(1.0, S * 0.02))
                                    mask2d = mask2d * face
                            elif protect_lips:
                                # No parser: geometrically exclude the mouth band.
                                mask2d = mask2d * _mouth_band_keep(S)
                            if do_blend:
                                restored = _restore_crop(crop, net, device, fidelity_weight)
                                restored = cv2.resize(restored, (S, S), interpolation=cv2.INTER_LINEAR)
                                mask = mask2d[..., None]
                                region = crop.astype(np.float32)
                                blended = region * (1.0 - mask) + restored.astype(np.float32) * mask
                                frame[sy:sy + S, sx:sx + S] = blended.round().astype(np.uint8)

                writer.write(frame)
                processed += 1
                if progress_cb:
                    progress_cb(processed, total)
                if log_cb and (processed % max(1, total // 10) == 0):
                    log_cb(f"CodeFormer: restored {processed}/{total} frames "
                           f"({100 * processed // total}%)")

            del window, boxes
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cap.release()
        if writer is not None:
            writer.release()

        if log_cb and parser is not None and skipped_occluded:
            log_cb(f"CodeFormer: {skipped_occluded}/{total} frames skipped "
                   f"(mouth occluded)")

        # ── mux original audio ───────────────────────────────────
        out_video.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(tmp_avi),
            "-i", str(in_video),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out_video),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            # No audio stream? retry video-only.
            subprocess.run([
                "ffmpeg", "-y", "-i", str(tmp_avi),
                "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                str(out_video),
            ], check=True, capture_output=True, text=True)
    finally:
        if cap.isOpened():
            cap.release()
        if writer is not None and writer.isOpened():
            writer.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_video


def frame_filter_from_plan(plan: dict | str | Path | None) -> Callable[[int], bool] | None:
    """Build a ``frame_filter`` for ``restore_video`` from a face plan.

    Returns ``None`` (= enhance everything) when no plan is given, so callers
    can pass the plan path straight through.
    """
    if plan is None:
        return None
    if isinstance(plan, (str, Path)):
        try:
            plan = json.loads(Path(plan).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    frames = plan.get("frames") or {}
    if not frames:
        return None
    keep = {int(k) for k in frames}
    return lambda i: i in keep


def occlusion_gate_video(
    orig_video: str | Path,
    lipsync_video: str | Path,
    out_video: str | Path,
    *,
    det_device: str = "cpu",
    det_every: int = 4,
    det_max_width: int = 640,
    face_det_batch_size: int = 8,
    conf_thresh: float = 0.9,
    occlusion_lip_thresh: float = 0.004,
    min_occluded_run: int = 6,
    parse_device: str | None = None,
    chunk_size: int = 600,
    occlusion_mode: str | None = None,
    full_lip_thresh: float | None = None,
    region_dilate_frac: float = 0.03,
    region_feather_frac: float = 0.02,
    cuts=None,
    stats: dict | None = None,
    cancel_check: Callable[[], bool] | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> Path:
    """Keep the ORIGINAL pixels wherever the mouth is occluded / no face found.

    MuseTalk pastes a generated mouth onto each original frame.  When the mouth
    is covered it paints a mouth onto the occluder.  Two modes:

    ``"frame"`` (v2): per frame, choose between the lip-synced frame (mouth
    visible) and the original frame (mouth occluded or no confident face)
    using the BiSeNet lip-presence gate, only inside runs of at least
    ``min_occluded_run`` occluded frames (the lip check misfires on isolated
    head-turn / motion-blur frames; a real hand-over-mouth lasts many).

    ``"region"`` (v3): parse both the original and the lip-synced lower
    face.  Pixels the original labels as hair / hat / clothing / background
    — or as non-face where the generated frame put lips — are the occluder;
    the original is pasted back only there (dilated, feathered), so the
    mouth keeps moving around a strand of hair instead of freezing for the
    whole run.  The whole frame is reverted only when the original has
    essentially no lip pixels at all (``full_lip_thresh``) for a sustained
    run.  BiSeNet labels hands as skin, so a hand is *not* caught — that
    limit is documented, not hidden.

    ``cuts`` are clip-local scdet cut indices (box interpolation never
    bridges one); ``stats`` receives per-clip counts.  Audio is taken from
    *lipsync_video*.  If the parser is unavailable the lip-synced video is
    passed through unchanged.
    """
    from ai_movie.config import OCCLUSION_FULL_LIP_THRESH, OCCLUSION_MODE
    from ai_movie.shots import local_cuts

    occlusion_mode = occlusion_mode or OCCLUSION_MODE
    full_lip_thresh = OCCLUSION_FULL_LIP_THRESH if full_lip_thresh is None else full_lip_thresh
    region_mode = occlusion_mode == "region"

    orig_video, lipsync_video, out_video = Path(orig_video), Path(lipsync_video), Path(out_video)
    if not face_parser_available():
        out_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(lipsync_video), str(out_video))
        return out_video

    from ai_movie.lip_sync import _detect_faces
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parse_dev = parse_device or device
    parser = _load_face_parser(parse_dev)

    capO = cv2.VideoCapture(str(orig_video))
    capL = cv2.VideoCapture(str(lipsync_video))
    if not capO.isOpened() or not capL.isOpened():
        raise RuntimeError("occlusion_gate: cannot open input video(s)")
    fps = capL.get(cv2.CAP_PROP_FPS) or 25.0
    total = min(int(capO.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(capL.get(cv2.CAP_PROP_FRAME_COUNT)))

    tmp_dir = Path(tempfile.mkdtemp(prefix="cf_occgate_"))
    tmp_avi = tmp_dir / "gated.avi"
    writer = None
    processed = 0
    reverted = 0
    region_frames = 0
    region_px = 0.0
    try:
        while processed < total:
            if cancel_check and cancel_check():
                capO.release(); capL.release()
                if writer is not None:
                    writer.release()
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return out_video
            base = processed
            ow, lw = [], []
            for _ in range(min(chunk_size, total - processed)):
                okO, fO = capO.read(); okL, fL = capL.read()
                if not (okO and okL):
                    break
                ow.append(fO); lw.append(fL)
            if not ow:
                break
            H, W = lw[0].shape[:2]
            if writer is None:
                writer = cv2.VideoWriter(str(tmp_avi), cv2.VideoWriter_fourcc(*"FFV1"), fps, (W, H))
                if not writer.isOpened():
                    writer = cv2.VideoWriter(str(tmp_avi), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))

            # Detect faces on the ORIGINAL frames (CPU), subsampled + interpolated.
            det_scale = det_max_width / W if W > det_max_width else 1.0
            n = len(ow)
            key_idx = list(range(0, n, max(1, det_every)))
            if key_idx[-1] != n - 1:
                key_idx.append(n - 1)

            def _prep(f):
                if det_scale == 1.0:
                    return f
                dw, dh = int(round(W * det_scale)), int(round(H * det_scale))
                return cv2.resize(f, (dw, dh), interpolation=cv2.INTER_AREA)

            fr = _detect_faces([_prep(ow[i]) for i in key_idx], device=det_device,
                               batch_size=face_det_batch_size, pads=(0, 0, 0, 0),
                               nosmooth=False, return_scores=True)
            inv = 1.0 / det_scale
            key_boxes = [(b[0] * inv, b[1] * inv, b[2] * inv, b[3] * inv) for _, b, _s in fr]
            key_valid = [s >= conf_thresh for _, _b, s in fr]
            boxes = _resolve_boxes(key_idx, key_boxes, key_valid, n, det_every,
                                   cuts=local_cuts(cuts, base, n))

            # First pass: per-frame occluded flag (no confident face OR mouth
            # has essentially no visible lip pixels).  In region mode the
            # per-pixel occluder is composited right here and only a total
            # loss of lip pixels counts as "occluded" for the frame revert.
            occluded = [False] * n
            for i in range(n):
                box = boxes[i]
                if box is None:
                    occluded[i] = True
                    continue
                y1, y2, x1, x2 = box
                bw, bh = x2 - x1, y2 - y1
                if bw > 8 and bh > 8:
                    S = int(max(bw, bh) * 1.4)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    sx = max(0, min(cx - S // 2, W - 1))
                    sy = max(0, min(cy - S // 2, H - 1))
                    S = min(S, W - sx, H - sy)
                    if S > 8:
                        crop_o = ow[i][sy:sy + S, sx:sx + S]
                        par = _parse_crop(crop_o, parser, parse_dev)
                        par = cv2.resize(par, (S, S), interpolation=cv2.INTER_NEAREST)
                        region_sel = _lower_face_mask(S) > 0.5
                        lip_px = int((np.isin(par, _LIP_CLASSES) & region_sel).sum())
                        if not region_mode:
                            occluded[i] = lip_px < occlusion_lip_thresh * S * S
                            continue
                        occluded[i] = lip_px < full_lip_thresh * S * S
                        if occluded[i]:
                            continue
                        crop_l = lw[i][sy:sy + S, sx:sx + S]
                        if crop_l.shape != crop_o.shape:
                            continue
                        par_l = _parse_crop(crop_l, parser, parse_dev)
                        par_l = cv2.resize(par_l, (S, S), interpolation=cv2.INTER_NEAREST)
                        occ = region_sel & (
                            np.isin(par, _OCCLUDER_CLASSES)
                            | (~np.isin(par, _FACE_CLASSES) & np.isin(par_l, _LIP_CLASSES)))
                        if not occ.any():
                            continue
                        m = occ.astype(np.uint8) * 255
                        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                        if not m.any():
                            continue
                        k = max(3, int(region_dilate_frac * S)) | 1
                        m = cv2.dilate(m, np.ones((k, k), np.uint8))
                        sig = max(1.0, region_feather_frac * S)
                        mf = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), sig)
                        mf = np.clip(mf, 0.0, 1.0)[..., None]
                        blended = crop_o.astype(np.float32) * mf + crop_l.astype(np.float32) * (1 - mf)
                        lw[i][sy:sy + S, sx:sx + S] = np.clip(blended, 0, 255).astype(np.uint8)
                        region_frames += 1
                        region_px += float(mf.mean())

            # Second pass: only revert frames in a SUSTAINED occluded run
            # (filters out sporadic BiSeNet misses on normal talking frames).
            revert = [False] * n
            j = 0
            while j < n:
                if occluded[j]:
                    k = j
                    while k < n and occluded[k]:
                        k += 1
                    if k - j >= min_occluded_run:
                        for m in range(j, k):
                            revert[m] = True
                    j = k
                else:
                    j += 1

            for i in range(n):
                writer.write(ow[i] if revert[i] else lw[i])
                if revert[i]:
                    reverted += 1
                processed += 1

            del ow, lw, boxes
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if log_cb:
                log_cb(f"occlusion-gate: {processed}/{total} frames "
                       f"({reverted} kept original)")

        capO.release(); capL.release()
        if writer is not None:
            writer.release()

        out_video.parent.mkdir(parents=True, exist_ok=True)
        res = subprocess.run([
            "ffmpeg", "-y", "-i", str(tmp_avi), "-i", str(lipsync_video),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_video),
        ], capture_output=True, text=True)
        if res.returncode != 0:
            subprocess.run([
                "ffmpeg", "-y", "-i", str(tmp_avi),
                "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(out_video),
            ], check=True, capture_output=True, text=True)
        if log_cb:
            log_cb(f"occlusion-gate done ({occlusion_mode}): {reverted}/{total} frames "
                   f"kept original"
                   + (f", {region_frames} frames region-patched" if region_mode else ""))
        if stats is not None:
            stats.update({"mode": occlusion_mode, "frames": int(total),
                          "reverted_frames": int(reverted),
                          "region_frames": int(region_frames),
                          "region_px_mean": round(region_px / region_frames, 4)
                          if region_frames else 0.0})
    finally:
        if capO.isOpened():
            capO.release()
        if capL.isOpened():
            capL.release()
        if writer is not None and writer.isOpened():
            writer.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_video
