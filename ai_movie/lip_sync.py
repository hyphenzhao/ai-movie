"""Lip-sync module: Wav2Lip + MuseTalk audio-driven face reenactment.

Two backends, auto-selected by the GUI:

- **Wav2Lip** (96×96, stable) — original GAN model, S3FD face detector.
- **MuseTalk** (256×256, HQ) — Tencent UNet-based, better lip accuracy.

Usage::

    from ai_movie.lip_sync import wav2lip_sync, musetalk_sync
    wav2lip_sync("input.mp4", "audio.wav", "output.mp4")
    musetalk_sync("input.mp4", "audio.wav", "output.mp4")
"""

import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable

import gc
import cv2
import numpy as np
import torch

# ── PyTorch 2.6+ safe globals fix ───────────────────────────────────
# torch.load defaults to weights_only=True since PyTorch 2.6.
# mmengine/mmpose checkpoints contain numpy objects not in the safe list.
try:
    torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
except (AttributeError, TypeError):
    pass
try:
    torch.serialization.add_safe_globals([np.ndarray, np.dtype])
except (AttributeError, TypeError):
    pass

# ── debug logging ───────────────────────────────────────────────────
_DEBUG = True
_LOG_FILE = None  # set by wav2lip_sync() each run


def _log(msg: str):
    """Print timestamped debug message to stderr and log file."""
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[Wav2Lip {ts}] {msg}"
    print(line, file=sys.stderr, flush=True)
    if _LOG_FILE:
        try:
            with open(_LOG_FILE, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass


# ── paths ──────────────────────────────────────────────────────────
_MODELS_DIR = Path(__file__).parent.parent / "models" / "wav2lip"
_CHECKPOINT = _MODELS_DIR / "checkpoints" / "wav2lip_gan.pth"

# ── model cache ────────────────────────────────────────────────────
_model = None


def _ensure_import_path():
    """Add Wav2Lip source dir to sys.path so face_detection / models are importable."""
    s = str(_MODELS_DIR)
    if s not in sys.path:
        sys.path.insert(0, s)
        _log(f"Added to sys.path: {s}")


def _load_model(checkpoint_path: str | Path | None = None):
    """Lazy-load the Wav2Lip GAN model (singleton, GPU when available)."""
    global _model
    if _model is not None:
        _log("Model already loaded (cached)")
        return _model

    _ensure_import_path()
    from models import Wav2Lip as Wav2LipNet

    ckpt = Path(checkpoint_path) if checkpoint_path else _CHECKPOINT
    _log(f"Loading checkpoint: {ckpt}")
    _log(f"Checkpoint exists: {ckpt.exists()}, size: {ckpt.stat().st_size / 1e6:.1f} MB" if ckpt.exists() else "MISSING!")

    if not ckpt.exists():
        raise FileNotFoundError(
            f"Wav2Lip checkpoint not found: {ckpt}\n"
            f"Download it from:\n"
            f"  huggingface-cli download camenduru/Wav2Lip --include 'checkpoints/wav2lip_gan.pth'\n"
            f"Place it at: models/wav2lip/checkpoints/wav2lip_gan.pth"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Device: {device}")
    _log(f"Loading state dict from checkpoint...")
    t0 = time.time()
    checkpoint = torch.load(str(ckpt), map_location=device if device == "cpu" else None, weights_only=False)
    s = checkpoint["state_dict"]
    new_s = {k.replace("module.", ""): v for k, v in s.items()}
    _log(f"State dict loaded ({time.time() - t0:.1f}s), {len(new_s)} keys")

    _model = Wav2LipNet()
    _model.load_state_dict(new_s)
    _model = _model.to(device).eval()
    _log(f"Model ready on {device}")
    return _model


# ── face detection ─────────────────────────────────────────────────

def _detect_faces(frames: list[np.ndarray],
                  device: str = "cuda",
                  batch_size: int = 16,
                  pads: tuple = (0, 10, 0, 0),
                  nosmooth: bool = False,
                  return_scores: bool = False,
                  ) -> list:
    """Detect faces in every frame; return [(face_crop, (y1,y2,x1,x2)), ...].

    If ``return_scores`` is True, each entry gains a third element — the S3FD
    detection confidence in [0, 1] (``0.0`` for frames with no face, where the
    full frame is used as fallback). Callers can use it to skip / down-weight
    unreliable detections instead of "restoring" a whole non-face frame.
    """
    _ensure_import_path()
    import face_detection

    _log(f"Loading face detector on {device}...")
    t0 = time.time()
    detector = face_detection.FaceAlignment(
        face_detection.LandmarksType._2D, flip_input=False, device=device,
    )
    _log(f"Face detector ready ({time.time() - t0:.1f}s)")

    # Run detection (auto-reduces batch size on OOM)
    total_frames = len(frames)
    _log(f"Detecting faces in {total_frames} frames (batch_size={batch_size})...")
    t0 = time.time()
    attempts = 0
    while True:
        predictions = []
        try:
            for i in range(0, total_frames, batch_size):
                predictions.extend(
                    detector.get_detections_for_batch(
                        np.array(frames[i : i + batch_size]),
                        return_scores=return_scores,
                    )
                )
        except RuntimeError as e:
            attempts += 1
            if batch_size == 1:
                raise RuntimeError(
                    "Image too big to run face detection on GPU. "
                    "Use resize_factor > 1."
                ) from e
            batch_size //= 2
            _log(f"Face detection OOM; reducing batch to {batch_size} (attempt {attempts})")
            continue
        break
    _log(f"Face detection done ({time.time() - t0:.1f}s)")

    _log(f"Face detection done ({time.time() - t0:.1f}s), "
         f"predictions={len(predictions)}, frames={total_frames}")

    if len(predictions) != total_frames:
        _log(f"WARNING: predictions count ({len(predictions)}) != frames count "
             f"({total_frames}), truncating to min")
        total_frames = min(len(predictions), total_frames)

    pady1, pady2, padx1, padx2 = pads
    results = []
    scores = []
    no_face_count = 0
    for idx in range(total_frames):
        rect = predictions[idx]
        image = frames[idx]
        if rect is None:
            no_face_count += 1
            scores.append(0.0)          # no face → zero confidence
            # Save faulty frame for debugging
            fault_path = tempfile.gettempdir() + f"/wav2lip_noface_{idx}.jpg"
            cv2.imwrite(fault_path, image)
            _log(f"No face in frame {idx}, saved to {fault_path}")
            # Use full frame as fallback instead of crashing
            h, w = image.shape[:2]
            rect = [0, 0, w, h]
        else:
            scores.append(float(rect[4]) if return_scores and len(rect) > 4 else 1.0)
        y1 = max(0, int(rect[1]) - pady1)
        y2 = min(image.shape[0], int(rect[3]) + pady2)
        x1 = max(0, int(rect[0]) - padx1)
        x2 = min(image.shape[1], int(rect[2]) + padx2)
        results.append([x1, y1, x2, y2])

    if no_face_count > 0:
        _log(f"WARNING: {no_face_count}/{total_frames} frames had no face detected (using full frame fallback)")

    boxes = np.array(results)
    if not nosmooth:
        boxes = _smooth_boxes(boxes, T=5)
    if return_scores:
        face_list = [
            [image[y1:y2, x1:x2], (y1, y2, x1, x2), sc]
            for image, (x1, y1, x2, y2), sc in zip(frames, boxes, scores)
        ]
    else:
        face_list = [
            [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
            for image, (x1, y1, x2, y2) in zip(frames, boxes)
        ]

    # Release face detector from GPU memory
    del detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return face_list


def _smooth_boxes(boxes: np.ndarray, T: int = 5) -> np.ndarray:
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


# ── audio processing ───────────────────────────────────────────────

def _prepare_mel_chunks(audio_path: str | Path, fps: float,
                        mel_step_size: int = 16) -> list:
    """Load audio and split mel-spectrogram into per-frame chunks."""
    _ensure_import_path()
    from audio import load_wav, melspectrogram

    _log(f"Loading audio: {audio_path}")
    t0 = time.time()
    wav = load_wav(str(audio_path), 16000)
    _log(f"Audio loaded: {len(wav)} samples ({len(wav)/16000:.1f}s), computing mel...")

    mel = melspectrogram(wav)
    _log(f"Mel shape: {mel.shape}, fps={fps}")

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            "Mel contains NaN! Try adding a small noise floor to the audio."
        )

    mel_chunks = []
    mel_idx_multiplier = 80.0 / fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > mel.shape[1]:
            mel_chunks.append(mel[:, mel.shape[1] - mel_step_size :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    _log(f"Mel chunks: {len(mel_chunks)} (took {time.time()-t0:.1f}s)")
    return mel_chunks


# ── main API ───────────────────────────────────────────────────────

def wav2lip_sync(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    checkpoint_path: str | Path | None = None,
    resize_factor: int = 1,
    pads: tuple = (0, 10, 0, 0),
    nosmooth: bool = False,
    face_det_batch_size: int = 16,
    wav2lip_batch_size: int = 32,
    img_size: int = 96,
    max_frames_in_memory: int = 600,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    debug_log: str | Path | None = None,
) -> Path:
    """Run Wav2Lip inference on a video + audio pair.

    For videos with more than *max_frames_in_memory*, processing is
    automatically split into sub-segments to bound peak memory.  Sub-
    segment temp files are concatenated before the final mux step.

    Parameters
    ----------
    video_path:
        Input video file (must contain at least one face in every frame).
    audio_path:
        Audio file (WAV recommended, 16 kHz mono).  Drives the mouth
        movements in the output.
    output_path:
        Destination ``.mp4`` path.
    checkpoint_path:
        Path to ``wav2lip_gan.pth``.  Defaults to the bundled checkpoint
        under ``models/wav2lip/checkpoints/``.
    resize_factor:
        Divide video resolution by this factor before face detection
        (e.g. ``2`` → 540p from 1080p).  Reduces VRAM and speeds up
        face detection.
    pads:
        ``(top, bottom, left, right)`` extra pixels around the detected
        face bbox.  Default ``(0, 10, 0, 0)`` includes some chin.
    nosmooth:
        Disable temporal smoothing of face bounding boxes.
    face_det_batch_size:
        Batch size for S3FD face detection.
    wav2lip_batch_size:
        Batch size for the Wav2Lip generator.
    img_size:
        Face crop size (default 96).  Must match the checkpoint.
    max_frames_in_memory:
        If total frames exceed this, split into sub-segments of this
        size.  Default 600 (∼ 24 s at 25 fps) keeps frame memory
        ≤ ~1 GB at 1080p.
    progress_cb:
        ``progress_cb(current_batch, total_batches)`` called after each
        generator batch.
    cancel_check:
        Return ``True`` to abort between batches.
    debug_log:
        Optional path to write a detailed debug log.

    Returns
    -------
    ``output_path`` on success.

    Raises
    ------
    FileNotFoundError
        If the checkpoint or input files are missing.
    ValueError
        If no face is detected in a frame, or audio produces NaN mel.
    RuntimeError
        If GPU OOM on face detection even at batch-size 1.
    """
    global _LOG_FILE
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    # Set up debug log
    if debug_log:
        _LOG_FILE = Path(debug_log)
    else:
        _LOG_FILE = output_path.parent / "wav2lip_debug.log"

    _log("=" * 50)
    _log(f"Wav2Lip sync START")
    _log(f"  video: {video_path} (exists={video_path.exists()}, size={video_path.stat().st_size/1e6:.1f}MB)" if video_path.exists() else f"  video: {video_path} MISSING!")
    _log(f"  audio: {audio_path} (exists={audio_path.exists()})" if audio_path.exists() else f"  audio: {audio_path} MISSING!")
    _log(f"  output: {output_path}")
    _log(f"  resize_factor={resize_factor}, batches={wav2lip_batch_size}, "
         f"max_frames_in_memory={max_frames_in_memory}")
    _log(f"  ROCm/CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        _log(f"  GPU: {torch.cuda.get_device_name(0)}")
        _log(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Probe video ──────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Decide whether to split into sub-segments
    if total_frame_count > max_frames_in_memory:
        _log(f"  {total_frame_count} frames > {max_frames_in_memory} limit — "
             f"splitting into sub-segments")
        return _wav2lip_sync_chunked(
            video_path=video_path,
            audio_path=audio_path,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            resize_factor=resize_factor,
            pads=pads,
            nosmooth=nosmooth,
            face_det_batch_size=face_det_batch_size,
            wav2lip_batch_size=wav2lip_batch_size,
            img_size=img_size,
            max_frames_per_chunk=max_frames_in_memory,
            fps=fps,
            total_frames=total_frame_count,
            device=device,
            progress_cb=progress_cb,
            cancel_check=cancel_check,
        )

    # ── Single-segment path (original behaviour) ─────────────────
    return _wav2lip_sync_single(
        video_path=video_path,
        audio_path=audio_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        resize_factor=resize_factor,
        pads=pads,
        nosmooth=nosmooth,
        face_det_batch_size=face_det_batch_size,
        wav2lip_batch_size=wav2lip_batch_size,
        img_size=img_size,
        device=device,
        fps=fps,
        progress_cb=progress_cb,
        cancel_check=cancel_check,
    )


def _wav2lip_sync_single(
    *,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    checkpoint_path: Path | None,
    resize_factor: int,
    pads: tuple,
    nosmooth: bool,
    face_det_batch_size: int,
    wav2lip_batch_size: int,
    img_size: int,
    device: str,
    fps: float,
    progress_cb: Callable[[int, int], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> Path:
    """Core Wav2Lip processing for a single (short) video segment."""

    try:
        # ── 1. Read video frames ──────────────────────────────────
        _log("Step 1/5: Reading video frames...")
        t0 = time.time()
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        read_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            read_count += 1
            if resize_factor > 1:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (w // resize_factor, h // resize_factor))
            frames.append(frame)
        cap.release()
        frame_shape = frames[0].shape if frames else (0, 0)
        frame_mem_mb = len(frames) * np.prod(frame_shape) * 1 / 1e6
        _log(f"  Read {read_count} frames, kept {len(frames)} "
             f"({time.time() - t0:.1f}s), frame shape: {frame_shape}, "
             f"est. RAM: {frame_mem_mb:.0f} MB")
        if frame_mem_mb > 4000:
            _log(f"  ⚠️  Large video — consider using resize_factor=2 to reduce memory")

        if not frames:
            raise ValueError(f"No frames read from {video_path}")

        # ── 2. Prepare mel chunks ─────────────────────────────────
        _log("Step 2/5: Computing mel spectrogram...")
        t0 = time.time()
        mel_chunks = _prepare_mel_chunks(audio_path, fps)
        count = min(len(frames), len(mel_chunks))
        frames = frames[:count]
        mel_chunks = mel_chunks[:count]
        _log(f"  Frames: {len(frames)}, mel_chunks: {len(mel_chunks)}")

        # ── 3. Face detection ─────────────────────────────────────
        _log("Step 3/5: Face detection...")
        t0 = time.time()
        face_results = _detect_faces(
            frames, device=device, batch_size=face_det_batch_size,
            pads=pads, nosmooth=nosmooth,
        )
        _log(f"  Face detection total: {time.time() - t0:.1f}s")

        # ── 4. Extract coords & crops ─────────────────────────────
        _log("Step 4/6: Prepping inference data...")
        t0 = time.time()
        face_coords = [coords for _, coords in face_results]
        face_crops = [face for face, _ in face_results]
        del face_results

        total_batches = (len(mel_chunks) + wav2lip_batch_size - 1) // wav2lip_batch_size
        _log(f"  {total_batches} batches ({time.time()-t0:.1f}s)")

        # ── 5. Wav2Lip inference ──────────────────────────────────
        _log("Step 5/6: Running Wav2Lip inference...")
        t0 = time.time()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = _load_model(checkpoint_path)

        frame_h, frame_w = frames[0].shape[:2]
        temp_dir = Path(tempfile.mkdtemp(prefix="wav2lip_"))
        temp_avi = temp_dir / "result.avi"
        _log(f"  Temp dir: {temp_dir}")

        def _batch_generator():
            ib, mb, idxs, cb = [], [], [], []
            for i, m in enumerate(mel_chunks):
                face_resized = cv2.resize(face_crops[i], (img_size, img_size))
                ib.append(face_resized)
                mb.append(m)
                idxs.append(i)
                cb.append(face_coords[i])
                if len(ib) >= wav2lip_batch_size:
                    yield _make_tensors(np.asarray(ib), np.asarray(mb),
                                        idxs, cb, img_size, device)
                    ib, mb, idxs, cb = [], [], [], []
            if ib:
                yield _make_tensors(np.asarray(ib), np.asarray(mb),
                                    idxs, cb, img_size, device)

        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        out = cv2.VideoWriter(str(temp_avi), fourcc, fps, (frame_w, frame_h))
        if not out.isOpened():
            raise RuntimeError(f"Cannot create temp video: {temp_avi}")

        try:
            total_frames_out = 0
            for batch_idx, (img_batch, mel_batch, frame_indices, coords) in enumerate(_batch_generator()):
                if cancel_check and cancel_check():
                    _log(f"  Cancelled at batch {batch_idx + 1}/{total_batches}")
                    out.release()
                    return None

                batch_t0 = time.time()
                with torch.no_grad():
                    pred = model(mel_batch, img_batch)

                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

                for p, idx, c in zip(pred, frame_indices, coords):
                    y1, y2, x1, x2 = c
                    p_resized = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    f = frames[idx]
                    f[y1:y2, x1:x2] = p_resized
                    out.write(f)
                    total_frames_out += 1

                if progress_cb:
                    progress_cb(batch_idx + 1, total_batches)

            out.release()
            _log(f"  Wav2Lip inference done: {total_frames_out} frames, "
                 f"{time.time() - t0:.1f}s total")

            # ── 6. Mux video + audio ──────────────────────────────
            _log("Step 6/6: Composing final video with FFmpeg...")
            t0 = time.time()
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(temp_avi),
                "-i", str(audio_path),
                "-c:v", "libx264", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                str(output_path),
            ], check=True, capture_output=True, text=True)
            _log(f"  FFmpeg done ({time.time() - t0:.1f}s), output: "
                 f"{output_path.stat().st_size / 1e6:.1f} MB")

        finally:
            if out.isOpened():
                out.release()
            shutil.rmtree(temp_dir, ignore_errors=True)
            _log(f"  Cleaned up temp dir")

    except Exception:
        _log(f"ERROR: {traceback.format_exc()}")
        raise

    _log(f"Wav2Lip sync DONE: {output_path}")
    return output_path


def _wav2lip_sync_chunked(
    *,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    checkpoint_path: Path | None,
    resize_factor: int,
    pads: tuple,
    nosmooth: bool,
    face_det_batch_size: int,
    wav2lip_batch_size: int,
    img_size: int,
    max_frames_per_chunk: int,
    fps: float,
    total_frames: int,
    device: str,
    progress_cb: Callable[[int, int], None] | None,
    cancel_check: Callable[[], bool] | None,
) -> Path:
    """Process a long video by splitting into sub-segments, each processed
    by ``_wav2lip_sync_single``, then concatenating the results."""

    video_duration = total_frames / fps
    chunk_duration = max_frames_per_chunk / fps
    overlap = 0.1  # 100ms overlap for smooth seams

    # Calculate chunk boundaries (start_sec, duration_sec)
    chunks: list[tuple[float, float]] = []
    cursor = 0.0
    while cursor < video_duration:
        dur = min(chunk_duration, video_duration - cursor)
        chunks.append((cursor, dur))
        cursor += dur - overlap  # small overlap to avoid 1-frame gaps

    total_chunks = len(chunks)
    _log(f"  Split into {total_chunks} chunks of ~{chunk_duration:.1f}s each")

    tmp_dir = Path(tempfile.mkdtemp(prefix="wav2lip_chunked_"))
    chunk_outputs: list[Path] = []

    try:
        for ci, (start, dur) in enumerate(chunks):
            if cancel_check and cancel_check():
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
                return None

            chunk_video = tmp_dir / f"chunk_{ci:04d}_video.mp4"
            chunk_audio = tmp_dir / f"chunk_{ci:04d}_audio.wav"
            chunk_out = tmp_dir / f"chunk_{ci:04d}_out.mp4"

            _log(f"  Chunk {ci+1}/{total_chunks}: [{start:.1f}s, {start+dur:.1f}s] "
                 f"({dur:.1f}s)")

            # Cut video & audio for this chunk
            _cut_video_clip(video_path, start, dur, chunk_video, reencode=True)
            _cut_audio_clip(audio_path, start, dur, chunk_audio)

            # Process this chunk
            _wav2lip_sync_single(
                video_path=chunk_video,
                audio_path=chunk_audio,
                output_path=chunk_out,
                checkpoint_path=checkpoint_path,
                resize_factor=resize_factor,
                pads=pads,
                nosmooth=nosmooth,
                face_det_batch_size=face_det_batch_size,
                wav2lip_batch_size=wav2lip_batch_size,
                img_size=img_size,
                device=device,
                fps=fps,
                progress_cb=None,  # chunk-level progress not useful
                cancel_check=cancel_check,
            )
            chunk_outputs.append(chunk_out)

            # Memory cleanup after each chunk
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Report overall progress
            if progress_cb:
                progress_cb(ci + 1, total_chunks)

        # ── Concatenate chunks ────────────────────────────────────
        _log(f"  Concatenating {len(chunk_outputs)} chunks…")
        concat_video = tmp_dir / "concat_video.mp4"
        _concatenate_videos(chunk_outputs, concat_video)

        # Mux with full audio
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(concat_video),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(output_path),
        ], check=True, capture_output=True, text=True)

        _log(f"Wav2Lip chunked sync DONE: {output_path}")

    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    return output_path


# ── internal helpers ───────────────────────────────────────────────

def _make_tensors(img_batch: np.ndarray, mel_batch: np.ndarray,
                  frame_batch: list, coords_batch: list,
                  img_size: int, device: str):
    """Build masked face tensor + mel tensor for one Wav2Lip forward pass."""
    # Half-face masking (lower half is the "unknown" region)
    img_masked = img_batch.copy()
    img_masked[:, img_size // 2 :, :] = 0
    img_input = np.concatenate((img_masked, img_batch), axis=3) / 255.0
    mel_input = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1],
                                        mel_batch.shape[2], 1])

    img_t = torch.FloatTensor(np.transpose(img_input, (0, 3, 1, 2))).to(device)
    mel_t = torch.FloatTensor(np.transpose(mel_input, (0, 3, 1, 2))).to(device)
    return img_t, mel_t, frame_batch, coords_batch


# ── MuseTalk backend (256×256, HQ) ──────────────────────────────────

_MUSETALK_DIR = Path(__file__).parent.parent / "models" / "musetalk"


def _find_musetalk_inference_script() -> Path | None:
    """Locate the MuseTalk inference entry point."""
    candidates = [
        _MUSETALK_DIR / "scripts" / "inference.py",
        _MUSETALK_DIR / "inference.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _find_musetalk_unet() -> tuple[Path | None, Path | None]:
    """Find MuseTalk UNet checkpoint and config."""
    # v1.5 (best quality)
    unet_path = _MUSETALK_DIR / "models" / "musetalkV15" / "unet.pth"
    cfg_path = _MUSETALK_DIR / "models" / "musetalkV15" / "musetalk.json"
    if unet_path.exists() and cfg_path.exists():
        return unet_path, cfg_path
    # v1.0 (fallback)
    unet_path = _MUSETALK_DIR / "models" / "musetalk" / "unet.pth"
    cfg_path = _MUSETALK_DIR / "models" / "musetalk" / "musetalk.json"
    if unet_path.exists() and cfg_path.exists():
        return unet_path, cfg_path
    return None, None


def musetalk_available() -> bool:
    """Return True if MuseTalk is installed and models are present."""
    script = _find_musetalk_inference_script()
    unet, cfg = _find_musetalk_unet()
    return script is not None and unet is not None


def _extract_speech_ranges(
    tts_results: list[dict],
    video_duration: float,
    padding_before: float = 0.5,
    padding_after: float = 0.3,
    merge_gap: float = 1.0,
    max_duration: float = 12.0,
    anchor_gender: str | None = None,
) -> list[tuple[float, float]]:
    """Extract and merge time ranges where TTS speech audio exists.

    Parameters
    ----------
    tts_results:
        Segment dicts with ``start``, ``end``, and ``audio`` keys.
        Segments with ``audio is None`` or missing file are treated as silent.
    video_duration:
        Total video length in seconds (used to clamp ranges).
    padding_before:
        Extra seconds before each speech range.
    padding_after:
        Extra seconds after each speech range.
    merge_gap:
        Merge adjacent ranges whose gap is ≤ this many seconds.
    max_duration:
        Maximum duration of any single segment in seconds.
        Segments longer than this are split into smaller chunks to limit
        per-call memory usage (frames × resolution × channels add up fast).
        Default 12 s keeps peak frame memory ≤ ~3 GB at 1080p.

    Returns
    -------
    List of ``(start_sec, end_sec)`` tuples, sorted ascending.
    """
    # 1. Collect raw ranges from segments with valid audio.
    #    When anchor_gender is set (person-anchoring), only segments whose
    #    speaker matches that gender are lip-synced; the rest pass through as
    #    original video (their dubbed audio still plays via build_speech_track).
    raw: list[tuple[float, float]] = []
    skipped_gender = 0
    for seg in tts_results:
        audio = seg.get("audio")
        if not audio or not Path(str(audio)).exists():
            continue
        if anchor_gender and seg.get("tts_gender") != anchor_gender:
            skipped_gender += 1
            continue
        raw.append((float(seg["start"]), float(seg["end"])))

    if anchor_gender:
        _log(f"Person-anchoring ({anchor_gender}-only): lip-syncing {len(raw)} "
             f"segments, {skipped_gender} other-gender segments pass through")

    if not raw:
        return []

    # 2. Sort by start time
    raw.sort(key=lambda x: x[0])

    # 3. Apply padding and merge overlapping/nearby ranges
    merged: list[tuple[float, float]] = []
    for s, e in raw:
        s = max(0.0, s - padding_before)
        e = min(video_duration, e + padding_after)
        if merged and s <= merged[-1][1] + merge_gap:
            # merge into previous range
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # 4. Clamp to video bounds
    merged = [(max(0.0, s), min(video_duration, e)) for s, e in merged]
    merged = [(s, e) for s, e in merged if e - s > 0.1]  # skip tiny ranges

    # 5. Split oversized segments to limit per-call memory
    #    This is CRITICAL: merge_gap can create one giant segment covering
    #    the whole video for dialogue-heavy content.  Splitting caps the
    #    number of frames any single model-call holds in memory.
    chunked: list[tuple[float, float]] = []
    for s, e in merged:
        dur = e - s
        if dur <= max_duration:
            chunked.append((s, e))
        else:
            n_chunks = int(np.ceil(dur / max_duration))
            chunk_len = dur / n_chunks
            for i in range(n_chunks):
                cs = s + i * chunk_len
                ce = min(e, s + (i + 1) * chunk_len)
                if ce - cs > 0.1:
                    chunked.append((cs, ce))
            _log(f"  Split [{s:.1f}s–{e:.1f}s] ({dur:.1f}s) into "
                 f"{n_chunks} chunks of ~{chunk_len:.1f}s each")

    return chunked


def _cut_video_clip(
    video_path: Path,
    start: float,
    duration: float,
    output_path: Path,
    *,
    reencode: bool = False,
    fps: float | None = None,
) -> Path:
    """Cut a precise clip from *video_path*.

    When ``reencode=False`` (default) uses stream copy for speed.
    When ``reencode=True`` re-encodes for frame-accurate cutting.  ``-ss`` is
    placed AFTER ``-i`` so the seek is frame-accurate (a fast pre-input seek
    with ``-c copy`` snaps to the previous keyframe and re-includes already
    shown frames → duplicated/stuttering seams).  ``fps`` forces the output
    frame rate so every clip in the timeline shares one rate for clean concat.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if reencode:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-ss", str(start), "-t", str(duration),
            "-c:v", "libx264", "-crf", "18",
            "-pix_fmt", "yuv420p",
        ]
        if fps:
            cmd += ["-r", f"{fps}"]
        cmd += ["-an", str(output_path)]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-i", str(video_path),
            "-t", str(duration),
            "-c", "copy", "-an",
            str(output_path),
        ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to cut video clip [{start:.2f}s, {start + duration:.2f}s]:\n"
            f"{result.stderr[-500:]}"
        )
    return output_path


def _cut_audio_clip(
    audio_path: Path,
    start: float,
    duration: float,
    output_path: Path,
) -> Path:
    """Cut a precise audio clip as 16kHz mono WAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", str(audio_path),
        "-t", str(duration),
        "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to cut audio clip [{start:.2f}s, {start + duration:.2f}s]:\n"
            f"{result.stderr[-500:]}"
        )
    return output_path


def _concatenate_videos(
    clip_paths: list[Path],
    output_path: Path,
) -> Path:
    """Concatenate video clips using FFmpeg concat demuxer.

    Falls back to re-encoding if stream copy fails (e.g. codec mismatch).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filelist = output_path.parent / f"{output_path.stem}_concat.txt"
    with open(filelist, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p.absolute()}'\n")

    # Try stream copy first
    result = subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(filelist),
        "-c", "copy",
        str(output_path),
    ], capture_output=True, text=True)

    if result.returncode == 0:
        filelist.unlink(missing_ok=True)
        return output_path

    # Fallback: re-encode
    _log("Concat stream-copy failed; re-encoding instead.")
    result = subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(filelist),
        "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ], capture_output=True, text=True, check=True)

    filelist.unlink(missing_ok=True)
    return output_path


def musetalk_sync(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    *,
    bbox_shift: int = 0,
    fps: int = 25,
    use_float16: bool = True,
    batch_size: int = 4,
    extra_margin: int = 10,
    parsing_mode: str = "jaw",
    left_cheek_width: int = 100,
    right_cheek_width: int = 100,
    audio_padding_length_left: int = 2,
    audio_padding_length_right: int = 2,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Path | None:
    """Run MuseTalk lip-sync (256×256, high quality).

    Requires MuseTalk to be cloned to ``models/musetalk/`` with weights
    downloaded.  See ``ai_movie/config.py`` for download URLs.

    Parameters
    ----------
    video_path:
        Input video (any format OpenCV can read).
    audio_path:
        Driving audio (16 kHz mono WAV recommended).
    output_path:
        Where to write the output MP4.
    bbox_shift:
        Mouth openness adjustment (-9 to +9, negative = less open).
        NOTE: only used by MuseTalk **v1.0**; the v1.5 inference script
        hard-codes bbox_shift=0 and ignores this value.
    fps:
        Output frame rate (25 is MuseTalk's native rate).
    use_float16:
        Load models in half precision (halves GPU VRAM, minimal quality loss).
    batch_size:
        Inference batch size (default 4 for ~8GB VRAM; reduce to 2 if OOM).
    extra_margin:
        Extra pixels of chin included below the face box (V15).  Larger
        values let the regenerated region follow the jaw further down,
        reducing the visible seam under the lower lip.  Range ~10–20.
    parsing_mode:
        Face-parsing blend mode: ``"jaw"`` (V15, follows the jawline —
        softest, most natural boundary) or ``"raw"`` (rectangular).
    left_cheek_width, right_cheek_width:
        Width (px) of the cheek region folded into the blend mask.
        Wider = the seam is pushed out onto flat cheek skin instead of
        cutting across the mouth corners, which removes the "boxy" edge.
        MuseTalk default is 90; 100 blends a little softer.
    audio_padding_length_left, audio_padding_length_right:
        Whisper audio context frames on each side (temporal smoothing of
        mouth motion).  Default 2.
    progress_cb:
        Called as ``progress_cb(current_step, total_steps)``.
    cancel_check:
        Return ``True`` to abort.

    Returns
    -------
    ``output_path`` on success, ``None`` if cancelled.
    """
    import yaml

    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    # Locate MuseTalk
    script = _find_musetalk_inference_script()
    if script is None:
        raise FileNotFoundError(
            f"MuseTalk inference script not found under {_MUSETALK_DIR}.\n"
            f"Clone it: git clone https://github.com/TMElyralab/MuseTalk "
            f"{_MUSETALK_DIR}"
        )

    unet_path, unet_cfg = _find_musetalk_unet()
    if unet_path is None:
        raise FileNotFoundError(
            f"MuseTalk UNet checkpoint not found under {_MUSETALK_DIR}/models/.\n"
            f"Download weights — see ai_movie/config.py for URLs."
        )

    # ── Build temp YAML config ──────────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="musetalk_"))
    result_dir = tmp_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "task_0": {
            "video_path": str(video_path.absolute()),
            "audio_path": str(audio_path.absolute()),
            "bbox_shift": bbox_shift,
        }
    }
    config_yaml = tmp_dir / "inference.yaml"
    with open(config_yaml, "w") as f:
        yaml.dump(config, f)

    # ── Build CLI command ────────────────────────────────────────
    python_exe = sys.executable
    cmd = [
        python_exe, "-m", "scripts.inference",
        "--inference_config", str(config_yaml),
        "--unet_model_path", str(unet_path),
        "--unet_config", str(unet_cfg),
        "--version", "v15" if "V15" in str(unet_path).upper() else "v1",
        "--fps", str(fps),
        "--batch_size", str(batch_size),
        "--extra_margin", str(extra_margin),
        "--parsing_mode", parsing_mode,
        "--left_cheek_width", str(left_cheek_width),
        "--right_cheek_width", str(right_cheek_width),
        "--audio_padding_length_left", str(audio_padding_length_left),
        "--audio_padding_length_right", str(audio_padding_length_right),
        "--result_dir", str(result_dir),
    ]
    if use_float16:
        cmd.append("--use_float16")

    museTalk_dir = str(_MUSETALK_DIR)
    env = {
        **__import__("os").environ,
        "PYTHONPATH": museTalk_dir,
    }

    _log(f"MuseTalk: float16={use_float16}, batch={batch_size}, bbox_shift={bbox_shift}, "
         f"extra_margin={extra_margin}, parsing={parsing_mode}, "
         f"cheek={left_cheek_width}/{right_cheek_width}")
    _log(f"MuseTalk cmd: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        cwd=museTalk_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # ── Monitor progress ─────────────────────────────────────────
    total_steps = 100
    current_step = 0
    output_lines: list[str] = []
    try:
        for line in proc.stdout:
            if cancel_check and cancel_check():
                proc.terminate()
                proc.wait(timeout=10)
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
                return None

            line_stripped = line.strip()
            if line_stripped:
                _log(f"[MuseTalk] {line_stripped}")
                output_lines.append(line_stripped)

            # Parse tqdm-style progress from stderr (merged into stdout)
            if "%" in line and "it/s" in line:
                try:
                    pct_str = line.split("%")[0].strip().split()[-1]
                    current_step = int(float(pct_str))
                    if progress_cb:
                        progress_cb(current_step, total_steps)
                except (ValueError, IndexError):
                    pass

        proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        raise RuntimeError("MuseTalk timed out after 10 minutes")

    if proc.returncode != 0:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        raise RuntimeError(f"MuseTalk exited with code {proc.returncode}")

    # ── Check for errors caught by MuseTalk's bare except ─────────
    muse_errors = [l for l in output_lines if "Error occurred during processing" in l]
    if muse_errors:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        raise RuntimeError(
            f"MuseTalk inference failed:\n" + "\n".join(muse_errors)
        )

    if progress_cb:
        progress_cb(total_steps, total_steps)

    # ── Locate output and copy to target ─────────────────────────
    result_files = list(result_dir.glob("**/*.mp4"))
    if not result_files:
        result_files = list(result_dir.glob("**/*.avi"))
    if not result_files:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        raise RuntimeError(f"MuseTalk produced no output in {result_dir}")

    shutil.copy2(str(result_files[0]), str(output_path))
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    _log(f"MuseTalk sync DONE: {output_path}")
    return output_path


def segment_based_lip_sync(
    video_path: str | Path,
    tts_results: list[dict],
    output_path: str | Path,
    *,
    backend: str = "auto",
    padding_before: float = 0.5,
    padding_after: float = 0.3,
    merge_gap: float = 1.0,
    max_segment_duration: float = 12.0,
    resize_factor: int | None = None,
    face_restore: bool = False,
    anchor_gender: str | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    detail_progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Path | None:
    """Run segment-based lip-sync — only on video portions that have speech.

    Silent portions pass through unchanged (stream copy).  Segments where
    face detection fails fall back to the original clip automatically.

    **Memory management**: Segments are capped to ``max_segment_duration``
    seconds each and the video is downscaled by ``resize_factor`` before
    lip-sync processing.  GPU cache is purged between segments.

    Parameters
    ----------
    video_path:
        Original input video.
    tts_results:
        TTS segment list from ``人声生成`` step data (each dict has
        ``start``, ``end``, ``audio`` keys).
    output_path:
        Where to write the output MP4.
    backend:
        ``"auto"`` (prefer MuseTalk if available), ``"musetalk"``, or
        ``"wav2lip"``.
    padding_before, padding_after:
        Extra seconds around each speech range for smooth transitions.
    merge_gap:
        Merge speech ranges whose gap ≤ this (reduces subprocess launches).
    max_segment_duration:
        Hard cap on segment length (seconds).  Segments longer than this
        are split into smaller chunks to limit peak frame memory.
        Default 12 s keeps ~750 frames at 1080p → ~4.5 GB peak.
    resize_factor:
        Divide video resolution by this factor before lip-sync (e.g. 2 →
        540p from 1080p).  Reduces frame memory by 4×.  Face crops are
        small (96² for Wav2Lip, 256² for MuseTalk) so this has minimal
        quality impact.
    progress_cb:
        ``progress_cb(processed_segments, total_segments)``.
    detail_progress_cb:
        Forwarded to the inner engine (wav2lip_sync / musetalk_sync) as
        its ``progress_cb``.  Reports per-batch (Wav2Lip) or per-percent
        (MuseTalk) progress within the current segment.
    cancel_check:
        Return ``True`` to abort.

    Returns
    -------
    ``output_path`` on success, ``None`` if cancelled.
    """
    from ai_movie.composer import build_speech_track

    video_path = Path(video_path)
    output_path = Path(output_path)

    # ── Resolve backend ──────────────────────────────────────────
    if backend == "auto":
        backend = "musetalk" if musetalk_available() else "wav2lip"

    # ── Resolve resize_factor (backend-aware) ────────────────────
    # MuseTalk crops & re-renders only the face at 256², so downscaling
    # the whole frame beforehand just throws away final-image resolution
    # and makes the pasted mouth look soft/mosaic'd — keep it at 1×.
    # Wav2Lip's 96² crops tolerate a 2× downscale for a big memory win.
    if resize_factor is None:
        resize_factor = 1 if backend == "musetalk" else 2
    # ── Resolve CodeFormer face-restore availability ─────────────
    # Sharpens the MuseTalk/Wav2Lip-regenerated lower face on close-up
    # shots.  Applied per-segment (only frames the lip-sync engine
    # touched) so silent/gap clips are never altered.
    if face_restore:
        from ai_movie import face_restore as _face_restore_mod
        if not _face_restore_mod.codeformer_available():
            _log("face_restore requested but CodeFormer arch/weights missing — skipping restore pass")
            face_restore = False

    _log(f"Segment-based lip-sync: backend={backend}, max_dur={max_segment_duration}s, "
         f"resize={resize_factor}x, face_restore={face_restore}, "
         f"padding=[-{padding_before}s, +{padding_after}s], merge_gap={merge_gap}s")

    # ── Build full speech track ──────────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="seg_lipsync_"))
    speech_track = tmp_dir / "full_speech_track.wav"
    build_speech_track(tts_results, speech_track)

    # ── Get video duration ───────────────────────────────────────
    dur_result = subprocess.run([
        "ffprobe", "-v", "quiet", "-show_entries",
        "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ], capture_output=True, text=True, check=True)
    video_duration = float(dur_result.stdout.strip())

    # ── Extract speech time ranges ───────────────────────────────
    speech_ranges = _extract_speech_ranges(
        tts_results, video_duration,
        padding_before=padding_before,
        padding_after=padding_after,
        merge_gap=merge_gap,
        max_duration=max_segment_duration,
        anchor_gender=anchor_gender,
    )

    if not speech_ranges:
        _log("No speech segments found — returning original video.")
        shutil.copy2(str(video_path), str(output_path))
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        return output_path

    total_segments = len(speech_ranges)
    _log(f"Speech ranges to process: {total_segments}")

    # ── Downscale whole video once (if resize_factor > 1) ────────
    working_video = video_path
    if resize_factor > 1:
        scaled_video = tmp_dir / "video_scaled.mp4"
        _log(f"Downscaling video {resize_factor}× for lip-sync…")
        _downscale_video(video_path, scaled_video, resize_factor)
        working_video = scaled_video

    # Uniform frame rate for the whole timeline: MuseTalk otherwise forces 25 fps
    # (re-timing speech clips → drift/stutter vs the gap clips).  Cut gaps and
    # drive MuseTalk at the source rate so every clip shares one fps.
    target_fps = _probe_fps(working_video)
    ms_fps = int(round(target_fps))

    # ── Process each speech range ────────────────────────────────
    processed_map: dict[tuple[float, float], Path] = {}
    # (start, end) → path to lip-synced or original clip

    for idx, (seg_start, seg_end) in enumerate(speech_ranges):
        if cancel_check and cancel_check():
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
            return None

        duration = seg_end - seg_start
        clip_name = f"speech_{idx:04d}"
        orig_clip = tmp_dir / f"{clip_name}_orig.mp4"
        audio_clip = tmp_dir / f"{clip_name}_audio.wav"
        lipsync_clip = tmp_dir / f"{clip_name}_lipsync.mp4"

        _log(f"[{idx+1}/{total_segments}] Segment [{seg_start:.1f}s–{seg_end:.1f}s] "
             f"({duration:.1f}s, ~{int(duration*25)} frames)")

        # Cut video clip from the (possibly downscaled) working video
        _cut_video_clip(working_video, seg_start, duration, orig_clip, reencode=True)

        # Cut audio clip
        _cut_audio_clip(speech_track, seg_start, duration, audio_clip)

        # Run lip-sync
        try:
            if backend == "musetalk":
                musetalk_sync(
                    orig_clip, audio_clip, lipsync_clip,
                    fps=ms_fps,
                    use_float16=True,
                    batch_size=4,
                    progress_cb=detail_progress_cb,
                    cancel_check=cancel_check,
                )
            else:
                wav2lip_sync(
                    orig_clip, audio_clip, lipsync_clip,
                    resize_factor=1,  # already downscaled above
                    progress_cb=detail_progress_cb,
                    cancel_check=cancel_check,
                )
            final_clip = lipsync_clip
            # ── Optional CodeFormer restore of the regenerated face ──
            if face_restore:
                restored_clip = tmp_dir / f"{clip_name}_restored.mp4"
                try:
                    _face_restore_mod.restore_video(
                        lipsync_clip, restored_clip,
                        progress_cb=detail_progress_cb,
                        cancel_check=cancel_check,
                        log_cb=_log,
                    )
                    if restored_clip.exists():
                        final_clip = restored_clip
                        _log(f"[{idx+1}/{total_segments}] CodeFormer restore OK")
                    else:  # cancelled mid-restore
                        if cancel_check and cancel_check():
                            shutil.rmtree(str(tmp_dir), ignore_errors=True)
                            return None
                except Exception as rexc:
                    _log(f"[{idx+1}/{total_segments}] CodeFormer restore FAILED: {rexc} "
                         f"— using un-restored lip-sync clip")
            processed_map[(seg_start, seg_end)] = final_clip
            _log(f"[{idx+1}/{total_segments}] Lip-sync OK")
        except Exception as exc:
            _log(f"[{idx+1}/{total_segments}] Lip-sync FAILED: {exc}")
            _log(f"[{idx+1}/{total_segments}] Using original clip as fallback")
            processed_map[(seg_start, seg_end)] = orig_clip

        # ── Aggressive memory cleanup between segments ───────────
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _log(f"  GPU mem after cleanup: "
                 f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB, "
                 f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")

        if progress_cb:
            progress_cb(idx + 1, total_segments)

    # ── Build complete clip timeline ─────────────────────────────
    # Gaps are cut from working_video (SAME resolution as the speech clips) and
    # re-encoded frame-accurately (stream-copy would keyframe-snap and duplicate
    # frames at the seams → the reported stutter/repeat).  MuseTalk clips are
    # forced to ms_fps, so gaps match that; Wav2Lip preserves the source fps, so
    # gaps inherit working_video's fps (fps=None).
    gap_fps = ms_fps if backend == "musetalk" else None
    all_clips: list[Path] = []
    cursor = 0.0
    min_gap = 0.05  # 50ms minimum to avoid zero-duration clips

    for seg_start, seg_end in speech_ranges:
        # Gap before this speech segment
        if seg_start - cursor > min_gap:
            gap_clip = tmp_dir / f"gap_{cursor:.3f}_{seg_start:.3f}.mp4"
            _cut_video_clip(working_video, cursor, seg_start - cursor, gap_clip,
                            reencode=True, fps=gap_fps)
            all_clips.append(gap_clip)

        # Speech clip (lip-synced or original fallback)
        all_clips.append(processed_map[(seg_start, seg_end)])
        cursor = seg_end

    # Trailing gap
    if video_duration - cursor > min_gap:
        gap_clip = tmp_dir / f"gap_{cursor:.3f}_{video_duration:.3f}.mp4"
        _cut_video_clip(working_video, cursor, video_duration - cursor, gap_clip,
                        reencode=True, fps=gap_fps)
        all_clips.append(gap_clip)

    # ── Concatenate all clips ────────────────────────────────────
    _log(f"Concatenating {len(all_clips)} clips")
    _concatenate_videos(all_clips, output_path)

    # ── Clean up ─────────────────────────────────────────────────
    shutil.rmtree(str(tmp_dir), ignore_errors=True)

    _log(f"Segment-based lip-sync DONE: {output_path}")
    return output_path


def _probe_fps(video_path: Path) -> float:
    """Return the average frame rate of *video_path* (fallback 25.0)."""
    try:
        out = subprocess.run([
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate",
            "-of", "default=nw=1:nk=1", str(video_path),
        ], capture_output=True, text=True).stdout.strip()
        num, _, den = out.partition("/")
        fps = float(num) / float(den) if den and float(den) else float(num)
        return fps if fps and fps > 0 else 25.0
    except Exception:
        return 25.0


def _downscale_video(
    src: Path,
    dst: Path,
    factor: int = 2,
) -> Path:
    """Downscale a video by *factor* using bicubic scaling via FFmpeg."""
    probe = subprocess.run([
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", str(src),
    ], capture_output=True, text=True)
    try:
        w, h = map(int, probe.stdout.strip().split(","))
    except (ValueError, AttributeError):
        w, h = 1920, 1080

    new_w, new_h = w // factor, h // factor
    # Ensure even dimensions for YUV 4:2:0
    new_w += new_w % 2
    new_h += new_h % 2

    _log(f"  Scaling: {w}×{h} → {new_w}×{new_h}")
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(src),
        "-vf", f"scale={new_w}:{new_h}:flags=bicubic",
        "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        str(dst),
    ], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to downscale video:\n{result.stderr[-500:]}")
    return dst
