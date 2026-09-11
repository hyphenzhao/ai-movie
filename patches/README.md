# MuseTalk patches

`models/musetalk/` is a downloaded vendor repo (not tracked by this project's git),
so local modifications are preserved here as patches and re-applied after a fresh
MuseTalk checkout.

## musetalk_rotation_align.patch — rotation-aware lip-sync (head-tilt fix)

MuseTalk crops the face with an **axis-aligned** box and never rotates, so on
**tilted heads** the mouth is out-of-distribution for the UNet → blurry / mosaic
mouth. This patch:
- `musetalk/utils/preprocessing.py::get_landmark_and_bbox` also returns a per-frame
  **head roll** (deg) from the eye landmarks (68-pt: R eye 36–41, L eye 42–47).
- `scripts/inference.py` rotates the crop **upright** before generation (when
  `|roll| >= ROLL_MIN = 8°`) and rotates the generated mouth **back** before the
  axis-aligned paste, so it aligns with the tilted face. Upright frames are
  unchanged (no regression).
- The other `get_landmark_and_bbox` callers are updated to unpack the extra return.

Apply from repo root:
```
cd models/musetalk && git apply ../../patches/musetalk_rotation_align.patch
```

## musetalk_target_face.patch — drive a *specific* face (person anchoring)

Upstream MuseTalk never chooses which face to animate: it takes DWPose's
`keypoints[0]` and S3FD's `d[0]`, i.e. whichever face happened to score
highest in that individual frame. In a two-person shot the mouth being driven
can therefore change from frame to frame. This patch lets the caller name the
target face per frame:

- `musetalk/utils/preprocessing.py::get_landmark_and_bbox(img_list,
  upperbondrange=0, target_boxes=None)`. With `target_boxes`:
  - **`inference_topdown` is now given an explicit person box.** Upstream calls
    it with no `bboxes=`, so DWPose treats the whole frame as one person and
    `keypoints[0]` is arbitrary when several faces are present. This is the
    core correction.
  - S3FD detections are taken from `detect_from_batch` (all faces) and the one
    with the highest IoU against the target is used; below IoU 0.25 the target
    box itself is used, so a missed detection never becomes a placeholder.
  - A frame whose `target_boxes` entry is `None` is skipped entirely (no
    DWPose, no S3FD — also a large speed-up) and emitted as `coord_placeholder`.
  - Without `target_boxes` the original behaviour is bit-for-bit unchanged.
- `scripts/inference.py`:
  - new `--target_bbox_json` (`{"frames": {frame_index: [x1,y1,x2,y2]}}`,
    indices local to the clip; missing frames pass through unmodified);
  - the coordinate pickle cache is bypassed in target-face mode — a cached
    coord list has no notion of *which* face it was for;
  - **fixes a pre-existing upstream bug**: the latent-building loop used
    `continue` on `coord_placeholder` frames while the paste loop indexes
    `coord_list_cycle[i % len(...)]`, so a single face-less frame shifted every
    subsequent mouth onto the wrong frame. Placeholder frames now contribute a
    stand-in latent to keep the lists index-aligned, and the paste loop writes
    the original frame for them.

Written by `ai_movie/faces.py` (`build_face_plan`) and passed through by
`ai_movie/lip_sync.py` (`musetalk_sync(..., target_bbox_json=...)`).

**Apply after the rotation patch** — both touch `get_landmark_and_bbox` and the
`inference.py` latent loop:
```
cd models/musetalk
git apply ../../patches/musetalk_rotation_align.patch
git apply ../../patches/musetalk_target_face.patch
```
A pristine copy of the three modified files is kept in `models/musetalk/.orig/`
for regenerating the patch with `diff -u`.

## musetalk_quality.patch — sharper, steadier, and 1:1 with the input

Applies on top of the two patches above (touches `scripts/inference.py` and
`musetalk/utils/audio_processor.py`):

- **Fractional fps / pinned frame count.** `AudioProcessor.get_whisper_chunk`
  did `fps = int(fps)`, so an NTSC 29.97 clip was driven at 29 fps: ~3.3% too
  few generated frames (the "MuseTalk runs a few % short" that
  `_fit_clip_to_duration` re-timed) *and* an audio→frame mapping that drifts one
  frame every 30 — measured on test_1 as the mouth lagging the voice by 7 frames
  (230 ms) 200 frames into a clip. `fps` is now a float and `num_frames` is
  pinned to the clip's real frame count, so the output is exactly 1:1 with the
  input video and the re-timing step becomes a no-op.
- **Crop-box smoothing** (`--box_smooth`, default 5 frames): DWPose's landmark
  box jitters a few px per frame and the regenerated lower face inherits that
  jitter. Boxes are averaged over a centred window, only inside runs of
  consecutive real boxes with IoU > 0.5 (never across a cut, a placeholder frame
  or a different face). Per task via `box_smooth:` in the inference yaml.
- **Cubic paste** (`--paste_interp`, default `cubic`): the 256² render was
  scaled up to big faces with bilinear; cubic for upscaling, area for downscaling.
- **Optional unsharp mask** (`--sharpen`, default 0 = off) on the generated face
  before paste; a knob for experiments, CodeFormer (人脸增强) is the real fix.

Apply after the other two:
```
cd models/musetalk && git apply ../../patches/musetalk_quality.patch
```
`models/musetalk/.orig/audio_processor.py` is the pristine copy.
