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
