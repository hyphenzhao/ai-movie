#!/usr/bin/env python
"""Overlapped-speech detection worker (runs inside vendor/osd_venv).

Reads a JSON job on stdin ``{"audio": path, "model": name, "device": "cpu"}``
and prints ONE JSON line on stdout: ``{"regions": [[s, e], ...], "model": …,
"method": …}`` or ``{"error": …}``.  The HF token comes from ``HF_TOKEN``.

Strategy: pyannote's ``OverlappedSpeechDetection`` pipeline when it accepts
the model; otherwise sliding-window ``Inference`` on the segmentation
model, converted to multilabel, with "≥ 2 active speakers" binarised into
regions (hysteresis + minimum durations).
"""

from __future__ import annotations

import json
import os
import sys


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _regions_from_mask(mask, frame_times, min_on=0.10, min_off=0.10):
    """Boolean per-frame mask → merged [start, end] regions (seconds)."""
    regions = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        s, e = float(frame_times[i]), float(frame_times[min(j, n - 1)])
        if regions and s - regions[-1][1] < min_off:
            regions[-1][1] = e
        else:
            regions.append([s, e])
        i = j
    return [[s, e] for s, e in regions if e - s >= min_on]


def main() -> int:
    job = json.loads(sys.stdin.read() or "{}")
    audio = job.get("audio")
    model_name = job.get("model", "pyannote/segmentation-3.0")
    device = job.get("device", "cpu")
    token = os.environ.get("HF_TOKEN")
    if not audio or not os.path.exists(audio):
        print(json.dumps({"error": f"audio missing: {audio}"}))
        return 1
    try:
        import torch
        torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
        from pyannote.audio import Model
    except Exception as exc:                            # noqa: BLE001
        print(json.dumps({"error": f"import failed: {type(exc).__name__}: {exc}"}))
        return 1

    model = None
    err = ""
    for kw in ({"token": token}, {"use_auth_token": token}):
        try:
            model = Model.from_pretrained(model_name, **kw)
            break
        except TypeError as exc:
            err = str(exc)
            continue
        except Exception as exc:                        # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            break
    if model is None:
        print(json.dumps({"error": f"model load failed: {err}"}))
        return 1
    model.to(torch.device(device))
    model.eval()

    # ── 1. official pipeline ──
    try:
        from pyannote.audio.pipelines import OverlappedSpeechDetection
        pipe = OverlappedSpeechDetection(segmentation=model)
        params = {"min_duration_on": 0.10, "min_duration_off": 0.10}
        try:
            pipe.instantiate({"onset": 0.5, "offset": 0.5, **params})
        except Exception:                               # noqa: BLE001
            pipe.instantiate(params)
        ann = pipe({"audio": audio})
        regions = [[float(seg.start), float(seg.end)]
                   for seg, _, _ in ann.itertracks(yield_label=True)]
        print(json.dumps({"regions": regions, "model": model_name,
                          "method": "OverlappedSpeechDetection"}))
        return 0
    except Exception as exc:                            # noqa: BLE001
        _log(f"[osd_worker] pipeline path failed ({type(exc).__name__}: {exc}); "
             f"falling back to raw inference")

    # ── 2. manual: sliding-window inference → multilabel → ≥2 speakers ──
    try:
        import numpy as np
        from pyannote.audio import Inference
        inf = Inference(model, duration=getattr(model.specifications, "duration", 10.0),
                        step=1.0, device=torch.device(device))
        out = inf({"audio": audio})
        data = np.asarray(out.data)
        # Powerset models come back as class probabilities unless Inference
        # converted them; detect by checking the specification.
        spec = model.specifications
        if getattr(spec, "powerset", False) and data.shape[-1] == len(spec.classes):
            # argmax over powerset classes → multilabel
            idx = data.argmax(-1)
            ml = np.zeros(data.shape[:-1] + (spec.num_powerset_classes if hasattr(spec, "num_powerset_classes") else 3,))
            mapping = model.powerset.mapping.cpu().numpy()        # (P, K)
            ml = mapping[idx]
            active = ml.sum(-1)
        else:
            active = (data > 0.5).sum(-1)
        mask = active >= 2
        sw = out.sliding_window
        times = [sw[i].start for i in range(len(mask))]
        regions = _regions_from_mask(mask, times)
        print(json.dumps({"regions": regions, "model": model_name,
                          "method": "inference_multilabel"}))
        return 0
    except Exception as exc:                            # noqa: BLE001
        print(json.dumps({"error": f"inference failed: {type(exc).__name__}: {exc}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
