"""Speaker diarization worker — runs inside ``vendor/osd_venv`` (pyannote 4).

Called by :mod:`ai_movie.diarize` as a subprocess, the same way
``osd_worker.py`` is, because pyannote.audio 4 cannot share a process with
the pipeline's transformers/torch stack.

    osd_venv/bin/python -m ai_movie.diar_worker <audio.wav> <out.json>
        [--num-speakers N] [--min-speakers N] [--max-speakers N]

Why the pipeline is *assembled* here instead of loaded by name: the two
official pipelines (``speaker-diarization-3.1`` and ``…-community-1``) are
gated and this account is not on their access list (403 on every file).
Their components are not gated — ``segmentation-3.0`` and the WeSpeaker
ResNet34 embedding — and ``SpeakerDiarization`` accepts them directly.  The
clustering hyper-parameters below are the published ones from 3.1's
``config.yaml``.  pyannote 4 also insists on loading a PLDA model (only used
by its VBx clustering) from the gated repo, so that loader is stubbed out;
agglomerative clustering never touches it.

The embedding weights live in ``models/wespeaker-voxceleb-resnet34-LM``
(config.yaml + pytorch_model.bin, fetched once via the VPS relay when the
direct HF download crawled at 0.4 MB/s).

Output: ``{"turns": [{"start", "end", "speaker"}], "speakers": [...],
"seconds": elapsed}``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_DIR = ROOT / "models" / "wespeaker-voxceleb-resnet34-LM"
SEGMENTATION = "pyannote/segmentation-3.0"

# pyannote/speaker-diarization-3.1/config.yaml, verbatim
PARAMS = {
    "segmentation": {"min_duration_off": 0.0},
    "clustering": {"method": "centroid", "min_cluster_size": 12,
                   "threshold": 0.7045654963945799},
}


def build_pipeline(token: str | None):
    import pyannote.audio.pipelines.speaker_diarization as sd
    sd.get_plda = lambda *a, **k: None          # PLDA feeds VBx only; we use AHC
    from pyannote.audio import Model
    from pyannote.audio.pipelines import SpeakerDiarization

    seg = Model.from_pretrained(SEGMENTATION, token=token)
    emb = Model.from_pretrained(str(EMBEDDING_DIR))
    pipe = SpeakerDiarization(segmentation=seg, embedding=emb,
                              clustering="AgglomerativeClustering",
                              embedding_exclude_overlap=True)
    pipe.instantiate(PARAMS)
    return pipe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:                               # noqa: BLE001
            token = None

    t0 = time.time()
    try:
        pipe = build_pipeline(token)
        kw = {k: v for k, v in (("num_speakers", args.num_speakers),
                                ("min_speakers", args.min_speakers),
                                ("max_speakers", args.max_speakers)) if v}
        out = pipe(args.audio, **kw)
        ann = getattr(out, "speaker_diarization", out)
        turns = [{"start": round(s.start, 3), "end": round(s.end, 3), "speaker": lab}
                 for s, _, lab in ann.itertracks(yield_label=True)]
        doc = {"turns": turns, "speakers": sorted({t["speaker"] for t in turns}),
               "seconds": round(time.time() - t0, 1),
               "backend": "pyannote-assembled-3.1"}
    except Exception as exc:                            # noqa: BLE001
        doc = {"turns": [], "speakers": [], "error": f"{type(exc).__name__}: {exc}",
               "seconds": round(time.time() - t0, 1)}
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in doc.items() if k != "turns"}, ensure_ascii=False))
    return 0 if doc["turns"] else 1


if __name__ == "__main__":
    sys.exit(main())
