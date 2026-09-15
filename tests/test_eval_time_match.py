"""Time-keyed subtitle ground truth reproduces the index-keyed baseline.

    .venv/bin/python tests/test_eval_time_match.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_against_subs as E                   # noqa: E402

ARCH = ROOT / "workspace" / "_archive_v3.0.0" / "output_test"
GT = ROOT / "inputs" / "subs" / "output_test.gt.json"


def _have() -> bool:
    if not (ARCH / "state.json").exists() or not GT.exists():
        print("  (skipped: needs the v3.0.0 archive and inputs/subs ground truth)")
        return False
    return True


def test_index_to_cues_merges_runs():
    segs = [{"start": 0, "end": 1}, {"start": 1, "end": 2}, {"start": 3, "end": 4},
            {"start": 5, "end": 6}]
    cues = E.index_gt_to_cues({"0": "あ", "1": "あ", "2": "", "3": "い"}, segs)
    assert [(c["ja"], c["start"], c["end"]) for c in cues] == [("あ", 0, 2), ("い", 5, 6)]


def test_cue_for_prefers_containment():
    cues = [{"id": 0, "start": 0.0, "end": 2.0, "ja": "a"},
            {"id": 1, "start": 2.1, "end": 6.0, "ja": "b"}]
    assert E.cue_for({"start": 1.5, "end": 2.4}, cues) == 0      # midpoint 1.95
    assert E.cue_for({"start": 7.0, "end": 8.0}, cues) is None


def test_reproduces_v300_baseline():
    if not _have():
        return
    state = json.loads((ARCH / "state.json").read_text(encoding="utf-8"))
    segs = E.final_segments(state)
    groups = E.build_groups(segs, E.load_gt(GT))
    s = E.summarise(groups, segs)
    assert s["cues"] == 87, s["cues"]
    assert abs(s["asr_median"] - 0.91) < 0.005, s["asr_median"]
    assert abs(s["asr_mean"] - 0.84) < 0.01, s["asr_mean"]
    assert s["asr_bad"] == 14, s["asr_bad"]


def test_resegmented_run_still_matches():
    """Merging neighbouring segments (a different segmentation) must keep
    every cue matched — the reason ground truth is keyed by time."""
    if not _have():
        return
    state = json.loads((ARCH / "state.json").read_text(encoding="utf-8"))
    segs = E.final_segments(state)
    merged = []
    for k in range(0, len(segs), 2):
        pair = segs[k:k + 2]
        merged.append({"start": pair[0]["start"], "end": pair[-1]["end"],
                       "text": "".join(p["text"] for p in pair),
                       "text_translated": "".join(p.get("text_translated", "") for p in pair)})
    s = E.summarise(E.build_groups(merged, E.load_gt(GT)), merged)
    assert s["cues"] >= 80, s["cues"]
    assert s["asr_median"] >= 0.75, s["asr_median"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
