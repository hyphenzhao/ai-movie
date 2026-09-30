"""run_vc_version._merge_attempts: per-speaker choice, failed attempts never win (no GPU).

    .venv/bin/python tests/test_vc_attempts.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("run_vc_version", ROOT / "scripts" / "run_vc_version.py")
rv = importlib.util.module_from_spec(spec); spec.loader.exec_module(rv)     # type: ignore[union-attr]


def attempt(k, per_line, ref):
    """per_line: {i: (vc, ok|None)} → (k, items, stats, ref_set)."""
    items = {i: {"audio": f"a{k}_{i}.wav", "vc": vc} for i, (vc, _) in per_line.items()}
    verdicts = {i: {"ok": ok, "judged": True, "reason": "" if ok else "lost_x"} for i, (vc, ok) in per_line.items() if vc and ok is not None}
    return (k, items, {"verdicts": verdicts, "checked": len(verdicts), "rejected": sum(1 for v in verdicts.values() if not v["ok"])},
            {"A": {"ref_audio": f"refA{ref}.wav"}, "B": {"ref_audio": f"refB{ref}.wav"}})


def test_per_speaker_choice_and_failed_attempt_loses():
    segs = [{"speaker": "A"}] * 4 + [{"speaker": "B"}] * 4
    a0 = attempt(0, {0: (True, True), 1: (True, True), 2: (True, True), 3: (True, False),      # A: 1/4 rejected
                     4: (True, False), 5: (True, False), 6: (True, True), 7: (True, False)}, "0")   # B: 3/4 rejected
    a1 = attempt(1, {0: (True, False), 1: (True, False), 2: (True, True), 3: (True, True),      # A: 2/4 (worse)
                     4: (True, True), 5: (True, True), 6: (True, True), 7: (True, False)}, "1")     # B: 1/4 (better)
    a2 = attempt(2, {i: (False, None) for i in range(8)}, "2")                                   # worker died: nothing judged
    items, refs, stats, chosen = rv._merge_attempts([a0, a1, a2], segs)
    assert chosen == {"A": 0, "B": 1}
    assert refs["A"]["ref_audio"] == "refA0.wav" and refs["B"]["ref_audio"] == "refB1.wav"
    assert items[0]["audio"] == "a0_0.wav" and items[4]["audio"] == "a1_4.wav"
    assert stats["checked"] == 8 and stats["rejected"] == 2
    # only the dead attempt available → it is still chosen (there is nothing else), with checked 0
    items, refs, stats, chosen = rv._merge_attempts([a2], segs)
    assert chosen == {"A": 2, "B": 2} and stats["checked"] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
