"""scripts/replay_content.py: the persisted 01_content.csv rows replay through classify exactly (no GPU).

    .venv/bin/python tests/test_content_replay.py
"""
from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("replay_content", ROOT / "scripts" / "replay_content.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)     # type: ignore[union-attr]

FIELDS = ["start", "end", "decision", "reason", "pass", "asr_conf", "no_speech_prob", "avg_logprob",
          "compression_ratio", "speaker", "text", "alt_text"]


def _csv(rows: list[dict]) -> Path:
    d = Path(tempfile.mkdtemp(prefix="replay_"))
    p = d / "01_content.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:          # BOM, like artifacts.export_csv
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    return p


def test_rows_round_trip_and_flip():
    rows = [
        # a moan the old rules dropped by window ratio: now nonlexical
        {"start": 10.0, "end": 12.0, "decision": "drop", "reason": "compression 23.89", "pass": "vad",
         "asr_conf": 0.7, "no_speech_prob": 0.3, "avg_logprob": -0.5, "compression_ratio": 23.89, "text": "アーッ、アーッ、アーッ、アーッ"},
        # a real line: unchanged
        {"start": 20.0, "end": 21.0, "decision": "speech", "reason": "content", "pass": "vad",
         "asr_conf": 0.9, "compression_ratio": 0.9, "text": "気持ちいい"},
        # energy drop: outside classify, held fixed
        {"start": 30.0, "end": 31.0, "decision": "drop", "reason": "energy floor or repeated line", "pass": "vad",
         "asr_conf": 0.9, "compression_ratio": 0.9, "text": "気持ちいい"},
        # sweep row whose second decode heard nothing: the CSV holds '' for None and '' alike, and only
        # the delivered reason says which it was — restored, nsp ≥ 0.5 drops it again
        {"start": 40.0, "end": 42.0, "decision": "drop", "reason": "only one decode heard text, nsp ≥ 0.5", "pass": "sweep",
         "asr_conf": 0.5, "no_speech_prob": 0.8, "avg_logprob": -0.9, "compression_ratio": 1.0, "text": "奥まで入っちゃうよ", "alt_text": ""},
        # …and a sweep row with no second decode at all (SONE-846 p14 56.9 s: nsp 0.8, delivered speech)
        {"start": 50.0, "end": 52.0, "decision": "speech", "reason": "content", "pass": "sweep",
         "asr_conf": 0.5, "no_speech_prob": 0.8, "avg_logprob": -0.9, "compression_ratio": 1.0, "text": "奥まで入っちゃうよ", "alt_text": ""},
    ]
    loaded = rc.load_rows(_csv(rows))
    assert [r["start"] for r in loaded] == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert loaded[0]["compression_ratio"] == 23.89 and loaded[1]["no_speech_prob"] is None
    assert loaded[3]["alt_text"] is None and loaded[4]["alt_text"] is None      # '' → None on load
    out = rc.replay_rows(loaded)
    assert [(r["old"], r["new"]) for r in out] == [("drop", "nonlexical"), ("speech", "speech"), ("drop", "drop"),
                                                   ("drop", "drop"), ("speech", "speech")]
    assert out[2]["new_reason"] == "energy floor or repeated line"
    assert out[3]["new_reason"].startswith("only one decode")


def test_loop_rule_replays_offline():
    base = {"decision": "drop", "reason": "energy floor or repeated line", "pass": "vad", "asr_conf": 0.9, "compression_ratio": 1.0}
    rows = [dict(base, start=10.0 + 5 * i, end=11.0 + 5 * i, text="ご飯食べました") for i in range(3)]
    out = rc.replay_rows(rc.load_rows(_csv(rows)))
    assert all(r["new"] == "drop" and r["new_reason"] == "repeated line (loop)" for r in out)
    # a looped *vocalisation* is exempt from the loop rule (classify_segments semantics)
    rows = [dict(base, decision="nonlexical", reason="vocalisation ×4", start=10.0 + 5 * i, end=11.0 + 5 * i, text="はぁはぁはぁはぁ") for i in range(3)]
    out = rc.replay_rows(rc.load_rows(_csv(rows)))
    assert all(r["new"] == "nonlexical" for r in out)


def test_replay_on_persisted_states():
    """The films on disk, when present: the short films must not flip at all."""
    if not (rc.WORKSPACE / "test_2" / "deliverables" / "01_content.csv").exists():
        print("  (skipped: no workspace)")
        return
    for film in ("output_test", "test_1", "test_2"):
        if (rc.WORKSPACE / film / "deliverables" / "01_content.csv").exists():
            res = rc.replay_film(film)
            assert res["rows"] and not res["flips"], (film, res["flips"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
