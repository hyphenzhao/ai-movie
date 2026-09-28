"""Run records: manifest, events, index, archive, error capture (no GPU).

    .venv/bin/python tests/test_runlog.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_movie.runlog import RunLog, append_event, snapshot_host      # noqa: E402


def test_record_of_a_failing_run():
    work = Path(tempfile.mkdtemp(prefix="runlog_"))
    (work / "state.json").write_text('{"asr": {"segments": []}}')
    (work / "deliverables").mkdir()
    (work / "deliverables" / "01_asr.ja.srt").write_text("1\n")
    (work / "deliverables" / "big.mp4").write_bytes(b"x" * 10)
    rl = RunLog(work, "t", ["run_pipeline.py", "in.mp4", "--name", "t"], ["asr", "translate", "tts"])
    rl.archive_state()
    rl.stage_cached("asr")
    rl.stage_start("translate", why="stale or missing")
    rl.stage_end("translate", 12.3, "abc123", {"segments": 5})
    rl.stage_start("tts")
    try:
        raise RuntimeError("GPU not available")
    except RuntimeError as exc:
        rl.stage_error("tts", exc)
    rl.close("failed")
    man = json.loads((rl.dir / "manifest.json").read_text())
    assert man["status"] == "failed" and man["failed_stage"] == "tts"
    assert man["stages"]["translate"]["fingerprint"] == "abc123" and man["stages"]["asr"]["status"] == "cached"
    assert "commit" in man["git"] and "mem_available_gb" in man["host"]
    ev = [json.loads(x) for x in (rl.dir / "events.jsonl").read_text().splitlines()]
    kinds = [e["kind"] for e in ev]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end" and "error" in kinds
    err = next(e for e in ev if e["kind"] == "error")
    assert "GPU not available" in err["error"] and "RuntimeError" in err["traceback"] and "host" in err
    idx = [json.loads(x) for x in (work / "runs" / "index.jsonl").read_text().splitlines()]
    assert idx[-1]["failed_stage"] == "tts" and idx[-1]["ran"] == ["translate"]
    hist = work / "history" / rl.run_id
    assert (hist / "state.json").exists() and (hist / "deliverables" / "01_asr.ja.srt").exists()
    assert not (hist / "deliverables" / "big.mp4").exists()            # media is not archived
    assert (work / "runs" / "latest").resolve() == rl.dir.resolve()


def test_append_event_and_snapshot():
    p = Path(tempfile.mkdtemp(prefix="runlog_")) / "sub" / "events.jsonl"
    append_event(p, "chunk_end", chunk="x_p01", rc="0")
    e = json.loads(p.read_text().splitlines()[0])
    assert e["kind"] == "chunk_end" and e["chunk"] == "x_p01"
    assert "mem_available_gb" in snapshot_host()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
