"""Run records: what ran, on which code, how far it got, and why it stopped.

Every ``run_pipeline.py`` invocation that executes at least one stage gets

    workspace/<name>/runs/<run_id>/manifest.json   who/what/where (written at start, closed at end)
    workspace/<name>/runs/<run_id>/events.jsonl    append-only: stage_start / cached / stage_end / error / run_end
    workspace/<name>/runs/index.jsonl              one line per run (id, steps, status, failing stage)
    workspace/<name>/history/<run_id>/             the state.json + text deliverables the run replaced

``run_id`` is ``YYYYmmdd-HHMMSS-<pid>``.  The manifest pins the git commit
(and whether the tree was dirty), the command line, the host state (GPU
visible to torch? free memory, load) and, per stage, the fingerprint hash
the stage cache uses — so an output can be traced to the exact code and
configuration, and a failure to the machine state at that moment.  Resuming
needs nothing from here (the stage cache decides), but ``scripts/runs.py``
reads these records to say where a run stopped and how to continue.

Writing a record must never break a run: every entry point swallows its own
errors.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEEP_HISTORY = 10
_TEXT_EXT = {".json", ".csv", ".srt", ".txt", ".md"}


def _git() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.rstrip()
        except Exception:                               # noqa: BLE001
            return ""
    dirty = [ln for ln in run("status", "--porcelain").splitlines() if ln and not ln.startswith("??")]
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "describe": run("describe", "--tags", "--always"), "dirty": bool(dirty),
            "dirty_files": [ln[3:] for ln in dirty][:20]}


def snapshot_host() -> dict:
    """Machine state right now (cheap; no model loading)."""
    d: dict = {"t": round(time.time(), 1)}
    try:
        mem = {k.strip(): v for k, v in (ln.split(":", 1) for ln in Path("/proc/meminfo").read_text().splitlines())}
        d["mem_available_gb"] = round(int(mem["MemAvailable"].split()[0]) / 1048576, 1)
        d["mem_total_gb"] = round(int(mem["MemTotal"].split()[0]) / 1048576, 1)
    except Exception:                                   # noqa: BLE001
        pass
    try:
        d["load1"] = round(os.getloadavg()[0], 1)
    except OSError:
        pass
    try:
        busy = sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"))
        if busy:
            d["gpu_busy_percent"] = int(busy[0].read_text().strip())
    except Exception:                                   # noqa: BLE001
        pass
    d["kfd"] = Path("/dev/kfd").exists()
    try:
        d["kfd_mtime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(Path("/dev/kfd").stat().st_mtime))
    except OSError:
        pass
    try:
        du = shutil.disk_usage(ROOT)
        d["disk_free_gb"] = round(du.free / 1e9, 1)
    except OSError:
        pass
    if "torch" in sys.modules:                          # never import torch just to look
        try:
            import torch
            d["cuda_available"] = bool(torch.cuda.is_available())
            d["torch"] = torch.__version__
        except Exception as exc:                        # noqa: BLE001
            d["cuda_available"] = f"error: {exc}"
    return d


class RunLog:
    def __init__(self, work: Path, name: str, argv: list[str], steps: list[str]):
        self.work = Path(work)
        self.name = name
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
        self.dir = self.work / "runs" / self.run_id
        self.steps = list(steps)
        self.t0 = time.time()
        self.stages: dict[str, dict] = {}
        self.failed: str | None = None
        self.ok = False
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.manifest = {"run_id": self.run_id, "name": name, "argv": argv, "steps": self.steps,
                             "started": time.strftime("%Y-%m-%d %H:%M:%S"), "git": _git(),
                             "host": snapshot_host(), "python": sys.version.split()[0],
                             "env": {k: os.environ[k] for k in ("PROFILES", "AI_MOVIE_UPLOAD", "HF_HUB_OFFLINE",
                                                                "AI_MOVIE_POLISH_MODEL", "HTTPS_PROXY") if k in os.environ},
                             "status": "running"}
            self._write_manifest()
            self.event("run_start", steps=self.steps)
            self.ok = True
        except Exception as exc:                        # noqa: BLE001
            print(f"[runlog] disabled: {exc}", file=sys.stderr)

    # ── writing ──
    def _write_manifest(self) -> None:
        tmp = self.dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tmp.replace(self.dir / "manifest.json")

    def event(self, kind: str, **fields) -> None:
        if not getattr(self, "dir", None):
            return
        try:
            rec = {"t": round(time.time(), 3), "at": time.strftime("%H:%M:%S"), "kind": kind, **fields}
            with open(self.dir / "events.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:                               # noqa: BLE001
            pass

    def archive_state(self) -> None:
        """Keep what this run is about to overwrite: state.json + text deliverables."""
        try:
            st = self.work / "state.json"
            if not st.exists():
                return
            dst = self.work / "history" / self.run_id
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(st, dst / "state.json")
            deliver = self.work / "deliverables"
            if deliver.is_dir():
                for p in deliver.rglob("*"):
                    if p.is_file() and p.suffix.lower() in _TEXT_EXT and p.stat().st_size < 20_000_000:
                        q = dst / "deliverables" / p.relative_to(deliver)
                        q.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, q)
            old = sorted((self.work / "history").iterdir())
            for d in old[:-KEEP_HISTORY]:
                shutil.rmtree(d, ignore_errors=True)
            self.event("archived", to=str(dst))
        except Exception as exc:                        # noqa: BLE001
            self.event("archive_failed", error=str(exc))

    # ── stages ──
    def stage_start(self, step: str, why: str | None = None) -> None:
        self.stages[step] = {"started": time.strftime("%H:%M:%S"), "status": "running", "why": why}
        self.event("stage_start", step=step, why=why, host=snapshot_host())

    def stage_cached(self, step: str) -> None:
        self.stages[step] = {"status": "cached"}
        self.event("cached", step=step)

    def stage_end(self, step: str, seconds: float, fp_hash: str | None = None, summary: dict | None = None) -> None:
        self.stages[step] = {**self.stages.get(step, {}), "status": "done", "seconds": round(seconds, 1),
                             "fingerprint": fp_hash, **({"summary": summary} if summary else {})}
        self.event("stage_end", step=step, seconds=round(seconds, 1), fingerprint=fp_hash, summary=summary)

    def stage_error(self, step: str, exc: BaseException) -> None:
        self.failed = step
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-6000:]
        host = snapshot_host()
        self.stages[step] = {**self.stages.get(step, {}), "status": "failed",
                             "error": f"{type(exc).__name__}: {exc}"[:500]}
        self.event("error", step=step, error=f"{type(exc).__name__}: {exc}"[:2000], traceback=tb, host=host)

    def close(self, status: str) -> None:
        if not getattr(self, "dir", None) or not self.ok:
            return
        try:
            self.manifest.update({"status": status, "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
                                  "seconds": round(time.time() - self.t0, 1), "stages": self.stages,
                                  "failed_stage": self.failed, "host_end": snapshot_host()})
            self._write_manifest()
            self.event("run_end", status=status, failed_stage=self.failed)
            with open(self.work / "runs" / "index.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"run_id": self.run_id, "started": self.manifest["started"], "status": status,
                                     "failed_stage": self.failed, "steps": self.steps,
                                     "ran": [s for s, v in self.stages.items() if v.get("status") == "done"],
                                     "commit": self.manifest["git"].get("commit", "")[:10],
                                     "dirty": self.manifest["git"].get("dirty"),
                                     "seconds": self.manifest["seconds"]}, ensure_ascii=False) + "\n")
            latest = self.work / "runs" / "latest"
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(self.run_id)
        except Exception as exc:                        # noqa: BLE001
            print(f"[runlog] close failed: {exc}", file=sys.stderr)


def append_event(path: Path, kind: str, **fields) -> None:
    """One JSON line into any events file (used by the shell orchestrators)."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": round(time.time(), 3), "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "kind": kind, **fields}, ensure_ascii=False, default=str) + "\n")
    except Exception:                                   # noqa: BLE001
        pass


if __name__ == "__main__":                              # python -m ai_movie.runlog <events.jsonl> <kind> k=v …
    if len(sys.argv) >= 3:
        kv = dict(a.split("=", 1) for a in sys.argv[3:] if "=" in a)
        append_event(Path(sys.argv[1]), sys.argv[2], host=snapshot_host(), **kv)
