"""Job manager: one pipeline subprocess at a time, live log tail, SSE fan-out.

A job is an ordered list of commands (argv lists) run sequentially in their
own process group with ``AI_MOVIE_JSON_LOG=1``; stdout+stderr go to a log
file that a tail thread parses into events (``log``, ``progress``, ``step``,
``job``).  Only one job runs at a time — the GPU is unified memory — and a
foreign pipeline process (started from a terminal) blocks new jobs unless
the caller overrides.  On server restart, running jobs are re-attached by
pid and their logs keep streaming.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from . import projects as P

JOBS_FILE = P.WORKSPACE / "_web" / "jobs.json"
FOREIGN_PATTERNS = ("run_pipeline.py", "run_vc_version.py", "auto_select_refs.py",
                    "vc_ref_probe.py", "scripts.inference", "run_v3.sh", "deliver.py")

_RE_TS = re.compile(r"^\[\d\d:\d\d:\d\d\] ")
_RE_START = re.compile(r"^▶ (\w+)$")
_RE_DONE = re.compile(r"^✓ (\w+)$")
_RE_TOOK = re.compile(r"^\s+(\w+) took (\d+)s$")
_RE_CACHED = re.compile(r"^· (\w+) \(cached\)$")
_RE_STALE = re.compile(r"^· (\w+) STALE \((.*)\)")
_RE_PROG = re.compile(r"^\s+(TTS|VC|lipsync|enhance|人脸检测|faces) (\d+)/(\d+)")
_RE_PROG2 = re.compile(r"^\s+([\w+.\-]+): (\d+)/(\d+)$")
_RE_PCT = re.compile(r"^\s+ASR (\d+)%")
_RE_STAGE = re.compile(r"^===== \[\d\d:\d\d:\d\d\] .* · (.*) =====$")
_RE_ERR = re.compile(r"^(Traceback|FAILED:|\s*\w+Error: |RuntimeError|MemoryError)")


def _kind_steps(kind: str, name: str, steps: list[str] | None, force: bool,
                version: str = "vc", with_v2: bool = True, with_deliver: bool = True) -> list[list[str]]:
    """Commands for a job kind (mirrors scripts/run_v3.sh)."""
    st = P.state_path(name)
    work = P.workdir(name)
    py = P.PY
    S = P.SCRIPTS
    cmds: list[list[str]] = []
    if kind == "steps":
        cmds.append(P.build_argv(name, steps, force))
    elif kind == "one_click":
        cmds.append(P.build_argv(name, steps or list(P.rp.ALL_STEPS), force))
        if with_v2:
            cmds.append([py, "-u", str(S / "auto_select_refs.py"), str(st)])
            cmds.append([py, "-u", str(S / "run_vc_version.py"), str(st),
                         "--refs-json", str(work / "refs_auto" / "refs.json")])
        cmds.append(P.build_argv(name, ["qc"], True))
        cmds.append([py, "-u", str(S / "eval_pipeline.py"), str(st)])
        if with_deliver:
            cmds.append([py, "-u", str(S / "deliver.py"), str(st), "--out", str(P.DELIVER),
                         "--version", "vc" if with_v2 else "v1"])
    elif kind == "v2":
        cmds.append([py, "-u", str(S / "auto_select_refs.py"), str(st)])
        cmds.append([py, "-u", str(S / "run_vc_version.py"), str(st),
                     "--refs-json", str(work / "refs_auto" / "refs.json")])
        cmds.append(P.build_argv(name, ["qc"], True))
    elif kind == "qc":
        cmds.append(P.build_argv(name, ["qc"], True))
    elif kind == "eval":
        cmds.append([py, "-u", str(S / "eval_pipeline.py"), str(st)])
    elif kind == "deliver":
        cmds.append([py, "-u", str(S / "deliver.py"), str(st), "--out", str(P.DELIVER),
                     "--version", version])
    elif kind == "fp_adopt":
        cmds.append(P.build_argv(name, None) + ["--fp-adopt"])
    elif kind == "review_speakers":
        cmds.append([py, "-u", str(S / "review_speakers.py"), str(st)])
    else:
        raise ValueError(f"unknown job kind {kind}")
    return cmds


class Job:
    def __init__(self, name: str, kind: str, cmds: list[list[str]], user: str = "lan",
                 meta: dict | None = None):
        self.id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.name = name
        self.kind = kind
        self.cmds = cmds
        self.user = user
        self.meta = meta or {}
        self.status = "queued"          # queued|running|done|failed|cancelled
        self.created = time.time()
        self.started: float | None = None
        self.ended: float | None = None
        self.exit_code: int | None = None
        self.pid: int | None = None
        self.cmd_index = 0
        self.current_step: str | None = None
        self.steps_done: list[str] = []
        self.progress: dict = {}
        self.error_tail: list[str] = []
        self.log_path = P.webdir(name) / "logs" / f"{self.id}.log"
        self.events: deque = deque(maxlen=6000)
        self.seq = 0
        self._cancel = False

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "id", "name", "kind", "user", "meta", "status", "created", "started", "ended",
            "exit_code", "pid", "cmd_index", "current_step", "steps_done", "progress",
            "error_tail")} | {"log_path": str(self.log_path), "n_cmds": len(self.cmds),
                              "cmds": [" ".join(Path(a).name if i == 0 else a for i, a in enumerate(c))
                                       for c in self.cmds]}


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.queue: deque[str] = deque()
        self.current: Job | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.subs: dict[str, set[asyncio.Queue]] = {}     # job id or "*" → queues
        self._lock = threading.Lock()
        self._runner_task: asyncio.Task | None = None
        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._load_history()

    # ── persistence ──
    def _load_history(self) -> None:
        if not JOBS_FILE.exists():
            return
        try:
            rows = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        except Exception:                               # noqa: BLE001
            return
        for r in rows[-200:]:
            j = Job.__new__(Job)
            j.__dict__.update({k: v for k, v in r.items() if k not in ("log_path", "n_cmds", "cmds")})
            j.cmds = []
            j.log_path = Path(r.get("log_path") or "")
            j.events = deque(maxlen=6000)
            j.seq = 0
            j._cancel = False
            j.meta = r.get("meta") or {}
            self.jobs[j.id] = j
            if j.status == "running" and j.pid and _alive(j.pid):
                j.cmds = [["(re-attached)"]]
                self.current = j
                threading.Thread(target=self._reattach, args=(j,), daemon=True).start()
            elif j.status in ("running", "queued"):
                j.status = "failed"
                j.error_tail = ["服务器重启时任务已不在运行"]
        self._save()

    def _save(self) -> None:
        rows = [j.to_dict() for j in sorted(self.jobs.values(), key=lambda j: j.created)][-200:]
        tmp = JOBS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, JOBS_FILE)

    # ── events ──
    def _push(self, job: Job, ev: dict) -> None:
        job.seq += 1
        ev = {"seq": job.seq, "job": job.id, **ev}
        job.events.append(ev)
        if self.loop is None:
            return
        for q in list(self.subs.get(job.id, ())) + list(self.subs.get("*", ())):
            self.loop.call_soon_threadsafe(q.put_nowait, ev)

    def subscribe(self, key: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.subs.setdefault(key, set()).add(q)
        return q

    def unsubscribe(self, key: str, q: asyncio.Queue) -> None:
        self.subs.get(key, set()).discard(q)

    # ── public ──
    def foreign_processes(self) -> list[dict]:
        own = {j.pid for j in self.jobs.values() if j.status == "running" and j.pid}
        out = []
        try:
            r = subprocess.run(["pgrep", "-af", "|".join(FOREIGN_PATTERNS)],
                               capture_output=True, text=True, timeout=10)
            for ln in r.stdout.splitlines():
                pid_s, _, cmd = ln.partition(" ")
                pid = int(pid_s)
                if pid == os.getpid() or pid in own or "pgrep" in cmd:
                    continue
                # children of our own jobs share the session id with the job pid
                try:
                    if own and os.getsid(pid) in own:
                        continue
                except Exception:                       # noqa: BLE001
                    pass
                out.append({"pid": pid, "cmd": cmd[:160]})
        except Exception:                               # noqa: BLE001
            pass
        return out

    def submit(self, name: str, kind: str, *, steps: list[str] | None = None, force: bool = False,
               version: str = "vc", with_v2: bool = True, with_deliver: bool = True,
               user: str = "lan", allow_foreign: bool = False) -> Job:
        if not allow_foreign:
            f = self.foreign_processes()
            if f:
                raise BusyError(f)
        cmds = _kind_steps(kind, name, steps, force, version, with_v2, with_deliver)
        job = Job(name, kind, cmds, user, {"steps": steps, "force": force, "version": version,
                                         "with_v2": with_v2, "with_deliver": with_deliver})
        with self._lock:
            self.jobs[job.id] = job
            self.queue.append(job.id)
        self._save()
        self._push(job, {"type": "job", "status": "queued", "position": len(self.queue)})
        self._ensure_runner()
        return job

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job.status == "queued":
            with self._lock:
                try:
                    self.queue.remove(job_id)
                except ValueError:
                    pass
            job.status = "cancelled"
            job.ended = time.time()
            self._save()
            self._push(job, {"type": "job", "status": "cancelled"})
            return True
        if job.status == "running":
            job._cancel = True
            _kill_group(job.pid)
            return True
        return False

    def running_for(self, name: str) -> Job | None:
        for j in self.jobs.values():
            if j.name == name and j.status in ("running", "queued"):
                return j
        return None

    def history(self, name: str | None = None, limit: int = 50) -> list[dict]:
        rows = [j.to_dict() for j in sorted(self.jobs.values(), key=lambda j: -j.created)
                if name is None or j.name == name]
        return rows[:limit]

    # ── runner ──
    def _ensure_runner(self) -> None:
        if self.loop is None:
            self.loop = asyncio.get_event_loop()
        if self._runner_task is None or self._runner_task.done():
            self._runner_task = self.loop.create_task(self._runner())

    async def _runner(self) -> None:
        while True:
            with self._lock:
                if self.current is not None and self.current.status == "running":
                    job_id = None
                else:
                    job_id = self.queue.popleft() if self.queue else None
            if job_id is None:
                if not self.queue and (self.current is None or self.current.status != "running"):
                    return
                await asyncio.sleep(1.0)
                continue
            job = self.jobs[job_id]
            self.current = job
            await asyncio.to_thread(self._run_job, job)
            P.invalidate_status(job.name)
            self.current = None

    def _run_job(self, job: Job) -> None:
        job.status = "running"
        job.started = time.time()
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._save()
        self._push(job, {"type": "job", "status": "running"})
        env = {**os.environ, "AI_MOVIE_JSON_LOG": "1", "PYTHONUNBUFFERED": "1"}
        rc = 0
        with open(job.log_path, "ab") as log_fh:
            for i, argv in enumerate(job.cmds):
                if job._cancel:
                    break
                job.cmd_index = i
                log_fh.write(f"===== [{time.strftime('%H:%M:%S')}] {job.name} · cmd {i + 1}/{len(job.cmds)}: "
                             f"{' '.join(Path(a).name if k == 0 else a for k, a in enumerate(argv))} =====\n".encode())
                log_fh.flush()
                proc = subprocess.Popen(argv, cwd=str(P.ROOT), env=env, stdout=log_fh,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                job.pid = proc.pid
                self._save()
                stop = threading.Event()
                tail = threading.Thread(target=self._tail, args=(job, stop), daemon=True)
                tail.start()
                rc = proc.wait()
                stop.set()
                tail.join(timeout=5)
                self._tail_flush(job)
                if rc != 0:
                    break
        job.pid = None
        job.ended = time.time()
        if job._cancel:
            job.status = "cancelled"
        elif rc == 0:
            job.status = "done"
        else:
            job.status = "failed"
        job.exit_code = rc
        job.current_step = None
        if job.kind in ("v2", "one_click") and job.status == "done":
            _record_vc_deps(job.name)
        self._save()
        self._push(job, {"type": "job", "status": job.status, "exit_code": rc})

    # ── tail / parse ──
    def _read_new(self, job: Job) -> None:
        """Parse whatever the log file gained since the last read."""
        pos = getattr(job, "_tail_pos", 0)
        try:
            with open(job.log_path, "rb") as fh:
                fh.seek(pos)
                data = fh.read()
                pos = fh.tell()
        except FileNotFoundError:
            return
        if not data:
            return
        # keep a partial trailing line for the next read
        cut = data.rfind(b"\n")
        if cut < 0:
            return
        job._tail_pos = pos - (len(data) - cut - 1)
        for raw in data[:cut].decode("utf-8", "replace").split("\n"):
            raw = raw.rstrip("\r")
            if raw:
                self._parse_line(job, raw)

    def _tail(self, job: Job, stop: threading.Event) -> None:
        while True:
            self._read_new(job)
            if stop.is_set():
                self._read_new(job)
                break
            time.sleep(0.4)

    def _tail_flush(self, job: Job) -> None:
        self._read_new(job)

    def _parse_line(self, job: Job, raw: str) -> None:
        msg, kind, obj = raw, None, None
        if raw.startswith("{"):
            try:
                obj = json.loads(raw)
                msg = obj.get("msg", "")
                kind = obj.get("kind")
            except Exception:                           # noqa: BLE001
                obj = None
        text = _RE_TS.sub("", msg) if msg else raw
        if kind:
            step = obj.get("step")
            if kind == "step_start":
                job.current_step = step
                job.progress = {}
                self._push(job, {"type": "step", "step": step, "status": "running"})
            elif kind == "step_done":
                if step and step not in job.steps_done:
                    job.steps_done.append(step)
                job.current_step = None
                self._push(job, {"type": "step", "step": step, "status": "done", "took": obj.get("took")})
                P.invalidate_status(job.name)
            elif kind == "cached":
                self._push(job, {"type": "step", "step": step, "status": "cached"})
            elif kind == "stale":
                self._push(job, {"type": "step", "step": step, "status": "stale", "reasons": obj.get("reasons")})
            elif kind == "error":
                job.error_tail.append(str(obj.get("error")))
                self._push(job, {"type": "step", "step": step, "status": "failed", "error": obj.get("error")})
            return
        if not text.strip():
            return
        m = _RE_STAGE.match(text)
        if m:
            self._push(job, {"type": "log", "line": text, "banner": True})
            return
        if (m := _RE_START.match(text)):
            job.current_step = m.group(1)
            self._push(job, {"type": "step", "step": m.group(1), "status": "running"})
        elif (m := _RE_DONE.match(text)):
            if m.group(1) not in job.steps_done:
                job.steps_done.append(m.group(1))
            job.current_step = None
            self._push(job, {"type": "step", "step": m.group(1), "status": "done"})
            P.invalidate_status(job.name)
        elif (m := _RE_CACHED.match(text)):
            self._push(job, {"type": "step", "step": m.group(1), "status": "cached"})
        elif (m := _RE_PROG.match(text)):
            job.progress = {"label": m.group(1), "done": int(m.group(2)), "total": int(m.group(3))}
            self._push(job, {"type": "progress", **job.progress, "step": job.current_step})
        elif (m := _RE_PROG2.match(text)):
            job.progress = {"label": m.group(1), "done": int(m.group(2)), "total": int(m.group(3))}
            self._push(job, {"type": "progress", **job.progress, "step": job.current_step})
        elif (m := _RE_PCT.match(text)):
            job.progress = {"label": "ASR", "done": int(m.group(1)), "total": 100}
            self._push(job, {"type": "progress", **job.progress, "step": job.current_step})
        if _RE_ERR.match(text):
            job.error_tail.append(text[:300])
            job.error_tail = job.error_tail[-40:]
        if re.search(r"MIOpen|Warning|warn\(|it/s\]|seconds/s\]", text):
            return                              # framework noise, not worth a browser message
        self._push(job, {"type": "log", "line": text[:2000]})

    def _reattach(self, job: Job) -> None:
        stop = threading.Event()
        job._tail_pos = 0
        t = threading.Thread(target=self._tail, args=(job, stop), daemon=True)
        t.start()
        while _alive(job.pid):
            time.sleep(2.0)
        stop.set()
        t.join(timeout=5)
        job.status = "done" if not job.error_tail else "failed"
        job.ended = time.time()
        job.exit_code = None
        job.pid = None
        if self.current is job:
            self.current = None
        self._save()
        P.invalidate_status(job.name)
        self._push(job, {"type": "job", "status": job.status, "exit_code": None})


class BusyError(Exception):
    def __init__(self, procs: list[dict]):
        super().__init__("gpu_busy")
        self.procs = procs


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _kill_group(pid: int | None) -> None:
    if not pid:
        return
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return

    def _later():
        time.sleep(15)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    threading.Thread(target=_later, daemon=True).start()


def _record_vc_deps(name: str) -> None:
    """Remember which v1 fingerprints the v2 film was built from."""
    try:
        st = P.load_state(name)
        fps = st.get("_fp") or {}
        st.setdefault("_web", {})["vc_deps"] = {k: (fps.get(k) or {}).get("hash") for k in ("fit", "compose")}
        P.rp.save_state(P.state_path(name), st)
    except Exception:                                   # noqa: BLE001
        pass


manager = JobManager()
