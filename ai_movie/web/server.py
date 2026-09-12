"""FastAPI server for the web UI.  Run: ``python -m ai_movie.web.server --host 0.0.0.0 --port 8000``."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from urllib.parse import quote

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import edits as E
from . import projects as P
from .jobs import BusyError, manager

STATIC = Path(__file__).resolve().parent / "static"
VERSION = "v3-web-1"

app = FastAPI(title="AI Movie Web", docs_url="/api/docs", redoc_url=None)


def _user(req: Request) -> str:
    return req.headers.get("remote-user") or "lan"


def _project_or_404(name: str) -> dict:
    if not P.valid_name(name):
        raise HTTPException(400, "invalid project name")
    st = P.load_state(name)
    if not st and P.find_video(name) is None:
        raise HTTPException(404, "project not found")
    return st


@app.on_event("startup")
async def _startup() -> None:
    manager.loop = asyncio.get_event_loop()
    manager._ensure_runner()


# ── basics ──

@app.get("/api/health")
def health():
    return {"ok": True, "version": VERSION, "gpu_busy": bool(manager.foreign_processes()),
            "running": manager.current.to_dict() if manager.current and manager.current.status == "running" else None,
            "time": time.time()}


@app.get("/api/projects")
def projects():
    rows = P.list_projects()
    for r in rows:
        j = manager.running_for(r["name"])
        r["job"] = j.to_dict() if j else None
    return rows


@app.get("/api/inputs")
def inputs():
    return P.list_inputs()


@app.post("/api/projects/import")
def import_project(body: dict = Body(...)):
    try:
        return P.import_input(body.get("file") or "")
    except FileNotFoundError:
        raise HTTPException(404, "file not found in inputs/")
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/uploads/init")
def upload_init(body: dict = Body(...)):
    try:
        return P.upload_init(body.get("filename") or "", int(body.get("size") or 0), body.get("name"))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.put("/api/uploads/{uid}/{index}")
async def upload_chunk(uid: str, index: int, request: Request):
    if not (P.UPLOADS / uid / "meta.json").exists():
        raise HTTPException(404, "upload not found")
    data = await request.body()
    return P.upload_chunk(uid, index, data)


@app.post("/api/uploads/{uid}/finalize")
def upload_finalize(uid: str):
    if not (P.UPLOADS / uid / "meta.json").exists():
        raise HTTPException(404, "upload not found")
    try:
        return P.upload_finalize(uid)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# ── project views ──

@app.get("/api/projects/{name}")
def project(name: str):
    st = _project_or_404(name)
    job = manager.running_for(name)
    raw = P.raw_status(name)
    last_failed = next((j for j in manager.history(name, 5) if j["status"] == "failed"), None)
    failed_step = (last_failed or {}).get("current_step") if last_failed and not job else None
    steps = P.derive_status(name, raw, st, running_step=job.current_step if job else None,
                            failed_step=failed_step, running_kind=job.kind if job else None)
    return {
        "name": name, "video": str(P.find_video(name, st) or ""), "options": P.load_options(name),
        "steps": steps, "step_order": list(P.rp.ALL_STEPS) + ["v2", "deliver"],
        "videos": P.videos_view(name, st), "job": job.to_dict() if job else None,
        "jobs": manager.history(name, 10), "engines": P.ENGINES,
        "summary": {
            "duration": (st.get("demux") or {}).get("duration"),
            "n_segments": len((st.get("asr") or {}).get("segments") or []),
            "speakers": ((st.get("asr") or {}).get("diarization") or {}).get("speakers") or {},
            "compact": {k: v for k, v in (st.get("compact") or {}).items() if k in ("attempted", "rewritten", "still_over")},
            "qc": {k: (v or {}).get("summary") for k, v in (st.get("qc") or {}).items()},
            "vc": {k: v for k, v in (st.get("vc") or {}).items() if k in ("converted", "note", "max_drift_ms")},
            "osd": {k: v for k, v in (st.get("osd") or {}).items() if k in ("available", "total_overlap_s", "reason")},
            "edits": st.get("_edits") or {},
        },
    }


@app.get("/api/projects/{name}/options")
def get_options(name: str):
    _project_or_404(name)
    return P.load_options(name)


@app.put("/api/projects/{name}/options")
def put_options(name: str, body: dict = Body(...)):
    _project_or_404(name)
    return P.save_options(name, body)


@app.get("/api/projects/{name}/segments")
def segments(name: str):
    st = _project_or_404(name)
    return P.segments_view(name, st)


@app.get("/api/projects/{name}/speakers")
def speakers(name: str):
    st = _project_or_404(name)
    return P.speakers_view(name, st)


@app.get("/api/projects/{name}/faces")
def faces(name: str):
    st = _project_or_404(name)
    return P.faces_view(name, st)


@app.get("/api/projects/{name}/glossary")
def glossary(name: str):
    st = _project_or_404(name)
    return st.get("glossary") or {}


@app.get("/api/projects/{name}/compact")
def compact(name: str):
    st = _project_or_404(name)
    c = st.get("compact") or {}
    return {"report": c.get("report") or [], "attempted": c.get("attempted"),
            "rewritten": c.get("rewritten"), "skipped": c.get("skipped", False)}


@app.get("/api/projects/{name}/qc")
def qc(name: str, key: str = "fit"):
    _project_or_404(name)
    return P.qc_view(name, key)


@app.get("/api/projects/{name}/videos")
def videos(name: str):
    st = _project_or_404(name)
    return P.videos_view(name, st)


@app.get("/api/projects/{name}/deliverables")
def deliverables(name: str):
    _project_or_404(name)
    return P.deliverables_view(name)


@app.get("/api/projects/{name}/review")
def review(name: str):
    _project_or_404(name)
    return P.review_view(name)


@app.get("/api/projects/{name}/acceptance")
def acceptance(name: str):
    _project_or_404(name)
    return {"text": P.acceptance_view(name)}


@app.get("/api/projects/{name}/jobs")
def project_jobs(name: str):
    _project_or_404(name)
    return manager.history(name)


# ── edits ──

def _no_job(name: str) -> None:
    j = manager.running_for(name)
    if j:
        raise HTTPException(409, f"任务 {j.id} 正在运行/排队，暂不能编辑")


def _edit(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except E.EditError as exc:
        raise HTTPException(exc.code, str(exc))


@app.put("/api/projects/{name}/segments/{idx}/translation")
def put_translation(name: str, idx: int, body: dict = Body(...)):
    _project_or_404(name)
    _no_job(name)
    return _edit(E.set_translation, name, idx, body.get("text", ""))


@app.put("/api/projects/{name}/segments/{idx}/speaker")
def put_speaker(name: str, idx: int, body: dict = Body(...)):
    _project_or_404(name)
    _no_job(name)
    return _edit(E.set_speaker, name, idx, body.get("speaker"), body.get("gender"))


@app.post("/api/projects/{name}/speakers")
def post_speaker(name: str, body: dict = Body(...)):
    _project_or_404(name)
    _no_job(name)
    return _edit(E.add_speaker, name, body.get("gender", ""))


@app.put("/api/projects/{name}/glossary")
def put_glossary(name: str, body: dict = Body(...)):
    _project_or_404(name)
    _no_job(name)
    return _edit(E.set_glossary, name, body.get("terms") or {}, bool(body.get("apply_to_translation")))


@app.put("/api/projects/{name}/faces/binding")
def put_binding(name: str, body: dict = Body(...)):
    _project_or_404(name)
    _no_job(name)
    return _edit(E.set_face_binding, name, body.get("binding") or {})


# ── jobs ──

@app.post("/api/projects/{name}/jobs")
def start_job(name: str, request: Request, body: dict = Body(...)):
    _project_or_404(name)
    if manager.running_for(name):
        raise HTTPException(409, "该工程已有任务在运行/排队")
    kind = body.get("kind") or "steps"
    steps = body.get("steps") or None
    if steps:
        bad = [s for s in steps if s not in P.rp.ALL_STEPS]
        if bad:
            raise HTTPException(400, f"unknown steps: {bad}")
    try:
        job = manager.submit(name, kind, steps=steps, force=bool(body.get("force")),
                             version=body.get("version") or "vc",
                             with_v2=body.get("with_v2", True), with_deliver=body.get("with_deliver", True),
                             user=_user(request), allow_foreign=bool(body.get("allow_foreign")))
    except BusyError as exc:
        return JSONResponse({"error": "gpu_busy", "procs": exc.procs}, status_code=409)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    return {"job": job.to_dict(), "position": len(manager.queue)}


@app.get("/api/jobs")
def all_jobs():
    return manager.history(None, 100)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    j = manager.jobs.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return j.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not manager.cancel(job_id):
        raise HTTPException(409, "job is not cancellable")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/log")
def job_log(job_id: str, tail: int = 400):
    j = manager.jobs.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    lines = []
    if j.log_path.exists():
        lines = j.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:]
    return {"lines": lines}


async def _sse(gen):
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, since: int = 0):
    j = manager.jobs.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")

    async def gen():
        q = manager.subscribe(job_id)
        try:
            for ev in list(j.events):
                if ev["seq"] > since:
                    yield f"event: {ev['type']}\nid: {ev['seq']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            if j.status in ("done", "failed", "cancelled"):
                yield f"event: job\ndata: {json.dumps({'type': 'job', 'status': j.status, 'job': j.id, 'final': True})}\n\n"
                return
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"event: {ev['type']}\nid: {ev['seq']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    if ev["type"] == "job" and ev.get("status") in ("done", "failed", "cancelled"):
                        return
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            manager.unsubscribe(job_id, q)
    return await _sse(gen())


@app.get("/api/events")
async def global_events():
    async def gen():
        q = manager.subscribe("*")
        try:
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    if ev["type"] in ("job", "step"):
                        yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            manager.unsubscribe("*", q)
    return await _sse(gen())


# ── media ──

_CT = {".mp4": "video/mp4", ".mkv": "video/x-matroska", ".mov": "video/quicktime", ".webm": "video/webm",
       ".wav": "audio/wav", ".mp3": "audio/mpeg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
       ".png": "image/png", ".json": "application/json", ".csv": "text/csv; charset=utf-8",
       ".srt": "text/plain; charset=utf-8", ".md": "text/plain; charset=utf-8",
       ".txt": "text/plain; charset=utf-8", ".log": "text/plain; charset=utf-8"}


def _file(p: str, attachment: bool):
    path = P.safe_media_path(p)
    if path is None:
        raise HTTPException(403, "path not allowed")
    headers = {"Accept-Ranges": "bytes"}
    if attachment:
        headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{quote(path.name)}"
    return FileResponse(str(path), media_type=_CT.get(path.suffix.lower(), "application/octet-stream"),
                        headers=headers)


@app.api_route("/api/media", methods=["GET", "HEAD"])
def media(p: str):
    return _file(p, False)


@app.api_route("/api/download", methods=["GET", "HEAD"])
def download(p: str):
    return _file(p, True)


# ── static ──

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--proxy-headers", action="store_true")
    ap.add_argument("--forwarded-allow-ips", default="127.0.0.1")
    a = ap.parse_args()
    uvicorn.run("ai_movie.web.server:app", host=a.host, port=a.port, workers=1,
                proxy_headers=a.proxy_headers, forwarded_allow_ips=a.forwarded_allow_ips,
                timeout_keep_alive=75, log_level="info")


if __name__ == "__main__":
    main()
