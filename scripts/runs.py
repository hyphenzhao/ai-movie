#!/usr/bin/env python3
"""Where did it stop, on which code, and how do I continue?

    python scripts/runs.py test_1                 # recent runs of one project
    python scripts/runs.py test_1 --run latest    # one run in detail (stages, error, host at failure)
    python scripts/runs.py --film SONE-846        # a long film: per-chunk state + film events
    python scripts/runs.py --all                  # last run of every project

Reads the records written by ``ai_movie/runlog.py`` (``workspace/<name>/runs``)
and, for long films, ``workspace/<film>/_split/events.jsonl`` + ``status.txt``.
Resuming never needs a run id: the stage cache skips what is valid, so the
command printed under "continue" is simply the original one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT / "workspace"


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out


def show_project(name: str, n: int = 8) -> None:
    rows = _jsonl(WS / name / "runs" / "index.jsonl")
    if not rows:
        print(f"{name}: no run records (runs before 2026-09-28 were not recorded; see workspace/{name}/run_v3.log)")
        return
    print(f"{name}: {len(rows)} recorded runs")
    for r in rows[-n:]:
        ran = ",".join(r.get("ran") or []) or "-"
        print(f"  {r['run_id']}  {r['status']:11s} {r.get('seconds', 0):>7.0f}s  commit {r.get('commit')}"
              f"{'*' if r.get('dirty') else ' '}  ran: {ran}"
              + (f"  FAILED at {r['failed_stage']}" if r.get("failed_stage") else ""))
    last = rows[-1]
    if last["status"] != "ok":
        show_run(name, last["run_id"])


def show_run(name: str, run: str) -> None:
    d = WS / name / "runs" / run
    if run == "latest":
        d = (WS / name / "runs" / "latest").resolve()
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8")) if (d / "manifest.json").exists() else {}
    if not man:
        print(f"no such run: {d}")
        return
    g = man.get("git") or {}
    print(f"\nrun {man['run_id']} of {man['name']} — {man.get('status')}")
    print(f"  started {man.get('started')}  ended {man.get('ended', '—')}  ({man.get('seconds', '?')} s)")
    print(f"  code    {g.get('describe')} ({g.get('commit', '')[:10]}){'  + uncommitted: ' + ', '.join(g.get('dirty_files') or []) if g.get('dirty') else ''}")
    print(f"  command {' '.join(man.get('argv') or [])}")
    h = man.get("host") or {}
    print(f"  host    mem free {h.get('mem_available_gb')} GB, load {h.get('load1')}, gpu device {'present' if h.get('kfd') else 'MISSING'}"
          f" (since {h.get('kfd_mtime')})")
    for s in man.get("steps") or []:
        v = (man.get("stages") or {}).get(s)
        if not v:
            print(f"    {s:9s} not reached")
            continue
        extra = ""
        if v.get("summary"):
            extra = "  " + ", ".join(f"{k}={x}" for k, x in v["summary"].items() if k not in ("video", "audio"))
        print(f"    {s:9s} {v.get('status'):8s} {str(v.get('seconds', '')):>8s}s  fp {str(v.get('fingerprint') or '')[:10]}{extra}"
              + (f"\n              ERROR {v.get('error')}" if v.get("error") else ""))
    errs = [e for e in _jsonl(d / "events.jsonl") if e.get("kind") == "error"]
    for e in errs[-1:]:
        eh = e.get("host") or {}
        print(f"  at failure: mem free {eh.get('mem_available_gb')} GB, gpu device {'present' if eh.get('kfd') else 'MISSING'},"
              f" torch sees GPU: {eh.get('cuda_available')}")
        tb = (e.get("traceback") or "").strip().splitlines()
        print("  traceback (tail):\n    " + "\n    ".join(tb[-6:]))
    if man.get("status") != "ok":
        argv = [a for a in (man.get("argv") or []) if a != "--force"]
        print(f"  continue: {ROOT / '.venv/bin/python'} -u {' '.join(argv)}     # valid stages are skipped")
    hist = WS / name / "history" / man["run_id"]
    if hist.exists():
        print(f"  replaced state kept in {hist}")


def show_film(film: str) -> None:
    split = WS / film / "_split"
    plan = json.loads((split / "plan.json").read_text()) if (split / "plan.json").exists() else {"chunks": []}
    status = {}
    for ln in (split / "status.txt").read_text().splitlines() if (split / "status.txt").exists() else []:
        p = ln.split(None, 1)
        if len(p) == 2:
            status[p[0]] = p[1]
    print(f"{film}: {len(plan['chunks'])} chunks")
    for c in plan["chunks"]:
        i = f"{c['index']:02d}"
        name = f"{film}_p{i}"
        rows = _jsonl(WS / name / "runs" / "index.jsonl")
        last = rows[-1] if rows else {}
        final = (WS / name / "output" / "v2_cloned_dubbed.mp4").exists()
        print(f"  p{i} {c['minutes']:5.1f} min speech {c['speech_minutes']:5.2f}  {status.get(i, '—'):22s} "
              f"final={'yes' if final else 'no ':3s} last run: {last.get('run_id', '—')} {last.get('status', '')}"
              + (f" FAILED at {last['failed_stage']}" if last.get("failed_stage") else ""))
    ev = _jsonl(split / "events.jsonl")
    if ev:
        print("  film events (last 8):")
        for e in ev[-8:]:
            print(f"    {e.get('at')} {e.get('kind'):12s} " + " ".join(f"{k}={v}" for k, v in e.items() if k not in ('t', 'at', 'kind', 'host')))
    print(f"  continue: bash scripts/run_long.sh {plan.get('video', '<video>')} {film}     # finished chunks are skipped")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?")
    ap.add_argument("--run", default=None)
    ap.add_argument("--film", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if args.film:
        show_film(args.film)
    elif args.all:
        for d in sorted(WS.iterdir()):
            rows = _jsonl(d / "runs" / "index.jsonl")
            if rows:
                r = rows[-1]
                print(f"{d.name:22s} {r['run_id']} {r['status']:11s} commit {r.get('commit')}"
                      + (f" FAILED at {r['failed_stage']}" if r.get("failed_stage") else ""))
    elif args.name and args.run:
        show_run(args.name, args.run)
    elif args.name:
        show_project(args.name)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
