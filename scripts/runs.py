#!/usr/bin/env python3
"""Where did it stop, on which code, and how do I continue?

    python scripts/runs.py test_1                 # recent runs of one project (+ manual-edit summary)
    python scripts/runs.py test_1 --run latest    # one run in detail (stages, error, host at failure)
    python scripts/runs.py test_1 --edits         # manual edits: who / when / what → what, undo hint
    python scripts/runs.py test_1 --edits --stage translate --idx 12 --who lan -n 20
    python scripts/runs.py --film SONE-846        # a long film: per-chunk state + film events + profile edits
    python scripts/runs.py --film SONE-846 --edits
    python scripts/runs.py --all                  # last run of every project

Reads the records written by ``ai_movie/runlog.py`` (``workspace/<name>/runs``),
the web edit log (``workspace/<name>/edits.jsonl``, ``ai_movie/web/edits.py``)
and, for long films, ``workspace/<film>/_split/events.jsonl`` + ``status.txt``.
Resuming never needs a run id: the stage cache skips what is valid, so the
command printed under "continue" is simply the original one.

The edit summary cannot decide staleness by itself: ``index.jsonl`` is written
only when a run closes, and a run may have executed only upstream steps.  It
says how many edits landed after the newest run started and which consumers
those edits name; ``run_pipeline.py --status-json`` is the authority.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT / "workspace"
EDIT_LOG = "edits.jsonl"


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


# ── manual edits ──

def read_edits(name: str) -> list[dict]:
    """The edit log of a project (or, for a film name, its profile edits); [] when absent."""
    return [r for r in _jsonl(WS / name / EDIT_LOG) if isinstance(r, dict)]


def _epoch(stamp: str | None) -> float | None:
    try:
        return time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S")) if stamp else None
    except (ValueError, TypeError, OverflowError):
        return None


def last_run_start(name: str) -> tuple[str | None, str | None]:
    """``(run_id, started)`` of the newest run by manifest — includes a run still running,
    which index.jsonl (written at close) does not list."""
    best: tuple[str | None, str | None, float] = (None, None, -1.0)
    for m in (WS / name / "runs").glob("*/manifest.json"):
        try:
            man = json.loads(m.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        ep = _epoch(man.get("started"))
        if ep is not None and ep > best[2]:
            best = (man.get("run_id"), man.get("started"), ep)
    return best[0], best[1]


def edits_summary(rows: list[dict], since_epoch: float | None = None) -> dict:
    """Counts for one line of output: total, per stage, last at/who, how many landed after
    *since_epoch* and the union of their ``consumers`` (what those edits say must be redone)."""
    by_stage = Counter(str(r.get("stage")) for r in rows)
    since = [r for r in rows if since_epoch is not None and float(r.get("t") or 0) > since_epoch]
    last = rows[-1] if rows else {}
    return {"n": len(rows), "by_stage": dict(by_stage), "last_at": last.get("at"), "last_who": last.get("who"),
            "since_last_run": len(since),
            "consumers_since": sorted({str(c) for r in since for c in (r.get("consumers") or [])})}


def _q(s) -> str:
    return "「" + str(s) + "」" if s is not None else "∅"


def fmt_edit(e: dict) -> str:
    """One line per record, newest last: when, who, stage, what → what, consumers."""
    kind = e.get("kind")
    b, a = e.get("before") or {}, e.get("after") or {}
    spk = e.get("speaker") or {}
    idx = f"#{e['idx']}" if e.get("idx") is not None else ""
    head = f"{e.get('at', '?'):19s}  {str(e.get('who') or '?'):8s} {str(e.get('stage')):9s} {idx:5s}"
    if kind == "set_translation":
        body = (f"[{spk.get('before')} {float(e.get('start') or 0):.1f}s]  {_q(b.get('text_translated'))} → "
                f"{_q(a.get('text_translated'))}")
        if b.get("text_translated_full"):
            body += f"  (compact original {_q(b['text_translated_full'])})"
    elif kind == "set_speaker":
        body = f"speaker {spk.get('before')}→{spk.get('after')}  gender {b.get('gender')}→{a.get('gender')}"
        if e.get("minted"):
            body += "  minted " + ",".join(e["minted"])
        body += "  propagated " + (",".join(e.get("propagated") or []) or "-")
        if e.get("not_propagated"):
            body += "  NOT propagated (drift): " + ",".join(e["not_propagated"])
    elif kind == "add_speaker":
        body = f"new speaker {a.get('speaker')} ({a.get('gender')})"
    elif kind == "set_glossary":
        body = (f"+{len(e.get('added') or [])} ~{len(e.get('changed') or [])} -{len(e.get('removed') or [])}"
                f"  apply={'yes' if e.get('apply_to_translation') else 'no'} applied={e.get('applied', 0)}")
        for ja, old, new in (e.get("changed") or [])[:3]:
            body += f"  {ja}: {old}→{new}"
        if e.get("restamped_stages") is not None:
            body += "  restamped " + (",".join(e["restamped_stages"]) or "-")
    elif kind == "set_face_binding":
        body = f"faces_bind {b.get('faces_bind') or 'none'} → {a.get('faces_bind') or 'none'}"
    elif kind == "set_speaker_profile":
        def _prof(p) -> str:                      # {"profile": "P0", "how": "manual"} | "P0" | None
            if isinstance(p, dict):
                return f"{p.get('profile')} ({p.get('how') or 'auto'})"
            return str(p) if p else "auto"
        body = f"{spk.get('before')} profile {_prof(b.get('profile'))} → {_prof(a.get('profile'))}"
        if not e.get("restamped"):
            body += "  (legacy state: enrol not restamped)"
    elif kind == "update_profile":
        body = f"{spk.get('before')}"
        if e.get("merged_into"):
            body += f" merged into {e['merged_into']}"
        for k in ("name", "ref_audio", "default_for_gender", "manual"):
            if b.get(k) != a.get(k):
                body += f"  {k} {b.get(k)} → {a.get(k)}"
        body += f"  (v{b.get('version')}→v{a.get('version')})"
    else:
        body = json.dumps({k: v for k, v in e.items() if k not in ("t", "at", "kind", "who", "stage")},
                          ensure_ascii=False)[:160]
    tail = ""
    if e.get("edits_n") is not None:
        tail += f"  edits_n={e['edits_n']}"
    if e.get("backup"):
        tail += f"  backup {e['backup']}"
    cons = e.get("consumers")
    if cons:
        tail += "  ⇒ " + ",".join(str(c) for c in cons)
    elif cons == []:
        tail += "  ⇒ nothing cached"
    if e.get("truncated"):
        tail += "  [truncated]"
    return head + " " + body + tail


def show_edits(name: str, n: int = 30, stage: str | None = None, idx: int | None = None,
               who: str | None = None, film: bool = False) -> None:
    all_rows = read_edits(name)
    rows = all_rows
    if stage:
        rows = [r for r in rows if r.get("stage") == stage]
    if idx is not None:
        rows = [r for r in rows if r.get("idx") == idx]
    if who:
        rows = [r for r in rows if r.get("who") == who]
    if not rows:
        print(f"{name}: no manual edits recorded" + (" (with these filters)" if (stage or idx is not None or who) else ""))
        return
    print(f"{name}: {len(rows)} manual edits" + (f" (showing last {n})" if len(rows) > n else ""))
    for e in rows[-n:]:
        print("  " + fmt_edit(e))
    if film:
        return
    # The undo hint is about the newest state edit of the whole log (a backup holds the
    # state before THAT edit), never about the newest line that matched a filter.
    newest = next((r for r in reversed(all_rows) if r.get("backup") and r.get("kind") != "update_profile"), None)
    if newest:
        bk = WS / name / "web" / "state_backups" / newest["backup"]
        run_id, started = last_run_start(name)
        print("  undo (restores state.json only — not asset/glossary.json, web/options.json, "
              "profiles.json or the refreshed deliverables):")
        if not bk.exists():
            print(f"    backup {newest['backup']} is gone (20 kept) — no exact undo")
        else:
            print(f"    cp {bk} {WS / name / 'state.json'}")
            ep = _epoch(started)
            if ep is not None and ep > float(newest.get("t") or 0):
                print(f"    WARNING: run {run_id} started {started}, after this edit — that state would point "
                      "stages at outputs the run overwrote; not an exact undo")


def _edits_line(name: str) -> str | None:
    rows = read_edits(name)
    if not rows:
        return None
    run_id, started = last_run_start(name)
    s = edits_summary(rows, _epoch(started))
    by = ", ".join(f"{k} {v}" for k, v in sorted(s["by_stage"].items()))
    line = f"edits: {s['n']} ({by}), last {s['last_at']} by {s['last_who']}"
    if run_id:
        line += f"; {s['since_last_run']} after the newest run started ({run_id} {started})"
        if s["consumers_since"]:
            line += "  ⇒ to redo: " + ",".join(s["consumers_since"])
    line += "   (python scripts/runs.py " + name + " --edits)"
    return line


def show_project(name: str, n: int = 8) -> None:
    rows = _jsonl(WS / name / "runs" / "index.jsonl")
    if not rows:
        print(f"{name}: no run records (runs before 2026-09-28 were not recorded; see workspace/{name}/run_v3.log)")
    else:
        print(f"{name}: {len(rows)} recorded runs")
        for r in rows[-n:]:
            ran = ",".join(r.get("ran") or []) or "-"
            print(f"  {r['run_id']}  {r['status']:11s} {r.get('seconds', 0):>7.0f}s  commit {r.get('commit')}"
                  f"{'*' if r.get('dirty') else ' '}  ran: {ran}"
                  + (f"  FAILED at {r['failed_stage']}" if r.get("failed_stage") else ""))
    el = _edits_line(name)
    if el:
        print("  " + el)
    if rows and rows[-1]["status"] != "ok":
        show_run(name, rows[-1]["run_id"])


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
        n_ed = len(read_edits(name))
        print(f"  p{i} {c['minutes']:5.1f} min speech {c['speech_minutes']:5.2f}  {status.get(i, '—'):22s} "
              f"final={'yes' if final else 'no ':3s} edits={n_ed:<3d} last run: {last.get('run_id', '—')} {last.get('status', '')}"
              + (f" FAILED at {last['failed_stage']}" if last.get("failed_stage") else ""))
    ev = _jsonl(split / "events.jsonl")
    if ev:
        print("  film events (last 8):")
        for e in ev[-8:]:
            print(f"    {e.get('at')} {e.get('kind'):12s} " + " ".join(f"{k}={v}" for k, v in e.items() if k not in ('t', 'at', 'kind', 'host')))
    fe = read_edits(film)
    if fe:
        print(f"  profile edits ({len(fe)}, last 5; python scripts/runs.py --film {film} --edits):")
        for e in fe[-5:]:
            print("    " + fmt_edit(e))
    print(f"  continue: bash scripts/run_long.sh {plan.get('video', '<video>')} {film}     # finished chunks are skipped")


def main() -> int:
    global WS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?")
    ap.add_argument("--run", default=None)
    ap.add_argument("--film", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--edits", action="store_true", help="manual-edit history of the project / film")
    ap.add_argument("--stage", default=None, help="--edits filter: translate / asr / glossary / faces / enrol")
    ap.add_argument("--idx", type=int, default=None, help="--edits filter: segment index")
    ap.add_argument("--who", default=None, help="--edits filter: user (lan / Remote-User login / cli)")
    ap.add_argument("-n", type=int, default=30, help="--edits: how many records (newest last)")
    ap.add_argument("--workspace", default=None, help="read another workspace root (e.g. a copied probe)")
    args = ap.parse_args()
    if args.workspace:
        WS = Path(args.workspace).resolve()
    if args.film and args.edits:
        show_edits(args.film, n=args.n, who=args.who, film=True)
    elif args.film:
        show_film(args.film)
    elif args.all:
        for d in sorted(WS.iterdir()):
            rows = _jsonl(d / "runs" / "index.jsonl")
            if rows:
                r = rows[-1]
                n_ed = len(read_edits(d.name))
                print(f"{d.name:22s} {r['run_id']} {r['status']:11s} commit {r.get('commit')}"
                      f"  edits {n_ed:<3d}"
                      + (f" FAILED at {r['failed_stage']}" if r.get("failed_stage") else ""))
    elif args.name and args.edits:
        show_edits(args.name, n=args.n, stage=args.stage, idx=args.idx, who=args.who)
    elif args.name and args.run:
        show_run(args.name, args.run)
    elif args.name:
        show_project(args.name)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
