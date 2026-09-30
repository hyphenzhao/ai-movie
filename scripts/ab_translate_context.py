#!/usr/bin/env python
"""A/B for the Sakura draft: speaker tags and prior-turn context (v3.4 T1/T9).

Re-runs ONLY the translate stage (Sakura draft → flagged-line polish →
glossary enforce, the exact ``run_pipeline._translate_by_units`` path with the
same unit grouping and split-back) in memory from each film's existing
state.json — asr segments + glossary — once per arm, and scores every arm
against the Chinese truth already in the repo: inputs/subs/output_test.zh.srt
for the short and workspace/SONE-846/_screen/scan.json for the long film.
Each arm is scored twice, on the Sakura drafts (what tags/context move) and on
the final line (what polish and enforce leave), with eval_against_subs
.pronoun_agreement / eval_long.pronoun_check — the metric behind release gates
H5 and L7.  Nothing under workspace/<film>/ is written; results go to
workspace/_ab/translate_ctx/<arm>/ and nothing downstream is regenerated.

Arms are config overrides on ai_movie.config (the translator reads them lazily):

  A0  today's requests + seed: tags off, 2 prior pairs frozen per block of 4   (baseline, re-run
      rather than taken from the shipped reports because the seed makes it a new baseline)
  A1  speaker tags only (+ the tag rule in the system prompt, inseparable from the tags)
  A2  context only: 4 prior pairs, block-append (SAKURA_CTX_APPEND, block 4)
  A3  tags + block-append context                                              (proposed default)
  A4  tags + exact sliding window of 4 (block 1): same context as A3 minus the prompt cache —
      run it only to measure the cache cost the review predicted (≈3× draft time)

    # shorts, all arms (≈ 8–12 min per arm: 248 units × ~1.2 s + ~50 polish calls + model swaps)
    .venv/bin/python scripts/ab_translate_context.py --arms A0,A1,A2,A3 --films output_test,test_1,test_2
    # long film, baseline and the best short-film arm (≈ 45–60 min per arm; 22 chunks, 1205 units)
    .venv/bin/python scripts/ab_translate_context.py --arms A0,A3 --long SONE-846
    # decision table from what is on disk (no Ollama)
    .venv/bin/python scripts/ab_translate_context.py --decide

Adopt rule (pre-registered; combined = output_test + SONE-846 final lines, all vs A0):
  1. mismatches ≤ 0.8 × A0 (shipped: 11 + 48 = 59 → ≤ 47) AND extra ≤ A0 + 2 (missing must not be
     "fixed" by spraying 你);
  2. draft-level mismatches also below A0 (no arm may win only through polish);
  3. kana_segments ≤ max(A0, 1) per film; fragmented_units not up;
  4. failed draft units (empty ∪ echo ∪ multi-line ∪ tag leak) ≤ 1 % of units;
  5. translate wall time ≤ 1.5 × A0;
  6. the user's spot-check of diff_<arm>_vs_A0_<film>.md finds no wrong-person / wrong-gender line
     that A0 got right.
The winning arm's overrides become the ai_movie.config defaults; all translate fingerprints are
already stale on v3.4-dev, so adoption costs no extra re-run.

Ollama holds Sakura 13 GB + Qwen-polish 24 GB + dolphin-mixtral:8x7b 27 GB (glossary enforce)
resident during a run: start it only when nothing else uses the GPU (the script refuses to start
while a pipeline process is running; --force overrides).
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_against_subs as subs                    # noqa: E402
import run_pipeline as rp                           # noqa: E402
from ai_movie import config as C                    # noqa: E402
from ai_movie import translator                     # noqa: E402
from ai_movie import units as units_mod             # noqa: E402

ENGINE = "sakura+qwen"
OUT_DIR = ROOT / "workspace" / "_ab" / "translate_ctx"
SHORTS = ["output_test", "test_1", "test_2"]

ARMS: dict[str, dict] = {
    "A0": {"SAKURA_SPEAKER_TAGS": False, "SAKURA_CTX_BEFORE": 2, "SAKURA_CTX_APPEND": False, "SAKURA_CTX_BLOCK": 4},
    "A1": {"SAKURA_SPEAKER_TAGS": True,  "SAKURA_CTX_BEFORE": 2, "SAKURA_CTX_APPEND": False, "SAKURA_CTX_BLOCK": 4},
    "A2": {"SAKURA_SPEAKER_TAGS": False, "SAKURA_CTX_BEFORE": 4, "SAKURA_CTX_APPEND": True,  "SAKURA_CTX_BLOCK": 4},
    "A3": {"SAKURA_SPEAKER_TAGS": True,  "SAKURA_CTX_BEFORE": 4, "SAKURA_CTX_APPEND": True,  "SAKURA_CTX_BLOCK": 4},
    "A4": {"SAKURA_SPEAKER_TAGS": True,  "SAKURA_CTX_BEFORE": 4, "SAKURA_CTX_APPEND": True,  "SAKURA_CTX_BLOCK": 1},
}

# adopt-rule constants (see the docstring)
ADOPT_MISMATCH_FRAC = 0.8
ADOPT_EXTRA_SLACK = 2
ADOPT_FAIL_RATE = 0.01
ADOPT_WALL_FRAC = 1.5


def _eval_long():
    spec = importlib.util.spec_from_file_location("eval_long", ROOT / "scripts" / "eval_long.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                    # type: ignore[union-attr]
    return mod


def guard_idle(force: bool) -> None:
    """Refuse to start next to a pipeline run: three Ollama models ≈ 64 GB resident."""
    pat = "run_pipeline.py|run_v3.sh|run_long.sh|run_vc_version.py|run_release.sh"
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    pids = [p for p in r.stdout.split() if p and int(p) != os.getpid()]
    if pids and not force:
        raise SystemExit(f"a pipeline process is running (pids {pids}); wait for the queue or pass --force")


# ── one film, one arm ──────────────────────────────────────────────

class _Shim:
    """Stands in for the translator module inside run_pipeline._translate_by_units,
    keeping every call's draft trace and timing."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def translate_segments(self, segments, **kw):
        trace: list[dict] = []
        t0 = time.time()
        out = translator.translate_segments(segments, trace=trace, **kw)
        self.calls.append({"n": len(segments), "trace": trace, "out": out, "seconds": time.time() - t0})
        return out


_SAKURA_SECONDS = [0.0]
_orig_sakura = translator._sakura_translate


def _timed_sakura(*a, **kw):
    t0 = time.time()
    try:
        return _orig_sakura(*a, **kw)
    finally:
        _SAKURA_SECONDS[0] += time.time() - t0


translator._sakura_translate = _timed_sakura


@contextlib.contextmanager
def arm_config(over: dict):
    saved = {k: getattr(C, k) for k in over}
    for k, v in over.items():
        setattr(C, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(C, k, v)


def drafts_on_segments(segs: list[dict], units: list[list[int]], calls: list[dict],
                       unit_rows: list[dict]) -> list[str]:
    """Per-segment Sakura drafts, split back the way _translate_by_units splits the final line.

    Call 0 translated the non-kept units in order; call 1 (if any) re-translated the members of
    units whose *final* line could not be split.  A unit whose draft cannot be split (but whose
    final could) keeps its whole draft on the first member — a cue-overlap metric joins the
    members anyway."""
    drafts = [""] * len(segs)
    if not calls:
        return drafts
    todo = [k for k, u in enumerate(units) if not segs[u[0]].get("keep_original")]
    main = {r["idx"]: r["zh"] for r in calls[0]["trace"]}
    fallback_units = {r["unit"] for r in unit_rows if r.get("fallback")}
    fb_segs: list[int] = [i for k in sorted(fallback_units) for i in units[k]]
    fb = {r["idx"]: r["zh"] for r in calls[1]["trace"]} if len(calls) > 1 else {}
    for p, k in enumerate(todo):
        u = units[k]
        if k in fallback_units:
            for i in u:
                drafts[i] = fb.get(fb_segs.index(i), "") if i in fb_segs else ""
            continue
        d = main.get(p, "")
        pieces = [d] if len(u) == 1 else units_mod.split_translation(
            d, [units_mod.visible_len(segs[i].get("text")) for i in u])
        if pieces is None:
            pieces = [d] + [""] * (len(u) - 1)
        for i, piece in zip(u, pieces):
            drafts[i] = piece
    return drafts


def run_film(state_path: Path, over: dict) -> dict:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    segs = [dict(s) for s in state["asr"]["segments"]]
    gloss = state.get("glossary") or {}
    units = units_mod.group_units(segs)
    shim = _Shim()
    polish_rows: list[dict] = []
    unit_rows: list[dict] = []
    _SAKURA_SECONDS[0] = 0.0
    t0 = time.time()
    with arm_config(over):
        final = rp._translate_by_units(shim, segs, units, ENGINE, gloss, polish_rows, unit_rows)
    wall = time.time() - t0
    drafts = drafts_on_segments(segs, units, shim.calls, unit_rows)
    rows = []
    for i, (s, zh, d) in enumerate(zip(segs, final, drafts)):
        s["text_translated"] = zh
        s["draft"] = d
        rows.append({"idx": i, "start": s["start"], "end": s["end"], "speaker": s.get("speaker"),
                     "gender": s.get("gender"), "overlap": s.get("overlap"), "unit": s.get("unit_id"),
                     "keep_original": bool(s.get("keep_original")), "tag": translator._speaker_tag(s),
                     "text": s.get("text"), "draft": d, "zh": zh})
    trace = [dict(r, call=n) for n, c in enumerate(shim.calls) for r in c["trace"]]
    return {"segments": segs, "rows": rows, "units": units, "polish_rows": polish_rows,
            "unit_rows": unit_rows, "trace": trace, "wall": wall, "sakura_wall": _SAKURA_SECONDS[0],
            "health": health(trace), "draft_f1": sum(1 for r in polish_rows if "F1_pronoun" in r["flags"]),
            "polish_mix": {st: sum(1 for r in polish_rows if "F1_pronoun" in r["flags"] and r["status"] == st)
                           for st in ("accepted", "rejected", "unchanged", "error")}}


def health(trace: list[dict]) -> dict:
    """Draft-side failure counters; ``failed`` is the union the adopt rule caps at 1 %."""
    failed = [r for r in trace if r.get("empty") or r.get("echo") or r.get("tag_leak") or (r.get("n_lines") or 0) > 1]
    return {"units": len(trace), "empty": sum(r.get("empty", 0) for r in trace),
            "echo": sum(r.get("echo", 0) for r in trace), "tag_leak": sum(r.get("tag_leak", 0) for r in trace),
            "multi_line": sum(1 for r in trace if (r.get("n_lines") or 0) > 1),
            "retry_bare": sum(r.get("retry_bare", 0) for r in trace),
            "failed": len(failed), "fail_rate": len(failed) / max(1, len(trace))}


# ── scoring ────────────────────────────────────────────────────────

def score_short(film: str, segs: list[dict], key: str) -> dict:
    """eval_against_subs metrics with *key* as the Chinese line (drafts or final)."""
    scored = [dict(s, text_translated=s.get(key) or "") for s in segs]
    gt = ROOT / "inputs" / "subs" / f"{film}.gt.json"
    srt = ROOT / "inputs" / "subs" / f"{film}.zh.srt"
    cues = subs.load_gt(gt) if gt.exists() else []
    groups = subs.build_groups(scored, cues)
    if srt.exists():
        subs.attach_reference_zh(groups, subs.parse_srt(srt))
    s = subs.summarise(groups, scored)
    out = {k: s[k] for k in ("pronoun_units", "kana_segments", "fragmented_units", "n_units")}
    if s.get("pronoun_agreement"):
        out["pronoun"] = {k: v for k, v in s["pronoun_agreement"].items() if k != "examples"}
        sims = [g["zh_sim"] for g in groups if g.get("zh_sim") is not None]
        out["zh_sim_mean"] = round(sum(sims) / len(sims), 4) if sims else None
    return out


def score_long(film: str, by_chunk: dict[int, list[dict]], key: str) -> dict:
    el = _eval_long()
    fsegs, _ = el.film_segments(film, "translate", segments_by_chunk=by_chunk)
    truth, _ = el.cues(film)
    pa = el.pronoun_check(truth, fsegs, key=key)
    tm = {"pronoun_units": 0, "kana_segments": 0, "fragmented_units": 0, "n_units": 0}
    for segs in by_chunk.values():
        m = subs.translation_metrics([dict(s, text_translated=s.get(key) or "") for s in segs])
        for k in tm:
            tm[k] += m[k]
    return {**tm, "pronoun": {k: v for k, v in pa.items() if k != "examples"}}


# ── reports ────────────────────────────────────────────────────────

def _load(arm: str, film: str) -> dict | None:
    p = OUT_DIR / arm / f"{film}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _fmt_p(p: dict | None) -> str:
    if not p:
        return "—"
    return f"{p['mismatch']} ({p['extra']}+{p['missing']}) /{p['scored']}"


def write_summary(arms: list[str], films: list[str]) -> str:
    lines = ["# Sakura draft A/B — speaker tags / prior-turn context", "",
             "mismatch = extra + missing pronoun classes (你/您、他、她) vs the reference Chinese; "
             "`draft` = Sakura output before polish, `final` = after polish + glossary enforce.", ""]
    for film in films:
        lines += [f"## {film}", "",
                  "| arm | config | draft mismatch (extra+missing)/scored | final mismatch | draft F1 rows | "
                  "F1 polish acc/rej/unch | pronoun_units | kana | fragmented | failed units | wall s (sakura s) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in arms:
            r = _load(arm, film)
            if not r:
                lines.append(f"| {arm} | {ARMS[arm]} | (not run) | | | | | | | | |")
                continue
            sd, sf, h, pm = r["scores"]["draft"], r["scores"]["final"], r["health"], r["polish_mix"]
            cfg = ARMS.get(arm, {})
            lines.append(
                f"| {arm} | tags={'on' if cfg.get('SAKURA_SPEAKER_TAGS') else 'off'} "
                f"ctx={cfg.get('SAKURA_CTX_BEFORE')}{'+append' if cfg.get('SAKURA_CTX_APPEND') else ''} "
                f"block={cfg.get('SAKURA_CTX_BLOCK')} | {_fmt_p(sd.get('pronoun'))} | {_fmt_p(sf.get('pronoun'))} | "
                f"{r['draft_f1']} | {pm['accepted']}/{pm['rejected']}/{pm['unchanged']} | "
                f"{sd['pronoun_units']}→{sf['pronoun_units']} | {sd['kana_segments']}→{sf['kana_segments']} | "
                f"{sd['fragmented_units']}→{sf['fragmented_units']} | "
                f"{h['failed']}/{h['units']} (empty {h['empty']}, echo {h['echo']}, leak {h['tag_leak']}, "
                f"multi {h['multi_line']}, retry {h['retry_bare']}) | {r['wall']:.0f} ({r['sakura_wall']:.0f}) |")
        lines.append("")
    lines += decide(arms, films)
    text = "\n".join(lines)
    (OUT_DIR / "summary.md").write_text(text, encoding="utf-8")
    return text


def decide(arms: list[str], films: list[str]) -> list[str]:
    """Evaluate the pre-registered adopt rule for every arm against A0 on the films present."""
    out = ["## Adopt rule vs A0 (films with a Chinese truth only: output_test, SONE-846)", ""]
    a0 = {f: _load("A0", f) for f in films}
    if not any(a0.values()):
        return out + ["A0 not run yet."]
    truth_films = [f for f in films if a0.get(f) and a0[f]["scores"]["final"].get("pronoun")]
    for arm in arms:
        if arm == "A0":
            continue
        rs = {f: _load(arm, f) for f in films}
        common = [f for f in films if rs.get(f) and a0.get(f)]
        if not common:
            out.append(f"- {arm}: not run")
            continue
        tf = [f for f in common if f in truth_films]

        def tot(res, which, k):
            return sum(res[f]["scores"][which]["pronoun"][k] for f in tf)

        def sm(res, which, k):
            return sum(res[f]["scores"][which][k] for f in common)

        checks = []
        if tf:
            m0, m1 = tot(a0, "final", "mismatch"), tot(rs, "final", "mismatch")
            e0, e1 = tot(a0, "final", "extra"), tot(rs, "final", "extra")
            d0, d1 = tot(a0, "draft", "mismatch"), tot(rs, "draft", "mismatch")
            checks.append((m1 <= ADOPT_MISMATCH_FRAC * m0, f"final mismatch {m1} ≤ {ADOPT_MISMATCH_FRAC}×{m0}={ADOPT_MISMATCH_FRAC * m0:.1f}"))
            checks.append((e1 <= e0 + ADOPT_EXTRA_SLACK, f"final extra {e1} ≤ {e0}+{ADOPT_EXTRA_SLACK}"))
            checks.append((d1 < d0, f"draft mismatch {d1} < {d0}"))
        else:
            checks.append((False, "no film with a Chinese truth in this arm yet"))
        k0 = {f: a0[f]["scores"]["final"]["kana_segments"] for f in common}
        k1 = {f: rs[f]["scores"]["final"]["kana_segments"] for f in common}
        checks.append((all(k1[f] <= max(k0[f], 1) for f in common), f"kana {k1} ≤ max(A0 {k0}, 1)"))
        checks.append((sm(rs, "final", "fragmented_units") <= sm(a0, "final", "fragmented_units"),
                       f"fragmented {sm(rs, 'final', 'fragmented_units')} ≤ {sm(a0, 'final', 'fragmented_units')}"))
        fu = sum(rs[f]["health"]["failed"] for f in common)
        nu = sum(rs[f]["health"]["units"] for f in common)
        checks.append((fu <= ADOPT_FAIL_RATE * nu, f"failed units {fu}/{nu} ≤ {ADOPT_FAIL_RATE:.0%}"))
        w0, w1 = sum(a0[f]["wall"] for f in common), sum(rs[f]["wall"] for f in common)
        checks.append((w1 <= ADOPT_WALL_FRAC * w0, f"wall {w1:.0f}s ≤ {ADOPT_WALL_FRAC}×{w0:.0f}s"))
        ok = all(c for c, _ in checks)
        partial = "" if set(common) == set(films) else f" (partial: {common})"
        out.append(f"- **{arm}: {'ADOPTABLE' if ok else 'not adoptable'}**{partial} — pending the user's diff spot-check")
        out += [f"  - {'✅' if c else '❌'} {msg}" for c, msg in checks]
    return out


def write_diff(arm: str, film: str) -> None:
    a0, r = _load("A0", film), _load(arm, film)
    if not a0 or not r or arm == "A0":
        return
    lines = [f"# {film}: {arm} vs A0 — units whose Sakura draft changed", "",
             "Only changed lines are listed (tag = what the arm's prompt carried). Read for wrong person / "
             "wrong gender that A0 got right.", ""]
    n = 0
    for x, y in zip(a0["rows"], r["rows"]):
        if x["draft"] == y["draft"] and x["zh"] == y["zh"]:
            continue
        n += 1
        lines += [f"**#{y['idx']} [{y['start']:.1f}s] {y['tag'] or y['speaker'] or ''}** {y['text']}",
                  f"- A0 draft：{x['draft']}" + (f"　→ final：{x['zh']}" if x["zh"] != x["draft"] else ""),
                  f"- {arm} draft：{y['draft']}" + (f"　→ final：{y['zh']}" if y["zh"] != y["draft"] else ""), ""]
    lines.insert(3, f"{n} of {len(r['rows'])} lines differ.")
    (OUT_DIR / f"diff_{arm}_vs_A0_{film}.md").write_text("\n".join(lines), encoding="utf-8")


# ── driver ─────────────────────────────────────────────────────────

def run_short(arm: str, film: str, redo: bool) -> None:
    out = OUT_DIR / arm / f"{film}.json"
    if out.exists() and not redo:
        print(f"[{arm}/{film}] exists, skipping (--redo to repeat)", flush=True)
        return
    state = ROOT / "workspace" / film / "state.json"
    print(f"[{arm}/{film}] translating {ARMS[arm]}", flush=True)
    r = run_film(state, ARMS[arm])
    r["scores"] = {"draft": score_short(film, r["segments"], "draft"),
                   "final": score_short(film, r["segments"], "text_translated")}
    del r["segments"]
    r.update({"arm": arm, "film": film, "config": ARMS[arm], "engine": ENGINE})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{arm}/{film}] wall {r['wall']:.0f}s draft {_fmt_p(r['scores']['draft'].get('pronoun'))} "
          f"final {_fmt_p(r['scores']['final'].get('pronoun'))} failed {r['health']['failed']}", flush=True)


def run_long(arm: str, film: str, redo: bool) -> None:
    out = OUT_DIR / arm / f"{film}.json"
    if out.exists() and not redo:
        print(f"[{arm}/{film}] exists, skipping (--redo to repeat)", flush=True)
        return
    plan = json.loads((ROOT / "workspace" / film / "_split" / "plan.json").read_text(encoding="utf-8"))
    chunk_dir = OUT_DIR / arm / film
    chunk_dir.mkdir(parents=True, exist_ok=True)
    by_chunk: dict[int, list[dict]] = {}
    chunks: dict[str, dict] = {}
    traces: list[dict] = []
    for c in plan["chunks"]:
        i = c["index"]
        cp = chunk_dir / f"p{i:02d}.json"
        state = ROOT / "workspace" / f"{film}_p{i:02d}" / "state.json"
        if not state.exists():
            print(f"[{arm}/{film}] p{i:02d}: no state, skipped", flush=True)
            continue
        if cp.exists() and not redo:
            r = json.loads(cp.read_text(encoding="utf-8"))
        else:
            print(f"[{arm}/{film}] p{i:02d} translating", flush=True)
            r = run_film(state, ARMS[arm])
            r["chunk_segments"] = [{k: s.get(k) for k in ("start", "end", "text", "text_translated", "draft",
                                                           "keep_original", "speaker", "gender", "overlap")}
                                   for s in r.pop("segments")]
            cp.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        by_chunk[i] = r["chunk_segments"]
        traces += r["trace"]
        chunks[f"p{i:02d}"] = {k: r[k] for k in ("wall", "sakura_wall", "health", "draft_f1", "polish_mix")}
    tot = {"arm": arm, "film": film, "config": ARMS[arm], "engine": ENGINE, "chunks": chunks,
           "wall": sum(c["wall"] for c in chunks.values()),
           "sakura_wall": sum(c["sakura_wall"] for c in chunks.values()),
           "draft_f1": sum(c["draft_f1"] for c in chunks.values()),
           "polish_mix": {k: sum(c["polish_mix"][k] for c in chunks.values())
                          for k in ("accepted", "rejected", "unchanged", "error")},
           "health": health(traces),
           "rows": [dict(s, idx=f"p{i:02d}#{n}", zh=s.get("text_translated") or "", tag=translator._speaker_tag(s))
                    for i in sorted(by_chunk) for n, s in enumerate(by_chunk[i])],
           "scores": {"draft": score_long(film, by_chunk, "draft"),
                      "final": score_long(film, by_chunk, "text_translated")}}
    out.write_text(json.dumps(tot, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{arm}/{film}] wall {tot['wall']:.0f}s draft {_fmt_p(tot['scores']['draft']['pronoun'])} "
          f"final {_fmt_p(tot['scores']['final']['pronoun'])} failed {tot['health']['failed']}", flush=True)


def install_fake_ollama(film_states: list[Path]) -> None:
    """--fake: answer every Ollama call from the shipped translations (plumbing check, no GPU)."""
    lookup: dict[str, str] = {}
    for p in film_states:
        st = json.loads(p.read_text(encoding="utf-8"))
        segs = (st.get("translate") or {}).get("segments") or []
        for u in units_mod.group_units(segs):            # the same pseudo units the harness will ask for
            ja = "".join((segs[i].get("text") or "") for i in u).strip()
            zh = "".join((segs[i].get("text_translated") or "") for i in u).strip()
            if ja and zh:
                lookup[ja] = zh
        for s in segs:
            if s.get("text") and s.get("text_translated"):
                lookup.setdefault(s["text"].strip(), s["text_translated"])

    def fake(model, messages, base_url, timeout=600, options=None, think=None):
        last = messages[-1]["content"]
        if "待改正：" in last or "草稿：" in last:
            return ""                                   # polish/enforce: leave the draft alone
        src = last.split("\n")[-1]
        src = src.split("]", 1)[1] if src.startswith("[") and "]" in src else src
        return lookup.get(src.strip(), "嗯。")

    translator._call_ollama_chat = fake

    @contextlib.contextmanager
    def no_engine(*a, **kw):
        yield
    translator.exclusive_engine = no_engine


def main() -> int:
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="A0,A1,A2,A3")
    ap.add_argument("--films", default=",".join(SHORTS), help="short films (workspace/<name>/state.json); '' for none")
    ap.add_argument("--long", default=None, help="long film split into workspace/<film>_pNN chunks")
    ap.add_argument("--redo", action="store_true", help="repeat arms/films that already have results")
    ap.add_argument("--decide", action="store_true", help="only rebuild summary.md / diffs from disk")
    ap.add_argument("--force", action="store_true", help="start even if a pipeline process is running")
    ap.add_argument("--fake", action="store_true", help="no Ollama: answer from the shipped translations (plumbing test)")
    ap.add_argument("--out", type=Path, default=None, help=f"results dir (default {OUT_DIR})")
    args = ap.parse_args()

    if args.out:
        OUT_DIR = args.out
    arms = [a for a in args.arms.split(",") if a]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {list(ARMS)}")
    films = [f for f in args.films.split(",") if f]
    all_films = films + ([args.long] if args.long else [])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.decide:
        if not args.fake:
            guard_idle(args.force)
        if args.fake:
            states = [ROOT / "workspace" / f / "state.json" for f in films]
            if args.long:
                states += sorted((ROOT / "workspace").glob(f"{args.long}_p*/state.json"))
            install_fake_ollama(states)
        for arm in arms:
            for film in films:
                run_short(arm, film, args.redo)
            if args.long:
                run_long(arm, args.long, args.redo)
    for arm in arms:
        for film in all_films:
            write_diff(arm, film)
    print(write_summary(arms, all_films))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
