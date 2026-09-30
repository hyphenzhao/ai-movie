#!/usr/bin/env python
"""Text-level A/B of the single-sentence rewrite model: compact + glossary enforce.

v3.4 (T5) points ``COMPACT_MODEL`` and ``GLOSSARY_ENFORCE_MODEL`` at the
flagged-line polish model (Qwen3.6, thinking-capable, ``think=False``)
instead of dolphin-mixtral:8x7b.  Nothing in the pipeline measured either
task: the compact stage only records what survived re-synthesis
(``no_gain_reverted`` is an audio verdict, not a text one) and enforce
recorded nothing at all.  This harness replays both tasks from the persisted
states against Ollama only — no TTS, no pipeline — and reports *text-level*
acceptance per model, which is the only thing the model swap can change.

Compact cases come from ``state["compact"]["report"]``: one attempt per
``rN:`` note that is not ``within_budget`` (14 rows / 16 attempts on the
three short films, 37 rows / 36 attempts on the 22 SONE-846 chunks).  The
round-1 budget is recomputed from the tts segments exactly as
``run_pipeline.step_compact`` does (the stored ``budget`` is the LAST
round's, ×0.8 in round 2); a round-2 attempt takes the stored round-2
budget and, as input, this arm's own round-1 candidate (the pipeline would
have fed the re-synthesized round-1 text) or the full line when round 1
returned nothing.

Enforce cases are a corruption set: no real in-the-wild pin miss exists once
occurrences are counted boundary-aware (the three "misses" were 「かんな」
inside 「わかんない」), so every line whose source names a pinned term and
whose translation carries the pin has the pin replaced by each documented
wrong rendering (卡娜 / 加奈 / 卡恩娜 / 小蓝华 / 镰鼬 / 小勘娜 / 小卡娜 for 坎娜,
电视剧 for 短剧, …) and ``enforce_glossary`` must put it back without
touching anything else.  Lines that recur across films (output_test is
SONE-846_p01's footage) are counted once.

Arms are model × repeat_penalty: the rewrite channels now send
``repeat_penalty 1.0`` (the 1.15 default penalised copying the sentence the
model was told to keep), so 1.15 is offered as the baseline arm.

    # nothing else may use Ollama meanwhile: a pipeline stage evicts the model mid-call
    .venv/bin/python scripts/ab_compact_enforce.py --dry-run            # case counts only, no Ollama
    .venv/bin/python scripts/ab_compact_enforce.py \\
        --models dolphin-mixtral:8x7b,"$(python -c 'from ai_movie.config import OLLAMA_POLISH_MODEL as m; print(m)')" \\
        --repeat-penalty 1.0,1.15

Writes ``workspace/_ab/compact_enforce/{cases_compact,cases_enforce,compact,enforce}.csv``,
``summary.json`` and ``summary.md`` (arm × task table plus the adopt rule).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_movie import translator as T                       # noqa: E402
from ai_movie.config import (                              # noqa: E402
    COMPACT_MIN_CHARS, COMPACT_TARGET_RATIO, OLLAMA_BASE_URL, OLLAMA_POLISH_MODEL,
    WORKSPACE_DIR,
)
from ai_movie.glossary import _term_pattern                # noqa: E402

SHORTS = ["output_test", "test_1", "test_2"]
BASELINE_MODEL = "dolphin-mixtral:8x7b"

# Wrong renderings measured on the release films (glossary.protect_terms /
# enforce_glossary docstrings, config.GLOSSARY_ENFORCE_MODEL) plus one
# plausible slip per remaining pin.  A pin absent here is skipped (counted).
WRONG_RENDERINGS: dict[str, list[str]] = {
    "坎娜": ["卡娜", "加奈", "卡恩娜", "小蓝华", "镰鼬", "小勘娜", "小卡娜"],
    "短剧": ["电视剧", "连续剧"],
    "梅拉尼斯": ["美拉尼斯", "梅拉妮丝"],
    "鸡鸡": ["小鸡鸡", "鸡巴"],
    "焦点": ["对焦", "焦距"],
    "S1": ["S一"],
}

_NOTE_RE = re.compile(r"^r(\d+):([a-z_]+)")
_QUOTE_RE = re.compile(r"[「」『』“”\"']")


# ── case sets (pure: no Ollama) ─────────────────────────────────────

def _load_state(film: str) -> dict | None:
    p = WORKSPACE_DIR / film / "state.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def compact_cases(film: str, state: dict) -> tuple[list[dict], list[str]]:
    """One case per (row, round) attempt of the persisted compact report."""
    from ai_movie.composer import _visible_chars, segment_slots, speaker_sec_per_char

    notes_out: list[str] = []
    comp = state.get("compact") or {}
    rows = comp.get("report") or []
    if not rows:
        return [], notes_out
    tts_segs = state["tts"]["segments"]
    csegs = comp.get("segments") or tts_segs
    slots = segment_slots(tts_segs)
    spc = speaker_sec_per_char(tts_segs)
    cases = []
    for r in sorted(rows, key=lambda r: r["idx"]):
        i = int(r["idx"])
        attempts = []
        for note in (r.get("notes") or "").split(";"):
            m = _NOTE_RE.match(note.strip())
            if m and m.group(2) != "within_budget":
                attempts.append((int(m.group(1)), m.group(2)))
        if not attempts:
            continue
        rate = spc.get(tts_segs[i].get("speaker") or "", 0.24)
        b1 = max(COMPACT_MIN_CHARS, int(slots[i] * COMPACT_TARGET_RATIO / max(rate, 0.05)))
        stored = int(r.get("budget") or 0)
        if int(r.get("round") or 1) == 1 and stored and stored != b1:
            notes_out.append(f"{film} #{i}: recomputed round-1 budget {b1} != stored {stored}; using stored")
            b1 = stored
        text_full = r.get("text_full") or csegs[i].get("text_translated_full") \
            or csegs[i].get("text_translated") or ""
        ctx = [(csegs[j].get("text_translated_full") or csegs[j].get("text_translated") or "")
               for j in range(max(0, i - 2), i)]
        for rnd, kind in attempts:
            budget = b1 if rnd == 1 else (stored if int(r.get("round") or 1) == 2
                                          else max(COMPACT_MIN_CHARS, int(b1 * 0.8)))
            cases.append({
                "film": film, "idx": i, "round": rnd, "speaker": tts_segs[i].get("speaker"),
                "slot": r.get("slot"), "budget": budget, "ja": tts_segs[i].get("text") or "",
                "text_full": text_full, "chars_full": _visible_chars(text_full),
                "context": ctx, "pipeline_note": kind, "glossary": state.get("glossary") or {},
            })
    return cases, notes_out


def enforce_cases(film: str, state: dict) -> tuple[list[dict], list[dict], list[str]]:
    """(corruption cases, real in-the-wild misses, skipped pins) for one film."""
    gloss = state.get("glossary") or {}
    segs = (state.get("translate") or {}).get("segments") or []
    cases, misses, skipped = [], [], []
    for i, s in enumerate(segs):
        src = s.get("text") or ""
        zh = s.get("text_translated") or ""
        if not src or not zh:
            continue
        pins = sorted({v["zh"] for k, v in gloss.items()
                       if v.get("zh") and k and _term_pattern(k).search(src)})
        for p in pins:
            if p not in zh:
                misses.append({"film": film, "idx": i, "ja": src, "zh": zh, "pin": p})
                continue
            wrongs = WRONG_RENDERINGS.get(p)
            if not wrongs:
                skipped.append(f"{film} #{i}: no wrong rendering listed for pin {p!r}")
                continue
            for w in wrongs:
                cases.append({"film": film, "idx": i, "ja": src, "zh_true": zh, "pin": p,
                              "wrong": w, "zh_corrupt": zh.replace(p, w), "glossary": gloss})
    return cases, misses, skipped


def build_cases(films: list[str], dedupe: bool = True) -> dict:
    out = {"compact": [], "enforce": [], "misses": [], "notes": [], "films": [],
           "report_rows": {}}
    seen: set[tuple] = set()
    for film in films:
        st = _load_state(film)
        if st is None:
            out["notes"].append(f"{film}: no state.json")
            continue
        out["films"].append(film)
        out["report_rows"][film] = len(((st.get("compact") or {}).get("report")) or [])
        cc, nn = compact_cases(film, st)
        out["compact"].extend(cc)
        out["notes"].extend(nn)
        ec, mm, sk = enforce_cases(film, st)
        out["misses"].extend(mm)
        out["notes"].extend(sk)
        for c in ec:
            key = (c["ja"], c["zh_true"], c["pin"], c["wrong"])
            if dedupe and key in seen:
                continue
            seen.add(key)
            out["enforce"].append(c)
    return out


def case_counts(cases: dict) -> dict:
    def _grp(f):
        return "shorts" if f in SHORTS else "sone"
    cc, ec = cases["compact"], cases["enforce"]
    rr = cases.get("report_rows") or {}
    return {
        # every row of 03_compact_report.csv, including within_budget-only ones (14 + 37)
        "compact_rows": sum(rr.values()),
        "compact_rows_shorts": sum(n for f, n in rr.items() if _grp(f) == "shorts"),
        "compact_rows_sone": sum(n for f, n in rr.items() if _grp(f) == "sone"),
        # rows that actually asked the model at least once
        "compact_rows_attempted": len({(c["film"], c["idx"]) for c in cc}),
        "compact_attempts": len(cc),
        "compact_attempts_shorts": sum(1 for c in cc if _grp(c["film"]) == "shorts"),
        "compact_attempts_sone": sum(1 for c in cc if _grp(c["film"]) == "sone"),
        "enforce_cases": len(ec),
        "enforce_lines": len({(c["ja"], c["zh_true"], c["pin"]) for c in ec}),
        "enforce_real_misses": len(cases["misses"]),
    }


# ── running one arm ───────────────────────────────────────────────

class _Recorder:
    """Wrap translator._call_ollama_chat to keep raw replies and latency per call."""

    def __init__(self):
        self.calls: list[dict] = []
        self._orig = T._call_ollama_chat

    def __enter__(self):
        rec = self

        def wrapped(model, messages, base_url, timeout=600, options=None, think=None):
            t0 = time.time()
            try:
                raw = rec._orig(model, messages, base_url, timeout=timeout,
                                options=options, think=think)
            except Exception as exc:                        # noqa: BLE001
                rec.calls.append({"raw": f"ERROR {type(exc).__name__}: {exc}",
                                  "secs": time.time() - t0, "error": True})
                raise
            rec.calls.append({"raw": raw, "secs": time.time() - t0, "error": False})
            return raw
        T._call_ollama_chat = wrapped
        return self

    def __exit__(self, *exc):
        T._call_ollama_chat = self._orig
        return False

    def take(self) -> list[dict]:
        calls, self.calls = self.calls, []
        return calls


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def run_arm(model: str, rp: float, cases: dict, only: str | None,
            log) -> tuple[list[dict], list[dict]]:
    from ai_movie.composer import _visible_chars

    opts = {"repeat_penalty": rp}
    crow: list[dict] = []
    erow: list[dict] = []
    with T.exclusive_engine("ollama", ollama_model=model, base_url=OLLAMA_BASE_URL, log_cb=log):
        # warm-up: the first call pays the model load, keep it out of the latencies
        try:
            T._call_ollama_chat(model, [{"role": "user", "content": "你好"}], OLLAMA_BASE_URL,
                                timeout=900, think=False, options={"num_predict": 4})
        except Exception as exc:                            # noqa: BLE001
            log(f"warm-up failed for {model}: {exc}")
        with _Recorder() as rec:
            if only in (None, "compact"):
                r1_out: dict[tuple, str | None] = {}
                for c in cases["compact"]:
                    key = (c["film"], c["idx"])
                    if c["round"] == 1:
                        text_in = c["text_full"]
                    else:
                        text_in = r1_out.get(key) or c["text_full"]
                    t0 = time.time()
                    cand = T.compact_translation(c["ja"], text_in, c["budget"],
                                                 glossary=c["glossary"], context=c["context"],
                                                 model=model, base_url=OLLAMA_BASE_URL,
                                                 options=opts)
                    calls = rec.take()
                    if c["round"] == 1:
                        r1_out[key] = cand
                    raws = [k["raw"] for k in calls]
                    crow.append({
                        "arm": f"{model}@rp{rp}", "model": model, "repeat_penalty": rp,
                        "film": c["film"], "idx": c["idx"], "round": c["round"],
                        "speaker": c["speaker"], "budget": c["budget"],
                        "chars_in": _visible_chars(text_in), "text_in": text_in, "ja": c["ja"],
                        "candidate": cand or "", "accepted": cand is not None,
                        "chars_out": _visible_chars(cand) if cand else "",
                        "within_budget": bool(cand) and _visible_chars(cand) <= c["budget"],
                        "kana_raw": any(T._KANA_RE.search(r) for r in raws),
                        "quote_raw": any(_QUOTE_RE.search(r) for r in raws),
                        "empty_raw": any(not r.strip() for r in raws),
                        "tries": len(calls), "errors": sum(1 for k in calls if k["error"]),
                        "secs": round(time.time() - t0, 2),
                        "raw_first": (raws[0] if raws else "")[:80].replace("\n", "⏎"),
                        "pipeline_note": c["pipeline_note"],
                    })
                    log(f"  compact {c['film']} #{c['idx']} r{c['round']} budget {c['budget']}: "
                        f"{'accepted' if cand else 'None'} {cand or ''}")
            if only in (None, "enforce"):
                for c in cases["enforce"]:
                    rows: list[dict] = []
                    t0 = time.time()
                    out = T.enforce_glossary([{"text": c["ja"], "seg_idx": c["idx"]}],
                                             [c["zh_corrupt"]], c["glossary"], model=model,
                                             base_url=OLLAMA_BASE_URL, report=rows, options=opts)
                    calls = rec.take()
                    row = rows[0] if rows else {"status": "not_attempted", "candidate": ""}
                    got = out[0]
                    exact = got == c["zh_true"]
                    collateral = _levenshtein(got.replace(c["pin"], ""),
                                              c["zh_true"].replace(c["pin"], ""))
                    raw = calls[0]["raw"] if calls else ""
                    erow.append({
                        "arm": f"{model}@rp{rp}", "model": model, "repeat_penalty": rp,
                        "film": c["film"], "idx": c["idx"], "pin": c["pin"], "wrong": c["wrong"],
                        "ja": c["ja"], "zh_true": c["zh_true"], "zh_corrupt": c["zh_corrupt"],
                        "candidate": row.get("candidate", ""), "status": row["status"],
                        "accepted": row["status"] == "accepted", "exact_restore": exact,
                        "collateral": collateral if row["status"] == "accepted" else "",
                        "kana_raw": bool(T._KANA_RE.search(raw)),
                        "quote_raw": bool(_QUOTE_RE.search(raw)),
                        "empty_raw": not raw.strip(),
                        "secs": round(time.time() - t0, 2),
                        "raw_first": raw[:80].replace("\n", "⏎"),
                    })
                    log(f"  enforce {c['film']} #{c['idx']} {c['wrong']}→{c['pin']}: "
                        f"{row['status']} {row.get('candidate', '')}")
    return crow, erow


# ── summary ───────────────────────────────────────────────────────

def _median(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(statistics.median(xs), 2) if xs else None


def summarise(crows: list[dict], erows: list[dict], counts: dict) -> dict:
    arms = sorted({r["arm"] for r in crows} | {r["arm"] for r in erows})
    out = {"counts": counts, "arms": {}}
    for arm in arms:
        c = [r for r in crows if r["arm"] == arm]
        e = [r for r in erows if r["arm"] == arm]
        cs = [r for r in c if r["film"] in SHORTS]
        cso = [r for r in c if r["film"] not in SHORTS]
        acc = [r for r in e if r["accepted"]]
        out["arms"][arm] = {
            "compact": {
                "attempts": len(c), "accepted": sum(r["accepted"] for r in c),
                "accepted_shorts": f"{sum(r['accepted'] for r in cs)}/{len(cs)}",
                "accepted_sone": f"{sum(r['accepted'] for r in cso)}/{len(cso)}",
                "within_budget": sum(r["within_budget"] for r in c),
                "median_chars_in": _median([r["chars_in"] for r in c]),
                "median_chars_out": _median([r["chars_out"] for r in c if r["accepted"]]),
                "kana_raw": sum(r["kana_raw"] for r in c),
                "quote_raw": sum(r["quote_raw"] for r in c),
                "empty_raw": sum(r["empty_raw"] for r in c),
                "errors": sum(r["errors"] for r in c),
                "median_secs": _median([r["secs"] for r in c]),
            },
            "enforce": {
                "cases": len(e), "accepted": len(acc),
                "rejected_no_pin": sum(r["status"] == "rejected_no_pin" for r in e),
                "rejected_rewrote": sum(r["status"] == "rejected_rewrote" for r in e),
                "error": sum(r["status"] == "error" for r in e),
                "exact_restore": sum(r["exact_restore"] for r in e),
                "accepted_with_collateral": sum(1 for r in acc if r["collateral"]),
                "kana_raw": sum(r["kana_raw"] for r in e),
                "quote_raw": sum(r["quote_raw"] for r in e),
                "empty_raw": sum(r["empty_raw"] for r in e),
                "median_secs": _median([r["secs"] for r in e]),
            },
        }
    return out


ADOPT_RULE = """\
Pre-registered adopt rule (text level; both tasks pooled over the three shorts and the
SONE-846 chunks, candidate = the OLLAMA_POLISH_MODEL arm at repeat_penalty 1.0):

1. compact: candidate `accepted` >= baseline `accepted` (dolphin-mixtral:8x7b at the same
   repeat_penalty) AND candidate `within_budget` >= baseline `within_budget`.
2. enforce: candidate `accepted` >= baseline `accepted` AND candidate `exact_restore`
   >= baseline `exact_restore` (the pin edit must not cost collateral edits).
3. Sanity: candidate `empty_raw` == 0 (an empty reply means think=False is not reaching the
   model) and `errors`/`error` == 0.

All three hold  -> keep COMPACT_MODEL = GLOSSARY_ENFORCE_MODEL = OLLAMA_POLISH_MODEL.
Any fails       -> set both back to the "dolphin-mixtral:8x7b" literal (think=False and
                   repeat_penalty 1.0 stay: harmless to a non-thinking model).
repeat_penalty: keep 1.0 if, for the adopted model, accepted@1.0 >= accepted@1.15 on both
tasks; otherwise drop the override from translator._REWRITE_OPTIONS.
Note: `no_gain_reverted` counts in the pipeline's 03_compact_report.csv are audio verdicts
after re-synthesis and are NOT comparable with these text-level numbers.
"""


def write_summary_md(summary: dict, path: Path) -> None:
    c = summary["counts"]
    lines = ["# compact + enforce text-level A/B", "",
             f"cases: compact {c['compact_rows']} rows / {c['compact_attempts']} attempts "
             f"(shorts {c['compact_rows_shorts']}/{c['compact_attempts_shorts']}, "
             f"SONE-846 {c['compact_rows_sone']}/{c['compact_attempts_sone']}); "
             f"enforce {c['enforce_cases']} corruption cases on {c['enforce_lines']} distinct "
             f"pinned lines; real in-the-wild misses: {c['enforce_real_misses']}", "",
             "| arm | compact acc | shorts | SONE | within budget | empty/kana/quote raw | s/line "
             "| enforce acc | no_pin | rewrote | error | exact | collateral | s/line |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, a in summary["arms"].items():
        cc, ee = a["compact"], a["enforce"]
        lines.append(
            f"| {arm} | {cc['accepted']}/{cc['attempts']} | {cc['accepted_shorts']} | "
            f"{cc['accepted_sone']} | {cc['within_budget']} | "
            f"{cc['empty_raw']}/{cc['kana_raw']}/{cc['quote_raw']} | {cc['median_secs']} | "
            f"{ee['accepted']}/{ee['cases']} | {ee['rejected_no_pin']} | {ee['rejected_rewrote']} | "
            f"{ee['error']} | {ee['exact_restore']} | {ee['accepted_with_collateral']} | "
            f"{ee['median_secs']} |")
    lines += ["", "```", ADOPT_RULE.rstrip(), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _pipeline_running() -> list[str]:
    try:
        out = subprocess.run(["pgrep", "-af", r"run_pipeline\.py|run_vc_version\.py|run_v3\.sh|run_long\.sh|build_profiles\.py"],
                             capture_output=True, text=True).stdout
    except Exception:                                       # noqa: BLE001
        return []
    return [ln for ln in out.splitlines() if "ab_compact_enforce" not in ln and ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--films", default=None,
                    help="comma list; default: the three shorts + every SONE-846_pNN state")
    ap.add_argument("--models", default=f"{BASELINE_MODEL},{OLLAMA_POLISH_MODEL}")
    ap.add_argument("--repeat-penalty", default="1.0",
                    help="comma list of arms, e.g. 1.0,1.15 (1.15 = the pre-v3.4 default)")
    ap.add_argument("--only", choices=["compact", "enforce"], default=None)
    ap.add_argument("--no-dedupe", action="store_true",
                    help="keep enforce lines that recur across films")
    ap.add_argument("--out", default=str(WORKSPACE_DIR / "_ab" / "compact_enforce"))
    ap.add_argument("--dry-run", action="store_true", help="build and count the cases; no Ollama")
    ap.add_argument("--limit", type=int, default=0,
                    help="first N cases of each task only (a one-minute smoke of the harness)")
    ap.add_argument("--force", action="store_true", help="run even if a pipeline process is visible")
    args = ap.parse_args()

    if args.films:
        films = [f.strip() for f in args.films.split(",") if f.strip()]
    else:
        films = SHORTS + sorted(Path(p).parent.name
                                for p in glob.glob(str(WORKSPACE_DIR / "SONE-846_p*" / "state.json")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    cases = build_cases(films, dedupe=not args.no_dedupe)
    counts = case_counts(cases)
    log("cases: " + json.dumps(counts, ensure_ascii=False))
    for n in cases["notes"]:
        log("  note: " + n)
    for m in cases["misses"]:
        log(f"  REAL MISS {m['film']} #{m['idx']} pin {m['pin']}: {m['ja']} | {m['zh']}")
    _write_csv([{k: v for k, v in c.items() if k not in ("context", "glossary")}
                | {"context": " | ".join(c["context"])}
                for c in cases["compact"]], out / "cases_compact.csv")
    _write_csv([{k: v for k, v in c.items() if k != "glossary"} for c in cases["enforce"]],
               out / "cases_enforce.csv")
    if args.dry_run:
        return 0

    busy = _pipeline_running()
    if busy and not args.force:
        log("refusing to run: a pipeline process would evict the model mid-call:")
        for ln in busy:
            log("  " + ln)
        return 2

    if args.limit:
        cases["compact"] = cases["compact"][:args.limit]
        cases["enforce"] = cases["enforce"][:args.limit]
        log(f"--limit {args.limit}: {len(cases['compact'])} compact / {len(cases['enforce'])} enforce cases")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rps = [float(x) for x in args.repeat_penalty.split(",") if x.strip()]
    crows: list[dict] = []
    erows: list[dict] = []
    for model in models:
        for rp in rps:
            log(f"=== arm {model} repeat_penalty {rp} ===")
            t0 = time.time()
            c, e = run_arm(model, rp, cases, args.only, log)
            crows.extend(c)
            erows.extend(e)
            log(f"arm done in {time.time() - t0:.0f}s")
            _write_csv(crows, out / "compact.csv")
            _write_csv(erows, out / "enforce.csv")
        # both candidates are below OLLAMA_EXCLUSIVE_ABOVE_GB, so exclusive_engine would
        # leave the previous model resident: evict it before the next one loads
        T.ollama_unload(model, OLLAMA_BASE_URL)
    summary = summarise(crows, erows, counts)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    write_summary_md(summary, out / "summary.md")
    print((out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
