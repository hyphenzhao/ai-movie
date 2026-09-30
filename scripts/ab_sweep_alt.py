#!/usr/bin/env python3
"""Offline A/B of the sweep's second decoder on the delivered states — nothing under
workspace/<name>/state.json is touched, every result is a side file.

    # decode the recorded sweep windows again with anime-whisper (GPU), re-classify, diff
    python scripts/ab_sweep_alt.py SONE-846_p05 --alt anime --alt-source mix --tag anime_mix
    python scripts/ab_sweep_alt.py SONE-846 --all --alt anime --alt-source mix --tag anime_mix   # every chunk
    python scripts/ab_sweep_alt.py output_test --alt anime --alt-source vocals --tag anime_voc
    # re-classify only (CPU): the current content.py rules over an earlier side file's words
    python scripts/ab_sweep_alt.py SONE-846 --all --alt none --from-tag anime_mix --tag rules_r1
    # film-level report (CPU): pinned chunk types, hit COUNTS, orphan COUNTS by parity, verdict
    python scripts/ab_sweep_alt.py SONE-846 --report anime_mix [--baseline base]

What one run does per workspace
  1. Rebuilds the pre-classification segment list the way step_asr did — asr_words.json
     (every sweep word carries its window's alt) → _finalize_segments with the stored
     diarization → silence gate → content.classify_segments — and checks that this
     reproduces state["asr"]["segments"] (printed as `baseline mismatch`; the only expected
     difference is a piece whose whisper alt was "" — segmenter._flush now carries it).
  2. Windows come from state["asr"]["sweep_windows"] (runs on this branch) or are
     reconstructed exactly as the pipeline computed them (Silero VAD on the CPU + the same
     energy floor + asr._sweep_windows); every sweep WORD is assigned to the window holding
     its midpoint (not the piece: split_into_sentences merges VAD and sweep words, and a
     mixed piece's midpoint can sit inside a VAD span).
  3. The chosen decoder re-reads each window that has words, on the mix or the vocals; the
     words get the new alt / alt_by, the list is rebuilt and re-classified.
  4. Writes workspace/<name>/state.ab_<tag>.json (state up to asr, segments replaced),
     asr_words.ab_<tag>.json, deliverables/ab_<tag>.csv (every pre-classification row: old →
     new decision, both alts, both similarities) and ab_<tag>.summary.json.

Measurement (the review's corrected protocol, --report)
  * chunk type pinned from plan.json (speech_minutes/minutes ≥ PROFILE_ENROL_DENSITY: p01,
    p14 interview) — never from the state under test, so p15/p17/p19 cannot flip and both
    L1 denominators are the same cue sets for every candidate;
  * L1 as hit COUNTS per type; orphans (kept dubbed line ≥ 6 folded chars whose nearest cue
    is 5–60 s away) as COUNTS, split sweep / vad — the 17 vad orphans are out of this item's
    reach and set the floor; L2 rate printed next to the kept-dubbed denominator so a shrink
    from lines moved to keep_original is visible;
  * everything by parity (odd = tuning set, even = report set) and per chunk;
  * every kept→drop / kept→nonlexical / drop→kept sweep line listed for the user's review.
Pre-registered adopt rule (even chunks, types pinned), printed as the verdict:
  (a) L1[scene] hits ≥ baseline − 1 and L1[interview] hits == baseline;
  (b) sweep-orphan count ≤ baseline;
  (c) film-level L2b median ≥ 0.73, and on even chunks L2b and scene similarity medians
      ≥ baseline − 0.01 (unchanged text → expected equal; a change means coverage moved);
  (e) no decoder error / fallback in any chunk;
  (d) output_test H4 (eval_against_subs --state …: median ≥ 0.90, < 0.70 count ≤ 15) and the
      test_1/test_2 change lists reviewed by the user — outside this script.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from ai_movie import asr as A                                   # noqa: E402
from ai_movie.content import classify, classify_segments, fold  # noqa: E402
from ai_movie.units import is_nonlexical                        # noqa: E402

SR = 16000
WORKSPACE = ROOT / "workspace"
OUT_ROOT: Path | None = None         # --out-dir: side files live under <out-dir>/<workspace name>/ instead
ADOPT_L2B_MIN = 0.73                 # the delivered v3.3 value on 28 interview cues
ADOPT_SCENE_SIM_TOL = 0.01           # "not lower": float noise from identical text


def side_dir(name: str) -> Path:
    return (OUT_ROOT / name) if OUT_ROOT else (WORKSPACE / name)


# ── per-workspace inputs ───────────────────────────────────────────

class WS:
    """One workspace's delivered asr evidence, read-only."""

    def __init__(self, name: str):
        self.name = name
        self.dir = WORKSPACE / name
        self.state = json.loads((self.dir / "state.json").read_text(encoding="utf-8"))
        wp = self.dir / "asr_words.json"
        if not wp.exists():
            raise SystemExit(f"{name}: asr_words.json missing (only runs since 0180f14 keep it)")
        self.words = json.loads(wp.read_text(encoding="utf-8"))
        asr = self.state.get("asr") or {}
        if not asr.get("segments"):
            raise SystemExit(f"{name}: no asr segments in state.json")
        self.audio = Path(self.state["demux"]["audio"])
        self.vocals = (self.state.get("separate") or {}).get("vocals")
        self.asr_tag = asr.get("asr_audio") or "mix"
        self.primary = self.audio if self.asr_tag == "mix" else Path(self.vocals)
        import run_pipeline as P
        self.trusted = P._vocals_trusted(self.audio, self.vocals)
        self._audio_cache: dict[str, "object"] = {}

    def load16k(self, which: str):
        """float32 16 kHz mono of "mix" | "vocals" (whisper.load_audio, as the pipeline did)."""
        import whisper
        if which not in self._audio_cache:
            p = self.audio if which == "mix" else self.vocals
            if not p or not Path(p).exists():
                raise SystemExit(f"{self.name}: no {which} track ({p})")
            self._audio_cache[which] = whisper.load_audio(str(p))
        return self._audio_cache[which]

    def content_rows(self) -> list[dict]:
        p = self.dir / "deliverables" / "01_content.csv"
        if not p.exists():
            return []
        with open(p, encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))


# ── windows ────────────────────────────────────────────────────────

def reconstruct_windows(ws: WS) -> tuple[list[dict], list[dict]]:
    """The sweep windows as _transcribe_whisper_gpu computed them for this file.

    step_asr never passed vocals_path, so the energy floor came from the
    primary track itself (asr._sweep_pass: floor_audio None → audio_np);
    VAD is Silero on the CPU with the config thresholds — deterministic.
    """
    import torch
    from ai_movie.config import (ASR_SWEEP_FLOOR_DBFS, ASR_SWEEP_MAX_WINDOW_S, ASR_SWEEP_MIN_GAP_S,
                                 ASR_VAD_MIN_SILENCE_DURATION_MS, ASR_VAD_MIN_SPEECH_DURATION_MS,
                                 ASR_VAD_SPEECH_PAD_MS, ASR_VAD_THRESHOLD)
    audio_np = ws.load16k(ws.asr_tag)
    spans = A._vad_detect(torch.from_numpy(audio_np), threshold=ASR_VAD_THRESHOLD,
                          min_silence_duration_ms=ASR_VAD_MIN_SILENCE_DURATION_MS,
                          min_speech_duration_ms=ASR_VAD_MIN_SPEECH_DURATION_MS,
                          speech_pad_ms=ASR_VAD_SPEECH_PAD_MS)
    if not spans:
        spans = [{"start": 0.0, "end": len(audio_np) / SR}]
    energy = A._frame_db(audio_np)
    wins = A._sweep_windows(spans, len(audio_np), energy, min_gap=ASR_SWEEP_MIN_GAP_S,
                            max_win=ASR_SWEEP_MAX_WINDOW_S, floor_db=ASR_SWEEP_FLOOR_DBFS)
    return wins, spans


def windows_for(ws: WS) -> tuple[list[dict], str]:
    rec = (ws.state.get("asr") or {}).get("sweep_windows")
    if rec:
        return [{"start": float(w["start"]), "end": float(w["end"]), "p95_db": w.get("p95_db")} for w in rec], "state"
    wins, _ = reconstruct_windows(ws)
    return wins, "reconstructed"


def assign_words(words: list[dict], wins: list[dict]) -> tuple[list[int | None], dict]:
    """Window index per word (sweep words by midpoint; vad words → None) + consistency counts.

    Every word of one window carries the same alt string, so a window whose
    assigned words disagree means the reconstruction is off — reported, not hidden.
    """
    import bisect
    starts = [w["start"] for w in wins]
    idx: list[int | None] = []
    n_sweep = n_out = 0
    for w in words:
        if w.get("pass") != "sweep":
            idx.append(None)
            continue
        n_sweep += 1
        mid = (float(w["s"]) + float(w["e"])) / 2
        j = bisect.bisect_right(starts, mid) - 1
        if j >= 0 and wins[j]["start"] <= mid <= wins[j]["end"]:
            idx.append(j)
        else:
            idx.append(None)
            n_out += 1
    alts_per_win: dict[int, set] = {}
    for w, j in zip(words, idx):
        if j is not None:
            alts_per_win.setdefault(j, set()).add(w.get("alt"))
    inconsistent = sum(1 for s in alts_per_win.values() if len(s) > 1)
    return idx, {"sweep_words": n_sweep, "unassigned": n_out, "windows_with_words": len(alts_per_win),
                 "alt_inconsistent_windows": inconsistent}


# ── rebuild + classify (what step_asr does after transcribe_all) ──────

def rebuild(ws: WS, words: list[dict]) -> tuple[list[dict], list[dict]]:
    """(pre-classification rows, kept rows) from a word stream, exactly as step_asr derives them."""
    import run_pipeline as P
    from ai_movie.config import ASR_SILENCE_DBFS
    diar = (ws.state.get("asr") or {}).get("diarization")
    segs = A._finalize_segments([], words, source=str(ws.audio), diarization=diar)
    if ws.trusted:
        segs = P._drop_silent_segments(segs, ws.vocals, ASR_SILENCE_DBFS)
    before = [dict(s) for s in segs]
    kept = classify_segments(segs, vocals=ws.vocals if ws.trusted else None)
    return before, kept


def _key(s: dict) -> tuple:
    return (round(float(s["start"]), 2), round(float(s["end"]), 2), s.get("text"))


def _sim(text: str, alt) -> float | None:
    if alt is None:
        return None
    fa = fold(alt)
    return round(difflib.SequenceMatcher(None, fold(text), fa).ratio(), 3) if fa else None


def decisions(before: list[dict], kept: list[dict]) -> dict[tuple, tuple[str, str]]:
    """(decision, reason) per pre-classification row, dropped rows re-derived as step_asr prints them."""
    k = {_key(s): s for s in kept}
    out = {}
    for s in before:
        x = k.get(_key(s))
        if x is not None:
            out[_key(s)] = (x.get("content"), "; ".join(x.get("content_reasons") or []))
        else:
            r = classify(s)
            out[_key(s)] = ("drop", "; ".join(r["reasons"]) if r["content"] == "drop" else "energy floor or repeated line")
    return out


def baseline_mismatch(ws: WS, kept: list[dict]) -> list[str]:
    """Rows where the rebuilt kept list differs from state["asr"]["segments"] (text/time/decision)."""
    want = {_key(s): s.get("content") for s in ws.state["asr"]["segments"]}
    have = {_key(s): s.get("content") for s in kept}
    out = []
    for k in sorted(set(want) | set(have)):
        if want.get(k) != have.get(k):
            out.append(f"{k[0]:.2f}-{k[1]:.2f} {k[2]!r}: state={want.get(k)} rebuilt={have.get(k)}")
    return out


# ── decoding ───────────────────────────────────────────────────────

_DECODER = None


def get_decoder(device: str):
    global _DECODER
    if _DECODER is None:
        t0 = time.time()
        _DECODER = A.AnimeWhisper(device=device)
        print(f"  anime-whisper loaded in {time.time() - t0:.1f} s (revision {_DECODER.revision}, {device})")
    return _DECODER


def decode_windows(ws: WS, wins: list[dict], need: list[int], source: str, device: str,
                   batch: int, limit: int | None) -> tuple[dict[int, str | None], dict]:
    """anime alt text per window index in *need*; errors as (None), one per failed batch."""
    audio = ws.load16k(source)
    dec = get_decoder(device)
    errors: list[str] = []
    todo = sorted(need)[:limit] if limit else sorted(need)
    out: dict[int, str | None] = {}
    t0 = time.time()
    for i in range(0, len(todo), batch):
        idx = todo[i:i + batch]
        chunks = [audio[int(wins[j]["start"] * SR):int(wins[j]["end"] * SR)] for j in idx]
        try:
            texts = dec.transcribe_batch(chunks)
            for j, t in zip(idx, texts):
                out[j] = t
        except Exception as exc:                                # noqa: BLE001
            errors.append(f"windows {wins[idx[0]]['start']}–{wins[idx[-1]]['end']}: {type(exc).__name__}: {exc}")
            for j in idx:
                out[j] = None
    dt = time.time() - t0
    return out, {"decoded": len(todo), "seconds": round(dt, 1),
                 "s_per_window": round(dt / max(1, len(todo)), 2), "errors": errors}


# ── one workspace ──────────────────────────────────────────────────

def run_one(name: str, *, alt: str, alt_source: str, tag: str, from_tag: str | None, device: str,
            batch: int, limit: int | None, all_windows: bool) -> dict:
    ws = WS(name)
    out_root = side_dir(name)
    (out_root / "deliverables").mkdir(parents=True, exist_ok=True)
    print(f"[{name}] asr on {ws.asr_tag}, vocals {'trusted' if ws.trusted else 'untrusted'}, "
          f"{len(ws.words)} words, {len(ws.state['asr']['segments'])} kept segments")

    # 1. baseline: the delivered words must reproduce the delivered pre-classification list; the
    #    decisions may differ where content.py changed since the state was written (v3.4-dev e077ed0
    #    keeps transcribed moans as original voice) — that is why the A/B's baseline is this rebuild
    #    (same rules, delivered alt), not state.json itself
    before0, kept0 = rebuild(ws, ws.words)
    mism = baseline_mismatch(ws, kept0)
    csv_rows = ws.content_rows()
    csv_keys = {(round(float(r["start"]), 2), round(float(r["end"]), 2), r["text"]) for r in csv_rows}
    pre_mism = len({_key(s) for s in before0} ^ csv_keys) if csv_rows else None
    print(f"  baseline rebuild: {len(before0)} rows pre-classification ({pre_mism} differ from 01_content.csv"
          f"{' — reconstruction is off!' if pre_mism else ''}), {len(kept0)} kept; "
          f"{len(mism)} decision(s) differ from state.json (current content.py rules vs the delivered run)")
    for m in mism[:5]:
        print("    ", m)

    # 2. the words the candidate starts from
    if alt == "none":
        src = out_root / f"asr_words.ab_{from_tag}.json" if from_tag else None
        words = json.loads(src.read_text(encoding="utf-8")) if src and src.exists() else [dict(w) for w in ws.words]
        print(f"  re-classify only: words from {src if src and src.exists() else 'asr_words.json'}")
        win_meta, dec_meta, wins = {}, {}, []
    else:
        wins, origin = windows_for(ws)
        idx, win_meta = assign_words(ws.words, wins)
        print(f"  windows: {len(wins)} ({origin}); sweep words {win_meta['sweep_words']}, "
              f"unassigned {win_meta['unassigned']}, windows with words {win_meta['windows_with_words']}, "
              f"alt-inconsistent {win_meta['alt_inconsistent_windows']}")
        need = sorted({j for j in idx if j is not None})
        if all_windows:
            need = list(range(len(wins)))
        if alt == "anime":
            texts, dec_meta = decode_windows(ws, wins, need, alt_source, device, batch, limit)
            print(f"  anime decoded {dec_meta['decoded']} windows on {alt_source} in {dec_meta['seconds']} s "
                  f"({dec_meta['s_per_window']} s/window), {len(dec_meta['errors'])} error(s)")
        else:                                                   # "whisper": the delivered alt is the whisper alt
            texts, dec_meta = {}, {"decoded": 0, "seconds": 0.0, "s_per_window": 0.0, "errors": []}
        words = []
        for w, j in zip(ws.words, idx):
            w2 = dict(w)
            if j is not None and j in texts:
                w2.pop("alt", None)
                if texts[j] is not None:
                    w2["alt"] = texts[j]
                w2["alt_by"] = "anime"
            elif w.get("pass") == "sweep" and w.get("alt_by") is None and w.get("alt") is not None:
                w2["alt_by"] = "whisper"
            words.append(w2)
        for j, w in enumerate(wins):
            w["text"] = "".join(x["w"] for x, k in zip(ws.words, idx) if k == j).strip()
            if j in texts:
                w["alt_text"], w["alt_by"] = texts[j], "anime"
            elif j in {k for k in idx if k is not None}:
                w["alt_text"] = next(x.get("alt") for x, k in zip(ws.words, idx) if k == j)
                w["alt_by"] = "whisper"
            else:
                w["alt_text"], w["alt_by"] = None, None

    # 3. rebuild + classify with the candidate alts
    before1, kept1 = rebuild(ws, words)
    d0, d1 = decisions(before0, kept0), decisions(before1, kept1)
    alt0 = {_key(s): s.get("alt_text") for s in before0}
    alt1 = {_key(s): s.get("alt_text") for s in before1}
    by1 = {_key(s): s.get("alt_by") for s in before1}
    trans: dict[str, int] = {}
    rows = []
    for s in before1:
        k = _key(s)
        old, new = d0.get(k, ("?", ""))[0], d1[k][0]
        changed = old != new
        if changed:
            trans[f"{old}→{new}"] = trans.get(f"{old}→{new}", 0) + 1
        rows.append({"chunk": name, "start": k[0], "end": k[1], "pass": s.get("pass", "vad"),
                     "speaker": s.get("speaker"), "text": k[2], "old": old, "new": new, "changed": int(changed),
                     "old_reason": d0.get(k, ("", ""))[1], "new_reason": d1[k][1],
                     "old_alt": alt0.get(k), "new_alt": alt1.get(k), "alt_by": by1.get(k),
                     "old_sim": _sim(k[2], alt0.get(k)), "new_sim": _sim(k[2], alt1.get(k)),
                     "asr_conf": s.get("asr_conf"), "avg_logprob": s.get("avg_logprob"),
                     "no_speech_prob": s.get("no_speech_prob")})
    n_changed = sum(r["changed"] for r in rows)
    print(f"  candidate: {len(kept1)} kept ({sum(1 for s in kept1 if s.get('keep_original'))} original voice); "
          f"{n_changed} decision(s) changed {trans}")

    # rescue candidates (phase-3 information only): large-v3 heard nothing, anime returned lexical text
    rescue = [w for w in wins if not w.get("text") and w.get("alt_text")
              and len(fold(w["alt_text"])) >= 5 and not is_nonlexical(w["alt_text"])] if wins else []

    # 4. side files — never state.json
    side = {k: ws.state[k] for k in ("_video", "demux", "separate", "osd") if k in ws.state}
    side["asr"] = dict(ws.state["asr"], segments=kept1, sweep_windows=wins or ws.state["asr"].get("sweep_windows"),
                       sweep_alt={"name": alt if alt != "none" else "reclassify", "source": alt_source,
                                  "tag": tag, "from_tag": from_tag,
                                  "revision": getattr(_DECODER, "revision", None) if alt == "anime" else None,
                                  "batch": batch, "decoder_errors": dec_meta.get("errors", [])},
                       chunk_errors=dec_meta.get("errors", []))
    side["ab"] = {"tag": tag, "alt": alt, "alt_source": alt_source, "from_tag": from_tag,
                  "baseline_mismatch": mism, "pre_rows_vs_csv": pre_mism, "windows": win_meta,
                  "decode": dec_meta, "transitions": trans, "changed": n_changed,
                  "kept": len(kept1), "kept_original": sum(1 for s in kept1 if s.get("keep_original")),
                  "kept_baseline": len(kept0), "rescue_candidates": len(rescue),
                  "rescue": [{"start": w["start"], "end": w["end"], "alt_text": w["alt_text"]} for w in rescue[:30]]}
    (out_root / f"state.ab_{tag}.json").write_text(json.dumps(side, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_root / f"asr_words.ab_{tag}.json").write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
    fields = list(rows[0].keys()) if rows else ["chunk", "start", "end", "pass", "speaker", "text", "old", "new", "changed"]
    with open(out_root / "deliverables" / f"ab_{tag}.csv", "w", encoding="utf-8-sig", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    (out_root / f"ab_{tag}.summary.json").write_text(json.dumps(side["ab"], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  → {out_root / f'state.ab_{tag}.json'}, deliverables/ab_{tag}.csv"
          + (f", {len(rescue)} rescue candidate(s)" if rescue else ""))
    return side["ab"]


# ── film-level report ──────────────────────────────────────────────

def chunk_types(film: str) -> dict[int, dict]:
    """plan.json chunks with the type pinned by the enrol density rule (run_long.sh's)."""
    from ai_movie.config import PROFILE_ENROL_DENSITY
    plan = json.loads((WORKSPACE / film / "_split" / "plan.json").read_text(encoding="utf-8"))
    out = {}
    for c in plan["chunks"]:
        dens = (c["speech_minutes"] / c["minutes"]) if c.get("minutes") else 0.0
        out[c["index"]] = dict(c, type="interview" if dens >= PROFILE_ENROL_DENSITY else "scene", density=round(dens, 3))
    return out


def film_place(film: str, state_file: str) -> tuple[list[dict], dict[int, dict]]:
    """Kept segments of every chunk on the film timeline (run_long's concat: chunk start − trimmed lead)."""
    import bisect
    import eval_long as E
    split = WORKSPACE / film / "_split"
    ks = json.loads((split / "keyframes.json").read_text()) if (split / "keyframes.json").exists() else []
    meta = chunk_types(film)
    segs = []
    for i, c in sorted(meta.items()):
        lead = 0.0
        f = Path(c.get("file", ""))
        if ks and f.exists():
            have = float(E._probe(f, "format=duration") or 0)
            if have - (c["end"] - c["start"]) > 1.0:
                j = bisect.bisect_left(ks, c["start"] - 1e-3)
                if j > 0 and abs(ks[j] - c["start"]) < 2e-3:
                    lead = c["start"] - ks[j - 1]
        cn = f"{film}_p{i:02d}"
        sp = (WORKSPACE / cn / state_file) if state_file == "state.json" else (side_dir(cn) / state_file)
        st = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
        c["state"] = bool(st)
        c["lead"] = lead
        c["errors"] = list(((st.get("asr") or {}).get("chunk_errors") or [])) + \
            list((((st.get("asr") or {}).get("sweep_alt") or {}).get("decoder_errors") or []))
        c["fallback"] = bool((((st.get("asr") or {}).get("sweep_alt") or {}).get("fallback")))
        off = c["start"] - lead
        for n, s in enumerate((st.get("asr") or {}).get("segments") or []):
            if s["end"] <= lead:
                continue
            segs.append(dict(s, chunk=i, idx=n, t0=off + s["start"], t1=off + s["end"]))
    return segs, meta


def measure(film: str, state_file: str) -> dict:
    """Per-chunk counts with pinned types, aggregated by parity."""
    import eval_long as E
    segs, meta = film_place(film, state_file)
    truth, read = E.cues(film)
    kept = [s for s in segs if not s.get("keep_original")]
    kept_iv = [(s["t0"], s["t1"]) for s in kept]
    cue_iv = sorted((c["start"], c["end"]) for c in truth)
    per: dict[int, dict] = {}
    for i, m in sorted(meta.items()):
        cs = [(c["start"], c["end"]) for c in truth if m["start"] <= c["start"] < m["end"]]
        hits = sum(E.coverage(cs, kept_iv)) if cs else 0
        mine = [s for s in kept if s["chunk"] == i]
        orphan_sweep = orphan_vad = posthoc = 0
        orphans = []
        for s in mine:
            r = classify({**s, "pass": s.get("pass", "vad")})
            if r["content"] == "drop":
                posthoc += 1
                continue
            vis = len(fold(s.get("text", "")))
            d = E.cue_distance(cue_iv, s["t0"], s["t1"])
            if vis >= 6 and 5.0 <= d < 60.0:
                if s.get("pass") == "sweep":
                    orphan_sweep += 1
                else:
                    orphan_vad += 1
                orphans.append({"t": round(s["t0"], 1), "pass": s.get("pass", "vad"), "text": s.get("text", ""),
                                "alt": s.get("alt_text"), "by": s.get("alt_by"), "d": round(d, 1)})
        sims = []
        for cid, (zh, ja) in read.items():
            c = next((c for c in truth if c["id"] == cid), None)
            if not c or not ja or not (m["start"] <= c["start"] < m["end"]):
                continue
            ov = [s for s in kept if min(c["end"], s["t1"]) - max(c["start"], s["t0"]) > 0.25]
            if ov:
                sims.append(difflib.SequenceMatcher(None, fold(ja), fold("".join(s.get("text", "") for s in ov))).ratio())
        per[i] = {"type": m["type"], "parity": "odd" if i % 2 else "even", "density": m["density"],
                  "state": m["state"], "cues": len(cs), "hits": hits, "kept_dubbed": len(mine),
                  "kept_original": sum(1 for s in segs if s["chunk"] == i and s.get("keep_original")),
                  "orphan_sweep": orphan_sweep, "orphan_vad": orphan_vad, "posthoc_drop": posthoc,
                  "sims": sims, "orphans": orphans, "errors": m["errors"], "fallback": m["fallback"]}
    agg = {}
    for parity in ("odd", "even", "all"):
        rows = [r for r in per.values() if parity == "all" or r["parity"] == parity]
        a = {"chunks": [i for i, r in per.items() if r in rows]}
        for kind in ("interview", "scene"):
            rr = [r for r in rows if r["type"] == kind]
            a[f"L1[{kind}]"] = {"hits": sum(r["hits"] for r in rr), "cues": sum(r["cues"] for r in rr)}
        kd = sum(r["kept_dubbed"] for r in rows)
        bad = sum(r["orphan_sweep"] + r["orphan_vad"] + r["posthoc_drop"] for r in rows)
        a["kept_dubbed"] = kd
        a["kept_original"] = sum(r["kept_original"] for r in rows)
        a["orphan_sweep"] = sum(r["orphan_sweep"] for r in rows)
        a["orphan_vad"] = sum(r["orphan_vad"] for r in rows)
        a["posthoc_drop"] = sum(r["posthoc_drop"] for r in rows)
        a["L2"] = {"bad": bad, "rate": round(bad / kd, 4) if kd else None}
        iv = [x for r in rows if r["type"] == "interview" for x in r["sims"]]
        sc = [x for r in rows if r["type"] == "scene" for x in r["sims"]]
        a["L2b"] = {"median": round(statistics.median(iv), 3) if iv else None, "n": len(iv)}
        a["scene_sim"] = {"median": round(statistics.median(sc), 3) if sc else None, "n": len(sc)}
        a["errors"] = sum(len(r["errors"]) for r in rows)
        a["fallback"] = sum(1 for r in rows if r["fallback"])
        agg[parity] = a
    return {"film": film, "state_file": state_file, "per_chunk": per, "agg": agg,
            "kept_dubbed_total": len(kept), "truth_cues": len(truth), "read_cues": len(read)}


def verdict(cand: dict, base: dict) -> tuple[bool, list[str]]:
    """The pre-registered adopt rule on EVEN chunks (types pinned)."""
    c, b = cand["agg"]["even"], base["agg"]["even"]
    checks = []
    ok_a = c["L1[scene]"]["hits"] >= b["L1[scene]"]["hits"] - 1 and c["L1[interview]"]["hits"] == b["L1[interview]"]["hits"]
    checks.append((ok_a, f"(a) L1[scene] hits {c['L1[scene]']['hits']}/{c['L1[scene]']['cues']} vs {b['L1[scene]']['hits']} "
                         f"(≥ base−1); L1[interview] {c['L1[interview]']['hits']}/{c['L1[interview]']['cues']} vs "
                         f"{b['L1[interview]']['hits']} (unchanged)"))
    ok_b = c["orphan_sweep"] <= b["orphan_sweep"]
    checks.append((ok_b, f"(b) sweep orphans {c['orphan_sweep']} vs {b['orphan_sweep']} (≤); vad orphans "
                         f"{c['orphan_vad']} vs {b['orphan_vad']} (floor, untouchable); L2 {c['L2']['bad']}/{c['kept_dubbed']} = "
                         f"{(c['L2']['rate'] or 0):.1%} vs {b['L2']['bad']}/{b['kept_dubbed']} = {(b['L2']['rate'] or 0):.1%}; "
                         f"kept original {c['kept_original']} vs {b['kept_original']}"))
    # (c): the 0.73 gate is a film-level number (the even half alone sits at 0.667 on 12 cues in the
    # delivered v3.3 state); on the even half the text is unchanged so both medians must not fall
    all_c, all_b = cand["agg"]["all"], base["agg"]["all"]
    l2b, l2b0 = c["L2b"]["median"], b["L2b"]["median"]
    ss, ss0 = c["scene_sim"]["median"], b["scene_sim"]["median"]
    film_l2b = all_c["L2b"]["median"]
    ok_c = ((film_l2b is None or film_l2b >= ADOPT_L2B_MIN)
            and (l2b is None or l2b0 is None or l2b >= l2b0 - ADOPT_SCENE_SIM_TOL)
            and (ss is None or ss0 is None or ss >= ss0 - ADOPT_SCENE_SIM_TOL))
    checks.append((ok_c, f"(c) L2b film-level {film_l2b} on {all_c['L2b']['n']} cues (≥ {ADOPT_L2B_MIN}; base "
                         f"{all_b['L2b']['median']}); even L2b {l2b} vs {l2b0}, even scene sim {ss} vs {ss0} on "
                         f"{c['scene_sim']['n']} cues (each ≥ base − {ADOPT_SCENE_SIM_TOL})"))
    ok_e = all_c["errors"] == 0 and all_c["fallback"] == 0
    checks.append((ok_e, f"(e) decoder errors {all_c['errors']}, fallbacks {all_c['fallback']} over all chunks (must be 0)"))
    checks.append((None, "(d) shorts: eval_against_subs output_test --state state.ab_<tag>.json (median ≥ 0.90, "
                         "< 0.70 ≤ 15) and the user's review of test_1/test_2 change lists — not measured here"))
    return all(ok for ok, _ in checks if ok is not None), [f"{'✅' if ok else '❌' if ok is False else '…'} {t}" for ok, t in checks]


def change_lists(film: str, tag: str, meta: dict[int, dict]) -> dict[str, list[dict]]:
    """Every sweep line whose decision changed, per parity, with its film time — for the user's review."""
    out = {"odd": [], "even": []}
    for i, m in sorted(meta.items()):
        p = side_dir(f"{film}_p{i:02d}") / "deliverables" / f"ab_{tag}.csv"
        if not p.exists():
            continue
        with open(p, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("changed") == "1":
                    t0 = m["start"] - m.get("lead", 0.0) + float(r["start"])
                    out["odd" if i % 2 else "even"].append(dict(r, film_t=round(t0, 1), chunk_type=m["type"]))
    return out


def report(film: str, tag: str, baseline: str) -> int:
    out_dir = side_dir(film) / "_ab"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_file = "state.json" if baseline == "base" else f"state.ab_{baseline}.json"
    print(f"[{film}] baseline {base_file} vs candidate state.ab_{tag}.json (chunk types pinned from plan.json)")
    if baseline == "base":
        print("  note: state.json carries the delivered run's decisions; the rules changed since (e077ed0), so "
              "compare against a null run (--alt whisper --tag whisper_other) to isolate the decoder")
    base = measure(film, base_file)
    cand = measure(film, f"state.ab_{tag}.json")
    missing = [i for i, r in cand["per_chunk"].items() if base["per_chunk"][i]["state"] and not r["state"]]
    if missing:
        print(f"  WARNING: candidate side file missing for chunks {missing} — their rows count as empty")
    lines = [f"# {film} sweep-alt A/B: {tag} vs {baseline}", "",
             "| set | L1[interview] | L1[scene] | sweep orphans | vad orphans | L2 (bad/kept dubbed) | kept original | L2b | scene sim | errors |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for parity in ("odd", "even", "all"):
        for label, m in (("base", base), (tag, cand)):
            a = m["agg"][parity]
            lines.append(f"| {parity} {label} | {a['L1[interview]']['hits']}/{a['L1[interview]']['cues']} | "
                         f"{a['L1[scene]']['hits']}/{a['L1[scene]']['cues']} | {a['orphan_sweep']} | {a['orphan_vad']} | "
                         f"{a['L2']['bad']}/{a['kept_dubbed']} = {(a['L2']['rate'] or 0):.1%} | {a['kept_original']} | "
                         f"{a['L2b']['median']} (n={a['L2b']['n']}) | {a['scene_sim']['median']} (n={a['scene_sim']['n']}) | "
                         f"{a['errors']}+{a['fallback']} |")
    lines += ["", "## per chunk (type · parity): cues hit, sweep/vad orphans, kept dubbed — base → candidate", ""]
    for i in sorted(cand["per_chunk"]):
        b, c = base["per_chunk"][i], cand["per_chunk"][i]
        lines.append(f"- p{i:02d} {c['type']:9s} {c['parity']:4s} L1 {b['hits']}→{c['hits']}/{c['cues']}  "
                     f"orph sweep {b['orphan_sweep']}→{c['orphan_sweep']}  vad {b['orphan_vad']}→{c['orphan_vad']}  "
                     f"kept {b['kept_dubbed']}→{c['kept_dubbed']} (+{b['kept_original']}→{c['kept_original']} original)"
                     + (f"  errors {len(c['errors'])}" if c["errors"] else "") + ("  FALLBACK" if c["fallback"] else ""))
    ok, checks = verdict(cand, base)
    tail = ["", "## adopt rule (even chunks, pinned types)", ""] + [f"- {t}" for t in checks]
    tail += ["", f"**verdict: {'ADOPT candidate' if ok else 'not adoptable as measured'}** — (d) pending"]
    changes = change_lists(film, tag, film_place(film, f"state.ab_{tag}.json")[1])
    for parity, rows in changes.items():
        p = out_dir / f"{tag}_changes_{parity}.csv"
        with open(p, "w", encoding="utf-8-sig", newline="") as fh:
            if rows:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                wr.writerows(rows)
        kinds: dict[str, int] = {}
        for r in rows:
            kinds[f"{r['old']}→{r['new']}"] = kinds.get(f"{r['old']}→{r['new']}", 0) + 1
        tail.append(f"- {len(rows)} changed line(s) on {parity} chunks {kinds} → {p}")
    orph = [(i, o) for i, r in cand["per_chunk"].items() for o in r["orphans"]]
    detail = ["", f"## orphans in the candidate ({len(orph)}; sweep ones are this item's reach)", ""]
    detail += [f"- p{i:02d} {o['t']}s [{o['pass']}] 「{o['text'][:30]}」 alt={str(o['alt'])[:30]!r} by={o['by']} (cue {o['d']} s away)"
               for i, o in orph[:80]]
    (out_dir / f"report_{tag}.md").write_text("\n".join(lines + tail + detail), encoding="utf-8")
    (out_dir / f"report_{tag}.json").write_text(json.dumps({"base": base, "cand": cand, "verdict": ok, "checks": checks},
                                                            ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("\n".join(lines[:4 + 6] + tail))
    print(f"  → {out_dir / f'report_{tag}.md'}")
    return 0 if ok else 1


# ── CLI ────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="workspace (SONE-846_p05, output_test) or film (with --all / --report)")
    ap.add_argument("--all", action="store_true", help="every <name>_pNN chunk that has a state.json")
    ap.add_argument("--alt", default="anime", choices=["anime", "whisper", "none"],
                    help="second decoder to replay: anime | whisper (= the delivered alt, a null run) | none (re-classify)")
    ap.add_argument("--alt-source", default="mix", choices=["mix", "vocals"], help="audio the second decoder hears")
    ap.add_argument("--tag", default=None, help="side-file tag (default: <alt>_<source>)")
    ap.add_argument("--from-tag", default=None, help="--alt none: take the words of this earlier side file")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=None, help="windows per generate() (config ASR_ANIME_BATCH)")
    ap.add_argument("--limit", type=int, default=None, help="decode only the first N text windows (smoke test)")
    ap.add_argument("--all-windows", action="store_true",
                    help="also decode windows where large-v3 heard nothing (rescue-window candidates, phase 3)")
    ap.add_argument("--out-dir", type=Path, default=None, help="write side files here instead of the workspace")
    ap.add_argument("--report", default=None, metavar="TAG", help="film-level report of this tag (no decoding)")
    ap.add_argument("--baseline", default="base", help="--report: 'base' (state.json) or another tag")
    args = ap.parse_args()
    global OUT_ROOT
    OUT_ROOT = args.out_dir.resolve() if args.out_dir else None

    if args.report:
        return report(args.name, args.report, args.baseline)
    tag = args.tag or (f"{args.alt}_{args.alt_source}" if args.alt != "none" else f"rules_{args.from_tag or 'base'}")
    if args.batch is None:
        from ai_movie.config import ASR_ANIME_BATCH
        args.batch = ASR_ANIME_BATCH
    names = [args.name]
    if args.all:
        names = sorted(p.parent.name for p in WORKSPACE.glob(f"{args.name}_p??/state.json"))
        if not names:
            raise SystemExit(f"no {args.name}_pNN/state.json under workspace/")
    summaries = {}
    t0 = time.time()
    try:
        for n in names:
            summaries[n] = run_one(n, alt=args.alt, alt_source=args.alt_source, tag=tag, from_tag=args.from_tag,
                                   device=args.device, batch=args.batch, limit=args.limit,
                                   all_windows=args.all_windows)
    finally:
        if _DECODER is not None:
            _DECODER.close()
    tot = sum(s["changed"] for s in summaries.values())
    errs = sum(len(s["decode"].get("errors", [])) for s in summaries.values())
    mism = sum(len(s["baseline_mismatch"]) for s in summaries.values())
    print(f"done: {len(names)} workspace(s) in {(time.time() - t0) / 60:.1f} min — {tot} decision(s) changed, "
          f"{errs} decoder error(s), {mism} decision(s) differ between state.json and the current rules")
    if args.all:
        print(f"next: python scripts/ab_sweep_alt.py {args.name} --report {tag} --baseline whisper_other"
              + (f" --out-dir {OUT_ROOT}" if OUT_ROOT else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
