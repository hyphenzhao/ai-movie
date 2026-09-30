#!/usr/bin/env python3
"""A/B: CosyVoice3 ``cross_lingual`` against the shipped SFT→VC route on one chunk's female lines.

    .venv/bin/python scripts/ab_cross_lingual.py SONE-846_p01 --out deliver/SONE-846_p01/ab_xl              # GPU, ~25 min
    .venv/bin/python scripts/ab_cross_lingual.py SONE-846_p01 --out deliver/SONE-846_p01/ab_xl --smoke      # 1 line, ~2 min
    .venv/bin/python scripts/ab_cross_lingual.py SONE-846_p01 --out deliver/SONE-846_p01/ab_xl --judge-only # CPU, re-judge
    .venv/bin/python scripts/ab_cross_lingual.py --decide deliver/SONE-846_p01/ab_xl/metrics.json --prefs deliver/SONE-846_p01/ab_xl/prefs.json

What differs between the arms is only where the speech tokens come from.  Both condition
the flow decoder on the same profile reference (speech tokens + mel + campplus of
``profiles/ref_P0.wav``):

  A   cross_lingual — the CosyVoice3 LLM turns the Chinese text into tokens; no prompt
      text, no prompt tokens reach the LLM (frontend_cross_lingual deletes them).
  A′  the same with ``ref_P0_alt0.wav`` (5.06 s ≥ TTS_REF_MIN_DURATION) — answers whether
      the 3.06 s profile clip limits A; it is a finding, never a winner.
  B   the shipped route — ``synthesized_vc/seg_XXXX.wav`` is VC of the built-in (SFT) line
      through tokenizer_v3, so no GPU work; ``fit.segments[i].audio_fit`` (the SFT line
      fitted to its slot) is the judge_line baseline for every arm.

Known input rule handled here, not in ai_movie: CosyVoice3's LLM asserts the
``<|endofprompt|>`` token is in its text (llm.py ``inference``); zero_shot carries it in
prompt_text, cross_lingual has no prompt_text, so the marker must be inside the tts text
(models/CosyVoice/example.py:81,100).  Without it the llm_job thread dies on the assert,
token2wav runs on 0 tokens and HiFT raises the "padded input size … kernel size" RuntimeError
that ai_movie.tts's comments call "cross_lingual is broken on this build".  ``--smoke``
records both the bare failure and the prefixed success.  A marker in the text also switches
the frontend's text normalisation off (frontend.py text_normalize), so lines with ASCII
letters/digits are excluded from the sample and counted.

Judges (CPU): vc_guard.judge_line vs the SFT line; pitch.gate vs the reference F0; whole-clip
ECAPA cosine to the reference and to the arm's centroid (one vector per line — embed_windows
would give 1–4 windows per line and mix populations); Whisper read-back (language, kana,
hanzi overlap); natural duration vs the fit stage's slot (composer.segment_slots, not
end−start).  The listening kit pairs A *fitted into its slot* (what would ship) with B's
fitted line.  ``decide`` holds the pre-registered rule; see AB_* below for the thresholds.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AB_N_PER_BIN = 10
AB_DUR_BINS = ((0.25, 0.7), (0.7, 1.5), (1.5, 2.5), (2.5, None))   # v1 fitted seconds; < 0.7 s is where CosyVoice2/3 collapses (42/170 lines)
AB_PREFIX = "You are a helpful assistant.<|endofprompt|>"
AB_MIN_VISIBLE = 2
AB_SIM_MARGIN = 0.05        # A's median ECAPA-to-reference must beat B's by this (take-to-take spread is 0.07–0.16: less is noise)
AB_CONSISTENCY_TOL = 0.05   # A's p10 (line vs arm centroid) may trail B's by this — relative to B, no absolute floor exists
AB_RATE_TOL = 0.05          # guard-ok / reference-ratio pass rates may trail B's by 2 lines in 40 (B ≈ 100 % on judged lines)
AB_SOFT_SPEEDUP_TOL = 0.05  # share of lines needing > TTS_FIT_MAX_SPEEDUP may exceed the SFT baseline's by 2 lines in 40
AB_HARD_SPEEDUP_TOL = 0.025 # share needing > TTS_FIT_MAX_SPEEDUP_HARD may exceed it by 1 line in 40 (baseline: 5.3 % / 2.4 % on all 170)
AB_PREF_ALPHA = 0.05        # listener preference: one-sided binomial p ≤ 0.05 (≥ 20/30, ≥ 26/40); 18/30 was p ≈ 0.18
AB_SMOKE_TEXT = "虽然已经开始了拍摄，但还精神吗？"
FOREIGN_PATTERNS = ("run_pipeline.py", "run_vc_version.py", "auto_select_refs.py", "vc_ref_probe.py",
                    "scripts.inference", "run_v3.sh", "deliver.py", "tts_worker.py", "run_long.sh")

_KANA = re.compile(r"[぀-ゟ゠-ヿ]")
_HANZI = re.compile(r"[一-鿿]")
_ASCII = re.compile(r"[A-Za-z0-9]")
_PUNCT = "。、，,．.！!？?…‥・「」『』（）()～~-— "


# ── pure helpers (tested) ─────────────────────────────────────────

def visible_chars(text: str) -> int:
    return sum(1 for ch in (text or "") if not ch.isspace() and ch not in _PUNCT)


def has_ascii(text: str) -> bool:
    return bool(_ASCII.search(text or ""))


def job_text(zh: str) -> str:
    """The text the worker feeds inference_cross_lingual: marker first, then the line."""
    return zh if "<|endofprompt|>" in zh else AB_PREFIX + zh


def min_take_dur(text: str) -> float:
    """tts_worker._synth_clean's floor: shorter than this is a collapsed take (hanzi only, so the prefix does not count)."""
    return sum(1 for ch in text if "一" <= ch <= "鿿") * 0.10 + 0.3


def dur_bin(dur: float, bins=AB_DUR_BINS) -> int | None:
    for k, (lo, hi) in enumerate(bins):
        if dur >= lo and (hi is None or dur < hi):
            return k
    return None


def stratify(cands: list[dict], n_per_bin: int = AB_N_PER_BIN, bins=AB_DUR_BINS) -> list[dict]:
    """Per duration bin, *n_per_bin* lines spread over the chunk (every k-th by unit_id), judgeable
    lines first so the guard has a baseline; short bins fill up with what exists."""
    out: list[dict] = []
    for k in range(len(bins)):
        pool = [c for c in cands if dur_bin(c["dur"], bins) == k]
        chosen: list[dict] = []
        for tier in (True, False):
            tp = sorted([c for c in pool if bool(c.get("judgeable", True)) == tier and c not in chosen],
                        key=lambda c: (c.get("unit_id", 0), c["index"]))
            need = n_per_bin - len(chosen)
            if need <= 0 or not tp:
                continue
            if len(tp) <= need:
                chosen += tp
            else:
                step = len(tp) / need
                chosen += [tp[int(j * step)] for j in range(need)]
        out += [dict(c, bin=k) for c in chosen]
    return out


def speedup_stats(durs: list[float], slots: list[float], soft: float, hard: float) -> dict:
    ratios = [d / s for d, s in zip(durs, slots) if s > 0]
    n = len(ratios)
    return {"n": n, "median_ratio": (statistics.median(ratios) if ratios else None),
            "gt_soft": sum(1 for r in ratios if r > soft) / n if n else None,
            "gt_hard": sum(1 for r in ratios if r > hard) / n if n else None}


def consistency(embs: list[np.ndarray]) -> dict:
    """Cosine of each line's whole-clip embedding to the arm centroid: p10, min, spread (p90 − p10)."""
    if len(embs) < 3:
        return {"n": len(embs), "p10": None, "min": None, "spread": None}
    m = np.stack(embs)
    c = m.mean(axis=0); c /= max(np.linalg.norm(c), 1e-9)
    cos = m @ c
    return {"n": len(embs), "p10": float(np.percentile(cos, 10)), "min": float(cos.min()),
            "spread": float(np.percentile(cos, 90) - np.percentile(cos, 10))}


def pref_threshold(n: int, alpha: float = AB_PREF_ALPHA) -> int:
    """Smallest k with P(X ≥ k | n, ½) ≤ alpha — the preference count that is evidence, not noise."""
    for k in range(n + 1):
        if sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n <= alpha:
            return k
    return n + 1


def _ge(a, b, tol=0.0) -> bool:
    return a is not None and b is not None and a >= b - tol - 1e-9


def _le(a, b, tol=0.0) -> bool:
    return a is not None and b is not None and a <= b + tol + 1e-9


def decide(metrics: dict, prefs: dict | None = None) -> dict:
    """Pre-registered rule.  A wins only if every condition holds; A′ is never a winner.

    (a) language: no kana and no non-zh line in A; median hanzi overlap ≥ B's.
    (b) guard ok-rate and reference-ratio pass-rate ≥ B's − AB_RATE_TOL on the same judged lines.
    (c) median ECAPA-to-reference ≥ B's + AB_SIM_MARGIN; consistency p10 ≥ B's − AB_CONSISTENCY_TOL.
    (d) natural duration vs slot: > soft share ≤ SFT baseline + AB_SOFT_SPEEDUP_TOL, > hard share ≤
        baseline + AB_HARD_SPEEDUP_TOL, no collapsed take.
    (e) listeners prefer A on ≥ pref_threshold(n) pairs (absent prefs → condition pending, no win).
    """
    A, B, base = metrics["arms"].get("A"), metrics["arms"].get("B"), metrics.get("baseline") or {}
    if not A or not B:
        return {"verdict": "keep_sft_vc", "reason": "arm A or B missing", "conditions": {}}
    cond = {
        "a_language": A["kana_lines"] == 0 and A["non_zh_lines"] == 0 and _ge(A["overlap_median"], B["overlap_median"]),
        "b_guard": _ge(A["guard_ok_rate"], B["guard_ok_rate"], AB_RATE_TOL) and _ge(A["ratio_pass_rate"], B["ratio_pass_rate"], AB_RATE_TOL),
        "c_timbre": _ge(A["sim_median"], (B["sim_median"] + AB_SIM_MARGIN) if B["sim_median"] is not None else None)
                    and _ge(A["consistency"]["p10"], B["consistency"]["p10"], AB_CONSISTENCY_TOL),
        "d_timing": _le(A["speedup"]["gt_soft"], base.get("speedup", {}).get("gt_soft"), AB_SOFT_SPEEDUP_TOL)
                    and _le(A["speedup"]["gt_hard"], base.get("speedup", {}).get("gt_hard"), AB_HARD_SPEEDUP_TOL)
                    and A["collapsed"] == 0,
    }
    if prefs:
        votes = [v for v in prefs.get("votes", {}).values() if v in ("A", "B")]
        need = pref_threshold(len(votes)) if votes else None
        cond["e_preference"] = bool(votes) and sum(1 for v in votes if v == "A") >= need
        pref = {"n": len(votes), "A": sum(1 for v in votes if v == "A"), "need": need}
    else:
        cond["e_preference"] = None
        pref = {"n": 0, "A": 0, "need": None}
    win = all(v is True for v in cond.values())
    alt = metrics["arms"].get("A2")
    finding = None
    if alt:
        # the reference-length question: the alternate (≥ 4 s) clip vs the 3.06 s profile clip, same texts
        finding = {"alt_ref_better_timbre": alt["sim_median"] is not None and A["sim_median"] is not None
                   and alt["sim_median"] > A["sim_median"] + 0.02,
                   "alt_ref_better_guard": (alt["guard_ok_rate"] or 0) > (A["guard_ok_rate"] or 0)}
    return {"verdict": "adopt_cross_lingual" if win else "keep_sft_vc", "conditions": cond, "preference": pref,
            "alt_reference_finding": finding}


# ── data access ───────────────────────────────────────────────────

def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)      # type: ignore[union-attr]
    return mod


def _dur(path: str | Path) -> float:
    import soundfile as sf
    try:
        return float(sf.info(str(path)).duration)
    except Exception:                                   # noqa: BLE001
        return 0.0


def pick_lines(state: dict, *, n_per_bin: int = AB_N_PER_BIN, log=print) -> tuple[list[dict], dict]:
    from ai_movie.composer import segment_slots
    from ai_movie.config import VC_GUARD_MIN_BASE_FRAMES
    from ai_movie.pitch import f0_median
    vs, fs = state["vc"]["segments"], state["fit"]["segments"]
    slots = segment_slots(fs)
    cands, excluded = [], {"ascii": 0, "short_text": 0, "not_vc": 0, "no_audio": 0}
    for i, (v, f) in enumerate(zip(vs, fs)):
        if v.get("gender") != "female" or v.get("keep_original"):
            continue
        zh = (v.get("text_translated") or "").strip()
        if not v.get("vc") or not v.get("audio"):
            excluded["not_vc"] += 1; continue
        if visible_chars(zh) < AB_MIN_VISIBLE:
            excluded["short_text"] += 1; continue
        if has_ascii(zh):
            excluded["ascii"] += 1; continue
        if not (f.get("audio_fit") and Path(f["audio_fit"]).exists() and f.get("audio") and Path(f["audio"]).exists()):
            excluded["no_audio"] += 1; continue
        f0v1, nv1 = f0_median(f["audio_fit"])
        cands.append({"index": i, "unit_id": v.get("unit_id", i), "zh": zh, "ja": v.get("text", ""),
                      "start": v["start"], "end": v["end"], "slot": slots[i], "dur": round(_dur(f["audio_fit"]), 3),
                      "sft_raw": f["audio"], "sft_raw_dur": round(_dur(f["audio"]), 3), "v1_fit": f["audio_fit"],
                      "v1_f0": f0v1 and round(f0v1, 1), "v1_voiced": nv1, "judgeable": nv1 >= VC_GUARD_MIN_BASE_FRAMES,
                      "b_raw": v["audio"], "b_fit": v.get("audio_fit"), "vc_guard": v.get("vc_guard")})
    lines = stratify(cands, n_per_bin)
    log(f"candidates {len(cands)} (excluded {excluded}) → {len(lines)} lines, per bin "
        f"{[sum(1 for l in lines if l['bin'] == k) for k in range(len(AB_DUR_BINS))]}")
    return lines, {"candidates": len(cands), "excluded": excluded}


# ── GPU work ──────────────────────────────────────────────────────

def gpu_busy() -> list[str]:
    out = subprocess.run(["pgrep", "-af", "python|bash"], capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if any(p in ln for p in FOREIGN_PATTERNS) and "ab_cross_lingual" not in ln]


def synth_arm(name: str, lines: list[dict], ref: Path, out_dir: Path, *, log=print) -> dict[int, dict]:
    from ai_movie.tts import run_isolated_synthesis
    rp = _load_script("run_pipeline")
    seg_texts = [(l["index"], job_text(l["zh"])) for l in lines]
    seg_refs = {l["index"]: (str(ref), None, "cross_lingual") for l in lines}
    t0 = time.time()
    res = run_isolated_synthesis(seg_texts, "cosyvoice3", str(ref), None, "cross_lingual", out_dir,
                                 progress_cb=lambda d, t: log(f"  {name} {d}/{t}") if d % 10 == 0 or d == t else None,
                                 seg_refs=seg_refs)
    for r in res.values():
        if r.get("audio"):
            rp._trim_silence_inplace(Path(r["audio"]))          # the tts stage's own post-step
    ok = sum(1 for r in res.values() if r.get("audio"))
    log(f"{name}: {ok}/{len(lines)} lines in {time.time() - t0:.0f} s")
    return res


def smoke(ref: Path, out_dir: Path, log=print) -> dict:
    """One worker job, two segments: the bare text (expected: the worker reports the HiFT
    "padded input size … kernel size" RuntimeError — the assert's own message only reaches the
    worker's stderr, which run_isolated_synthesis discards when the exit code is 0) and the
    prefixed text (expected: audio)."""
    from ai_movie.tts import run_isolated_synthesis
    texts = {"bare": AB_SMOKE_TEXT, "prefixed": job_text(AB_SMOKE_TEXT)}
    order = list(texts)
    res = run_isolated_synthesis([(k, texts[t]) for k, t in enumerate(order)], "cosyvoice3", str(ref), None,
                                 "cross_lingual", out_dir / "smoke", seg_refs={k: (str(ref), None, "cross_lingual") for k in range(len(order))})
    rep = {}
    for k, tag in enumerate(order):
        r = res.get(k) or {}
        rep[tag] = {"text": texts[tag], "audio": r.get("audio"), "dur": round(_dur(r["audio"]), 2) if r.get("audio") else None,
                    "error": r.get("tts_error")}
        log(f"smoke {tag}: {rep[tag]}")
    rep["marker_required"] = bool(rep["bare"]["error"]) and bool(rep["prefixed"]["audio"])
    return rep


# ── judges ────────────────────────────────────────────────────────

class Judges:
    def __init__(self, ref: Path, log=print):
        from ai_movie.asr import _load_cpu_model
        from ai_movie.config import ASR_MODEL_SIZE
        from ai_movie.diarize import _load_encoder, _load_mono16k
        from ai_movie.pitch import f0_median
        self.log = log
        log(f"loading whisper {ASR_MODEL_SIZE} (CPU int8 unless CUDA answers) …")
        self.asr = _load_cpu_model(ASR_MODEL_SIZE)
        self.enc = _load_encoder("cpu")
        self._mono = _load_mono16k
        self.ref_f0, _ = f0_median(ref)
        self.ref_emb = self.embed(ref)

    def embed(self, path: str | Path) -> np.ndarray | None:
        import torch
        a = self._mono(path)
        if len(a) < int(0.5 * 16000):
            return None
        with torch.no_grad():
            e = self.enc.encode_batch(torch.from_numpy(a[None, :])).squeeze().cpu().numpy()
        return e / max(np.linalg.norm(e), 1e-9)

    def hear(self, path: str | Path) -> tuple[str, str]:
        segs, info = self.asr.transcribe(str(path), language=None, beam_size=5, vad_filter=False)
        return "".join(s.text for s in segs).strip(), info.language

    def judge(self, path: str | Path, line: dict) -> dict:
        from ai_movie.pitch import f0_median, gate
        from ai_movie.vc_guard import judge_line
        g = judge_line(str(path), line["v1_fit"], "female")
        f0 = g.get("f0_conv")
        pg = gate(self.ref_f0, f0, "female")
        heard, lang = self.hear(path)
        want = set(_HANZI.findall(line["zh"]))
        emb = self.embed(path)
        dur = _dur(path)
        return {"index": line["index"], "dur": round(dur, 3), "slot": line["slot"], "speedup": round(dur / line["slot"], 3) if line["slot"] else None,
                "collapsed": dur < min_take_dur(line["zh"]),
                "guard": g, "gate": pg, "f0": f0, "lang": lang, "kana": len(_KANA.findall(heard)),
                "overlap": round(len(want & set(_HANZI.findall(heard))) / max(1, len(want)), 3), "heard": heard[:60],
                "sim_ref": (round(float(emb @ self.ref_emb), 4) if emb is not None else None), "_emb": emb}


def summarize(rows: list[dict], soft: float, hard: float) -> dict:
    judged = [r for r in rows if r["guard"].get("judged")]
    ok = [r for r in judged if r["guard"]["ok"]]
    reasons: dict[str, int] = {}
    for r in judged:
        if not r["guard"]["ok"]:
            reasons[r["guard"]["reason"].split("_")[0]] = reasons.get(r["guard"]["reason"].split("_")[0], 0) + 1
    gates = [r for r in rows if r["f0"] is not None]
    sims = [r["sim_ref"] for r in rows if r["sim_ref"] is not None]
    return {"n": len(rows), "judged": len(judged), "guard_ok_rate": (len(ok) / len(judged)) if judged else None, "guard_reasons": reasons,
            "ratio_pass_rate": (sum(1 for r in gates if r["gate"]["ok"]) / len(gates)) if gates else None,
            "f0_median": (statistics.median(r["f0"] for r in gates) if gates else None),
            "sim_median": (statistics.median(sims) if sims else None),
            "consistency": consistency([r["_emb"] for r in rows if r["_emb"] is not None]),
            "kana_lines": sum(1 for r in rows if r["kana"] > 0), "non_zh_lines": sum(1 for r in rows if r["lang"] != "zh"),
            "overlap_median": statistics.median(r["overlap"] for r in rows) if rows else None,
            "collapsed": sum(1 for r in rows if r["collapsed"]),
            "speedup": speedup_stats([r["dur"] for r in rows], [r["slot"] for r in rows], soft, hard),
            "dur_median": statistics.median(r["dur"] for r in rows) if rows else None}


def _strip_emb(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "_emb"} for r in rows]


# ── listening kit ─────────────────────────────────────────────────

def listening_kit(lines: list[dict], a_files: dict[int, str], out: Path, seed: int = 20260930) -> None:
    from ai_movie.composer import fit_audio_to_slot
    pairs = out / "pairs"; pairs.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    key, rows = {}, []
    for k, l in enumerate(sorted(lines, key=lambda l: l["index"])):
        a = a_files.get(l["index"]); b = l.get("b_fit") or l["b_raw"]
        if not a or not Path(a).exists() or not Path(b).exists():
            continue
        a_fit = pairs / f"_A_{l['index']:04d}.fit.wav"
        fit_audio_to_slot(Path(a), l["slot"], a_fit)
        letters = ["X", "Y"]; rng.shuffle(letters)
        for src, letter in ((a_fit, letters[0]), (Path(b), letters[1])):
            (pairs / f"{k:02d}_{letter}.wav").write_bytes(Path(src).read_bytes())
        key[f"{k:02d}"] = {letters[0]: "A", letters[1]: "B", "index": l["index"]}
        rows.append({"pair": f"{k:02d}", "index": l["index"], "zh": l["zh"], "ja": l["ja"], "slot": l["slot"], "v1_dur": l["dur"]})
    (out / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["pair"]); w.writeheader(); w.writerows(rows)
    (out / "prefs.json").write_text(json.dumps({"votes": {p: "" for p in key}}, ensure_ascii=False, indent=1), encoding="utf-8")
    need = pref_threshold(len(key))
    (out / "README.md").write_text("\n".join([
        "# 试听：cross_lingual（A）对 SFT→VC（B）", "",
        f"`pairs/NN_X.wav` 与 `pairs/NN_Y.wav` 是同一句台词的两种合成，X/Y 随机分配（答案在 `key.json`，先别看）。",
        "两者都已按该句的时隙压缩到会出片的时长。每对听完，在 `prefs.json` 的 `votes` 里填 `X` 或 `Y`（更像同一个人、更自然者），无法区分填 `=`。",
        f"填好后：`.venv/bin/python scripts/ab_cross_lingual.py --decide metrics.json --prefs prefs.json` 把 X/Y 换算成 A/B。",
        f"预登记：{len(key)} 对里 A 至少赢 {need} 对（单侧二项 p ≤ {AB_PREF_ALPHA}）才算听感占优；其余条件见 metrics.json 的 decision。", ""]), encoding="utf-8")


def prefs_to_arms(prefs: dict, key: dict) -> dict:
    votes = {}
    for p, v in (prefs.get("votes") or {}).items():
        v = (v or "").strip().upper()
        votes[p] = key.get(p, {}).get(v, "=" if v == "=" else "") if v in ("X", "Y") else ("=" if v == "=" else "")
    return {"votes": votes}


# ── main ──────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chunk", nargs="?", default="SONE-846_p01", help="workspace dir with state.json (vc + fit stages)")
    ap.add_argument("--out", type=Path, default=None, help="default deliver/<chunk>/ab_xl")
    ap.add_argument("--ref", type=Path, default=None, help="profile reference (default: state vc.refs of the female speakers)")
    ap.add_argument("--alt-ref", type=Path, default=None, help="A′ reference (default: <ref stem>_alt0.wav next to it)")
    ap.add_argument("--no-alt", action="store_true")
    ap.add_argument("--n-per-bin", type=int, default=AB_N_PER_BIN)
    ap.add_argument("--smoke", action="store_true", help="only the marker check on one line")
    ap.add_argument("--judge-only", action="store_true", help="skip synthesis; judge the wavs already in --out")
    ap.add_argument("--decide", type=Path, default=None, help="metrics.json to (re)decide")
    ap.add_argument("--prefs", type=Path, default=None, help="prefs.json filled in by the listener (X/Y)")
    args = ap.parse_args()

    if args.decide:
        metrics = json.loads(args.decide.read_text(encoding="utf-8"))
        prefs = None
        if args.prefs and args.prefs.exists():
            key = json.loads((args.decide.parent / "key.json").read_text(encoding="utf-8"))
            prefs = prefs_to_arms(json.loads(args.prefs.read_text(encoding="utf-8")), key)
        metrics["decision"] = decide(metrics, prefs)
        args.decide.write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(metrics["decision"], ensure_ascii=False, indent=1))
        return 0

    from ai_movie.config import TTS_FIT_MAX_SPEEDUP, TTS_FIT_MAX_SPEEDUP_HARD
    ws = ROOT / "workspace" / args.chunk
    state = json.loads((ws / "state.json").read_text(encoding="utf-8"))
    out = args.out or (ROOT / "deliver" / args.chunk / "ab_xl")
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "run.log", "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    ref = args.ref
    if ref is None:
        refs = {r.get("ref_audio") for r in (state["vc"].get("refs") or {}).values() if r.get("gender") == "female" and r.get("ref_audio")}
        if len(refs) != 1:
            raise SystemExit(f"need exactly one female reference in vc.refs, got {refs}; pass --ref")
        ref = Path(refs.pop())
    alt = args.alt_ref or ref.with_name(f"{ref.stem}_alt0{ref.suffix}")
    log(f"reference {ref} ({_dur(ref):.2f} s); alt {alt if alt.exists() and not args.no_alt else '—'}")

    if not args.judge_only:
        busy = gpu_busy()
        if busy:
            raise SystemExit("GPU work is running, refusing to start:\n  " + "\n  ".join(busy))
        from ai_movie.translator import free_gpu_for_local_work
        free_gpu_for_local_work(log_cb=log)
    if args.smoke:
        rep = smoke(ref, out, log)
        (out / "smoke.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"marker_required={rep['marker_required']}")
        return 0 if rep["marker_required"] else 1

    lines, pick_info = pick_lines(state, n_per_bin=args.n_per_bin, log=log)
    (out / "lines.json").write_text(json.dumps({"lines": lines, **pick_info}, ensure_ascii=False, indent=1), encoding="utf-8")

    arms_files: dict[str, dict[int, str]] = {"B": {l["index"]: l["b_raw"] for l in lines}}
    arm_refs = {"A": ref} | ({"A2": alt} if alt.exists() and not args.no_alt else {})
    for name, r in arm_refs.items():
        d = out / name
        if args.judge_only:
            files = {l["index"]: str(d / f"seg_{l['index'] + 1:04d}.wav") for l in lines}
            arms_files[name] = {i: p for i, p in files.items() if Path(p).exists()}
        else:
            res = synth_arm(name, lines, r, d, log=log)
            arms_files[name] = {i: v["audio"] for i, v in res.items() if v.get("audio")}
            errs = {i: v.get("tts_error") for i, v in res.items() if not v.get("audio")}
            if errs:
                log(f"{name}: {len(errs)} lines failed: {json.dumps(errs, ensure_ascii=False)[:500]}")

    J = Judges(ref, log)
    by_index = {l["index"]: l for l in lines}
    metrics: dict = {"chunk": args.chunk, "reference": str(ref), "alt_reference": (str(alt) if "A2" in arm_refs else None),
                     "ref_f0": J.ref_f0 and round(J.ref_f0, 1), "n_lines": len(lines), "pick": pick_info,
                     "thresholds": {"soft": TTS_FIT_MAX_SPEEDUP, "hard": TTS_FIT_MAX_SPEEDUP_HARD, "sim_margin": AB_SIM_MARGIN,
                                    "consistency_tol": AB_CONSISTENCY_TOL, "rate_tol": AB_RATE_TOL,
                                    "soft_tol": AB_SOFT_SPEEDUP_TOL, "hard_tol": AB_HARD_SPEEDUP_TOL, "pref_alpha": AB_PREF_ALPHA},
                     "arms": {}, "lines": {}}
    # SFT baseline for timing: the built-in line's natural duration against the same slots
    metrics["baseline"] = {"speedup": speedup_stats([l["sft_raw_dur"] for l in lines], [l["slot"] for l in lines],
                                                    TTS_FIT_MAX_SPEEDUP, TTS_FIT_MAX_SPEEDUP_HARD)}
    for name, files in arms_files.items():
        rows = []
        for i, p in sorted(files.items()):
            rows.append(J.judge(p, by_index[i]))
            log(f"  {name} {i:4d} dur {rows[-1]['dur']:.2f} f0 {rows[-1]['f0']} guard {rows[-1]['guard']['ok']} {rows[-1]['guard']['reason']} "
                f"lang {rows[-1]['lang']} kana {rows[-1]['kana']} ov {rows[-1]['overlap']} sim {rows[-1]['sim_ref']}")
        metrics["arms"][name] = summarize(rows, TTS_FIT_MAX_SPEEDUP, TTS_FIT_MAX_SPEEDUP_HARD)
        metrics["lines"][name] = _strip_emb(rows)
    metrics["decision"] = decide(metrics, None)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    if arms_files.get("A"):
        listening_kit(lines, arms_files["A"], out)
    for name, s in metrics["arms"].items():
        log(f"{name}: guard ok {s['guard_ok_rate']} ({s['judged']} judged, {s['guard_reasons']}) ratio-pass {s['ratio_pass_rate']} "
            f"f0 {s['f0_median']} sim {s['sim_median']} p10 {s['consistency']['p10']} kana {s['kana_lines']} non-zh {s['non_zh_lines']} "
            f"overlap {s['overlap_median']} collapsed {s['collapsed']} speedup>{TTS_FIT_MAX_SPEEDUP} {s['speedup']['gt_soft']} "
            f">{TTS_FIT_MAX_SPEEDUP_HARD} {s['speedup']['gt_hard']}")
    log(f"baseline SFT speedup: {metrics['baseline']['speedup']}")
    log(f"decision (before listening): {json.dumps(metrics['decision'], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
