#!/usr/bin/env python3
"""Read a long film's burned-in subtitle rows with a local OCR model into a
``screen.json``-compatible truth file — without touching the human entries.

    python scripts/screen_ocr.py SONE-846 --out _scratch/ocr            # CPU, bundled PP-OCRv3 (zh rows only, see below)
    python scripts/screen_ocr.py SONE-846 --out _scratch/ocr --source both --rec-model models/ocr/japan_PP-OCRv3_rec_infer.onnx
    python scripts/screen_ocr.py SONE-846 --out _scratch/ocr --engine ollama --ollama-model <vision model>   # GPU
    python scripts/screen_ocr.py SONE-846 --out _scratch/ocr --ids 3,4,5 --min-conf 0.7

Why: eval_long's L2b gate (ASR similarity to the screen JA) is decided by the
cues a person has read from the tiles — 28 interview cues today out of 514
probes with a subtitle — so the gate is a small-sample estimate.  This script
reads every strip that ``scripts/screen_subs.py`` cut, keeps the readings the
OCR engine is confident about, and writes them next to the human file:

    <out>/screen_ocr.json       {"id": ["zh", "ja"]}  — kept rows only, "" for a
                                dropped row; ``eval_long.read_truth`` merges it
                                under ``screen.json`` (a human entry always wins)
    <out>/screen_ocr.meta.json  every reading with its score, source and flags,
                                so the gate can be re-cut without re-running OCR
    <out>/reconcile.md          OCR vs the human entries (the accuracy check),
                                the score-cutoff curve and the disagreements

Rows: ``scan.json`` gives each row's frame-y bounds.  The 5 human one-row
entries fix the rule — a row starting above ``ZH_JA_SPLIT_Y`` (990) is the
Chinese line, at or below it the Japanese line (top rows start at y 929–957,
bottom rows at 1000–1013).

Engines: ``rapid`` runs any PP-OCR recognition model through the installed
``rapidocr_onnxruntime`` recogniser (recognition only: the strip *is* the text
box, and bypassing the package's detector also bypasses its silent
``text_score`` 0.5 cut, so every score reaches the meta file).  The bundled
model is the Chinese PP-OCRv3 whose dictionary holds 1 hiragana and 4 katakana
— it reads the zh row and cannot read the ja row, which is why ja rows are
dropped with ``dict_no_kana`` unless ``--rec-model`` points at a recogniser
whose dictionary covers kana (``OCR_MIN_KANA_IN_DICT``; keep downloaded
recognisers under the main tree's git-excluded ``models/ocr/``, a worktree's
``models/`` vanishes with it).  ``ollama`` sends each row image to a local
vision model; it returns no score, so its confidence is the agreement between
two renderings of the same row (and the raw crop with ``--source both``).

Inputs: ``--source strips`` reads the glyph masks (what the tiles were made
from; the glyphs are eroded and hollow), ``video`` reads the raw band crop of
the source file (better glyphs, picture content stays on this machine),
``both`` keeps the higher-scoring reading per row.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import re
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_movie.content import fold        # noqa: E402  — the normaliser L2b compares with

ZH_JA_SPLIT_Y = 990          # frame y of a row's top: above = zh row, at/below = ja row (human one-row ids 239/241/246/406/866)
OCR_ROW_PAD = 6              # px of margin around a row before recognition (and the strip's inter-row gap)
OCR_UPSCALE = 2              # PP-OCR rec resizes rows to 48 px; strips are 20–58 px, 2× keeps thin strokes
OCR_MIN_SCORE = 0.60         # default --min-conf.  SONE-846 zh rows vs 118 human entries (bundled model, --source both):
                             # the curve is flat 0.50–0.70 (kept 99→94 %, gross 1 %, mean CER 0.011) and coverage
                             # falls off above 0.75 (81 %) — 0.60 keeps 98 % at 1 % gross; reconcile.md prints the curve
OCR_MAX_GROSS_ERR = 0.05     # adopt rule: share of kept rows with CER > OCR_GROSS_CER
OCR_GROSS_CER = 0.20
OCR_MAX_MEAN_CER = 0.05      # adopt rule: mean CER of kept rows vs the human entries
OCR_MIN_KANA_IN_DICT = 100   # a rec dictionary with fewer kana cannot read the ja row (ch PP-OCRv3/v4: 5)
OCR_STRIP_GAP = 6            # screen_subs.py:107 stacks rows with a 6-px zero gap
EXCLUDE_FLAGS = ("empty", "low_score", "row_dup", "dict_no_kana")    # advisory flags (no_kana, unscored) keep the row

_KANA = re.compile(r"[぀-ゟ゠-ヿ]")
_HANZI = re.compile(r"[一-鿿]")

_RAPID_DEFAULT_MODEL = "ch_PP-OCRv3_rec_infer.onnx"       # bundled with rapidocr_onnxruntime 1.2.3 (dict embedded, 5 kana)
OLLAMA_PROMPTS = {
    "ja": "画像は映画に焼き込まれた日本語字幕の1行です。書かれている文字を一字一句そのまま書き起こしてください。"
          "説明・翻訳・引用符は不要で、字幕の文字列だけを出力してください。",
    "zh": "图片是电影里烧录的一行中文字幕。请逐字原样抄写这一行文字，不要解释、不要翻译、不要加引号，只输出字幕文本。",
}


# ── pure helpers (tested) ─────────────────────────────────────────

def row_role(y0: int) -> str:
    return "zh" if y0 < ZH_JA_SPLIT_Y else "ja"


def strip_rows(strip: np.ndarray, lines: list[tuple[int, int]], gap: int = OCR_STRIP_GAP) -> list[np.ndarray]:
    """Undo screen_subs.py's stacking: row k starts at Σ_{j<k}(h_j + gap) and is h_k = y1 − y0 + 1 tall."""
    out, y = [], 0
    for a, b in lines:
        h = int(b) - int(a) + 1
        out.append(strip[y:y + h])
        y += h + gap
    return out


def prep_row(row: np.ndarray, *, upscale: int = OCR_UPSCALE, invert: bool = True, pad: int = OCR_ROW_PAD,
             dilate: int = 0) -> np.ndarray:
    """Glyph mask (white on black) → BGR text line the recogniser expects (dark text on light)."""
    g = row if row.ndim == 2 else cv2.cvtColor(row, cv2.COLOR_BGR2GRAY)
    if dilate:
        g = cv2.dilate(g, np.ones((dilate + 1, dilate + 1), np.uint8))
    if invert:
        g = 255 - g
    if upscale != 1:
        g = cv2.resize(g, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    g = cv2.copyMakeBorder(g, pad, pad, pad * 2, pad * 2, cv2.BORDER_CONSTANT, value=255 if invert else 0)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def prep_raw(crop: np.ndarray, *, upscale: int = OCR_UPSCALE) -> np.ndarray:
    """Raw band crop (white outlined glyphs on picture) → BGR line, upscaled like the mask."""
    if upscale != 1:
        crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return crop


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a or not b:
        return len(a) + len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    """Character error rate after ``fold`` (punctuation/katakana-insensitive, like L2b)."""
    h, r = fold(hyp or ""), fold(ref or "")
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(h, r) / len(r)


_JUNK = " 　\t：:·・‘’“”\"'`／/|｜"     # stray mask pixels at the row edge read as these (4+3+2 rows of 492)


def clean_reading(text: str) -> str:
    """Strip whitespace and edge junk; sentence punctuation (？。、，) stays — fold() handles it."""
    return (text or "").strip(_JUNK)


def agreement(a: str, b: str) -> float:
    """Symmetric similarity of two readings after ``fold``: 1 − edits / longer length."""
    fa, fb = fold(a or ""), fold(b or "")
    n = max(len(fa), len(fb))
    return 1.0 if n == 0 else 1.0 - levenshtein(fa, fb) / n


def kana_count(text: str) -> int:
    return len(_KANA.findall(text or ""))


def row_flags(role: str, text: str, score: float | None, *, min_score: float = OCR_MIN_SCORE,
              other: str | None = None, dict_has_kana: bool = True) -> list[str]:
    """Why a reading is (not) truth.  Only EXCLUDE_FLAGS drop the row; the rest are recorded."""
    flags: list[str] = []
    t = fold(text or "")
    if not t:
        flags.append("empty")
    if score is None:
        flags.append("unscored")
    elif score < min_score:
        flags.append("low_score")
    if other is not None and t and t == fold(other):
        flags.append("row_dup")
    if role == "ja":
        if not dict_has_kana:
            flags.append("dict_no_kana")
        elif t and kana_count(t) == 0:
            flags.append("no_kana")          # 「今回撮影、何本目?」 is valid kana-free Japanese: advisory only
    return flags


def is_kept(flags: list[str]) -> bool:
    return not any(f in EXCLUDE_FLAGS for f in flags)


def calibrate_cutoff(pairs: list[tuple[float, float]], *, max_gross: float = OCR_MAX_GROSS_ERR,
                     gross_cer: float = OCR_GROSS_CER, floor: float = 0.5, step: float = 0.05,
                     min_pairs: int = 10) -> float | None:
    """Smallest score cutoff (≥ *floor*) at which the gross-error share among the KEPT rows is
    ≤ *max_gross* — the quantity the adopt rule uses, so a few gross errors cannot force the
    cutoff to their maximum score and collapse coverage.  None when no cutoff on the grid
    achieves it or there are fewer than *min_pairs* (score, CER) pairs."""
    if len(pairs) < min_pairs:
        return None
    grid = [round(floor + k * step, 2) for k in range(int(round((1.0 - floor) / step)) + 1)]
    for c in grid:
        kept = [e for s, e in pairs if s >= c]
        if not kept:
            return None
        if sum(1 for e in kept if e > gross_cer) / len(kept) <= max_gross:
            return c
    return None


def merge_truth(human: dict[int, list[str]], ocr: dict[int, list[str]]) -> tuple[dict[int, list[str]], dict[int, str]]:
    """screen.json ∪ screen_ocr.json: a human entry always wins (even with an empty row — the
    person judged that row unreadable); an OCR entry is used only for ids the person never read."""
    read = {int(k): list(v) for k, v in human.items()}
    src = {k: "human" for k in read}
    for k, v in ocr.items():
        k = int(k)
        if k in read or not (v[0] or v[1]):
            continue
        read[k] = list(v)
        src[k] = "ocr"
    return read, src


def best_reading(readings: list[dict]) -> dict | None:
    """Among the readings of one row (one per source), keep the highest score; unscored engines
    get the agreement between their renderings as the score."""
    if not readings:
        return None
    scored = [r for r in readings if r.get("score") is not None]
    if scored:
        return max(scored, key=lambda r: r["score"])
    texts = [r["text"] for r in readings]
    agree = None
    if len(texts) >= 2:
        agree = statistics.mean(agreement(texts[i], texts[j]) for i in range(len(texts)) for j in range(i + 1, len(texts)))
    return dict(readings[0], score=agree, agreement=agree)


# ── engines ───────────────────────────────────────────────────────

class RapidRec:
    """Recognition-only PP-OCR through rapidocr_onnxruntime's TextRecognizer (any rec .onnx)."""

    def __init__(self, model_path: str | Path | None = None, keys_path: str | Path | None = None):
        import rapidocr_onnxruntime as ro
        from rapidocr_onnxruntime.ch_ppocr_v3_rec import TextRecognizer
        pkg = Path(ro.__file__).resolve().parent
        self.model_path = str(model_path or pkg / "models" / _RAPID_DEFAULT_MODEL)
        cfg = {"use_cuda": False, "model_path": self.model_path, "rec_img_shape": [3, 48, 320], "rec_batch_num": 8}
        if keys_path:
            cfg["keys_path"] = str(keys_path)
        self.rec = TextRecognizer(cfg)
        chars = self.rec.postprocess_op.character
        self.dict_size = len(chars)
        self.dict_kana = sum(1 for c in chars if len(c) == 1 and _KANA.match(c))
        self.dict_has_kana = self.dict_kana >= OCR_MIN_KANA_IN_DICT

    def describe(self) -> dict:
        return {"engine": "rapid", "model": self.model_path, "dict_size": self.dict_size,
                "dict_kana": self.dict_kana, "dict_has_kana": self.dict_has_kana}

    def read(self, images: list[np.ndarray]) -> list[tuple[str, float]]:
        if not images:
            return []
        res, _ = self.rec(images)
        return [(str(t), float(s)) for t, s in res]


class OllamaReader:
    """A local vision LLM reads one row image (no score: see best_reading)."""

    def __init__(self, model: str, base_url: str | None = None, timeout: float = 120.0):
        from ai_movie.config import OLLAMA_BASE_URL
        self.model, self.base_url, self.timeout = model, (base_url or OLLAMA_BASE_URL).rstrip("/"), timeout
        self.dict_has_kana = True

    def describe(self) -> dict:
        return {"engine": "ollama", "model": self.model, "base_url": self.base_url, "dict_has_kana": True}

    def read_one(self, image: np.ndarray, role: str) -> str:
        import urllib.request
        ok, png = cv2.imencode(".png", image)
        if not ok:
            return ""
        payload = json.dumps({"model": self.model, "prompt": OLLAMA_PROMPTS[role], "stream": False, "think": False,
                              "images": [base64.b64encode(png.tobytes()).decode("ascii")],
                              "options": {"temperature": 0, "num_predict": 96}, "keep_alive": "10m"}).encode()
        req = urllib.request.Request(f"{self.base_url}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            text = json.loads(resp.read().decode("utf-8")).get("response", "")
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        line = next((ln.strip() for ln in text.strip().splitlines() if ln.strip()), "")
        return line.strip("「」『』\"'` ")


# ── frame access ──────────────────────────────────────────────────

def _screen_subs():
    spec = importlib.util.spec_from_file_location("screen_subs", ROOT / "scripts" / "screen_subs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                   # type: ignore[union-attr]
    return mod


def raw_rows(frame: np.ndarray, lines: list[tuple[int, int]], xr: tuple[int, int], pad: int = OCR_ROW_PAD) -> list[np.ndarray]:
    """Raw band crops per row: scan.json bounds are inclusive, so y1 + 1 + pad."""
    h = frame.shape[0]
    return [frame[max(0, a - pad):min(h, b + 1 + pad), xr[0]:xr[1]] for a, b in lines]


# ── the run ───────────────────────────────────────────────────────

def read_film(scr: Path, engine, *, source: str = "strips", video: Path | None = None, ids: set[int] | None = None,
              min_conf: float = OCR_MIN_SCORE, log=print) -> dict[str, dict]:
    """Per probe id: ``{"zh": reading, "ja": reading, "lines": [...]}`` with every source's reading kept."""
    scan = json.loads((scr / "scan.json").read_text(encoding="utf-8"))
    probes = [p for p in scan if p.get("sub") and (ids is None or p["id"] in ids)]
    ss = _screen_subs() if source in ("video", "both") else None
    if ss is not None and (video is None or not video.exists()):
        raise SystemExit(f"--source {source} needs the source video (--video), got {video}")
    rows: dict[str, dict] = {}
    t0 = time.time()
    for n, p in enumerate(probes, 1):
        cid, lines = p["id"], [tuple(x) for x in p["lines"]]
        rec: dict = {"lines": lines}
        per_role: dict[str, list[dict]] = {}
        variants: list[tuple[str, str, np.ndarray]] = []       # (role, src, image)
        if source in ("strips", "both"):
            strip = cv2.imread(str(scr / "strips" / f"c{cid:04d}.png"), 0)
            if strip is None:
                rec["error"] = "strip missing"
            else:
                for (a, b), row in zip(lines, strip_rows(strip, lines)):
                    variants.append((row_role(a), "mask", prep_row(row)))
                    if isinstance(engine, OllamaReader):
                        variants.append((row_role(a), "mask3x", prep_row(row, upscale=3, dilate=1)))
        if source in ("video", "both"):
            t = p.get("t", (p["start"] + p["end"]) / 2)
            frame = ss.frame_at(video, t)
            if frame is None:
                rec["error"] = "frame missing"
            else:
                for (a, b), crop in zip(lines, raw_rows(frame, lines, ss.XR)):
                    variants.append((row_role(a), "raw", prep_raw(crop)))
        if isinstance(engine, OllamaReader):
            results = [(engine.read_one(img, role), None) for role, _, img in variants]
        else:
            results = engine.read([img for _, _, img in variants])
        for (role, src, _), (text, score) in zip(variants, results):
            per_role.setdefault(role, []).append({"text": clean_reading(text), "score": score, "src": src})
        for role, readings in per_role.items():
            best = best_reading(readings)
            rec[role] = dict(best, readings=readings)
        for role in ("zh", "ja"):
            if role in rec:
                other = rec.get("zh" if role == "ja" else "ja")
                rec[role]["flags"] = row_flags(role, rec[role]["text"], rec[role]["score"], min_score=min_conf,
                                               other=other["text"] if other else None,
                                               dict_has_kana=getattr(engine, "dict_has_kana", True))
        rows[str(cid)] = rec
        if n % 50 == 0 or n == len(probes):
            log(f"  {n}/{len(probes)} probes read ({time.time() - t0:.0f} s)")
    return rows


def truth_from_rows(rows: dict[str, dict], min_conf: float) -> dict[str, list[str]]:
    """screen.json shape; a row that fails the gate becomes "" (the id stays if the other row passed)."""
    out: dict[str, list[str]] = {}
    for cid, rec in rows.items():
        vals = []
        for role in ("zh", "ja"):
            r = rec.get(role)
            keep = bool(r) and is_kept(row_flags(role, r["text"], r["score"], min_score=min_conf,
                                                 other=(rec.get("zh" if role == "ja" else "ja") or {}).get("text"),
                                                 dict_has_kana="dict_no_kana" not in (r.get("flags") or [])))
            vals.append(r["text"].strip() if keep else "")
        if vals[0] or vals[1]:
            out[cid] = vals
    return out


def reconcile(rows: dict[str, dict], human: dict[int, list[str]], tile_index: list | None, min_conf: float) -> dict:
    """OCR vs the human entries per role: CER stats, the cutoff curve, the disagreements."""
    tile_of = {}
    for t, ids in (tile_index or []):
        for cid in ids:
            tile_of[int(cid)] = t
    rep: dict = {"roles": {}, "disagreements": [], "min_conf": min_conf}
    for k, role in enumerate(("zh", "ja")):
        pairs, kept_pairs, items = [], [], []
        for cid, hv in human.items():
            r = (rows.get(str(cid)) or {}).get(role)
            if not r or not (hv[k] or "").strip():
                continue
            e = cer(r["text"], hv[k])
            sc = r["score"] if r["score"] is not None else 0.0
            pairs.append((sc, e))
            kept = is_kept(r.get("flags") or []) and sc >= min_conf
            if kept:
                kept_pairs.append((sc, e))
            items.append({"id": cid, "tile": tile_of.get(cid), "score": round(sc, 3), "cer": round(e, 3), "kept": kept,
                          "ocr": r["text"], "human": hv[k], "src": r.get("src"), "flags": r.get("flags") or []})
        if not pairs:
            continue
        errs = [e for _, e in pairs]
        kerrs = [e for _, e in kept_pairs]
        curve = []
        for c in [round(0.5 + 0.05 * i, 2) for i in range(11)]:
            sub = [e for s, e in pairs if s >= c]
            curve.append({"cutoff": c, "kept": len(sub) / len(pairs),
                          "gross": (sum(1 for e in sub if e > OCR_GROSS_CER) / len(sub)) if sub else None,
                          "mean_cer": (statistics.mean(sub) if sub else None)})
        rep["roles"][role] = {
            "n": len(pairs), "median_cer": statistics.median(errs), "mean_cer": statistics.mean(errs),
            "le_010": sum(1 for e in errs if e <= 0.10) / len(errs), "gt_020": sum(1 for e in errs if e > OCR_GROSS_CER) / len(errs),
            "kept_n": len(kept_pairs), "kept_mean_cer": (statistics.mean(kerrs) if kerrs else None),
            "kept_gross": (sum(1 for e in kerrs if e > OCR_GROSS_CER) / len(kerrs)) if kerrs else None,
            "curve": curve, "calibrated_cutoff": calibrate_cutoff(pairs),
            "adopt": bool(kerrs) and statistics.mean(kerrs) <= OCR_MAX_MEAN_CER
                     and (sum(1 for e in kerrs if e > OCR_GROSS_CER) / len(kerrs)) <= OCR_MAX_GROSS_ERR,
        }
        rep["disagreements"] += [dict(it, role=role) for it in items if it["cer"] > OCR_GROSS_CER]
    return rep


def reconcile_md(rep: dict, engine_desc: dict, coverage: dict) -> str:
    L = ["# screen_ocr — OCR 与人工读取的对照", "", f"engine: `{json.dumps(engine_desc, ensure_ascii=False)}`",
         f"min-conf: {rep['min_conf']}", "",
         f"coverage: {coverage['ids']} ids with a kept row / {coverage['probes']} probes; zh kept {coverage['zh']}, ja kept {coverage['ja']}", ""]
    for role, r in rep["roles"].items():
        L += [f"## {role} row — n={r['n']} human pairs",
              f"- CER median {r['median_cer']:.3f}, mean {r['mean_cer']:.3f}, ≤0.10: {r['le_010']:.0%}, >0.20: {r['gt_020']:.0%}",
              f"- kept at min-conf: {r['kept_n']} (mean CER {r['kept_mean_cer'] if r['kept_mean_cer'] is None else round(r['kept_mean_cer'], 3)}, "
              f"gross {r['kept_gross'] if r['kept_gross'] is None else round(r['kept_gross'], 3)}) → adopt: {'yes' if r['adopt'] else 'no'}",
              f"- calibrated cutoff (gross ≤ {OCR_MAX_GROSS_ERR:.0%} among kept): {r['calibrated_cutoff']}",
              "", "| cutoff | kept | gross>0.20 | mean CER |", "|---|---|---|---|"]
        for c in r["curve"]:
            gross = "—" if c["gross"] is None else f"{c['gross']:.0%}"
            mean = "—" if c["mean_cer"] is None else f"{c['mean_cer']:.3f}"
            L.append(f"| {c['cutoff']:.2f} | {c['kept']:.0%} | {gross} | {mean} |")
        L.append("")
    if rep["disagreements"]:
        L += ["## 分歧（CER > 0.20；人工读取来自 3/4 缩放的 tile，本身也可能错）", "",
              "| id | tile | row | score | CER | OCR | human |", "|---|---|---|---|---|---|---|"]
        for d in sorted(rep["disagreements"], key=lambda x: (x["role"], x["id"])):
            L.append(f"| {d['id']} | {d['tile']} | {d['role']} | {d['score']:.2f} | {d['cer']:.2f} | {d['ocr']} | {d['human']} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("film")
    ap.add_argument("--screen", type=Path, default=None, help="_screen dir (default workspace/<film>/_screen)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: the _screen dir)")
    ap.add_argument("--engine", choices=("rapid", "ollama"), default="rapid")
    ap.add_argument("--rec-model", type=Path, default=None, help="PP-OCR recognition .onnx (default: bundled ch PP-OCRv3)")
    ap.add_argument("--keys", type=Path, default=None, help="character dict for --rec-model when not embedded in the onnx")
    ap.add_argument("--ollama-model", default=None)
    ap.add_argument("--source", choices=("strips", "video", "both"), default="strips")
    ap.add_argument("--video", type=Path, default=None, help="source video for --source video/both")
    ap.add_argument("--ids", default=None, help="comma-separated probe ids (a sample)")
    ap.add_argument("--min-conf", type=float, default=OCR_MIN_SCORE)
    ap.add_argument("--no-reconcile", action="store_true")
    args = ap.parse_args()

    scr = args.screen or (ROOT / "workspace" / args.film / "_screen")
    out = args.out or scr
    out.mkdir(parents=True, exist_ok=True)
    if args.engine == "ollama":
        if not args.ollama_model:
            raise SystemExit("--engine ollama needs --ollama-model")
        engine = OllamaReader(args.ollama_model)
    else:
        engine = RapidRec(args.rec_model, args.keys)
        print(f"rec dict: {engine.dict_size} chars, {engine.dict_kana} kana → ja rows {'readable' if engine.dict_has_kana else 'NOT readable (dict_no_kana)'}")
    video = args.video
    if video is None and args.source != "strips":
        cands = sorted(ROOT.glob(f"inputs/{args.film}*.mp4"))
        video = cands[0] if cands else None
    ids = {int(x) for x in args.ids.split(",")} if args.ids else None

    rows = read_film(scr, engine, source=args.source, video=video, ids=ids, min_conf=args.min_conf)
    truth = truth_from_rows(rows, args.min_conf)
    cov = {"probes": len(rows), "ids": len(truth), "zh": sum(1 for v in truth.values() if v[0]),
           "ja": sum(1 for v in truth.values() if v[1])}
    (out / "screen_ocr.json").write_text(json.dumps(truth, ensure_ascii=False, indent=0), encoding="utf-8")
    meta = {"film": args.film, "engine": engine.describe(), "source": args.source, "min_conf": args.min_conf,
            "coverage": cov, "rows": rows}
    (out / "screen_ocr.meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"kept: {cov['ids']} ids ({cov['zh']} zh, {cov['ja']} ja) of {cov['probes']} probes → {out / 'screen_ocr.json'}")

    if not args.no_reconcile and (scr / "screen.json").exists():
        human = {int(k): v for k, v in json.loads((scr / "screen.json").read_text(encoding="utf-8")).items()}
        idx_p = scr / "tiles" / "index.json"
        tile_index = json.loads(idx_p.read_text()) if idx_p.exists() else None
        rep = reconcile(rows, human, tile_index, args.min_conf)
        (out / "reconcile.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "reconcile.md").write_text(reconcile_md(rep, engine.describe(), cov), encoding="utf-8")
        for role, r in rep["roles"].items():
            print(f"{role}: n={r['n']} CER median {r['median_cer']:.3f} mean {r['mean_cer']:.3f} ≤0.10 {r['le_010']:.0%} >0.20 {r['gt_020']:.0%}"
                  f" | kept {r['kept_n']} mean {r['kept_mean_cer'] and round(r['kept_mean_cer'], 3)} gross {r['kept_gross'] and round(r['kept_gross'], 3)}"
                  f" | calibrated cutoff {r['calibrated_cutoff']} | adopt {r['adopt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
