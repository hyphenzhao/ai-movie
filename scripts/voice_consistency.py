#!/usr/bin/env python
"""Voice-consistency report for the cloned (v2) dub — is each delivered voice one voice?

    python scripts/voice_consistency.py workspace/<name>/state.json [--profiles P.json]
    python scripts/voice_consistency.py --film SONE-846 [--profiles workspace/SONE-846/profiles.json]

Chunk mode embeds every delivered v2 line and its built-in v1 source with the ECAPA
encoder (CPU, GPU hidden, 4 threads, nice — it runs beside CosyVoice/MuseTalk jobs on
the same unified memory), computes the V1–V6 gates of ``ai_movie.voice_consistency``
and writes
  state["vc"]["consistency"]          (summary, per-line numbers keyed by str(index), gates)
  deliverables/07_voice_consistency.{json,md,csv}   (the CSV is the listening list, worst first)
  deliverables/07_voice_anchor_<voice>.wav          (3 typical converted lines — the reference
                                                     the listener compares each line against)
Film mode walks ``_split/plan.json`` like eval_long, pools every chunk's rows per voice,
adds the leave-one-chunk-out gate (V4) and writes ``deliver/<film>_full/VOICE_CONSISTENCY.*``
— which ``scripts/eval_long.py`` reads (L3b–L3f).  Embeddings are cached per project in
``workspace/<name>/consistency_emb.npz`` keyed by resolved path + mtime + size, so re-evaluating
a film after re-cloning one chunk re-embeds only that chunk.

Exit 2 (never a failure) when the encoder is missing, the state has no vc version, or vc is
stale against fit (``--steps fit --force`` after a v2 run leaves vc pointing at old lines).
"""

from __future__ import annotations

import os

# Before torch is imported: this is an evaluation, it must never touch the GPU or take every core.
for _k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
    os.environ[_k] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse          # noqa: E402
import csv               # noqa: E402
import json              # noqa: E402
import sys               # noqa: E402
import time              # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                    # noqa: E402

from ai_movie import voice_consistency as vcm         # noqa: E402

THREADS = 4
CACHE_NAME = "consistency_emb.npz"


def cache_path(work: Path, cache_dir: Path | None) -> Path:
    """The project's embedding cache; ``--cache-dir`` moves it out of the workspace (read-only evaluation)."""
    return (Path(cache_dir) / f"{work.name}.npz") if cache_dir else (work / CACHE_NAME)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _tc(t: float) -> str:
    h, rem = divmod(float(t), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def _resolve(p: str | None) -> str | None:
    if not p:
        return None
    fp = Path(p)
    if not fp.is_absolute():
        fp = ROOT / fp
    try:
        return str(fp.resolve())
    except OSError:
        return str(fp)


def wav_duration(p: str | None) -> float | None:
    """Duration of a wav (the rule ``run_vc_conversion`` applies), None when unreadable."""
    if not p:
        return None
    fp = Path(_resolve(p) or "")
    if not fp.exists():
        return None
    try:
        import soundfile as sf
        return float(sf.info(str(fp)).duration)
    except Exception:                                   # noqa: BLE001
        return None


def atomic_write_json(path: Path, doc: dict, *, indent: int = 1) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, ensure_ascii=False, indent=indent))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ── embedding cache ────────────────────────────────────────────────

class EmbeddingCache:
    """``resolved path → embedding`` keyed by (path, mtime_ns, size); NaN rows record files that
    could not be embedded (too short, unreadable) so they are not retried every run."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[tuple[str, int, int], np.ndarray] = {}
        if path.exists():
            try:
                z = np.load(path, allow_pickle=False)
                for p, m, s, e in zip(z["paths"], z["mtime_ns"], z["size"], z["emb"]):
                    self.rows[(str(p), int(m), int(s))] = e.astype(np.float32)
            except Exception as exc:                    # noqa: BLE001
                log(f"cache {path.name} unreadable ({exc}) — rebuilding")
                self.rows = {}

    @staticmethod
    def key(resolved: str) -> tuple[str, int, int] | None:
        try:
            st = os.stat(resolved)
        except OSError:
            return None
        return (resolved, int(st.st_mtime_ns), int(st.st_size))

    def lookup(self, paths: list[str | None]) -> tuple[np.ndarray, list[int]]:
        """Embeddings for *paths* (NaN where unknown) and the indices still to embed."""
        E = np.full((len(paths), vcm.DIM), np.nan, np.float32)
        todo = []
        for i, p in enumerate(paths):
            r = _resolve(p)
            k = self.key(r) if r else None
            if k is None:
                continue
            e = self.rows.get(k)
            if e is None:
                todo.append(i)
            else:
                E[i] = e
        return E, todo

    def fill(self, paths: list[str | None], todo: list[int], E: np.ndarray, *, device: str = "cpu") -> None:
        if not todo:
            return
        from ai_movie import diarize
        import torch
        torch.set_num_threads(THREADS)
        sub = [_resolve(paths[i]) for i in todo]
        emb, keep = diarize.embed_files(sub, device=device, batch=16)
        got = {todo[j]: emb[n] for n, j in enumerate(keep)}
        for i in todo:
            k = self.key(_resolve(paths[i]) or "")
            if k is None:
                continue
            e = got.get(i)
            E[i] = e if e is not None else np.nan
            self.rows[k] = E[i].copy()

    def save(self) -> None:
        if not self.rows:
            return
        keys = list(self.rows)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".npz.tmp")
        with open(tmp, "wb") as fh:                     # a file handle: np.savez would append ".npz" to a name
            np.savez(fh, paths=np.array([k[0] for k in keys]), mtime_ns=np.array([k[1] for k in keys], np.int64),
                     size=np.array([k[2] for k in keys], np.int64), emb=np.stack([self.rows[k] for k in keys]))
        os.replace(tmp, self.path)


def embed_rows(rows: list[dict], cache: EmbeddingCache, *, label: str = "") -> tuple[np.ndarray, np.ndarray]:
    p2 = [r["wav2"] for r in rows]
    p1 = [r["wav1"] for r in rows]
    E2, t2 = cache.lookup(p2)
    E1, t1 = cache.lookup(p1)
    if t2 or t1:
        t0 = time.time()
        cache.fill(p2, t2, E2)
        cache.fill(p1, t1, E1)
        cache.save()
        log(f"{label}embedded {len(t2)} v2 + {len(t1)} v1 lines in {time.time() - t0:.1f} s (cache {cache.path})")
    return E2, E1


# ── per-project pieces ─────────────────────────────────────────────

def load_profiles(path: Path | None) -> tuple[dict | None, dict[str, np.ndarray], dict[str, bool]]:
    """(document, {profile: voice vector}, {profile: has reference clip})."""
    if not path or not Path(path).exists():
        return None, {}, {}
    from ai_movie import profiles as pmod
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    vecs, has_ref = {}, {}
    for pid, p in pmod.load_vectors(Path(path), doc).items():
        if p.get("voice") is not None:
            vecs[pid] = np.asarray(p["voice"], np.float32)
        has_ref[pid] = bool(p.get("ref_audio"))
    return doc, vecs, has_ref


def project_rows(state: dict, work: Path, *, chunk: int | None, profiles_doc: dict | None,
                 v1_only: bool = False) -> list[dict]:
    profile_of = vcm.profile_map(state, profiles_doc)
    return vcm.collect_lines(state, profile_of, dur_of=wav_duration, chunk=chunk, v1_only=v1_only)


def write_anchor(rows: list[dict], E2: np.ndarray, group: dict, out: Path, *, min_line_s: float) -> Path | None:
    """Concatenate the 3 long converted lines nearest the voice's centroid — the listener's referent."""
    c = group.get("_c_conv")
    if c is None:
        return None
    cands = []
    for k, r in enumerate(rows):
        if r["key"] == group["key"] and r["vc"] and r["dur_heard"] >= min_line_s and not np.isnan(E2[k]).any():
            cands.append((vcm.cosd(E2[k], c), k))
    cands.sort()
    if not cands:
        return None
    try:
        import soundfile as sf
        pieces, sr0 = [], None
        for _, k in cands[:3]:
            a, sr = sf.read(_resolve(rows[k]["wav2"]), dtype="float32")
            if a.ndim > 1:
                a = a.mean(axis=1)
            if sr0 is None:
                sr0 = sr
            if sr != sr0:
                continue
            pieces += [a, np.zeros(int(0.3 * sr0), np.float32)]
        if not pieces:
            return None
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), np.concatenate(pieces), sr0)
        return out
    except Exception as exc:                            # noqa: BLE001
        log(f"anchor for {group['key']} not written: {exc}")
        return None


def _safe(key: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in key)


def listening_rows(rows: list[dict], summary: dict) -> list[dict]:
    """The CSV: converted lines worst-first, then the fallback lines (each one an audible flip)."""
    by = {(ln["chunk"], ln["i"]): ln for ln in summary["lines"]}
    out = []
    for r in rows:
        ln = by.get((r["chunk"], r["i"])) or {}
        out.append({"chunk": "" if r["chunk"] is None else r["chunk"], "idx": r["i"], "tc": _tc(r["start"]),
                    "speaker": r["speaker"], "gender": r["gender"], "key": r["key"],
                    "dur": r["dur_heard"], "reason": r["reason"] + (f":{r['detail']}" if r.get("detail") else ""),
                    "d_self": ln.get("d_self"), "d_v1": ln.get("d_v1"), "d_builtin": ln.get("d_builtin"),
                    "outlier": int(bool(ln.get("outlier"))), "long": int(bool(ln.get("long"))),
                    "wav2": r["wav2"], "wav1": r["wav1"] or "", "text": r["text"]})
    # judged (long) converted lines first, worst first; then the short ones (distance inflates
    # under 1 s, so they would otherwise crowd the top); then every fallback line
    out.sort(key=lambda x: (0 if x["reason"] == "converted" else 1, -x["long"],
                            -(x["d_self"] if x["d_self"] is not None else -1.0)))
    return out


CSV_COLS = ["chunk", "idx", "tc", "speaker", "gender", "key", "dur", "reason", "d_self", "d_v1", "d_builtin",
            "outlier", "long", "wav2", "wav1", "text"]


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def render_md(title: str, summary: dict, gate_rows: list[dict], cfg: dict, *, film: bool, notes: list[str]) -> str:
    md = [f"# {title} — 音色一致性", "",
          "每个声音（档案或性别）一组。距离 = ECAPA 余弦距离；同一内置音色自身约 0.20–0.25，"
          "内置女声↔内置男声 0.83–0.90。V3 只报告不判定；ok 为空 = 未判定（无参考音/未转换/长句不足）。", "",
          "| 门 | 声音 | 条件 | 结果 | 数据 | 备注 |", "|---|---|---|---|---|---|"]
    for r in gate_rows:
        mark = "—" if r["ok"] is None else ("✅" if r["ok"] else "❌")
        md.append(f"| {r['gate']} | {r['key']} | {r['desc']} | {mark} | {r['value']} | {r['note'] or ''} |")
    md += ["", "## 各声音", ""]
    for key, g in summary["groups"].items():
        st, fb = g["stats"], g["fallback"]
        md.append(f"### {key} ({g['gender']}, speakers {', '.join(g['speakers'])}, ref {'yes' if g['has_ref'] else 'no'})")
        md.append(f"- 台词 {g['n']}，已转换 {g['n_conv']}（≥{cfg['min_line_s']} s 的 {g['n_long']} 句参与判定），"
                  f"交付秒数 {g['seconds']:.1f} s")
        md.append(f"- 回退内置音：{fb['lines']} 句 / {fb['seconds']:.1f} s"
                  + (f"（{fb['share_sec']:.1%} 秒数）" if fb['share_sec'] is not None else "") + f"，原因 {fb['by_reason']}")
        if st["median_long"] is not None:
            md.append(f"- d_self 中位数 {st['median_long']:.3f}（p90 {st['p90_long']:.3f}，p95 {st['p95_long']:.3f}）；"
                      f"内置音同句中位数 {st['builtin_median_long']}；超额 {st['median_excess']}；"
                      f"短句中位数 {st['median_short']}（{st['n_short']} 句）；"
                      f"离群 {st['outliers']}（>{cfg['outlier']}）；与 v1 几乎未变 {st['n_unchanged']} 句")
        if g["shift"] is not None:
            md.append(f"- 偏离内置音色 {g['shift']:.3f}" + (f"；档案向量：转换 {g['profile']['d_conv']:.3f} vs 内置 {g['profile']['d_builtin']:.3f}（增益 {g['profile']['gain']:+.3f}）" if g["profile"] else ""))
        if film and g["chunks"]:
            md.append("- 分块（留一法距离；≥%d 句才判定）：" % cfg["min_chunk_lines"] + ", ".join(
                f"p{c}: n{v['n']} d{'' if v['d_loo'] is None else f'={v['d_loo']:.3f}'}{'*' if v['gated'] else ''}"
                for c, v in g["chunks"].items()))
        md.append("")
    if summary.get("builtin"):
        md.append("内置音色自身（v1 句 → 同性别内置质心）：" + "; ".join(
            f"{g}: n{v['n']} median {v['median']} p95 {v['p95']}" for g, v in summary["builtin"].items()))
    if summary.get("max_dv1_nonvc") is not None:
        md.append(f"未转换句 v2↔v1 最大距离 {summary['max_dv1_nonvc']}（应为 0：索引配对检查）")
    if notes:
        md += ["", "## 备注"] + [f"- {n}" for n in notes]
    md += ["", "L3c/V4 超过阈值 = 去听那一块（换了备选参考音），不是直接否决发布。"]
    return "\n".join(md) + "\n"


# ── chunk mode ─────────────────────────────────────────────────────

def chunk_report(state_path: Path, *, profiles: Path | None = None, write: bool = True,
                 cache_dir: Path | None = None) -> dict:
    """Compute the metric for one project.  Returns the report dict; ``rc`` says why when not computed."""
    from ai_movie.diarize import ecapa_available
    state_path = Path(state_path)
    work = state_path.parent
    state = json.loads(state_path.read_text(encoding="utf-8"))
    mtime0 = state_path.stat().st_mtime_ns
    cfg = vcm.default_cfg()
    if not ecapa_available():
        return {"rc": 2, "why": "ECAPA encoder not available"}
    problem = vcm.pairing_problem(state)
    if problem:
        return {"rc": 2, "why": problem}
    doc, vecs, has_ref = load_profiles(profiles)
    rows = project_rows(state, work, chunk=None, profiles_doc=doc)
    notes = []
    vc = state.get("vc") or {}
    if not vc.get("refs"):
        notes.append("no conversion attempted (vc.refs is empty) — every gate not judged")
    if not rows:
        return {"rc": 2, "why": "no dubbed lines with audio in state.vc"}
    cache = EmbeddingCache(cache_path(work, cache_dir))
    E2, E1 = embed_rows(rows, cache, label=f"{work.name}: ")
    summary = vcm.summarize(rows, E2, E1, profile_vecs=vecs, profile_has_ref=has_ref, cfg=cfg)
    gate_rows = vcm.gates(summary, cfg, film=False)
    lines = {str(ln["i"]): {k: v for k, v in ln.items() if k not in ("chunk", "i")} for ln in summary["lines"]}
    doc_out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "project": work.name,
               "vc_sig": vcm.vc_signature(state), "cfg": cfg,
               "summary": vcm.public_summary(summary), "gates": gate_rows, "notes": notes}
    if write:
        deliver = work / "deliverables"
        deliver.mkdir(parents=True, exist_ok=True)
        anchors = {}
        for key, g in summary["groups"].items():
            if g["n_conv"]:
                a = write_anchor(rows, E2, g, deliver / f"07_voice_anchor_{_safe(key)}.wav", min_line_s=cfg["min_line_s"])
                if a:
                    anchors[key] = str(a)
        doc_out["anchors"] = anchors
        atomic_write_json(deliver / "07_voice_consistency.json", doc_out)
        write_csv(deliver / "07_voice_consistency.csv", listening_rows(rows, summary))
        (deliver / "07_voice_consistency.md").write_text(
            render_md(work.name, summary, gate_rows, cfg, film=False, notes=notes), encoding="utf-8")
        # state.json: re-read, make sure nobody rewrote the vc version meanwhile, then inject.
        if state_path.stat().st_mtime_ns != mtime0:
            fresh = json.loads(state_path.read_text(encoding="utf-8"))
            if vcm.vc_signature(fresh) != doc_out["vc_sig"]:
                return {"rc": 2, "why": "state.json changed during evaluation — rerun", "report": doc_out}
            state = fresh
        state["vc"]["consistency"] = {k: doc_out[k] for k in ("generated_at", "vc_sig", "cfg", "summary", "gates", "notes")}
        state["vc"]["consistency"]["lines"] = lines
        atomic_write_json(state_path, state)
        log(f"{work.name}: wrote state.vc.consistency + {deliver / '07_voice_consistency.*'}")
    doc_out["lines"] = lines
    doc_out["rc"] = 0
    return doc_out


# ── film mode ──────────────────────────────────────────────────────

def film_report(film: str, *, profiles: Path | None = None, write: bool = True,
                cache_dir: Path | None = None, substitute: dict[int, Path] | None = None) -> dict:
    """Pool every chunk of a long film per voice; V4 leave-one-chunk-out; report under deliver/<film>_full.

    *substitute* (``{chunk: dir}``) is the detection-power check: the converted
    lines of that chunk are replaced, in order, by the ``seg_*.wav`` of *dir* — the
    same lines converted onto another reference (``profiles/real_P0/c*``) — so the
    report shows what a re-cloned chunk would score.  Nothing is written then.
    """
    from ai_movie.diarize import ecapa_available, embed_files
    if not ecapa_available():
        return {"rc": 2, "why": "ECAPA encoder not available"}
    substitute = substitute or {}
    if substitute:
        write = False
    cfg = vcm.default_cfg()
    split = ROOT / "workspace" / film / "_split"
    plan_path = split / "plan.json"
    if not plan_path.exists():
        return {"rc": 2, "why": f"{plan_path} missing"}
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if profiles is None and (ROOT / "workspace" / film / "profiles.json").exists():
        profiles = ROOT / "workspace" / film / "profiles.json"
    doc, vecs, has_ref = load_profiles(profiles)
    rows: list[dict] = []
    E2s, E1s = [], []
    chunks_meta: dict[str, dict] = {}
    notes: list[str] = []
    for c in plan["chunks"]:
        i = int(c["index"])
        work = ROOT / "workspace" / f"{film}_p{i:02d}"
        sp = work / "state.json"
        if not sp.exists():
            continue                                    # passthrough chunk: original sound
        state = json.loads(sp.read_text(encoding="utf-8"))
        vc = state.get("vc") or {}
        v1_only = not (vc.get("segments") and vc.get("refs"))
        if not v1_only:
            problem = vcm.pairing_problem(state)
            if problem:
                notes.append(f"p{i:02d}: {problem} — chunk skipped")
                continue
        if v1_only and not ((state.get("fit") or {}).get("segments")):
            continue
        crow = project_rows(state, work, chunk=i, profiles_doc=doc, v1_only=v1_only)
        if v1_only:
            notes.append(f"p{i:02d}: delivered without a cloned version — {len(crow)} lines count as chunk_v1_only")
        cache = EmbeddingCache(cache_path(work, cache_dir))
        e2, e1 = embed_rows(crow, cache, label=f"p{i:02d}: ")
        if i in substitute:
            wavs = sorted(str(w) for w in Path(substitute[i]).glob("seg_*.wav"))
            emb, keep = embed_files(wavs, device="cpu")
            targets = [k for k, r in enumerate(crow) if r["vc"]]
            for n, k in enumerate(targets[:len(keep)]):
                e2[k] = emb[n]
                crow[k]["wav2"] = wavs[keep[n]]
            notes.append(f"p{i:02d}: SUBSTITUTED {min(len(keep), len(targets))} converted lines with {substitute[i]} (detection check)")
        rows += crow
        E2s.append(e2)
        E1s.append(e1)
        chunks_meta[str(i)] = {"state_mtime_ns": sp.stat().st_mtime_ns, "vc_sig": vcm.vc_signature(state),
                               "profiles_sha1": vc.get("profiles_sha1"), "v1_only": v1_only, "n_rows": len(crow),
                               "refs": {k: Path(str((v or {}).get("ref_audio") or "")).name for k, v in (vc.get("refs") or {}).items()}}
    if not rows:
        return {"rc": 2, "why": "no chunk with dubbed lines"}
    E2 = np.concatenate(E2s) if E2s else np.full((0, vcm.DIM), np.nan, np.float32)
    E1 = np.concatenate(E1s) if E1s else np.full((0, vcm.DIM), np.nan, np.float32)
    summary = vcm.summarize(rows, E2, E1, profile_vecs=vecs, profile_has_ref=has_ref, cfg=cfg)
    gate_rows = vcm.gates(summary, cfg, film=True)
    out = ROOT / "deliver" / f"{film}_full"
    doc_out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "film": film,
               "profiles": str(profiles) if profiles else None, "cfg": cfg, "chunks": chunks_meta,
               "summary": vcm.public_summary(summary), "gates": gate_rows, "notes": notes,
               "lines": summary["lines"]}
    if write:
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out / "VOICE_CONSISTENCY.json", doc_out)
        write_csv(out / "VOICE_CONSISTENCY.csv", listening_rows(rows, summary))
        (out / "VOICE_CONSISTENCY.md").write_text(
            render_md(film, summary, gate_rows, cfg, film=True, notes=notes), encoding="utf-8")
        log(f"{film}: wrote {out / 'VOICE_CONSISTENCY.*'}")
    doc_out["rc"] = 0
    return doc_out


def _print_gates(gate_rows: list[dict]) -> None:
    for r in gate_rows:
        mark = "  — " if r["ok"] is None else ("PASS" if r["ok"] else "FAIL")
        print(f"{mark}  {r['id']:<18} {r['value']}" + (f"  ({r['note']})" if r["note"] else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state", nargs="?", help="workspace/<name>/state.json (chunk mode)")
    ap.add_argument("--film", default=None, help="long film name (film mode: pools workspace/<film>_pNN)")
    ap.add_argument("--profiles", type=Path, default=None, help="film-wide profiles.json (film mode defaults to workspace/<film>/profiles.json)")
    ap.add_argument("--no-write", action="store_true", help="compute and print only (no state/deliverable writes)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="keep the embedding caches here instead of workspace/<name>/ (read-only evaluation)")
    ap.add_argument("--substitute", action="append", default=[], metavar="CHUNK=DIR",
                    help="film mode, detection check: score the film as if CHUNK's converted lines were the "
                         "seg_*.wav of DIR (e.g. 7=workspace/SONE-846/profiles/real_P0/c8); implies --no-write")
    ap.add_argument("--nice", type=int, default=19, help="process niceness (default 19: never compete with a GPU job)")
    args = ap.parse_args()
    if args.nice:
        try:
            os.nice(args.nice)
        except OSError:
            pass
    if bool(args.state) == bool(args.film):
        ap.error("give exactly one of <state> or --film")
    t0 = time.time()
    if args.film:
        subs = {}
        for item in args.substitute:
            c, _, d = item.partition("=")
            subs[int(c)] = Path(d)
        rep = film_report(args.film, profiles=args.profiles, write=not args.no_write, cache_dir=args.cache_dir,
                          substitute=subs)
    else:
        rep = chunk_report(Path(args.state), profiles=args.profiles, write=not args.no_write, cache_dir=args.cache_dir)
    if rep.get("rc"):
        log(f"not computed: {rep.get('why')}")
        return 2
    _print_gates(rep["gates"])
    for n in rep.get("notes") or []:
        print(f"note: {n}")
    log(f"done in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
