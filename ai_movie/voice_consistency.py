"""Voice-consistency metric for the cloned (v2) dub: is each delivered voice *one* voice?

Why this exists
---------------
``run_vc_version`` converts every built-in-voice line onto one reference clip per
speaker profile and ``vc_guard`` drops the lines whose pitch collapsed.  Nothing
measured whether the lines that were *kept* sound like one person across a film:
the v3.2 long-film failure — every chunk cloned onto a different clip of the same
actress — was heard, not measured, and the guard cannot see it.  Of the eleven
candidate clips for SONE-846's P0 six pass the guard with 0/6 rejected lines, yet
three of those (c8 / c9 / c10) sit 0.72 / 0.52 / 0.43 from the shipped clip on the
ECAPA scale where built-in female vs built-in male is 0.83–0.90.  ECAPA separates
them; the guard's pitch ratio does not.

What is measured
----------------
One row per dubbed v2 line (``state["vc"]["segments"][i]``), paired 1:1 by index
with its built-in source (``state["fit"]["segments"][i]`` — ``run_vc_version``
copies that list).  Rows are grouped into *voices*: the enrol profile of the
speaker (``P0``) or, without profiles, ``gender:<g>``.  Embeddings ``E2``
(delivered) and ``E1`` (built-in) come from :func:`ai_movie.diarize.embed_files`
on CPU; this module only does numpy on them, so it is testable without a model.

Per voice (all distances are cosine distances between unit vectors):

  c_conv     unit mean of E2 over the converted lines (every duration)
  c_builtin  unit mean of E1 over all lines of the gender, film-wide — the SFT
             voice's own centroid
  d_self     1 − E2·c_conv per line, judged on lines whose *delivered* audio is
             ≥ ``VC_CONSIST_MIN_LINE_S`` (short lines inflate the distance:
             median 0.45 under 1 s vs 0.33 above 1.5 s)
  V1  share of the voice's DELIVERED seconds (fit_end − start) left in the
      built-in voice — each such line is a ≈0.8 voice flip; ``by_reason`` says
      why (guard / short / error / no_ref / chunk_v1_only / other)
  V2  median d_self of the long converted lines MINUS the same lines' built-in
      median (d(E1, c_builtin)) — the conversion's *excess* instability over the
      one-voice floor.  An absolute median cannot see a two-reference mix (a
      50/50 pool at pair distance 0.92 only reaches 0.385); V4 can.
  V3  share of long lines with d_self > ``VC_CONSIST_OUTLIER`` — the listening
      list; reported, not gated, until a blind listening check confirms 0.55
  V4  (film) leave-one-chunk-out: d(chunk centroid, centroid of all OTHER
      chunks' converted lines).  The only gate that sees a per-chunk reference
      switch.  A centroid that includes the chunk sits at the half-angle and
      halves the distance (two equal chunks 0.92 apart would read 0.265).
  V5  d(c_conv, c_builtin) — a silent no-op conversion scores 0
  V6  (profiles) d(c_builtin, P.voice) − d(c_conv, P.voice): did the lines move
      toward the enrolled voice (cross-domain vector, direction only)

Thresholds and the calibration behind them are recorded in ``config.py``
(``VC_CONSIST_*``).  A gate is ``None`` ("not judged") rather than failed when
the voice has no reference (built-in by design), nothing was converted, or too
few long lines exist to place a median.

What it cannot see: the converted wavs of guard-dropped lines score 0.28–0.45
from the centroid — inside the normal spread.  ECAPA sees *reference-level*
voice changes (centroid shifts of 0.4–0.9), not per-line octave collapse, so
``vc_guard`` stays and must not be weakened on this metric's evidence.  Lines
converted together (``tts._vc_chunks``, ≈1.5 s groups) share one inference
call, so outliers arrive in adjacent runs — count events, not lines, when
reading the listening list.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable

import numpy as np

GATE_IDS = ("V1", "V2", "V3", "V4", "V5", "V6")
REASONS = ("guard", "short", "error", "no_ref", "chunk_v1_only", "other")
DIM = 192


def default_cfg() -> dict:
    """Thresholds from ``config`` (every name has a fallback so an older config still evaluates)."""
    from ai_movie import config as c
    return {
        "min_line_s": getattr(c, "VC_CONSIST_MIN_LINE_S", 1.5),
        "min_lines": getattr(c, "VC_CONSIST_MIN_LINES", 8),
        "max_median_excess": getattr(c, "VC_CONSIST_MAX_MEDIAN_EXCESS", 0.15),
        "outlier": getattr(c, "VC_CONSIST_OUTLIER", 0.55),
        "max_outlier_share": getattr(c, "VC_CONSIST_MAX_OUTLIER_SHARE", None),
        "min_chunk_lines": getattr(c, "VC_CONSIST_MIN_CHUNK_LINES", 8),
        "min_rest_lines": getattr(c, "VC_CONSIST_MIN_REST_LINES", 20),
        "max_chunk_dist": getattr(c, "VC_CONSIST_MAX_CHUNK_DIST", 0.35),
        "max_fallback_sec": getattr(c, "VC_CONSIST_MAX_FALLBACK_SEC", 0.10),
        "min_shift": getattr(c, "VC_CONSIST_MIN_SHIFT", 0.30),
        "min_profile_gain": getattr(c, "VC_CONSIST_MIN_PROFILE_GAIN", 0.10),
        "unchanged": getattr(c, "VC_CONSIST_UNCHANGED", 0.30),
        # tts.run_vc_conversion(min_seconds=0.7): a line whose *fitted wav* is shorter keeps the
        # built-in voice; the slot (end − start) disagrees with the wav on 20 % of lines.
        "vc_min_seconds": 0.7,
    }


# ── vectors ────────────────────────────────────────────────────────

def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cosd(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))


def centroid(E: np.ndarray) -> np.ndarray | None:
    """Unit mean of the valid (non-NaN) rows, or None when there are none."""
    if E is None or len(E) == 0:
        return None
    ok = ~np.isnan(E).any(axis=1)
    if not ok.any():
        return None
    return unit(E[ok].mean(axis=0))


def _valid(E: np.ndarray, i: int) -> bool:
    return E is not None and i < len(E) and not np.isnan(E[i]).any()


def _pct(xs: list[float], q: float) -> float | None:
    return float(np.percentile(xs, q)) if xs else None


def _median(xs: list[float]) -> float | None:
    return float(np.median(xs)) if xs else None


# ── state → rows ───────────────────────────────────────────────────

def pairing_problem(state: dict) -> str | None:
    """Why ``state["vc"]["segments"]`` cannot be paired 1:1 with ``state["fit"]["segments"]``.

    ``run_vc_version`` builds the v2 list as a copy of the fitted list, so the
    pairing holds by construction — unless ``fit`` was re-run afterwards (``--steps
    fit --force``), in which case ``vc`` still points at the old lines and every
    v1↔v2 comparison here would be between different sentences.
    """
    vc = (state.get("vc") or {}).get("segments") or []
    fit = (state.get("fit") or {}).get("segments") or []
    if not vc:
        return "state has no vc segments"
    if not fit:
        return "state has no fit segments"
    if len(vc) != len(fit):
        return f"vc has {len(vc)} segments, fit has {len(fit)} — vc is stale vs fit"
    for i, (a, b) in enumerate(zip(vc, fit)):
        for k in ("start", "end", "text_translated"):
            if a.get(k) != b.get(k):
                return f"segment {i}: vc.{k}={a.get(k)!r} ≠ fit.{k}={b.get(k)!r} — vc is stale vs fit"
    return None


def vc_signature(state: dict) -> str:
    """SHA-1 of what the metric was computed from (references, per-line audio, conversion flags),
    so a film report can be recognised as stale after one chunk is re-cloned."""
    vc = state.get("vc") or {}
    body = {
        "refs": {k: str((v or {}).get("ref_audio")) for k, v in (vc.get("refs") or {}).items()},
        "converted": vc.get("converted"),
        "profiles_sha1": vc.get("profiles_sha1"),
        "vc_audio": [(s.get("audio_fit") or s.get("audio"), bool(s.get("vc")))
                     for s in (vc.get("segments") or [])],
        "fit_audio": [s.get("audio_fit") or s.get("audio")
                      for s in ((state.get("fit") or {}).get("segments") or [])],
    }
    return hashlib.sha1(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def profile_map(state: dict, profiles_doc: dict | None = None) -> dict[str, str]:
    """speaker → profile id: the enrol assignment, else the profile ``run_vc_version`` recorded in
    ``vc.refs``, else (with a profiles document) the gender's default profile — the same rule as
    ``run_vc_version._refs_from_profiles``, so a chunk run without enrol does not split a voice
    into ``P0`` and ``gender:female``."""
    out: dict[str, str] = {}
    assigned = ((state.get("enrol") or {}).get("speaker_profile")) or {}
    refs = (state.get("vc") or {}).get("refs") or {}
    default = {}
    if profiles_doc:
        default = {p.get("gender"): pid for pid, p in (profiles_doc.get("profiles") or {}).items()
                   if p.get("default_for_gender")}
    speakers = (((state.get("asr") or {}).get("diarization") or {}).get("speakers")) or {}
    names = set(speakers) | set(assigned) | set(refs)
    for spk in names:
        a = assigned.get(spk)
        pid = (a.get("profile") if isinstance(a, dict) else a) or (refs.get(spk) or {}).get("profile")
        if not pid and default:
            g = (speakers.get(spk) or {}).get("gender") or (refs.get(spk) or {}).get("gender")
            pid = default.get(g)
        if pid:
            out[spk] = str(pid)
    return out


def collect_lines(state: dict, profile_of: dict[str, str] | None = None, *,
                  dur_of: Callable[[str], float | None] | None = None,
                  chunk: int | None = None, v1_only: bool = False) -> list[dict]:
    """One row per dubbed line of the v2 version (or of v1 when *v1_only*).

    A row needs a translation, no ``keep_original`` and delivered audio.  ``key``
    is the speaker's profile (``profile_of``) or ``gender:<g>``.  ``reason``
    explains a line that kept the built-in voice, in the order the conversion
    itself decides: ``no_ref`` (speaker has no reference in ``vc.refs``),
    ``guard`` (``vc_guard`` dropped it), ``short`` / ``error`` (``vc_skip``, or —
    for states written before that field existed — the fitted v1 wav under 0.7 s
    via *dur_of*, matching ``tts.run_vc_conversion``'s own rule; the slot length
    is NOT that rule), ``chunk_v1_only`` (a film chunk delivered without a
    cloned version), else ``other``.  ``dur`` is the slot, ``dur_heard`` the
    delivered seconds (``fit_end − start``), which is what V1 counts.
    """
    profile_of = profile_of or {}
    vc = state.get("vc") or {}
    fit = (state.get("fit") or {}).get("segments") or []
    segs = fit if v1_only else (vc.get("segments") or [])
    refs = {} if v1_only else (vc.get("refs") or {})
    min_s = default_cfg()["vc_min_seconds"]
    rows = []
    for i, s in enumerate(segs):
        if s.get("keep_original") or not (s.get("text_translated") or "").strip():
            continue
        wav2 = s.get("audio_fit") or s.get("audio")
        if not wav2:
            continue
        f = fit[i] if i < len(fit) else {}
        wav1 = f.get("audio_fit") or f.get("audio") or (wav2 if v1_only else None)
        spk = s.get("speaker") or ""
        g = s.get("gender") or s.get("tts_gender") or f.get("gender") or "unknown"
        key = profile_of.get(spk) or f"gender:{g}"
        start, end = float(s.get("start") or 0.0), float(s.get("end") or 0.0)
        fit_end = s.get("fit_end")
        heard = (float(fit_end) - start) if fit_end is not None else (end - start)
        conv = bool(s.get("vc")) and not v1_only
        reason, detail = "converted", None
        if not conv:
            if v1_only:
                reason = "chunk_v1_only"
            elif spk not in refs:
                reason = "no_ref"
            elif s.get("vc_guard"):
                reason, detail = "guard", str(s.get("vc_guard"))
            elif s.get("vc_skip"):
                detail = str(s.get("vc_skip"))
                reason = "short" if "short" in detail.lower() else "error"
            else:
                d1 = dur_of(wav1) if (dur_of and wav1) else None
                reason = "short" if (d1 is not None and d1 < min_s) else "other"
        rows.append({
            "i": i, "chunk": chunk, "speaker": spk, "gender": g, "key": key,
            "vc": conv, "reason": reason, "detail": detail,
            "dur": round(end - start, 3), "dur_heard": round(max(0.0, heard), 3),
            "start": start, "end": end,
            "wav2": wav2, "wav1": wav1, "text": s.get("text_translated") or "",
            "has_ref": spk in refs,
        })
    return rows


# ── rows + embeddings → summary ────────────────────────────────────

def summarize(rows: list[dict], E2: np.ndarray, E1: np.ndarray, *,
              chunk_of: dict[int, object] | None = None,
              profile_vecs: dict[str, np.ndarray] | None = None,
              profile_has_ref: dict[str, bool] | None = None,
              cfg: dict | None = None) -> dict:
    """Per-voice statistics.  ``E2[k]`` / ``E1[k]`` belong to ``rows[k]`` (NaN rows = not embedded).

    *chunk_of* maps a row index to its film chunk (film mode; ``rows[k]["chunk"]``
    is used when absent); *profile_vecs* maps a group key to the enrolled voice
    vector (``profiles.load_vectors``); *profile_has_ref* says which profile keys
    own a reference clip (a chunk where no line of the profile was converted
    still belongs to a voice that has one).
    """
    cfg = cfg or default_cfg()
    profile_vecs = profile_vecs or {}
    profile_has_ref = profile_has_ref or {}
    n = len(rows)
    if E2 is None:
        E2 = np.full((n, DIM), np.nan, np.float32)
    if E1 is None:
        E1 = np.full((n, DIM), np.nan, np.float32)

    # built-in centroid per gender, film-wide, over every dubbed line's v1 wav
    by_gender: dict[str, list[int]] = {}
    for k, r in enumerate(rows):
        by_gender.setdefault(r["gender"], []).append(k)
    c_builtin: dict[str, np.ndarray | None] = {}
    builtin_stats: dict[str, dict] = {}
    for g, idx in by_gender.items():
        c = centroid(E1[idx])
        c_builtin[g] = c
        ds = [cosd(E1[k], c) for k in idx if c is not None and _valid(E1, k)]
        builtin_stats[g] = {"n": len(ds), "median": _median(ds), "p95": _pct(ds, 95)}

    groups: dict[str, dict] = {}
    lines: list[dict] = []
    max_dv1_nonvc = None
    for key in sorted({r["key"] for r in rows}):
        idx = [k for k, r in enumerate(rows) if r["key"] == key]
        conv = [k for k in idx if rows[k]["vc"]]
        conv_v = [k for k in conv if _valid(E2, k)]
        c_conv = centroid(E2[conv_v]) if conv_v else None
        genders = [rows[k]["gender"] for k in idx]
        g_major = max(set(genders), key=genders.count) if genders else "unknown"
        cb = c_builtin.get(g_major)

        total_sec = sum(rows[k]["dur_heard"] for k in idx)
        fb = [k for k in idx if not rows[k]["vc"]]
        fb_sec = sum(rows[k]["dur_heard"] for k in fb)
        by_reason: dict[str, int] = {}
        for k in fb:
            by_reason[rows[k]["reason"]] = by_reason.get(rows[k]["reason"], 0) + 1

        d_long, d_short, b_long, n_unchanged, n_outl = [], [], [], 0, 0
        for k in idx:
            r = rows[k]
            rec = {"i": r["i"], "chunk": r["chunk"], "key": key, "speaker": r["speaker"],
                   "vc": r["vc"], "reason": r["reason"], "long": r["dur_heard"] >= cfg["min_line_s"],
                   "d_self": None, "d_v1": None, "d_builtin": None, "outlier": False}
            if _valid(E2, k) and _valid(E1, k):
                rec["d_v1"] = round(cosd(E2[k], E1[k]), 4)
            if r["vc"] and _valid(E2, k) and c_conv is not None:
                d = cosd(E2[k], c_conv)
                rec["d_self"] = round(d, 4)
                if cb is not None:
                    rec["d_builtin"] = round(cosd(E2[k], cb), 4)
                if rec["d_v1"] is not None and rec["d_v1"] < cfg["unchanged"]:
                    n_unchanged += 1
                if rec["long"]:
                    d_long.append(d)
                    if d > cfg["outlier"]:
                        rec["outlier"] = True
                        n_outl += 1
                    gb = c_builtin.get(r["gender"])
                    if gb is not None and _valid(E1, k):
                        b_long.append(cosd(E1[k], gb))
                else:
                    d_short.append(d)
            elif not r["vc"] and rec["d_v1"] is not None:
                max_dv1_nonvc = max(max_dv1_nonvc or 0.0, rec["d_v1"])
            lines.append(rec)

        median_long, builtin_median = _median(d_long), _median(b_long)
        chunks: dict = {}
        conv_chunks: dict[object, list[int]] = {}
        for k in conv_v:
            c = chunk_of.get(k) if chunk_of is not None else rows[k]["chunk"]
            if c is not None:
                conv_chunks.setdefault(c, []).append(k)
        for c, ks in sorted(conv_chunks.items(), key=lambda kv: (0, int(kv[0])) if str(kv[0]).isdigit() else (1, str(kv[0]))):
            own = set(ks)
            rest = [k for k in conv_v if k not in own]
            d_loo = None
            if len(ks) >= 3 and len(rest) >= cfg["min_rest_lines"]:
                d_loo = round(cosd(centroid(E2[ks]), centroid(E2[rest])), 4)
            chunks[str(c)] = {"n": len(ks), "n_rest": len(rest), "d_loo": d_loo,
                              "gated": d_loo is not None and len(ks) >= cfg["min_chunk_lines"]}

        prof = None
        pv = profile_vecs.get(key)
        if pv is not None and c_conv is not None and cb is not None:
            pv = unit(np.asarray(pv, np.float32))
            d_c, d_b = cosd(c_conv, pv), cosd(cb, pv)
            prof = {"d_conv": round(d_c, 4), "d_builtin": round(d_b, 4), "gain": round(d_b - d_c, 4)}

        groups[key] = {
            "key": key, "gender": g_major,
            "speakers": sorted({rows[k]["speaker"] for k in idx}),
            "has_ref": any(rows[k]["has_ref"] for k in idx) or bool(profile_has_ref.get(key)),
            "n": len(idx), "n_conv": len(conv), "n_embedded": len(conv_v), "n_long": len(d_long),
            "seconds": round(total_sec, 2),
            "fallback": {"lines": len(fb), "seconds": round(fb_sec, 2),
                         "share_lines": round(len(fb) / len(idx), 4) if idx else None,
                         "share_sec": round(fb_sec / total_sec, 4) if total_sec > 0 else None,
                         "by_reason": dict(sorted(by_reason.items()))},
            "stats": {"median_long": None if median_long is None else round(median_long, 4),
                      "p90_long": None if not d_long else round(_pct(d_long, 90), 4),
                      "p95_long": None if not d_long else round(_pct(d_long, 95), 4),
                      "median_short": None if not d_short else round(_median(d_short), 4),
                      "n_short": len(d_short),
                      "builtin_median_long": None if builtin_median is None else round(builtin_median, 4),
                      "median_excess": (None if median_long is None or builtin_median is None
                                        else round(median_long - builtin_median, 4)),
                      "outliers": n_outl,
                      "outlier_share": round(n_outl / len(d_long), 4) if d_long else None,
                      "n_unchanged": n_unchanged},
            "shift": None if (c_conv is None or cb is None) else round(cosd(c_conv, cb), 4),
            "profile": prof,
            "chunks": chunks,
            "_c_conv": c_conv,
        }
    return {"groups": groups, "lines": lines, "builtin": builtin_stats,
            "max_dv1_nonvc": None if max_dv1_nonvc is None else round(max_dv1_nonvc, 4),
            "n_rows": n}


# ── summary → gates ────────────────────────────────────────────────

def gates(summary: dict, cfg: dict | None = None, *, film: bool = False) -> list[dict]:
    """V1–V6 per voice.  ``ok`` is True / False / None (not judged, with ``note`` saying why).

    V4 exists only in film mode (it needs chunks).  V6 exists only when the voice
    has an enrolled vector.  V3 is reported with ``ok=None`` unless
    ``max_outlier_share`` is set.
    """
    cfg = cfg or default_cfg()
    out: list[dict] = []
    for key, g in summary["groups"].items():
        st, fb = g["stats"], g["fallback"]

        def add(gid: str, desc: str, ok: bool | None, value: str, note: str | None = None) -> None:
            out.append({"id": f"{gid}[{key}]", "gate": gid, "key": key, "desc": desc,
                        "ok": ok, "value": value, "note": note})

        skip = None
        if not g["has_ref"]:
            skip = "no reference for this voice — built-in by design"
        elif g["n_conv"] == 0:
            skip = "no conversion attempted"
        few = g["n_long"] < cfg["min_lines"]
        few_note = f"only {g['n_long']} converted lines ≥ {cfg['min_line_s']} s (need {cfg['min_lines']})"

        # V1 needs dubbed lines, not converted ones: a voice whose conversion mostly failed is
        # exactly what it must still see.
        v1 = fb["share_sec"]
        add("V1", f"seconds left in the built-in voice ≤ {cfg['max_fallback_sec']:.0%}",
            None if (skip or v1 is None or g["n"] < cfg["min_lines"]) else v1 <= cfg["max_fallback_sec"],
            f"{fb['seconds']:.1f}/{g['seconds']:.1f} s = {v1:.1%}; {fb['lines']}/{g['n']} lines {fb['by_reason']}"
            if v1 is not None else "no seconds",
            skip or (f"only {g['n']} dubbed lines (need {cfg['min_lines']})" if g["n"] < cfg["min_lines"] else None))

        ex = st["median_excess"]
        add("V2", f"median d(line, own centroid) − built-in median ≤ {cfg['max_median_excess']}",
            None if (skip or few or ex is None) else ex <= cfg["max_median_excess"],
            (f"{ex:+.3f} (converted {st['median_long']:.3f} vs built-in {st['builtin_median_long']:.3f}, "
             f"{g['n_long']} lines ≥ {cfg['min_line_s']} s)") if ex is not None else "—",
            skip or (few_note if few else None))

        osh = st["outlier_share"]
        lim = cfg["max_outlier_share"]
        add("V3", f"lines with d > {cfg['outlier']} (listening list)" + (f" ≤ {lim:.0%}" if lim is not None else " — reported only"),
            None if (skip or few or osh is None or lim is None) else osh <= lim,
            f"{st['outliers']}/{g['n_long']} = {osh:.1%}" if osh is not None else "—",
            skip or (few_note if few else ("not gated until the listening check confirms the cutoff" if lim is None else None)))

        if film:
            gated = {c: v for c, v in g["chunks"].items() if v["gated"]}
            reported = {c: v for c, v in g["chunks"].items() if not v["gated"] and v["d_loo"] is not None}
            worst = max(gated.items(), key=lambda kv: kv[1]["d_loo"]) if gated else None
            val = (f"max {worst[1]['d_loo']:.3f} (chunk {worst[0]}, {worst[1]['n']} lines) over {len(gated)} gated chunks"
                   if worst else f"no chunk with ≥ {cfg['min_chunk_lines']} converted lines and ≥ {cfg['min_rest_lines']} elsewhere")
            if reported:
                val += "; reported only: " + ", ".join(f"{c}={v['d_loo']:.2f}(n{v['n']})" for c, v in reported.items())
            add("V4", f"leave-one-chunk-out centroid distance ≤ {cfg['max_chunk_dist']} (chunks with ≥ {cfg['min_chunk_lines']} lines)",
                None if (skip or not gated) else all(v["d_loo"] <= cfg["max_chunk_dist"] for v in gated.values()),
                val, skip or (None if gated else "no gated chunk"))

        sh = g["shift"]
        add("V5", f"converted centroid moved ≥ {cfg['min_shift']} from the built-in voice",
            None if (skip or few or sh is None) else sh >= cfg["min_shift"],
            f"{sh:.3f}" if sh is not None else "—",
            skip or (few_note if few else None))

        if g["profile"] is not None:
            gain = g["profile"]["gain"]
            add("V6", f"moved toward the enrolled voice by ≥ {cfg['min_profile_gain']}",
                None if (skip or few) else gain >= cfg["min_profile_gain"],
                f"{gain:+.3f} (converted {g['profile']['d_conv']:.3f} vs built-in {g['profile']['d_builtin']:.3f} from P.voice)",
                skip or (few_note if few else None))
    return out


def gates_by_id(gate_rows: list[dict]) -> dict[str, dict[str, dict]]:
    """``{"V1": {key: row}, …}`` for the eval scripts."""
    out: dict[str, dict[str, dict]] = {}
    for r in gate_rows:
        out.setdefault(r["gate"], {})[r["key"]] = r
    return out


def public_summary(summary: dict) -> dict:
    """The summary without numpy payloads (``_``-prefixed keys), ready for JSON."""
    return {"groups": {k: {kk: vv for kk, vv in g.items() if not kk.startswith("_")}
                       for k, g in summary["groups"].items()},
            "builtin": summary["builtin"], "max_dv1_nonvc": summary["max_dv1_nonvc"],
            "n_rows": summary["n_rows"]}
