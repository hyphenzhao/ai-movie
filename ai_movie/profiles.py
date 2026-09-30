"""Film-wide speaker profiles: who is who across the chunks of a long film.

Each chunk of a long film is diarized on its own, so the same actress is a
fresh ``S0`` in every chunk and every chunk cloned a different voice.  Voice
embeddings alone do not link them — the same person measured in an
interview and in a scene scores ≈ 0.28 cosine — so a profile carries both a
**voice centroid** (ECAPA, the embedding ``diarize`` clusters with) and a
**face identity** (ArcFace, from the face track the speaker was bound to),
plus the reference clip every chunk clones from.

Pure functions here (numpy only) so the merging and assignment rules are
unit-testable; the I/O lives in ``scripts/build_profiles.py`` and the
``enrol`` stage of ``scripts/run_pipeline.py``.

``profiles.json``::

    {"version": 1, "film": "...", "built_from": ["<film>_p01", ...],
     "profiles": {"P0": {"gender": "female", "f0_median": 250.1, "seconds": 574.0,
                         "voice_centroid": "profiles/P0.voice.npy",
                         "face_embedding": "profiles/P0.face.npy", "face_thumb": "profiles/P0.jpg",
                         "ref_audio": "profiles/ref_P0.wav", "ref_alternatives": [...], "f0_ref": 244.0,
                         "sources": [{"chunk": "...", "speaker": "S0", "seconds": 203.1, "track": 6}],
                         "default_for_gender": true, "manual": false}}}
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ai_movie.config import (PROFILE_FACE_MIN_COS, PROFILE_MARGIN, PROFILE_MIN_SCORE,
                             PROFILE_MIN_SPEECH_S, PROFILE_VOICE_LINK_DIST)


def _cos(a, b) -> float | None:
    if a is None or b is None:
        return None
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else None


def speaker_centroids(diar: dict, audio: np.ndarray, *, min_speech: float | None = None,
                      embed=None) -> dict[str, dict]:
    """``{speaker: {"voice": 192-d, "seconds", "gender", "f0_median"}}`` from a
    chunk's diarization units (``conf ≥ 0.5``, ``overlap ≤ 0.2``, ``≥ 1 s``).

    *embed* defaults to ``diarize.embed_windows`` (injectable for tests).
    """
    from ai_movie import diarize as dz
    embed = embed or (lambda a, spans: dz.embed_windows(a, spans)[0])
    min_speech = PROFILE_MIN_SPEECH_S if min_speech is None else min_speech
    by: dict[str, list[tuple[float, float]]] = {}
    for u in diar.get("units") or []:
        if u.get("conf", 1.0) < 0.5 or u.get("overlap", 0.0) > 0.2:
            continue
        if float(u["end"]) - float(u["start"]) < 1.0:
            continue
        by.setdefault(u["speaker"], []).append((float(u["start"]), float(u["end"])))
    out = {}
    for spk, spans in by.items():
        secs = sum(b - a for a, b in spans)
        meta = (diar.get("speakers") or {}).get(spk) or {}
        if secs < min_speech:
            continue
        embs = embed(audio, spans)
        if embs is None or len(embs) == 0:
            continue
        c = np.mean(np.asarray(embs, np.float32), axis=0)
        n = float(np.linalg.norm(c))
        out[spk] = {"voice": (c / n) if n > 0 else c, "seconds": round(secs, 1),
                    "gender": meta.get("gender"), "f0_median": meta.get("f0_median")}
    return out


def merge_sources(nodes: list[dict], *, voice_link_dist: float | None = None,
                  face_min_cos: float | None = None) -> list[list[int]]:
    """Union-find over (chunk, speaker) nodes.

    Two nodes are the same person when their voice centroids are closer than
    *voice_link_dist* (cosine distance, same scale as diarization's merge)
    **or** they share a gender and their bound faces match (cosine ≥
    *face_min_cos*).  Different genders never merge.  Each node:
    ``{"gender", "voice": vec|None, "face": vec|None}``.  Returns groups of
    node indices.
    """
    vd = PROFILE_VOICE_LINK_DIST if voice_link_dist is None else voice_link_dist
    fc = PROFILE_FACE_MIN_COS if face_min_cos is None else face_min_cos
    parent = list(range(len(nodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if a.get("gender") and b.get("gender") and a["gender"] != b["gender"]:
                continue
            cv = _cos(a.get("voice"), b.get("voice"))
            cf = _cos(a.get("face"), b.get("face"))
            if (cv is not None and 1 - cv < vd) or (cf is not None and cf >= fc):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(nodes)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: (-sum(nodes[k].get("seconds", 0) for k in g), g[0]))


def pitch_match(f0_a: float | None, f0_b: float | None) -> float:
    if not f0_a or not f0_b:
        return 0.5
    return 1.0 - min(1.0, abs(math.log2(f0_a / f0_b)) / 0.5)


def assign_profiles(speakers: dict[str, dict], profiles: dict[str, dict], *,
                    min_score: float | None = None, margin: float | None = None) -> dict[str, dict]:
    """Map each chunk speaker to a profile id.

    ``speakers[S] = {"gender", "f0_median", "voice": vec|None, "face": vec|None}``;
    ``profiles[P] = {"gender", "f0_median", "voice", "face", "default_for_gender"}``.

    Score = 0.6·cos(voice) + 0.3·cos(face) (when a face exists on both sides,
    else the voice weight is 0.9) + 0.1·pitch match.  A profile wins when its
    score ≥ *min_score* and beats the runner-up by *margin*, or when it is the
    only profile of that gender; otherwise the gender's default profile.
    Returns ``{S: {"profile", "score", "how"}}``.
    """
    ms = PROFILE_MIN_SCORE if min_score is None else min_score
    mg = PROFILE_MARGIN if margin is None else margin
    default = {p.get("gender"): pid for pid, p in profiles.items() if p.get("default_for_gender")}
    out = {}
    for spk, sp in speakers.items():
        g = sp.get("gender")
        cands = [(pid, p) for pid, p in profiles.items()
                 if not g or not p.get("gender") or p.get("gender") == g or sp.get("f0_median") is None]
        scored = []
        for pid, p in cands:
            cv = _cos(sp.get("voice"), p.get("voice"))
            cf = _cos(sp.get("face"), p.get("face"))
            wv, wf = (0.6, 0.3) if cf is not None else (0.9, 0.0)
            score = wv * (cv if cv is not None else 0.0) + wf * (cf or 0.0) + 0.1 * pitch_match(sp.get("f0_median"), p.get("f0_median"))
            scored.append((score, pid, cv, cf))
        scored.sort(reverse=True)
        if len(scored) == 1 and scored[0][1] and (g is None or profiles[scored[0][1]].get("gender") == g):
            out[spk] = {"profile": scored[0][1], "score": round(scored[0][0], 3), "how": "only profile of gender"}
            continue
        if scored and scored[0][0] >= ms and (len(scored) == 1 or scored[0][0] - scored[1][0] >= mg):
            s0 = scored[0]
            out[spk] = {"profile": s0[1], "score": round(s0[0], 3), "how": "matched",
                        "voice_cos": None if s0[2] is None else round(s0[2], 3),
                        "face_cos": None if s0[3] is None else round(s0[3], 3)}
            continue
        pid = default.get(g)
        out[spk] = {"profile": pid, "score": round(scored[0][0], 3) if scored else None,
                    "how": "gender default" if pid else "no profile"}
    return out


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_vectors(path: Path, doc: dict) -> dict[str, dict]:
    """Profiles with their ``voice`` / ``face`` arrays loaded (paths relative to *path*)."""
    base = Path(path).parent
    out = {}
    for pid, p in (doc.get("profiles") or {}).items():
        q = dict(p)
        for key in ("voice_centroid", "face_embedding"):
            f = p.get(key)
            fp = (base / f) if f and not Path(f).is_absolute() else (Path(f) if f else None)
            q["voice" if key == "voice_centroid" else "face"] = np.load(fp) if fp and fp.exists() else None
        out[pid] = q
    return out


def save(path: Path, doc: dict) -> None:
    Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def assignment_signature(path: Path) -> str | None:
    """SHA-1 of what ``assign_profiles`` reads: per profile the gender, pitch, default flag and the
    voice/face vectors.  Reference clips and gate rows are excluded on purpose (see run_pipeline)."""
    import hashlib
    try:
        doc = load(path)
    except (OSError, ValueError):
        return None
    h = hashlib.sha1()
    base = Path(path).parent
    for pid, p in sorted((doc.get("profiles") or {}).items()):
        h.update(json.dumps([pid, p.get("gender"), p.get("f0_median"), bool(p.get("default_for_gender")),
                             p.get("manual"), sorted((s.get("chunk"), s.get("speaker")) for s in (p.get("sources") or []))],
                            sort_keys=True, default=str).encode())
        for key in ("voice_centroid", "face_embedding"):
            f = p.get(key)
            fp = (base / f) if f and not Path(f).is_absolute() else (Path(f) if f else None)
            if fp and fp.exists():
                h.update(fp.read_bytes())
    return h.hexdigest()
