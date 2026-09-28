"""Manual edits to a project's state (translation, speaker labels, glossary).

Every edit: back up state.json → mutate → ``run_pipeline.restamp_after_edit``
(bumps the edited stage's revision inside its fingerprint so downstream
stages go STALE at the next run) → atomic save → refresh the cheap
deliverables (SRT / speaker CSV / glossary JSON).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from . import projects as P

STAGES_WITH_SEGMENTS = ("asr", "translate", "tts", "compact", "fit")
SEED_PATH_OVERRIDE: Path | None = None       # tests point this at a temp file


class EditError(Exception):
    def __init__(self, msg: str, code: int = 400):
        super().__init__(msg)
        self.code = code


def _backup(name: str, state: dict) -> Path:
    d = P.webdir(name) / "state_backups"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    old = sorted(d.glob("*.json"))[:-20]
    for o in old:
        o.unlink(missing_ok=True)
    return p


def _load(name: str) -> dict:
    st = P.load_state(name)
    if not st:
        raise EditError("工程还没有 state.json", 404)
    return st


def _commit(name: str, state: dict, edited: str, keep_valid: list[str]) -> None:
    try:
        P.rp.restamp_after_edit(state, edited, keep_valid)
    except KeyError as exc:
        raise EditError(f"该工程的 {edited} 步没有指纹，请先执行“指纹采用”: {exc}", 409)
    P.rp.save_state(P.state_path(name), state)
    P.invalidate_status(name)


def _refresh_deliverables(name: str, state: dict, what: str) -> None:
    try:
        from ai_movie import artifacts
        deliver = P.workdir(name) / "deliverables"
        deliver.mkdir(parents=True, exist_ok=True)
        if what in ("asr", "translate"):
            segs = (state.get("translate") or {}).get("segments") or (state.get("asr") or {}).get("segments") or []
            if segs:
                artifacts.export_speaker_csv(segs, deliver / "01_speakers.csv")
                artifacts.export_srt(segs, deliver / "01_asr.ja.srt", speaker_prefix=True)
                chosen = (state.get("translate") or {}).get("chosen")
                if chosen and any(s.get("text_translated") for s in segs):
                    artifacts.export_srt(segs, deliver / f"02_zh_{chosen}.srt",
                                         text_key="text_translated", speaker_prefix=True)
        if what == "glossary":
            artifacts.export_json(state.get("glossary") or {}, deliver / "02_glossary.json")
    except Exception:                                   # noqa: BLE001
        pass


# ── translation ──

def set_translation(name: str, idx: int, text: str) -> dict:
    st = _load(name)
    tr = st.get("translate") or {}
    segs = tr.get("segments") or []
    if not (0 <= idx < len(segs)):
        raise EditError("segment index out of range", 404)
    text = (text or "").strip()
    if not text:
        raise EditError("译文不能为空")
    _backup(name, st)
    segs[idx]["text_translated"] = text
    for k in ("tts", "compact", "fit"):
        lst = (st.get(k) or {}).get("segments") or []
        if idx < len(lst):
            lst[idx]["text_translated"] = text
            lst[idx].pop("text_translated_full", None)
    chosen = tr.get("chosen")
    if chosen and chosen in (tr.get("variants") or {}):
        var = tr["variants"][chosen]
        if idx < len(var):
            var[idx] = text
    _commit(name, st, "translate", keep_valid=[])
    _refresh_deliverables(name, st, "translate")
    return {"idx": idx, "text_translated": text}


# ── speaker / gender ──

def set_speaker(name: str, idx: int, speaker: str | None = None, gender: str | None = None) -> dict:
    st = _load(name)
    asr = st.get("asr") or {}
    segs = asr.get("segments") or []
    if not (0 <= idx < len(segs)):
        raise EditError("segment index out of range", 404)
    diar = asr.setdefault("diarization", {}) or {}
    asr["diarization"] = diar
    spks = diar.setdefault("speakers", {})
    if gender not in (None, "male", "female"):
        raise EditError("gender must be male/female")
    _backup(name, st)
    seg = segs[idx]
    if speaker:
        if speaker not in spks:
            raise EditError(f"未知说话人 {speaker}", 404)
        seg["speaker"] = speaker
        g = spks[speaker].get("gender") or gender or seg.get("gender")
        seg["gender"] = g
        seg["tts_gender"] = g
    elif gender:
        from ai_movie.diarize import assign_speaker_for_gender
        assign_speaker_for_gender(diar, seg, gender, minted_by="web edit")
    else:
        raise EditError("need speaker or gender")
    _propagate_labels(st, idx, seg)
    _commit(name, st, "asr", keep_valid=["glossary", "translate"])
    _refresh_deliverables(name, st, "asr")
    return {"idx": idx, "speaker": seg.get("speaker"), "gender": seg.get("gender"),
            "speakers": spks}


def _propagate_labels(state: dict, idx: int, src: dict) -> None:
    for k in ("translate", "tts", "compact", "fit"):
        lst = (state.get(k) or {}).get("segments") or []
        if idx < len(lst):
            tgt = lst[idx]
            if abs(float(tgt.get("start", -1)) - float(src.get("start", -2))) > 0.05:
                continue            # index drift — never relabel a different line
            for f in ("speaker", "gender", "tts_gender"):
                if f in src:
                    tgt[f] = src[f]


# ── film-wide speaker profiles (long films) ──

def _film_of(name: str) -> str | None:
    """``SONE-846_p07`` → ``SONE-846`` when that film has profiles."""
    import re as _re
    m = _re.match(r"^(.+)_p\d{2}$", name)
    if m and (P.ROOT / "workspace" / m.group(1) / "profiles.json").exists():
        return m.group(1)
    return None


def set_speaker_profile(name: str, speaker: str, profile: str | None) -> dict:
    """Pin a chunk speaker to a profile (``None`` = back to the automatic choice).

    Marks the enrol stage edited so tts / faces / vc go stale; translate and
    compact stay valid (the words did not change).
    """
    st = _load(name)
    spks = ((st.get("asr") or {}).get("diarization") or {}).get("speakers") or {}
    if speaker not in spks:
        raise EditError(f"未知说话人 {speaker}", 404)
    film = _film_of(name)
    if not film:
        raise EditError("该项目没有全片档案（profiles.json）", 404)
    doc = json.loads((P.ROOT / "workspace" / film / "profiles.json").read_text(encoding="utf-8"))
    if profile is not None and profile not in (doc.get("profiles") or {}):
        raise EditError(f"未知档案 {profile}", 404)
    _backup(name, st)
    enrol = st.setdefault("enrol", {})
    sp = enrol.setdefault("speaker_profile", {})
    if profile is None:
        sp.pop(speaker, None)
    else:
        sp[speaker] = {"profile": profile, "how": "manual", "manual": True}
    if "enrol" in (st.get("_fp") or {}):
        _commit(name, st, "enrol", keep_valid=["glossary", "translate", "compact"])
    else:
        P.state_path(name).write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"speaker": speaker, "profile": profile, "speaker_profile": sp}


def update_profile(film: str, pid: str, patch: dict) -> dict:
    """Edit a film profile: ``ref_audio`` (an existing wav, e.g. one of the
    alternatives), ``merge_into`` (fold this profile into another) or ``name``.
    Bumps ``version`` and marks ``manual`` — every chunk's enrol fingerprint
    (which hashes profiles.json) goes stale at its next run."""
    path = P.ROOT / "workspace" / film / "profiles.json"
    if not path.exists():
        raise EditError("没有全片档案", 404)
    doc = json.loads(path.read_text(encoding="utf-8"))
    profiles = doc.setdefault("profiles", {})
    if pid not in profiles:
        raise EditError(f"未知档案 {pid}", 404)
    prof = profiles[pid]
    if "ref_audio" in patch:
        ref = patch["ref_audio"]
        rp = Path(ref) if ref and Path(ref).is_absolute() else (path.parent / ref if ref else None)
        if rp and not rp.exists():
            raise EditError("参考音文件不存在", 404)
        prof["ref_audio"] = ref
    if patch.get("merge_into"):
        tgt = patch["merge_into"]
        if tgt not in profiles or tgt == pid:
            raise EditError(f"无法并入 {tgt}")
        profiles[tgt].setdefault("sources", []).extend(prof.get("sources") or [])
        profiles[tgt]["seconds"] = round(float(profiles[tgt].get("seconds") or 0) + float(prof.get("seconds") or 0), 1)
        if prof.get("default_for_gender") and not profiles[tgt].get("default_for_gender"):
            profiles[tgt]["default_for_gender"] = True
        del profiles[pid]
        prof = profiles[tgt]
    if "name" in patch:
        prof["name"] = patch["name"]
    if "default_for_gender" in patch and patch["default_for_gender"]:
        for q in profiles.values():
            if q.get("gender") == prof.get("gender"):
                q["default_for_gender"] = False
        prof["default_for_gender"] = True
    prof["manual"] = True
    doc["version"] = int(doc.get("version") or 1) + 1
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    return doc


def add_speaker(name: str, gender: str) -> dict:
    if gender not in ("male", "female"):
        raise EditError("gender must be male/female")
    st = _load(name)
    asr = st.get("asr") or {}
    diar = asr.setdefault("diarization", {}) or {}
    asr["diarization"] = diar
    spks = diar.setdefault("speakers", {})
    _backup(name, st)
    new_id = f"S{max((int(k[1:]) for k in spks if k.startswith('S') and k[1:].isdigit()), default=-1) + 1}"
    spks[new_id] = {"gender": gender, "f0_median": None, "total_speech": 0.0, "n_turns": 0,
                    "synthesized_by": "web edit"}
    P.rp.save_state(P.state_path(name), st)      # no fingerprint change: nothing bound yet
    return {"speaker": new_id, "speakers": spks}


# ── glossary ──

def set_glossary(name: str, terms: dict, apply_to_translation: bool = False) -> dict:
    st = _load(name)
    clean: dict[str, dict] = {}
    for ja, v in (terms or {}).items():
        ja = str(ja).strip()
        if not ja:
            continue
        if isinstance(v, str):
            v = {"zh": v}
        zh = str((v or {}).get("zh") or "").strip()
        if not zh:
            continue
        clean[ja] = {"zh": zh, "kind": (v or {}).get("kind") or "name",
                     "source": (v or {}).get("source") or "user"}
    _backup(name, st)
    old = st.get("glossary") or {}
    st["glossary"] = clean
    # Seed file so a re-run of the glossary stage keeps the user's terms.
    from ai_movie.config import GLOSSARY_PATH
    seed_p = SEED_PATH_OVERRIDE or Path(GLOSSARY_PATH)
    try:
        seed = json.loads(seed_p.read_text(encoding="utf-8")) if seed_p.exists() else {}
    except Exception:                                   # noqa: BLE001
        seed = {}
    for ja, v in clean.items():
        if v.get("source") == "user" or ja in seed:
            seed[ja] = {"zh": v["zh"], "kind": v.get("kind", "name")}
    seed_p.parent.mkdir(parents=True, exist_ok=True)
    seed_p.write_text(json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8")

    changed = 0
    if apply_to_translation:
        # Replace old renderings with the new ones in the existing translation
        # (string level, no re-translation), then only the translate stage moves.
        repl = {}
        for ja, v in clean.items():
            o = (old.get(ja) or {}).get("zh")
            if o and o != v["zh"]:
                repl[o] = v["zh"]
        if repl:
            for k in ("translate", "tts", "compact", "fit"):
                for seg in (st.get(k) or {}).get("segments") or []:
                    t = seg.get("text_translated") or ""
                    t2 = t
                    for o, n_ in repl.items():
                        t2 = t2.replace(o, n_)
                    if t2 != t:
                        seg["text_translated"] = t2
                        if k == "translate":
                            changed += 1
        if "_fp" in st and "glossary" in st["_fp"]:
            P.rp.restamp_after_edit(st, "glossary", keep_valid=["translate"])
        if changed and "_fp" in st and "translate" in st["_fp"]:
            P.rp.restamp_after_edit(st, "translate", keep_valid=[])
        P.rp.save_state(P.state_path(name), st)
        P.invalidate_status(name)
    else:
        _commit(name, st, "glossary", keep_valid=[])
    _refresh_deliverables(name, st, "glossary")
    if changed:
        _refresh_deliverables(name, st, "translate")
    return {"glossary": clean, "applied": changed}


# ── face binding (stored as an option; faces goes STALE via its fingerprint extra) ──

def set_face_binding(name: str, binding: dict) -> dict:
    parts = []
    for spk, tid in (binding or {}).items():
        if tid in (None, "", "none", "null"):
            parts.append(f"{spk}=none")
        else:
            parts.append(f"{spk}={int(tid)}")
    spec = ",".join(parts) if parts else None
    opts = P.save_options(name, {"faces_bind": spec})
    return {"faces_bind": opts.get("faces_bind")}
