"""Manual edits to a project's state (translation, speaker labels, glossary).

Every edit: validate → back up state.json → mutate → ``run_pipeline.restamp_after_edit``
(bumps the edited stage's revision inside its fingerprint so downstream
stages go STALE at the next run) → atomic save → refresh the cheap
deliverables (SRT / speaker CSV / glossary JSON) → append one line to the
edit log.

The edit log (``workspace/<name>/edits.jsonl``; film-wide profile edits go to
``workspace/<film>/edits.jsonl``) records who changed what, when, from what
to what, and which stages the edit invalidates — the ``_edits`` counters in
state.json only say *how many* times a stage was touched.  It is written
after the save succeeded, so it never claims an edit that did not land, and
it is a side channel: no stage reads it, no fingerprint hashes it (the name
is a module constant on purpose — config.py names are hashed when listed in
``STEP_CONFIG``), and ``runlog.archive_state`` does not copy it.  Readers:
``scripts/runs.py <name> --edits`` and ``GET /api/projects/<name>/edits``.

What an edit invalidates is not always what it restamps: a translation edit
restamps ``translate`` but the stage whose output is now wrong is ``tts``
(compact / fit follow through ``up``); a profile pin restamps ``enrol`` but no
cached stage depends on enrol (``STEP_DEPS``) — the consumer is the v2 clone
(``run_vc_version.py`` reads ``enrol.speaker_profile``).  Each record therefore
carries both ``restamped`` (was ``restamp_after_edit`` called) and
``consumers`` (what has to be redone), so a reader never infers staleness from
a line count.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import projects as P

STAGES_WITH_SEGMENTS = ("asr", "translate", "tts", "compact", "fit")
SEED_PATH_OVERRIDE: Path | None = None       # tests point this at a temp file

EDIT_LOG_NAME = "edits.jsonl"
# Record bounds.  Measured over every workspace state (2026-09-30): the longest
# text_translated is 64 chars (SONE-846_p05), the largest glossary 7 terms, the
# most segments 185 — so 500 chars keeps any real line whole and only a pasted
# essay is cut; 100 list entries keep a whole glossary diff and bound a bulk
# import (the counts n_before/n_after/applied are always complete).  A record
# is written with one os.write on an O_APPEND fd (runlog.append_event), so two
# threadpool workers cannot interleave inside a line whatever its length.
_MAX_TEXT = 500
_MAX_LIST = 100
_BACKUP_KEEP = 20


class EditError(Exception):
    def __init__(self, msg: str, code: int = 400):
        super().__init__(msg)
        self.code = code


# ── paths / log ──

def _edit_log_path(name: str) -> Path:
    return P.workdir(name) / EDIT_LOG_NAME


def _film_dir(film: str) -> Path:
    return P.WORKSPACE / film


def _film_log_path(film: str) -> Path:
    return _film_dir(film) / EDIT_LOG_NAME


def _clip(obj, flag: list | None = None):
    """Bound a record payload: strings to _MAX_TEXT chars, lists/dicts to _MAX_LIST entries.
    Appends to *flag* when something was cut so the record can say ``truncated``."""
    if isinstance(obj, str):
        if len(obj) > _MAX_TEXT:
            if flag is not None:
                flag.append(True)
            return obj[:_MAX_TEXT] + "…"
        return obj
    if isinstance(obj, dict):
        items = list(obj.items())
        if len(items) > _MAX_LIST:
            if flag is not None:
                flag.append(True)
            items = items[:_MAX_LIST]
        return {str(k): _clip(v, flag) for k, v in items}
    if isinstance(obj, (list, tuple, set)):
        lst = list(obj)
        if len(lst) > _MAX_LIST:
            if flag is not None:
                flag.append(True)
            lst = lst[:_MAX_LIST]
        return [_clip(v, flag) for v in lst]
    return obj


def _log_edit(path: Path, op: str, stage: str, *, idx=None, before=None, after=None,
              speaker: dict | None = None, who: str | None = "cli", state: dict | None = None,
              backup: Path | None = None, restamped: bool = True, consumers=(), **extra) -> None:
    """Append one record.  ``stage`` is the stage the edit belongs to (translate / asr /
    glossary / faces / enrol / profiles), ``consumers`` the stages or products that must be
    redone because of it.  Never raises (append_event swallows its own errors)."""
    from ai_movie.runlog import append_event
    trunc: list = []
    rec = {"stage": stage, "idx": idx, "before": _clip(before, trunc), "after": _clip(after, trunc),
           "speaker": speaker or {"before": None, "after": None}, "who": who or "cli",
           "edits_n": (int(((state or {}).get("_edits") or {}).get(stage, 0) or 0)
                       if state is not None else None),
           "backup": backup.name if backup else None, "restamped": bool(restamped),
           "consumers": list(consumers), **_clip(extra, trunc)}
    if trunc:
        rec["truncated"] = True
    append_event(path, op, **rec)


def read_edits(name: str, n: int | None = 50, stage: str | None = None, idx: int | None = None,
               who: str | None = None, film: bool = False) -> list[dict]:
    """Tail of a project's (or, with ``film=True``, a film's) edit log, oldest first.
    Absent/unreadable file → ``[]``; a corrupt line is skipped."""
    from ai_movie.runlog import read_jsonl
    rows = read_jsonl(_film_log_path(name) if film else _edit_log_path(name))
    if stage:
        rows = [r for r in rows if r.get("stage") == stage]
    if idx is not None:
        rows = [r for r in rows if r.get("idx") == idx]
    if who:
        rows = [r for r in rows if r.get("who") == who]
    return rows[-n:] if n else rows


# ── backups ──

def _write_backup(d: Path, data: bytes) -> Path:
    """``<d>/<YYYYmmdd-HHMMSS>-NNN.json``: the suffix is claimed with O_EXCL so two edits in
    the same second (or two threadpool workers) never share a file and the log's ``backup``
    pointer never lies; the newest _BACKUP_KEEP files are kept."""
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for n in range(1, 1000):
        p = d / f"{stamp}-{n:03d}.json"
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        break
    else:                                                   # pragma: no cover — 999 edits in one second
        raise EditError("备份目录在同一秒内已满", 500)
    for o in sorted(d.glob("*.json"))[:-_BACKUP_KEEP]:
        o.unlink(missing_ok=True)
    return p


def _backup(name: str, state: dict) -> Path:
    return _write_backup(P.webdir(name) / "state_backups", json.dumps(state, ensure_ascii=False).encode("utf-8"))


def _backup_file(path: Path) -> Path:
    """Pre-edit copy of a film-level file (profiles.json → profiles_backups/)."""
    return _write_backup(path.parent / f"{path.stem}_backups", path.read_bytes())


# ── state helpers ──

def _load(name: str) -> dict:
    st = P.load_state(name)
    if not st:
        raise EditError("工程还没有 state.json", 404)
    return st


def _require_fp(state: dict, stage: str) -> None:
    """Refuse a legacy state before anything is written (backup, seed file), not after."""
    if stage not in (state.get("_fp") or {}):
        raise EditError(f"该工程的 {stage} 步没有指纹，请先执行“指纹采用”", 409)


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


def _segs(state: dict, stage: str) -> list:
    return (state.get(stage) or {}).get("segments") or []


# ── translation ──

def set_translation(name: str, idx: int, text: str, *, who: str = "cli") -> dict:
    st = _load(name)
    tr = st.get("translate") or {}
    segs = tr.get("segments") or []
    if not (0 <= idx < len(segs)):
        raise EditError("segment index out of range", 404)
    text = (text or "").strip()
    if not text:
        raise EditError("译文不能为空")
    _require_fp(st, "translate")
    seg = segs[idx]
    old = seg.get("text_translated")
    # text_translated_full is created by the compact stage on the compact/tts/fit lists
    # (the translate list never carries it): capture it before it is popped below.
    full = next((_segs(st, k)[idx].get("text_translated_full") for k in ("compact", "tts", "fit")
                 if idx < len(_segs(st, k)) and _segs(st, k)[idx].get("text_translated_full")), None)
    touched = [k for k in ("tts", "compact", "fit") if idx < len(_segs(st, k))]
    bk = _backup(name, st)
    seg["text_translated"] = text
    for k in ("tts", "compact", "fit"):
        lst = _segs(st, k)
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
    _log_edit(_edit_log_path(name), "set_translation", "translate", idx=idx,
              before={"text_translated": old, "text_translated_full": full},
              after={"text_translated": text},
              speaker={"before": seg.get("speaker"), "after": seg.get("speaker")},
              who=who, state=st, backup=bk, restamped=True, consumers=["tts"],
              start=seg.get("start"), end=seg.get("end"), text=seg.get("text"), variant=chosen,
              stages_touched=touched)
    return {"idx": idx, "text_translated": text}


# ── speaker / gender ──

def set_speaker(name: str, idx: int, speaker: str | None = None, gender: str | None = None,
                *, who: str = "cli") -> dict:
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
    if speaker and speaker not in spks:
        raise EditError(f"未知说话人 {speaker}", 404)
    if not speaker and not gender:
        raise EditError("need speaker or gender")
    _require_fp(st, "asr")
    seg = segs[idx]
    before = {k: seg.get(k) for k in ("speaker", "gender", "tts_gender")}
    spks_before = set(spks)
    bk = _backup(name, st)
    if speaker:
        seg["speaker"] = speaker
        g = spks[speaker].get("gender") or gender or seg.get("gender")
        seg["gender"] = g
        seg["tts_gender"] = g
    else:
        from ai_movie.diarize import assign_speaker_for_gender
        assign_speaker_for_gender(diar, seg, gender, minted_by="web edit")
    propagated = _propagate_labels(st, idx, seg)
    present = [k for k in ("translate", "tts", "compact", "fit") if idx < len(_segs(st, k))]
    _commit(name, st, "asr", keep_valid=["glossary", "translate"])
    _refresh_deliverables(name, st, "asr")
    after = {k: seg.get(k) for k in ("speaker", "gender", "tts_gender")}
    _log_edit(_edit_log_path(name), "set_speaker", "asr", idx=idx, before=before, after=after,
              speaker={"before": before["speaker"], "after": after["speaker"]},
              who=who, state=st, backup=bk, restamped=True, consumers=["tts"],
              keep_valid=["glossary", "translate"], requested={"speaker": speaker, "gender": gender},
              minted=sorted(set(spks) - spks_before), propagated=propagated,
              not_propagated=[k for k in present if k not in propagated],
              start=seg.get("start"), end=seg.get("end"), text=seg.get("text"))
    return {"idx": idx, "speaker": seg.get("speaker"), "gender": seg.get("gender"),
            "speakers": spks}


def _propagate_labels(state: dict, idx: int, src: dict) -> list[str]:
    """Copy speaker / gender / tts_gender of segment *idx* into the downstream lists.
    Returns the stages that were relabelled; a stage whose line at *idx* has drifted
    (> 0.05 s start difference) is skipped and therefore absent from the list."""
    done: list[str] = []
    for k in ("translate", "tts", "compact", "fit"):
        lst = _segs(state, k)
        if idx < len(lst):
            tgt = lst[idx]
            if abs(float(tgt.get("start", -1)) - float(src.get("start", -2))) > 0.05:
                continue            # index drift — never relabel a different line
            for f in ("speaker", "gender", "tts_gender"):
                if f in src:
                    tgt[f] = src[f]
            done.append(k)
    return done


# ── film-wide speaker profiles (long films) ──

def _film_of(name: str) -> str | None:
    """``SONE-846_p07`` → ``SONE-846`` when that film has profiles."""
    import re as _re
    m = _re.match(r"^(.+)_p\d{2}$", name)
    if m and (_film_dir(m.group(1)) / "profiles.json").exists():
        return m.group(1)
    return None


def set_speaker_profile(name: str, speaker: str, profile: str | None, *, who: str = "cli") -> dict:
    """Pin a chunk speaker to a profile (``None`` = back to the automatic choice).

    Marks the enrol stage edited when it has a fingerprint, but no cached stage
    depends on enrol (``STEP_DEPS``: tts follows translate only, so a pin does
    not re-synthesize anything).  The consumer is the v2 clone —
    ``run_vc_version.py`` reads ``enrol.speaker_profile`` — which the UI shows
    as stale through ``_web.vc_deps`` (enrol hash recorded when v2 was built).
    Translate and compact stay valid (the words did not change).
    """
    st = _load(name)
    spks = ((st.get("asr") or {}).get("diarization") or {}).get("speakers") or {}
    if speaker not in spks:
        raise EditError(f"未知说话人 {speaker}", 404)
    film = _film_of(name)
    if not film:
        raise EditError("该项目没有全片档案（profiles.json）", 404)
    doc = json.loads((_film_dir(film) / "profiles.json").read_text(encoding="utf-8"))
    if profile is not None and profile not in (doc.get("profiles") or {}):
        raise EditError(f"未知档案 {profile}", 404)
    bk = _backup(name, st)
    enrol = st.setdefault("enrol", {})
    sp = enrol.setdefault("speaker_profile", {})
    before = {"profile": dict(sp[speaker]) if isinstance(sp.get(speaker), dict) else sp.get(speaker)}
    if profile is None:
        sp.pop(speaker, None)
    else:
        sp[speaker] = {"profile": profile, "how": "manual", "manual": True}
    restamped = "enrol" in (st.get("_fp") or {})
    if restamped:
        _commit(name, st, "enrol", keep_valid=["glossary", "translate", "compact"])
    else:                                   # legacy state: no fingerprint to bump, still an atomic save
        P.rp.save_state(P.state_path(name), st)
        P.invalidate_status(name)
    _log_edit(_edit_log_path(name), "set_speaker_profile", "enrol", idx=None, before=before,
              after={"profile": sp.get(speaker)}, speaker={"before": speaker, "after": speaker},
              who=who, state=st, backup=bk, restamped=restamped, consumers=["v2"],
              film=film, requested=profile)
    return {"speaker": speaker, "profile": profile, "speaker_profile": sp}


_PROFILE_KEYS = ("name", "ref_audio", "default_for_gender", "manual", "gender")


def _profile_snapshot(doc: dict, pid: str) -> dict:
    prof = (doc.get("profiles") or {}).get(pid) or {}
    return {**{k: prof.get(k) for k in _PROFILE_KEYS}, "version": doc.get("version"),
            "n_sources": len(prof.get("sources") or []), "seconds": prof.get("seconds")}


def update_profile(film: str, pid: str, patch: dict, *, who: str = "cli") -> dict:
    """Edit a film profile: ``ref_audio`` (an existing wav, e.g. one of the
    alternatives), ``merge_into`` (fold this profile into another), ``name`` or
    ``default_for_gender``.  Bumps ``version`` and marks ``manual``.

    What goes stale: every chunk's enrol fingerprint hashes
    ``profiles.assignment_signature`` — gender, pitch, default flag, manual
    flag, sources and the centroid files — but NOT ref_audio or name.  So the
    first edit of a profile (manual False → True), a merge or a default change
    makes enrol stale on every chunk of the film; a later ref_audio / name
    edit changes nothing cached and is consumed only by the v2 clone.  The
    record's ``consumers`` says which of the two happened.  The pre-edit file
    is kept in ``profiles_backups/`` (a merge deletes a profile).
    """
    path = _film_dir(film) / "profiles.json"
    if not path.exists():
        raise EditError("没有全片档案", 404)
    doc = json.loads(path.read_text(encoding="utf-8"))
    profiles = doc.setdefault("profiles", {})
    if pid not in profiles:
        raise EditError(f"未知档案 {pid}", 404)
    prof = profiles[pid]
    tgt = patch.get("merge_into") or None
    if tgt and (tgt not in profiles or tgt == pid):
        raise EditError(f"无法并入 {tgt}")
    if "ref_audio" in patch:
        ref = patch["ref_audio"]
        rp = Path(ref) if ref and Path(ref).is_absolute() else (path.parent / ref if ref else None)
        if rp and not rp.exists():
            raise EditError("参考音文件不存在", 404)
    from ai_movie.profiles import assignment_signature
    sig_before = assignment_signature(path)
    before = _profile_snapshot(doc, pid)
    bk = _backup_file(path)
    if "ref_audio" in patch:
        prof["ref_audio"] = patch["ref_audio"]
    if tgt:
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
    P.rp.save_state(path, doc)              # atomic (tmp + fsync + replace), same indent=1 layout
    sig_after = assignment_signature(path)
    chunks = sorted(d.name for d in P.WORKSPACE.glob(f"{film}_p??") if (d / "state.json").exists())
    changed = sig_before != sig_after
    _log_edit(_film_log_path(film), "update_profile", "profiles", idx=None, before=before,
              after=_profile_snapshot(doc, tgt or pid), speaker={"before": pid, "after": tgt or pid},
              who=who, state=None, backup=bk, restamped=False,
              consumers=[f"enrol@{c}" for c in chunks] if changed else ["v2"],
              film=film, patch=patch, merged_into=tgt, deleted=[pid] if tgt else [],
              signature_before=sig_before, signature_after=sig_after, chunks=chunks)
    return doc


def add_speaker(name: str, gender: str, *, who: str = "cli") -> dict:
    if gender not in ("male", "female"):
        raise EditError("gender must be male/female")
    st = _load(name)
    asr = st.get("asr") or {}
    diar = asr.setdefault("diarization", {}) or {}
    asr["diarization"] = diar
    spks = diar.setdefault("speakers", {})
    bk = _backup(name, st)
    new_id = f"S{max((int(k[1:]) for k in spks if k.startswith('S') and k[1:].isdigit()), default=-1) + 1}"
    spks[new_id] = {"gender": gender, "f0_median": None, "total_speech": 0.0, "n_turns": 0,
                    "synthesized_by": "web edit"}
    P.rp.save_state(P.state_path(name), st)      # no fingerprint change: nothing bound yet
    _log_edit(_edit_log_path(name), "add_speaker", "asr", idx=None, before=None,
              after={"speaker": new_id, "gender": gender}, speaker={"before": None, "after": new_id},
              who=who, state=st, backup=bk, restamped=False, consumers=[])
    return {"speaker": new_id, "speakers": spks}


# ── glossary ──

def set_glossary(name: str, terms: dict, apply_to_translation: bool = False, *, who: str = "cli") -> dict:
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
    if not apply_to_translation:
        _require_fp(st, "glossary")
    old = st.get("glossary") or {}
    added = sorted(ja for ja in clean if ja not in old)
    removed = sorted(ja for ja in old if ja not in clean)
    changed_terms = [[ja, (old[ja] or {}).get("zh"), clean[ja]["zh"]] for ja in clean
                     if ja in old and (old[ja] or {}).get("zh") != clean[ja]["zh"]]
    bk = _backup(name, st)
    st["glossary"] = clean
    # Seed file so a re-run of the glossary stage keeps the user's terms.
    from ai_movie.config import GLOSSARY_PATH
    seed_p = SEED_PATH_OVERRIDE or Path(GLOSSARY_PATH)
    try:
        seed = json.loads(seed_p.read_text(encoding="utf-8")) if seed_p.exists() else {}
    except Exception:                                   # noqa: BLE001
        seed = {}
    seed_written = 0
    for ja, v in clean.items():
        if v.get("source") == "user" or ja in seed:
            seed[ja] = {"zh": v["zh"], "kind": v.get("kind", "name")}
            seed_written += 1
    seed_p.parent.mkdir(parents=True, exist_ok=True)
    seed_p.write_text(json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8")

    changed = 0
    changed_idx: list[int] = []
    repl: dict[str, str] = {}
    restamped_stages: list[str] = []
    fps = st.get("_fp") or {}
    if apply_to_translation:
        # Replace old renderings with the new ones in the existing translation
        # (string level, no re-translation), then only the translate stage moves.
        for ja, v in clean.items():
            o = (old.get(ja) or {}).get("zh")
            if o and o != v["zh"]:
                repl[o] = v["zh"]
        if repl:
            for k in ("translate", "tts", "compact", "fit"):
                for i, seg in enumerate(_segs(st, k)):
                    t = seg.get("text_translated") or ""
                    t2 = t
                    for o, n_ in repl.items():
                        t2 = t2.replace(o, n_)
                    if t2 != t:
                        seg["text_translated"] = t2
                        if k == "translate":
                            changed += 1
                            changed_idx.append(i)
        # Restamp is conditional here: a legacy state (no fingerprints) is saved without
        # any stale marking — the record's restamped_stages says which happened.
        if "glossary" in fps:
            P.rp.restamp_after_edit(st, "glossary", keep_valid=["translate"])
            restamped_stages.append("glossary")
        if changed and "translate" in fps:
            P.rp.restamp_after_edit(st, "translate", keep_valid=[])
            restamped_stages.append("translate")
        P.rp.save_state(P.state_path(name), st)
        P.invalidate_status(name)
        consumers = ["tts"] if changed else []
    else:
        _commit(name, st, "glossary", keep_valid=[])
        restamped_stages.append("glossary")
        consumers = ["translate"]
    _refresh_deliverables(name, st, "glossary")
    if changed:
        _refresh_deliverables(name, st, "translate")
    _log_edit(_edit_log_path(name), "set_glossary", "glossary", idx=None,
              before={ja: (old.get(ja) or {}).get("zh") for ja in [c[0] for c in changed_terms] + removed},
              after={ja: clean[ja]["zh"] for ja in [c[0] for c in changed_terms] + added},
              who=who, state=st, backup=bk, restamped=bool(restamped_stages), consumers=consumers,
              restamped_stages=restamped_stages, n_before=len(old), n_after=len(clean),
              added=added, removed=removed, changed=changed_terms,
              apply_to_translation=bool(apply_to_translation), applied=changed, replacements=repl,
              changed_idx=changed_idx, seed_path=str(seed_p), seed_written=seed_written)
    return {"glossary": clean, "applied": changed}


# ── face binding (stored as an option; faces goes STALE via its fingerprint extra) ──

def set_face_binding(name: str, binding: dict, *, who: str = "cli") -> dict:
    parts = []
    for spk, tid in (binding or {}).items():
        if tid in (None, "", "none", "null"):
            parts.append(f"{spk}=none")
        else:
            try:
                parts.append(f"{spk}={int(tid)}")
            except (TypeError, ValueError):
                raise EditError(f"track id 必须是整数: {spk}={tid!r}")
    spec = ",".join(parts) if parts else None
    before = P.load_options(name).get("faces_bind")
    opts = P.save_options(name, {"faces_bind": spec})
    # No state.json change (no backup, no restamp): faces goes stale through
    # _args_extra("faces")["faces_bind"] in run_pipeline.step_fingerprint.
    _log_edit(_edit_log_path(name), "set_face_binding", "faces", idx=None,
              before={"faces_bind": before}, after={"faces_bind": opts.get("faces_bind")},
              who=who, state=None, backup=None, restamped=False, consumers=["faces"],
              binding=binding or {})
    return {"faces_bind": opts.get("faces_bind")}
