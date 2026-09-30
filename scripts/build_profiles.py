#!/usr/bin/env python3
"""Build film-wide speaker profiles from a long film's enrolment chunks.

    python scripts/build_profiles.py SONE-846            # → workspace/SONE-846/profiles.json

Enrolment chunks are the dialogue-dense ones (``speech_minutes/minutes ≥
PROFILE_ENROL_DENSITY``, at least the two densest) — interviews, where one
person talks on camera at a time.  They must already have run
``demux,separate,osd,asr`` (run_long.sh does that first).  For each of their
speakers: an ECAPA voice centroid (``profiles.speaker_centroids``), the face
track it binds to and that track's ArcFace identity (``faces``), and
reference-clip candidates (``diarize.extract_speaker_references``).  Speakers
are merged into people (``profiles.merge_sources``), the pooled candidates
are gated by the VC probe (``auto_select_refs.qualify``), and the result is
written as ``profiles.json`` + ``profiles/*.npy|wav|jpg``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_movie import diarize as dz, faces as faces_mod, profiles as prof   # noqa: E402
from ai_movie.config import PROFILE_ENROL_DENSITY, OSD_REF_MAX_OVERLAP, TTS_GENDER_HZ   # noqa: E402

_spec = importlib.util.spec_from_file_location("auto_select_refs", ROOT / "scripts" / "auto_select_refs.py")
asr_refs = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(asr_refs)   # type: ignore[union-attr]


def log(m: str) -> None:
    print(m, flush=True)


def enrolment_chunks(plan: dict) -> list[dict]:
    dense = [c for c in plan["chunks"] if c["minutes"] and c["speech_minutes"] / c["minutes"] >= PROFILE_ENROL_DENSITY]
    top = sorted(plan["chunks"], key=lambda c: -c["speech_minutes"])[:2]
    seen, out = set(), []
    for c in sorted(dense + top, key=lambda c: c["index"]):
        if c["index"] not in seen:
            seen.add(c["index"]); out.append(c)
    return out


def chunk_faces(name: str, video: str, st: dict, segs: list[dict]) -> tuple[dict, dict[int, np.ndarray], dict]:
    """Face plan (cached tracks reused by the faces stage later), identities per track, speaker→track."""
    work = ROOT / "workspace" / name
    plan = faces_mod.build_face_plan(video, segs, out_json=None, tracks_cache=work / "face_tracks_cache.json",
                                     progress_cb=lambda m: log(f"    {m}"))
    tracks = plan["_tracks_full"]
    ids = faces_mod.embed_track_identity(video, tracks, progress_cb=lambda m: log(f"    {m}"))
    return plan, ids, plan["speaker_track"]


def real_probe(members: list[dict], cands: list[str], gender: str, out_dir: Path, n_lines: int = 10) -> list[dict]:
    """Convert real built-in lines of this person onto each candidate; rank by the guard's rejection rate.

    Lines come from any enrolment chunk whose tts/fit stage is done (``audio_fit``); with none, returns [].
    """
    from ai_movie import tts as tts_mod
    from ai_movie.vc_guard import guard_lines
    lines = []
    for m in members:
        sp = ROOT / "workspace" / m["chunk"] / "state.json"
        if not sp.exists():
            continue
        st = json.loads(sp.read_text())
        segs_ = (st.get("fit") or {}).get("segments") or (st.get("tts") or {}).get("segments") or []
        for sg in segs_:
            wav = sg.get("audio_fit") or sg.get("audio")
            if sg.get("speaker") == m["speaker"] and wav and Path(wav).exists() \
                    and not sg.get("keep_original") and (float(sg["end"]) - float(sg["start"])) >= 1.5:
                lines.append(dict(sg, speaker="X", gender=gender, audio_fit=wav))
    if len(lines) < 4:
        log(f"    real probe skipped ({len(lines)} usable lines)")
        return []
    lines = sorted(lines, key=lambda sg: -(float(sg["end"]) - float(sg["start"])))[:n_lines]
    ranked = []
    for k, c in enumerate(cands):
        segs = [dict(sg) for sg in lines]
        items = tts_mod.run_vc_conversion(segs, {"X": {"ref_audio": c, "gender": gender}}, out_dir / f"c{k}")
        stats = guard_lines(segs, items, lines)
        rate = stats["rejected"] / max(1, stats["checked"])
        ranked.append({"path": c, "reject_rate": round(rate, 3), "checked": stats["checked"], "reasons": stats["reasons"]})
        log(f"    real probe {Path(c).name}: {stats['rejected']}/{stats['checked']} lines rejected {stats['reasons'] or ''}")
    # a candidate whose conversion produced nothing measurable (worker died) is not "0 % rejected"
    measured = [r for r in ranked if r["checked"] > 0]
    if not measured:
        log("    real probe measured nothing — keeping the generic probe's pick")
        return []
    measured.sort(key=lambda r: (r["reject_rate"], -r["checked"]))
    return measured + [r for r in ranked if r["checked"] == 0]


def reselect_refs(film: str, *, no_probe: bool = False) -> int:
    """Re-choose ref_audio / ref_alternatives of every profile; everything else in profiles.json stays."""
    pj = ROOT / "workspace" / film / "profiles.json"
    doc = prof.load(pj)
    out_dir = pj.parent / "profiles"
    for pid, p in (doc.get("profiles") or {}).items():
        gender = p.get("gender")
        members = p.get("sources") or []
        cands = []
        for m in members:
            d = ROOT / "workspace" / m["chunk"] / "refs_profile"
            if not d.exists():
                continue
            cands += [str(x) for x in sorted(d.glob(f"ref_{m['speaker']}*.wav"))]
            cands += [str(x) for x in sorted((ROOT / "workspace" / m["chunk"] / "refs_auto").glob(f"cand_{gender}_seg*.wav"))]
        cands = sorted(set(cands))
        rows = []
        for c in cands:
            med, voiced = asr_refs._f0_median(c)
            d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", c],
                                     capture_output=True, text=True).stdout.strip() or 0)
            rows.append({"path": c, "dur": round(d, 2), "voiced": voiced, "f0": med, "reject": asr_refs.candidate_ok(d, voiced, med, gender) if gender in TTS_GENDER_HZ else "no_gender"})
        ok_rows = [r for r in rows if r["reject"] is None]
        log(f"  {pid} ({gender}): {len(cands)} candidates, {len(ok_rows)} pass the static gate")
        ranked = real_probe([dict(m) for m in members], [r["path"] for r in ok_rows], gender, out_dir / f"real_{pid}") if ok_rows and not no_probe else []
        if not ranked:
            log(f"  {pid}: no measurable candidate — reference unchanged"); continue
        picked = ranked[0]["path"]; alts = [r["path"] for r in ranked[1:4] if r["checked"] > 0]
        shutil.copy2(picked, out_dir / f"ref_{pid}.wav")
        for i, a in enumerate(alts):
            shutil.copy2(a, out_dir / f"ref_{pid}_alt{i}.wav")
        p["ref_audio"] = f"profiles/ref_{pid}.wav"
        p["ref_alternatives"] = [f"profiles/ref_{pid}_alt{i}.wav" for i in range(len(alts))]
        p["ref_source"] = Path(picked).name
        p["f0_ref"] = next((r["f0"] for r in rows if r["path"] == picked), None)
        p["ref_gate"] = [{**{k: v for k, v in r.items() if k != "path"}, "file": Path(r["path"]).name,
                          **({"real_reject_rate": rr["reject_rate"], "real_lines": rr["checked"]} if (rr := next((x for x in ranked if x["path"] == r["path"]), None)) else {})}
                         for r in rows]
        log(f"  {pid}: reference ← {Path(picked).name} (real-line rejection {ranked[0]['reject_rate']:.0%} on {ranked[0]['checked']} lines), {len(alts)} alternates")
    doc["version"] = int(doc.get("version") or 1) + 1
    prof.save(pj, doc)
    log(f"updated {pj}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("film")
    ap.add_argument("--no-probe", action="store_true", help="skip the VC probe gate (tests)")
    ap.add_argument("--reselect-refs", action="store_true",
                    help="keep the existing profiles (identities, assignments) and only re-choose each profile's "
                         "reference clip from the enrolment chunks' candidates (generic gate + real-line probe)")
    args = ap.parse_args()
    film = args.film
    if args.reselect_refs:
        return reselect_refs(film, no_probe=args.no_probe)
    split = ROOT / "workspace" / film / "_split"
    plan = json.loads((split / "plan.json").read_text())
    out_dir = ROOT / "workspace" / film / "profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = enrolment_chunks(plan)
    log(f"enrolment chunks: {[c['index'] for c in chunks]}")

    nodes, cand_refs = [], {}
    for c in chunks:
        name = f"{film}_p{c['index']:02d}"
        sp = ROOT / "workspace" / name / "state.json"
        if not sp.exists():
            log(f"  {name}: no state — run demux,separate,osd,asr first"); continue
        st = json.loads(sp.read_text())
        asr = st.get("asr") or {}
        diar = asr.get("diarization") or {}
        segs = asr.get("segments") or []
        if not segs or not diar:
            log(f"  {name}: no asr/diarization"); continue
        voc = (st.get("separate") or {}).get("vocals")
        mix = (st.get("demux") or {}).get("audio")
        mix_a = dz._load_mono16k(mix)
        audio = mix_a
        if voc and Path(voc).exists():                      # the vocals, unless separation gutted them
            audio = dz._pick_embed_source(mix_a, dz._load_mono16k(voc), [(0.0, len(mix_a) / 16000.0)])
        log(f"  {name}: voice centroids…")
        cents = prof.speaker_centroids(diar, audio)
        log(f"    {len(cents)} speakers with ≥ 20 s: {[(k, v['gender'], v['seconds']) for k, v in cents.items()]}")
        log(f"  {name}: faces…")
        fplan, ids, s2t = chunk_faces(name, st["_video"], st, segs)
        log(f"  {name}: reference candidates…")
        refs = dz.extract_speaker_references(diar, segs, mix, vocals_path=voc,
                                             out_dir=ROOT / "workspace" / name / "refs_profile",
                                             n_alternatives=4, overlap_regions=(st.get("osd") or {}).get("regions"))
        for spk, cinfo in cents.items():
            tid = s2t.get(spk)
            nodes.append({"chunk": name, "speaker": spk, "gender": cinfo["gender"], "f0_median": cinfo["f0_median"],
                          "seconds": cinfo["seconds"], "voice": cinfo["voice"],
                          "face": ids.get(tid) if tid is not None else None, "track": tid,
                          "thumb_frame": None})
            r = refs.get(spk) or {}
            alts = [(a.get("ref_audio") or a.get("path")) if isinstance(a, dict) else a for a in (r.get("alternatives") or [])]
            cand_refs[(name, spk)] = [p for p in [r.get("ref_audio"), *alts] if p and Path(p).exists()]
    if not nodes:
        log("no enrolment speakers — nothing to build"); return 1

    groups = prof.merge_sources(nodes)
    doc = {"version": 1, "film": film, "built_from": [f"{film}_p{c['index']:02d}" for c in chunks], "profiles": {}}
    default_seen = set()
    for gi, g in enumerate(groups):
        pid = f"P{gi}"
        members = [nodes[k] for k in g]
        gender = max((m["gender"] for m in members if m["gender"]), key=lambda x: sum(m["seconds"] for m in members if m["gender"] == x), default=None)
        voice = np.mean([m["voice"] for m in members], axis=0); voice /= max(1e-9, np.linalg.norm(voice))
        faces_ = [m["face"] for m in members if m["face"] is not None]
        face = None
        if faces_:
            face = np.mean(faces_, axis=0); face /= max(1e-9, np.linalg.norm(face))
        f0s = [m["f0_median"] for m in members if m["f0_median"]]
        np.save(out_dir / f"{pid}.voice.npy", voice.astype(np.float32))
        if face is not None:
            np.save(out_dir / f"{pid}.face.npy", face.astype(np.float32))
        # reference: pool the members' candidates, gate by the VC probe, keep the best + 3 alternates
        cands = [p for m in members for p in cand_refs.get((m["chunk"], m["speaker"]), [])]
        picked, alts, gate_rows = None, [], []
        if cands and gender in TTS_GENDER_HZ:
            rows = []
            for p in cands:
                med, voiced = asr_refs._f0_median(p)
                d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                         capture_output=True, text=True).stdout.strip() or 0)
                why = asr_refs.candidate_ok(d, voiced, med, gender)
                rows.append({"path": p, "dur": round(d, 2), "voiced": voiced, "f0": med, "reject": why})
            ok_rows = [r for r in rows if r["reject"] is None]
            if ok_rows and not args.no_probe:
                probe_out = out_dir / f"probe_{pid}"
                srcs = [str(ROOT / s) for s in asr_refs.SOURCES[gender]]
                r = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "vc_ref_probe.py"),
                                    "--refs", *[x["path"] for x in ok_rows], "--sources", *srcs, "--out", str(probe_out)],
                                   capture_output=True, text=True, timeout=3600)
                if r.returncode == 0:
                    manifest = json.loads((probe_out / "manifest.json").read_text())
                    by_ref: dict[str, list] = {}
                    for rec in manifest:
                        by_ref.setdefault(rec["ref"], []).append(rec)
                    for x in ok_rows:
                        outs = [asr_refs._f0_median(rec["vc"])[0] for rec in by_ref.get(x["path"], [])]
                        meas = [o for o in outs if o is not None]
                        x["out_f0"] = [o and round(o, 1) for o in outs]
                        x["ratio"] = (round(sorted(meas)[len(meas) // 2] / x["f0"], 3) if meas and x["f0"] else None)
                    good = asr_refs.qualify(ok_rows, gender)
                else:
                    log(f"    probe failed for {pid}: {r.stderr[-300:]}"); good = []
            else:
                good = sorted(ok_rows, key=lambda x: -x["voiced"]) if args.no_probe else []
            gate_rows = rows
            if good:
                picked = good[0]["path"]; alts = [x["path"] for x in good[1:4]]
        # Real-line probe: the generic probe passes clips that wreck real lines (ref_S1 of the first film:
        # 7 of 10 long lines came out without a measurable pitch).  When an enrolment chunk already has its
        # built-in lines synthesized, convert up to ten of this person's real lines onto every gated
        # candidate and keep the one that loses the fewest; the others become the alternates.
        if good and not args.no_probe:
            ranked = real_probe(members, [x["path"] for x in good], gender, out_dir / f"real_{pid}")
            if ranked:
                picked = ranked[0]["path"]; alts = [r["path"] for r in ranked[1:4]]
                for x in gate_rows:
                    rr = next((r for r in ranked if r["path"] == x["path"]), None)
                    if rr:
                        x["real_reject_rate"] = rr["reject_rate"]; x["real_lines"] = rr["checked"]
        ref_rel = None
        if picked:
            shutil.copy2(picked, out_dir / f"ref_{pid}.wav"); ref_rel = f"profiles/ref_{pid}.wav"
            for i, a in enumerate(alts):
                shutil.copy2(a, out_dir / f"ref_{pid}_alt{i}.wav")
        prof_doc = {"gender": gender, "f0_median": round(float(np.median(f0s)), 1) if f0s else None,
                    "seconds": round(sum(m["seconds"] for m in members), 1),
                    "voice_centroid": f"profiles/{pid}.voice.npy",
                    "face_embedding": f"profiles/{pid}.face.npy" if face is not None else None,
                    "ref_audio": ref_rel, "ref_alternatives": [f"profiles/ref_{pid}_alt{i}.wav" for i in range(len(alts))],
                    "f0_ref": next((r["f0"] for r in gate_rows if r["path"] == picked), None),
                    "ref_gate": [{k: v for k, v in r.items() if k != "path"} | {"file": Path(r["path"]).name} for r in gate_rows],
                    "sources": [{"chunk": m["chunk"], "speaker": m["speaker"], "seconds": m["seconds"], "track": m["track"]} for m in members],
                    "default_for_gender": gender not in default_seen, "manual": False}
        if gender:
            default_seen.add(gender)
        doc["profiles"][pid] = prof_doc
        log(f"  {pid}: {gender} {prof_doc['seconds']} s from {[(m['chunk'][-3:], m['speaker']) for m in members]} "
            f"ref={'yes' if picked else 'none'} face={'yes' if face is not None else 'no'}")
    prof.save(ROOT / "workspace" / film / "profiles.json", doc)
    log(f"wrote workspace/{film}/profiles.json ({len(doc['profiles'])} profiles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
