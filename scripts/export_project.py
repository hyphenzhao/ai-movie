#!/usr/bin/env python
"""Write a headless run's results into a GUI-loadable ``.aimovie.json``.

``scripts/run_pipeline.py`` keeps its own ``state.json`` so stages can be
re-run individually.  The GUI reads a different document (``ProjectLog``:
``steps`` + ``step_data`` keyed by the Chinese step names).  Without this
bridge a headless run is invisible in the GUI — you would still see whatever
the last GUI session left behind.

    python scripts/export_project.py workspace/output_test/state.json
    python scripts/export_project.py workspace/output_test/state.json \
        --out projects/output_test.aimovie.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_movie.config import PROJECTS_DIR, WORKSPACE_DIR   # noqa: E402
from ai_movie.project_log import ProjectLog               # noqa: E402


def build(state: dict, name: str, video: str | None) -> ProjectLog:
    log = ProjectLog(name=name)
    log.video_path = video or state.get("_video")
    log.workspace_dir = str(WORKSPACE_DIR / name)

    demux = state.get("demux") or {}
    asr = state.get("asr") or {}
    tr = state.get("translate") or {}
    tts = state.get("tts") or {}
    fit = state.get("fit") or {}
    mix = state.get("mix") or {}
    sep = state.get("separate") or {}
    faces = state.get("faces") or {}
    ls = state.get("lipsync") or {}
    comp = state.get("compose") or {}

    def done(step: str, data: dict, detail: str = "") -> None:
        log.set_step_data(step, data)
        log.mark_step(step, "done")
        log.add_entry(step, "done", detail or "imported from headless run")

    # ── 拆分音轨 ────────────────────────────────────────────────
    if demux.get("audio"):
        done("拆分音轨", {
            "results": [{
                "video": demux.get("video"),
                "audio": demux.get("audio"),
                "duration": demux.get("duration"),
                "label": "original",
                "source": log.video_path,
            }],
            "error_count": 0,
        }, f"{demux.get('duration', 0):.1f}s")

    # ── 转换文字 ────────────────────────────────────────────────
    segs = asr.get("segments") or []
    if segs:
        diar = asr.get("diarization") or {}
        data = {
            "language": asr.get("language", "ja"),
            "results": [{
                "source": demux.get("audio"),
                "language": asr.get("language", "ja"),
                "segments": segs,
            }],
        }
        if diar:
            data["diarization"] = diar
        spk = "、".join(f"{k}={v.get('gender')}"
                        for k, v in (diar.get("speakers") or {}).items())
        done("转换文字", data, f"{len(segs)} 段{('，' + spk) if spk else ''}")

    # ── 人声分离 ────────────────────────────────────────────────
    if sep.get("vocals"):
        done("人声分离", {"vocals": sep.get("vocals"),
                          "background": sep.get("background"),
                          "vocals_full": sep.get("vocals_full"),
                          "background_full": sep.get("background_full"),
                          "bed_sr": sep.get("bed_sr"),
                          "bed_channels": sep.get("bed_channels")})

    # ── 文本翻译 ────────────────────────────────────────────────
    tsegs = tr.get("segments") or []
    if tsegs:
        done("文本翻译", {
            "target_lang": "Chinese",
            "engine": tr.get("chosen") or "sakura",
            "segments": tsegs,
            "count": len(tsegs),
            "glossary": state.get("glossary") or {},
            "variants": tr.get("variants") or {},
        }, f"{len(tsegs)} 段 / {tr.get('chosen')}")

    # ── 人声生成（含时长适配后的 audio_fit）────────────────────
    compact = state.get("compact") or {}
    gsegs = (fit.get("segments") or compact.get("segments") or tts.get("segments") or [])
    if gsegs:
        ok = sum(1 for s in gsegs if s.get("audio"))
        data = {"results": gsegs, "ok": ok}
        if tts.get("refs"):
            data["speaker_refs"] = tts["refs"]
        if tts.get("quality"):
            data["clone_quality"] = tts["quality"]
        ratios = [s.get("fit_ratio", 1.0) for s in gsegs if s.get("audio_fit")]
        if ratios:
            data["fit"] = {"fitted": len(ratios),
                           "max_ratio": round(max(ratios), 3)}
        done("人声生成", data, f"{ok}/{len(gsegs)} 段")

    # ── 重新混音 ────────────────────────────────────────────────
    if mix.get("audio"):
        done("重新混音", {"audio": mix["audio"], "output": mix["audio"]})

    # ── 人物锚定 ────────────────────────────────────────────────
    if faces.get("plan_path"):
        done("人物锚定", {
            "face_plan": faces["plan_path"],
            "speaker_track": faces.get("speaker_track") or {},
            "tracks": faces.get("tracks") or [],
            "occlusion_gate": True,
        }, f"{len(faces.get('tracks') or [])} 条人脸轨迹")

    # ── 口型匹配 ────────────────────────────────────────────────
    if ls.get("video"):
        done("口型匹配", {"output_video": ls["video"],
                          "driving_audio_source": "tts_speech_track"})

    # ── 合成视频 ────────────────────────────────────────────────
    if comp.get("video"):
        qc = state.get("qc") or {}
        qs = ((qc.get("fit") or {}).get("summary")) or {}
        detail = ""
        if qs:
            detail = f"QC {qs.get('PASS', 0)}/{qs.get('WARN', 0)}/{qs.get('FAIL', 0)}"
        done("合成视频", {"output_video": comp["video"],
                          "output": comp["video"], "qc": qc}, detail)

    return log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("state", help="workspace/<name>/state.json")
    ap.add_argument("--out", default=None,
                    help="target .aimovie.json (default: projects/<name>.aimovie.json)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    state_path = Path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    name = args.name or state_path.parent.name
    out = Path(args.out) if args.out else Path(PROJECTS_DIR) / f"{name}.aimovie.json"

    log = build(state, name, args.video)

    if out.exists():
        backup = out.with_suffix(out.suffix + ".bak")
        backup.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"existing project backed up to {backup}")

    log.save(out)
    print(f"wrote {out}")
    print(f"  steps: {log.steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
