#!/usr/bin/env python3
"""Deliver this run's completed films for review, without granting release acceptance.

    python scripts/publish_evaluation.py v3.4 --run-id RUN --shorts output_test test_1 test_2 --long SONE-846
    python scripts/publish_evaluation.py v3.4 --run-id RUN --shorts-only

Run evidence lives in workspace/_release/VERSION/runs/RUN. Quality failures
are included visibly; missing films, invalid evidence and execution errors stop
delivery. This script never writes to the formal releases directory.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from publish_smb import component, require_mount, wait_for_uploads  # noqa: E402

DEFAULT_SHORTS = ("output_test", "test_1", "test_2")


def destination_parent(mount: Path, version: str) -> Path:
    """An existing intermediate symlink must never redirect delivery off the share."""
    parent = mount
    for name in ("AI-Movie", "evaluations", version):
        parent = parent / name
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise RuntimeError(f"evaluation destination is not a real directory: {parent}")
    if not parent.resolve().is_relative_to(mount.resolve()):
        raise RuntimeError("evaluation destination escapes the LAN mount")
    return parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"not a JSON object: {path}")
    return data


def resolve(value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError("evidence contains no file path")
    p = Path(value)
    return (p if p.is_absolute() else ROOT / p).resolve(strict=True)


def checked_record(record: dict, *, expected: Path | None = None) -> Path:
    if not isinstance(record, dict):
        raise RuntimeError("missing file evidence")
    p = resolve(record.get("path"))
    if expected is not None and p != expected.resolve(strict=True):
        raise RuntimeError(f"evidence refers to a different file: {p}")
    if not p.is_file() or p.stat().st_size == 0:
        raise RuntimeError(f"evidence file is missing or empty: {p}")
    if record.get("size", p.stat().st_size) != p.stat().st_size:
        raise RuntimeError(f"file size changed since completion: {p}")
    # SMB can normalize timestamp precision after a queued upload. Identity is
    # bound to the complete content hash and size, not the stored timestamp.
    if record.get("sha256") != sha256(p):
        raise RuntimeError(f"file content changed since completion: {p}")
    return p


def probe_video(path: Path) -> dict:
    proc = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,width,height,sample_rate,channels",
        "-of", "json", str(path),
    ], capture_output=True, text=True, timeout=120)
    try:
        result = json.loads(proc.stdout)
        duration = float(result["format"]["duration"])
        streams = result["streams"]
        video = any(s.get("codec_type") == "video" and int(s.get("width", 0)) > 0
                    and int(s.get("height", 0)) > 0 for s in streams)
        audio = any(s.get("codec_type") == "audio" and int(s.get("sample_rate", 0)) > 0
                    and int(s.get("channels", 0)) > 0 for s in streams)
    except (ValueError, KeyError, TypeError):
        raise RuntimeError(f"ffprobe returned no valid film: {path}") from None
    if proc.returncode or not math.isfinite(duration) or duration <= 0 or not video or not audio:
        raise RuntimeError(f"film must have valid video, audio and duration: {path}")
    return {"duration_seconds": duration, "streams": streams}


def collect(version: str, run_id: str, shorts: tuple[str, ...], long: str | None,
            *, shorts_only: bool) -> tuple[dict, list[dict], dict[Path, Path]]:
    """Validate all evidence before creating a destination directory."""
    run_dir = ROOT / "workspace" / "_release" / version / "runs" / run_id
    certified: dict[str, str] = {}

    def document(path: Path) -> dict:
        raw = path.read_bytes()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise RuntimeError(f"not a JSON object: {path}")
        key, digest = str(path.resolve()), hashlib.sha256(raw).hexdigest()
        if key in certified and certified[key] != digest:
            raise RuntimeError(f"evidence changed while validating: {path}")
        certified[key] = digest
        return data

    def verify(record: dict, *, expected: Path | None = None) -> Path:
        p = checked_record(record, expected=expected)
        if str(p) in certified and certified[str(p)] != record["sha256"]:
            raise RuntimeError(f"conflicting evidence for: {p}")
        certified[str(p)] = record["sha256"]
        return p

    run = document(run_dir / "RUN.json")
    if run.get("version") != version or run.get("run_id") != run_id:
        raise RuntimeError("RUN.json belongs to a different version or run")
    if not isinstance(run.get("started_at_ns"), int) or not run.get("code_hash"):
        raise RuntimeError("RUN.json lacks the run start or code identity")
    if set(run.get("films") or []) != set(shorts) or (not shorts_only and run.get("long") != long):
        raise RuntimeError("requested films do not match RUN.json")
    files = {Path("receipts/RUN.json"): run_dir / "RUN.json"}
    seen: dict[str, tuple[dict, Path]] = {}

    def receipt(name: str, *, is_long: bool = False) -> tuple[dict, Path]:
        if name in seen:
            return seen[name]
        p = run_dir / "receipts" / f"{name}.json"
        rec = document(p)
        if (rec.get("status") != "completed" or rec.get("run_id") != run_id
                or rec.get("code_hash") != run["code_hash"]
                or type(rec.get("completed_at_ns")) is not int
                or rec["completed_at_ns"] < run["started_at_ns"]):
            raise RuntimeError(f"no successful completion evidence for this run: {name}")
        if is_long:
            if rec.get("state") is not None:
                raise RuntimeError("long-film receipt must have state=null")
            expected_video = ROOT / "deliver" / f"{name}_dubbed_full.mp4"
        else:
            sp = verify(rec.get("state"), expected=ROOT / "workspace" / name / "state.json")
            state = document(sp)
            expected_video = resolve((state.get("vc") or {}).get("video"))
            files[Path("receipts/states") / f"{name}.json"] = sp
        video = rec.get("video")
        if not isinstance(video, dict) or not all(k in video for k in ("size", "sha256", "mtime_ns")):
            raise RuntimeError(f"incomplete video evidence: {name}")
        vp = verify(video, expected=expected_video)
        if rec.get("profiles") is not None:
            pp = verify(rec["profiles"])
            files[Path("receipts/profiles") / f"{name}.json"] = pp
        files[Path("receipts") / p.name] = p
        seen[name] = (rec, vp)
        return rec, vp

    films = []
    for name in shorts:
        rec, vp = receipt(name)
        films.append({"name": name, "kind": "short", "source": str(vp),
                      "path": f"videos/{name}.mp4", "sha256": rec["video"]["sha256"],
                      "size": rec["video"]["size"], "probe": probe_video(vp)})
        files[Path(f"videos/{name}.mp4")] = vp
    if long:
        rec, vp = receipt(long, is_long=True)
        plan_path = ROOT / "workspace" / long / "_split" / "plan.json"
        plan = document(plan_path)
        chunks = plan.get("chunks") or []
        threshold = rec.get("min_speech_min")
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0:
            raise RuntimeError("long-film receipt has an invalid speech threshold")
        names = [f"{long}_p{c['index']:02d}" for c in chunks if float(c.get("speech_minutes", threshold)) >= threshold]
        passthrough = [{"name": f"{long}_p{c['index']:02d}", "source": c.get("file"),
                        "speech_minutes": c["speech_minutes"]} for c in chunks
                       if float(c.get("speech_minutes", threshold)) < threshold]
        if not chunks or len(set(names)) != len(names) or set(rec.get("chunks") or {}) != set(names):
            raise RuntimeError("long-film evidence does not cover the complete chunk plan")
        for name in names:
            receipt(name)
            if sha256(run_dir / "receipts" / f"{name}.json") != rec["chunks"][name]:
                raise RuntimeError(f"long-film chunk completion evidence changed: {name}")
        verify(rec.get("plan"), expected=plan_path)
        if not isinstance(rec.get("passthrough"), dict) or set(rec["passthrough"]) != {p["name"] for p in passthrough}:
            raise RuntimeError("long-film evidence does not cover every passthrough source")
        for p in passthrough:
            original = verify(rec["passthrough"][p["name"]], expected=resolve(p["source"]))
            p["source"] = str(original)
            p["sha256"] = rec["passthrough"][p["name"]]["sha256"]
        cp = verify(rec.get("concat"), expected=plan_path.parent / "concat_manifest.json")
        concat = document(cp)

        def same(a: dict, b: dict) -> bool:
            return (isinstance(a, dict) and isinstance(b, dict) and
                    resolve(a.get("path")) == resolve(b.get("path")) and
                    a.get("size") == b.get("size") and a.get("sha256") == b.get("sha256"))

        if (concat.get("version") != 1 or concat.get("film") != long
                or concat.get("min_speech_min") != threshold
                or not same(concat.get("plan"), rec["plan"])
                or not same(concat.get("video"), rec["video"])
                or [p.get("name") for p in concat.get("parts", [])] != [f"{long}_p{c['index']:02d}" for c in chunks]):
            raise RuntimeError("concat manifest does not describe this complete film in plan order")
        for part in concat["parts"]:
            name = part["name"]
            expected = seen[name][0]["video"] if name in names else rec["passthrough"][name]
            if name in names:
                child_profile, film_profile = seen[name][0].get("profiles"), rec.get("profiles")
                if ((child_profile is None) != (film_profile is None)
                        or (child_profile is not None and not same(child_profile, film_profile))):
                    raise RuntimeError(f"long-film profile differs from chunk completion evidence: {name}")
            if part.get("mode") != ("dub" if name in names else "pass") or not same(part.get("source"), expected):
                raise RuntimeError(f"concat source differs from completed evidence: {name}")
            for key in ("source", "processed", "transport"):
                verify(part.get(key))
        files[Path("receipts") / f"{long}.concat{cp.suffix}"] = cp
        files[Path("receipts") / f"{long}.plan.json"] = plan_path
        films.append({"name": long, "kind": "long", "source": str(vp),
                      "path": f"videos/{long}_dubbed_full.mp4", "sha256": rec["video"]["sha256"],
                      "size": rec["video"]["size"], "probe": probe_video(vp),
                      "dubbed_chunks": names, "passthrough_chunks": passthrough,
                      "min_speech_min": threshold})
        files[Path(f"videos/{long}_dubbed_full.mp4")] = vp

    ep = run_dir / "evaluation.json"
    evaluation = document(ep)
    if (evaluation.get("run_id") != run_id or type(evaluation.get("exit_code")) is not int
            or evaluation["exit_code"] not in (0, 1)
            or set(evaluation.get("films") or []) != set(shorts) or evaluation.get("long") != long):
        raise RuntimeError("evaluation evidence is missing, failed to execute, or covers another run/scope")
    report_name = "shorts-ACCEPTANCE.json" if shorts_only else "ACCEPTANCE.json"
    report = verify(evaluation.get("report"), expected=run_dir / "reports" / report_name)
    # Unlike a reused successful rendering stage, evaluation must be newly run.
    # SMB timestamps may lose sub-second precision.
    latest_completion = max([run["started_at_ns"], *(rec["completed_at_ns"] for rec, _ in seen.values())])
    if report.stat().st_mtime_ns + 1_000_000_000 < latest_completion:
        raise RuntimeError("acceptance report predates this run's completed films")
    acceptance = document(report)
    gates = acceptance.get("gates")
    if acceptance.get("version") != version or type(acceptance.get("passed")) is not bool or not isinstance(gates, list) or not gates:
        raise RuntimeError("acceptance report is missing its version, outcome or gates")
    if any(not isinstance(g, dict) or type(g.get("ok")) is not bool for g in gates):
        raise RuntimeError("acceptance report contains incomplete gate results")
    if acceptance["passed"] != all(g["ok"] for g in gates) or evaluation["exit_code"] != (0 if acceptance["passed"] else 1):
        raise RuntimeError("acceptance report and evaluator exit status disagree")
    keys = {g.get("key", "") for g in gates}
    if any(f"H1[{name}]" not in keys for name in shorts) or (long and not any(k.startswith(f"L[{long}] ") for k in keys)):
        raise RuntimeError("acceptance report does not cover every requested film")
    files[Path("receipts/evaluation.json")] = ep
    if not report.with_suffix(".md").is_file():
        raise RuntimeError("acceptance Markdown report is missing")
    for p in sorted((run_dir / "reports").iterdir()):
        if p.is_file() and not p.is_symlink():
            files[Path("reports") / p.name] = p
    # Copy exactly the validated snapshot, and detect modifications during publication.
    snapshots = {}
    for p in set(files.values()):
        digest = sha256(p)
        if str(p.resolve()) in certified and digest != certified[str(p.resolve())]:
            raise RuntimeError(f"evidence changed while validating: {p}")
        snapshots[str(p)] = {"sha256": digest, "size": p.stat().st_size,
                             "mtime_ns": p.stat().st_mtime_ns}
    return {"run": run, "acceptance": acceptance, "report": f"reports/{report.name}",
            "snapshots": snapshots, "certified": certified}, films, files


def entry_pages(version: str, run_id: str, films: list[dict], acceptance: dict, report: str,
                *, shorts_only: bool, reports: list[str] | None = None) -> tuple[str, str]:
    scope = f"{len(films)} 部短片；长片不在本包范围内" if shorts_only else "短片与长片"
    quality = "质量门通过，但本包仍只是评测候选，并非正式发布" if acceptance["passed"] else "质量门未通过：请结合失败项评测"
    failed = [g for g in acceptance["gates"] if not g["ok"]]
    md = [f"# {version} 成片评测 · EVALUATION / NOT ACCEPTED", "",
          f"运行编号：`{run_id}`。范围：{scope}。", "", quality + "。", "",
          "本包仅含本次运行成功收据对应、且已验证可读取音视频信息的实际成片。质量未通过不代表生成失败；本包不授予正式发布验收。", "",
          "打开 `00_从这里开始.html` 播放；`FINAL_FILMS.json` 给出唯一影片路径，`MANIFEST.json` 记录大小和 SHA256。", "",
          "## 成片", ""]
    cards = []
    for f in films:
        md.append(f"- [{f['name']}]({f['path']}) · {f['probe']['duration_seconds']:.1f} 秒 · {f['size']} 字节")
        if f.get("passthrough_chunks"):
            md.append("  保留原片音轨的低对白块：" + "、".join(c["name"] for c in f["passthrough_chunks"]) + "。这些块未宣称重新配音。")
        cards.append(f'<article><h2>{html.escape(f["name"])}</h2><video controls preload="none" src="{html.escape(f["path"], quote=True)}"></video><p><a href="{html.escape(f["path"], quote=True)}">直接打开影片</a> · {f["probe"]["duration_seconds"]:.1f} 秒</p></article>')
    md += ["", "## 验收报告", "", f"[本次验收 JSON]({report})", "", "失败项："]
    md += [f"- {g.get('key', '')}: {g.get('desc', '')} — {g.get('detail', '')}" for g in failed] or ["- 无；仍未作为正式 release 发布。"]
    md += ["", "全部本次报告：", ""] + [f"- [{Path(p).name}]({p})" for p in reports or []]
    fail_html = "".join(f'<li>{html.escape(str(g.get("key", "")))}：{html.escape(str(g.get("desc", "")))} — {html.escape(str(g.get("detail", "")))}</li>' for g in failed)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(version)} 成片评测</title><style>body{{max-width:1000px;margin:32px auto;padding:0 20px;font:17px/1.65 system-ui,sans-serif;background:#f5f6f8;color:#17212b}}header,article,section{{background:white;padding:24px;margin:18px 0;border-radius:12px}}.badge{{font-weight:bold;color:#9a3700}}video{{width:100%;max-height:70vh;background:#111}}a{{color:#075ac0}}li{{margin:.5em 0}}</style><header><div class="badge">EVALUATION / NOT ACCEPTED · 仅供评测，未作为正式版本验收</div><h1>{html.escape(version)} 成片评测</h1><p>运行编号：{html.escape(run_id)}<br>范围：{html.escape(scope)}</p><p>{html.escape(quality)}。</p><p>以下为本次运行凭据对应的实际影片。浏览器无法播放时，可点击“直接打开影片”使用本地播放器。</p></header>{''.join(cards)}<section><h2>质量结果与原始报告</h2><ul>{fail_html or '<li>质量门通过；本包仍是评测候选。</li>'}</ul><p><a href="{html.escape(report, quote=True)}">本次验收 JSON</a> · <a href="{html.escape(str(Path(report).with_suffix('.md')), quote=True)}">验收说明</a> · <a href="README.md">交付说明</a> · <a href="FINAL_FILMS.json">成片路径清单</a> · <a href="MANIFEST.json">文件校验清单</a></p></section></html>'''
    links = "".join(f'<li><a href="{html.escape(p, quote=True)}">{html.escape(Path(p).name)}</a></li>' for p in reports or [])
    if links:
        page = page.replace("</body>", "") if "</body>" in page else page
        page = page.replace("</html>", f"<section><h2>全部本次报告</h2><ul>{links}</ul></section></html>")
    return "\n".join(md) + "\n", page


def publish(version: str, run_id: str, shorts: tuple[str, ...] = DEFAULT_SHORTS,
            long: str | None = "SONE-846", *, shorts_only: bool = False,
            upload_timeout: float = 1800) -> Path:
    for value in (version, run_id, *shorts, *([long] if long else [])):
        component(value)
    if not shorts or len(set(shorts)) != len(shorts) or (long in shorts) or upload_timeout <= 0:
        raise ValueError("invalid film selection or upload timeout")
    if shorts_only:
        long = None
    elif not long:
        raise ValueError("full evaluation requires --long; use --shorts-only for shorts")
    mount = require_mount()
    parent = destination_parent(mount, version)
    # Also settle the pipeline's source writes before taking a validation snapshot.
    wait_for_uploads(upload_timeout)
    evidence, films, files = collect(version, run_id, shorts, long, shorts_only=shorts_only)
    package_id = run_id + ("-shorts" if shorts_only else "")
    target = parent / package_id
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"evaluation already exists; refusing overwrite: {target}")
    destination_parent(require_mount(), version)
    partial = parent / (f".{package_id}.partial-{uuid.uuid4().hex[:12]}")
    partial.mkdir(parents=True)
    inventory = []
    for relative, source in sorted(files.items()):
        destination_parent(require_mount(), version)
        snapshot = evidence["snapshots"][str(source)]
        if source.stat().st_size != snapshot["size"] or source.stat().st_mtime_ns != snapshot["mtime_ns"]:
            raise RuntimeError(f"source changed before copy: {source}")
        dest = partial / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        if sha256(dest) != snapshot["sha256"] or sha256(source) != snapshot["sha256"]:
            raise RuntimeError(f"source or copy changed: {source}")
        inventory.append({"path": relative.as_posix(), "size": dest.stat().st_size, "sha256": snapshot["sha256"]})
    readme, page = entry_pages(version, run_id, films, evidence["acceptance"], evidence["report"], shorts_only=shorts_only,
                               reports=[p.as_posix() for p in sorted(files) if p.parts[0] == "reports"])
    pointers = {"status": "EVALUATION / NOT ACCEPTED", "version": version, "run_id": run_id,
                "scope": "shorts-only" if shorts_only else "full", "entry": "00_从这里开始.html",
                "films": [{k: v for k, v in f.items() if k != "source"} for f in films]}
    for name, contents in {"README.md": readme, "00_从这里开始.html": page,
                           "FINAL_FILMS.json": json.dumps(pointers, ensure_ascii=False, indent=2) + "\n"}.items():
        destination_parent(require_mount(), version)
        p = partial / name
        p.write_text(contents, encoding="utf-8")
        inventory.append({"path": name, "size": p.stat().st_size, "sha256": sha256(p)})
    manifest = {**pointers, "formal_release_accepted": False,
                "quality_passed": evidence["acceptance"]["passed"], "code_hash": evidence["run"]["code_hash"],
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "files": inventory}
    (partial / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    wait_for_uploads(upload_timeout)
    destination_parent(require_mount(), version)
    for path, digest in evidence["certified"].items():
        if sha256(Path(path)) != digest:
            raise RuntimeError(f"run evidence changed before delivery completed: {path}")
    if target.exists():
        raise RuntimeError(f"evaluation appeared while copying: {target}")
    partial.rename(target)
    wait_for_uploads(upload_timeout)
    # Only a confirmed complete directory becomes discoverable through the latest pointer.
    latest = {**pointers, "directory": package_id, "manifest": f"{package_id}/MANIFEST.json",
              "manifest_sha256": sha256(target / "MANIFEST.json"),
              "entry": f"{package_id}/00_从这里开始.html",
              "films": [{**f, "path": f"{package_id}/{f['path']}"} for f in pointers["films"]]}
    temp = parent / f".LATEST_EVALUATION-{uuid.uuid4().hex[:12]}.json"
    destination_parent(require_mount(), version)
    temp.write_text(json.dumps(latest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    wait_for_uploads(upload_timeout)
    os.replace(temp, parent / "LATEST_EVALUATION.json")
    wait_for_uploads(upload_timeout)
    print(f"EVALUATION / NOT ACCEPTED: {len(films)} completed films delivered to {target}")
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("version")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--shorts", nargs="+", default=list(DEFAULT_SHORTS))
    ap.add_argument("--long", default="SONE-846")
    ap.add_argument("--shorts-only", action="store_true")
    ap.add_argument("--upload-timeout", type=float, default=1800)
    args = ap.parse_args()
    try:
        publish(args.version, args.run_id, tuple(args.shorts), args.long,
                shorts_only=args.shorts_only, upload_timeout=args.upload_timeout)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"EVALUATION_DELIVERY_FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
