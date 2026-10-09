#!/usr/bin/env python
"""Publish an accepted release to the mounted LAN share.

    python scripts/publish_smb.py --check-mount
    python scripts/publish_smb.py v3.4 --long SONE-846

The mount is managed by the user's ai-movie-smb.service. Credentials stay in
its private rclone configuration; this script never reads them. Incomplete
copies remain under a .partial directory and never become a release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOUNT = Path.home() / "mnt" / "PrivateFiles"
MOUNT_SOURCE = "ai-movie-lan:PrivateFiles"
DEFAULT_FILMS = ("output_test", "test_1", "test_2")
VFS_STATS_URL = "http://127.0.0.1:5573/vfs/stats"


def component(value: str) -> str:
    """Only accept a single visible path component, including release tags."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"invalid release/film name: {value!r}")
    return value


def _unescape_mount(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def require_mount() -> Path:
    """Fail before creating anything unless the exact rclone mount is present."""
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, sep, after = line.partition(" - ")
        fields, fs = before.split(), after.split()
        if not sep or len(fields) < 6 or len(fs) < 3:
            continue
        target, source = _unescape_mount(fields[4]), _unescape_mount(fs[1])
        if target == str(MOUNT) and source == MOUNT_SOURCE and fs[0] == "fuse.rclone":
            if "rw" not in fields[5].split(","):
                raise RuntimeError(f"LAN share is mounted read-only: {MOUNT}")
            return MOUNT
    raise RuntimeError(f"LAN share is not mounted at {MOUNT}; start ai-movie-smb.service")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vfs_stats() -> dict:
    request = urllib.request.Request(
        VFS_STATS_URL, data=json.dumps({"fs": MOUNT_SOURCE}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def wait_for_uploads(timeout: float = 1800) -> None:
    """Checksums may read VFS cache: also wait for rclone to finish SMB writes."""
    if timeout <= 0:
        raise ValueError("upload timeout must be positive")
    deadline = time.monotonic() + timeout
    next_notice = 0.0
    while True:
        require_mount()
        stats = _vfs_stats()
        if stats.get("fs") != MOUNT_SOURCE:
            raise RuntimeError("rclone VFS stats refer to a different filesystem")
        cache = stats.get("diskCache") or {}
        keys = ("uploadsInProgress", "uploadsQueued", "erroredFiles")
        counts = [cache.get(k) for k in keys]
        if any(type(n) is not int or n < 0 for n in counts) or type(cache.get("outOfSpace")) is not bool:
            raise RuntimeError("rclone VFS stats do not contain complete upload status")
        if cache["outOfSpace"]:
            raise RuntimeError("rclone VFS cache is out of space; release not confirmed")
        if all(n == 0 for n in counts):
            return
        now = time.monotonic()
        status = ", ".join(f"{key}={cache[key]}" for key in keys)
        if now >= deadline:
            raise RuntimeError(f"timed out waiting for SMB uploads ({status}); release not confirmed")
        if now >= next_notice:
            print(f"waiting for SMB uploads: {status}", flush=True)
            next_notice = now + 30
        time.sleep(min(1, deadline - now))


def _accepted(version: str, films: tuple[str, ...], long: str | None) -> Path:
    report = ROOT / "workspace" / "_release" / version / "ACCEPTANCE.json"
    data = json.loads(report.read_text(encoding="utf-8"))
    gates = data.get("gates") or []
    # Check every row independently: older accept_release versions computed
    # their aggregate `passed` before adding the long-film gates.
    if data.get("version") != version or data.get("passed") is not True or not gates:
        raise RuntimeError(f"release has no successful acceptance report: {report}")
    if any(g.get("ok") is not True for g in gates):
        raise RuntimeError(f"release has failed acceptance gates: {report}")
    keys = {g.get("key", "") for g in gates}
    if any(f"H1[{film}]" not in keys for film in films):
        raise RuntimeError("acceptance report does not cover all requested short films")
    if long and not any(k.startswith(f"L[{long}] ") for k in keys):
        raise RuntimeError("acceptance report does not cover the requested long film")
    return report


def publish(version: str, films: tuple[str, ...], long: str | None = None,
            *, upload_timeout: float = 1800) -> Path:
    component(version)
    for film in films:
        component(film)
    if long:
        component(long)
    mount = require_mount()
    accepted = _accepted(version, films, long)
    deliver = ROOT / "deliver"
    files: dict[Path, Path] = {}

    def add(source: Path, relative: Path, *, optional: bool = False, final: bool = False) -> None:
        if optional and not source.exists():
            return
        if source.is_symlink() or not source.exists():
            raise RuntimeError(f"missing or symlinked release artifact: {source}")
        if source.is_dir():
            for child in sorted(source.iterdir()):
                add(child, relative / child.name)
            return
        if not source.is_file():
            raise RuntimeError(f"unsupported release artifact: {source}")
        if final and (source.stat().st_size == 0 or source.stat().st_mtime_ns > accepted.stat().st_mtime_ns):
            raise RuntimeError(f"final is empty or newer than its acceptance report: {source}")
        files[relative] = source

    for film in films:
        add(deliver / f"{film}_dubbed.mp4", Path(f"{film}_dubbed.mp4"), final=True)
        add(deliver / f"{film}_dubbed", Path(f"{film}_dubbed"))
        add(deliver / film / "09_subs_eval.md", Path(f"{film}_dubbed/09_subs_eval.md"), optional=True)
    if long:
        add(deliver / f"{long}_dubbed_full.mp4", Path(f"{long}_full/{long}_dubbed_full.mp4"), final=True)
        add(deliver / f"{long}_full", Path(f"{long}_full"))
        add(ROOT / "workspace" / long / "profiles.json", Path(f"{long}_full/profiles.json"), optional=True)
    add(accepted.parent, Path("_release"))

    require_mount()
    releases = mount / "AI-Movie" / "releases"
    target = releases / version
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"release already exists; refusing to overwrite: {target}")
    partial = releases / f".{version}.partial-{uuid.uuid4().hex[:12]}"
    partial.mkdir(parents=True)
    manifest = []
    for relative, source in sorted(files.items()):
        require_mount()
        dest = partial / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        digest = _sha256(source)
        if digest != _sha256(dest):
            raise RuntimeError(f"copy verification failed: {relative}; incomplete copy: {partial}")
        manifest.append({"path": relative.as_posix(), "size": dest.stat().st_size, "sha256": digest})
    require_mount()
    (partial / "MANIFEST.json").write_text(json.dumps({
        "version": version, "published_at": datetime.now(timezone.utc).isoformat(),
        "films": list(films), "long": long, "files": manifest,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    wait_for_uploads(upload_timeout)
    if target.exists():
        raise RuntimeError(f"release appeared during copy; incomplete copy: {partial}")
    partial.rename(target)
    wait_for_uploads(upload_timeout)
    print(f"published {len(manifest)} verified files to {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?")
    parser.add_argument("--check-mount", action="store_true")
    parser.add_argument("--films", default=",".join(DEFAULT_FILMS))
    parser.add_argument("--long", default=None)
    parser.add_argument("--upload-timeout", type=float, default=1800,
                        help="seconds to wait for rclone's SMB upload queue (default: 1800)")
    args = parser.parse_args()
    try:
        if args.check_mount:
            print(f"LAN share mounted: {require_mount()}")
            return 0
        if not args.version:
            parser.error("version is required unless --check-mount is used")
        films = tuple(f for f in args.films.split(",") if f)
        if not films and not args.long:
            parser.error("at least one film is required")
        if args.upload_timeout <= 0:
            parser.error("--upload-timeout must be positive")
        publish(args.version, films, args.long, upload_timeout=args.upload_timeout)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"SMB_PUBLISH_FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
