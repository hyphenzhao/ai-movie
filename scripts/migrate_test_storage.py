#!/usr/bin/env python3
"""Verify and switch test data to the explicitly configured LAN storage.

This tool never copies data. The operator first runs rclone copy --links, then
rclone check --download --one-way --links --combined REPORT for the same source
root and selected paths. Keep REPORT outside both source and destination.

  python3 scripts/migrate_test_storage.py scan
  python3 scripts/migrate_test_storage.py verify --download-report REPORT --download-exit-code 0
  python3 scripts/migrate_test_storage.py switch
  python3 scripts/migrate_test_storage.py cleanup --delete-verified-hold

switch keeps all original bytes in .migration-hold/2026-10-04; rollback restores
them. cleanup is the only command that deletes original files.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import time

SOURCE = Path("/home/ai-movie/ai-movie")
MOUNT = Path("/home/ai-movie/mnt/PrivateFiles")
TARGET = MOUNT / "AI-Movie"
REMOTE = "ai-movie-lan:PrivateFiles"
STATE = Path("/home/ai-movie/.local/state/ai-movie-storage/migration-2026-10-04")
HOLD = SOURCE / ".migration-hold" / "2026-10-04"
EXCLUDED = {"_cache", "_web", "web.log"}
BASE_UNITS = ["inputs", "test_data", "deliver", "projects", "output_test.aimovie.json"]


class MigrationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise MigrationError(message)


def lexists(path):
    return os.path.lexists(path)


def safe_relative(value):
    p = PurePosixPath(value)
    require(bool(value) and not p.is_absolute() and ".." not in p.parts
            and str(p) == value and "\n" not in value and "\r" not in value,
            f"Unsafe relative path: {value!r}")
    return value


def atomic_json(path, data):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def locked():
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(STATE.resolve() == STATE, "State directory must not contain symlinks")
    with open(STATE / "migration.lock", "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MigrationError("Another migration command is running") from exc
        yield


def check_mount():
    require(SOURCE.is_dir() and SOURCE.resolve() == SOURCE,
            "Unexpected source root or source-root symlink")
    require((SOURCE / ".git").is_dir(), "Source is not the expected repository")
    require(MOUNT.resolve() == MOUNT and TARGET.resolve() == TARGET,
            "Mount and destination must not contain symlinks")
    result = subprocess.run(
        ["findmnt", "-J", "-T", str(TARGET), "-o", "TARGET,SOURCE,FSTYPE"],
        capture_output=True, text=True, check=False)
    require(result.returncode == 0, "Cannot determine destination mount")
    mounts = json.loads(result.stdout).get("filesystems", [])
    require(len(mounts) == 1, "Ambiguous destination mount")
    mount = mounts[0]
    require(mount.get("target") == str(MOUNT)
            and mount.get("source") == REMOTE
            and mount.get("fstype") == "fuse.rclone",
            "Destination is not the configured rclone SMB mount")
    require(TARGET.is_dir(), "AI-Movie destination directory does not exist")


def discover_units():
    require((SOURCE / "workspace").is_dir()
            and not (SOURCE / "workspace").is_symlink(), "workspace must stay local")
    units = BASE_UNITS + ["workspace/" + p.name for p in (SOURCE / "workspace").iterdir()
                          if p.name not in EXCLUDED]
    for unit in units:
        safe_relative(unit)
        require(lexists(SOURCE / unit), f"Missing source unit: {unit}")
    return sorted(units)


def inventory(base, units):
    """lstat only: never open file contents or follow symlinks."""
    rows = {}
    stack = [(base / unit, unit) for unit in units]
    while stack:
        path, rel = stack.pop()
        safe_relative(rel)
        st = path.lstat()
        row = {"path": rel, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        if stat.S_ISLNK(st.st_mode):
            row.update(type="symlink", target=os.readlink(path))
        elif stat.S_ISDIR(st.st_mode):
            row["type"] = "directory"
            with os.scandir(path) as children:
                stack.extend((Path(child.path), rel + "/" + child.name) for child in children)
        elif stat.S_ISREG(st.st_mode):
            require(st.st_nlink == 1, f"Hard-linked file is outside this migration plan: {rel}")
            row["type"] = "file"
        else:
            raise MigrationError(f"Special file cannot be migrated: {rel}")
        require(rel not in rows, f"Duplicate path in inventory: {rel}")
        rows[rel] = row
    return dict(sorted(rows.items()))


def differences(expected, actual, remote=False, remote_mtime=False):
    problems = []
    for path in sorted(set(expected) | set(actual)):
        a, b = expected.get(path), actual.get(path)
        if a is None or b is None:
            problems.append(f"{'Unexpected' if a is None else 'Missing'}: {path}")
            continue
        keys = ("type",) if remote else ("type", "size", "mtime_ns")
        if remote and a["type"] == "file":
            keys += ("size",)
            if remote_mtime:
                keys += ("mtime_ns",)
        if a["type"] == "symlink":
            keys += ("target",)
        if any(a.get(key) != b.get(key) for key in keys):
            problems.append(f"Changed: {path}")
    return problems


def assert_inventory(expected, actual, label, remote=False, remote_mtime=False):
    problems = differences(expected, actual, remote, remote_mtime)
    require(not problems, f"{label}: {len(problems)} differences; " + "; ".join(problems[:10]))


def load_manifest():
    path = STATE / "manifest.json"
    require(path.is_file(), "Run scan first")
    data = json.loads(path.read_text(encoding="utf-8"))
    require(data.get("version") == 1 and data.get("source") == str(SOURCE)
            and data.get("target") == str(TARGET) and data.get("remote") == REMOTE,
            "Manifest belongs to another migration")
    units = data["units"]
    require(len(units) == len(set(units)) and set(BASE_UNITS).issubset(units),
            "Manifest has an invalid unit list")
    for unit in units:
        safe_relative(unit)
        require(unit in BASE_UNITS or (len(PurePosixPath(unit).parts) == 2
                and unit.startswith("workspace/") and unit.split("/")[1] not in EXCLUDED),
                f"Unit is outside the migration scope: {unit}")
    for path, row in data["entries"].items():
        safe_relative(path)
        require(row["path"] == path and any(path == u or path.startswith(u + "/") for u in units),
                f"Manifest path is outside its units: {path}")
        require(row["type"] in {"file", "directory", "symlink"}, "Invalid manifest entry type")
    require(all(u in data["entries"] for u in units), "Manifest lacks a top-level unit")
    return data


def subset(manifest, unit):
    return {p: row for p, row in manifest["entries"].items()
            if p == unit or p.startswith(unit + "/")}


def check_source(manifest, base=None):
    if base is None:
        base = SOURCE
    if base == SOURCE:
        require(discover_units() == manifest["units"], "Source unit list changed after scan")
    assert_inventory(manifest["entries"], inventory(base, manifest["units"]), "Source changed")


def check_remote(manifest, snapshot=None, units=None):
    check_mount()
    selected = units or manifest["units"]
    expected = {p: row for p, row in manifest["entries"].items()
                if any(p == u or p.startswith(u + "/") for u in selected)}
    actual = inventory(TARGET, selected)
    assert_inventory(expected, actual,
                     "Mounted destination differs", remote=True)
    if snapshot is not None:
        verified = {p: snapshot[p] for p in expected}
        assert_inventory(verified, actual, "Destination changed since download verification",
                         remote=True, remote_mtime=True)
    return actual


def check_report(manifest, report):
    require(report.is_file() and not report.is_symlink(), "Missing or symlinked download report")
    require(report.stat().st_mtime_ns >= manifest["created_ns"],
            "Download report predates the source manifest")
    expected = {path + (".rclonelink" if row["type"] == "symlink" else "")
                for path, row in manifest["entries"].items() if row["type"] != "directory"}
    matched = set()
    for line in report.read_text(encoding="utf-8").splitlines():
        require(line.startswith("= "), f"Download check contains a failure or invalid line: {line[:180]}")
        name = line[2:]
        safe_relative(name)
        require(name in expected, f"Unexpected download-check path: {name}")
        require(name not in matched, f"Duplicate download-check path: {name}")
        matched.add(name)
    missing = expected - matched
    require(not missing, f"Download check omits {len(missing)} files/links: {sorted(missing)[:5]}")


def require_verification(manifest):
    path = STATE / "verified.json"
    require(path.is_file(), "Run verify successfully before switching")
    proof = json.loads(path.read_text(encoding="utf-8"))
    require(proof.get("manifest_sha256") == digest(STATE / "manifest.json"), "Manifest changed after verify")
    report = Path(proof["download_report"])
    require(proof.get("download_exit_code") == 0 and proof.get("download_report_sha256") == digest(report),
            "Download-check evidence changed after verify")
    require(proof.get("download_report_mtime_ns") == report.stat().st_mtime_ns,
            "Download-check report timestamp changed after verify")
    check_report(manifest, report)
    snapshot = proof.get("remote_entries")
    require(isinstance(snapshot, dict) and set(snapshot) == set(manifest["entries"]),
            "Verification lacks its remote metadata snapshot; run verify again")
    assert_inventory(manifest["entries"], snapshot, "Invalid verified destination snapshot", remote=True)
    return proof


def scan(_args):
    check_mount()
    require(not (STATE / "manifest.json").exists(), "Manifest already exists; it will not be overwritten")
    require(not lexists(HOLD), "Existing hold directory needs recovery before a new scan")
    units = discover_units()
    entries = inventory(SOURCE, units)
    assert_inventory(entries, inventory(SOURCE, units), "Source changed during scan")
    data = {"version": 1, "created_ns": time.time_ns(), "source": str(SOURCE),
            "target": str(TARGET), "remote": REMOTE, "units": units, "entries": entries}
    atomic_json(STATE / "manifest.json", data)
    print(json.dumps({"manifest": str(STATE / "manifest.json"), "units": len(units),
                      "entries": len(entries), "bytes": sum(r["size"] for r in entries.values()
                                                            if r["type"] == "file")}))


def verify(args):
    require(args.download_exit_code == 0, "rclone check --download did not exit successfully")
    require(not (STATE / "switch.json").exists(), "verify is only allowed before switching")
    manifest = load_manifest()
    report = Path(args.download_report).absolute()
    check_report(manifest, report)
    report_hash = digest(report)
    report_mtime_ns = report.stat().st_mtime_ns
    check_source(manifest)
    snapshot = check_remote(manifest)
    # rclone copy preserves source file times. Allow SMB timestamp precision
    # loss here, then compare the remote's own exact timestamp after verify.
    # This guards normal rewrites, not deliberately restored old timestamps.
    for path, row in snapshot.items():
        if row["type"] != "file":
            continue
        require(abs(row["mtime_ns"] - manifest["entries"][path]["mtime_ns"]) < 1_000_000_000,
                f"Destination file time no longer matches copied source: {path}")
        require(row["mtime_ns"] <= report_mtime_ns,
                f"Destination file is newer than its download-check report: {path}")
    check_source(manifest)
    check_remote(manifest, snapshot)
    require(digest(report) == report_hash and report.stat().st_mtime_ns == report_mtime_ns,
            "Download-check report changed during verify")
    atomic_json(STATE / "verified.json", {
        "verified_ns": time.time_ns(), "manifest_sha256": digest(STATE / "manifest.json"),
        "download_report": str(report), "download_report_sha256": report_hash,
        "download_report_mtime_ns": report_mtime_ns, "remote_entries": snapshot,
        "download_exit_code": args.download_exit_code})
    print("Verified: download-check coverage, original source metadata, destination files and links")


def journal_write(journal):
    atomic_json(STATE / "switch.json", journal)


def expected_link(unit):
    path = SOURCE / unit
    return path.is_symlink() and os.readlink(path) == str(TARGET / unit)


def rollback_units(manifest, journal):
    errors = []
    for unit in reversed(manifest["units"]):
        source, held = SOURCE / unit, HOLD / unit
        if not lexists(held):
            continue
        try:
            # Never overwrite newly written local data or an unexpected link.
            require(not lexists(source) or expected_link(unit), f"Cannot overwrite changed source: {unit}")
            if expected_link(unit):
                source.unlink()
            source.parent.mkdir(parents=True, exist_ok=True)
            os.rename(held, source)
        except Exception as exc:
            errors.append(f"{unit}: {exc}")
    journal["phase"] = "rollback_failed" if errors else "rolled_back"
    journal["errors"] = errors
    journal_write(journal)
    require(not errors, "Rollback needs manual recovery: " + "; ".join(errors[:5]))


def switch(_args):
    manifest = load_manifest()
    proof = require_verification(manifest)
    require(not (STATE / "switch.json").exists(), "A switch journal already exists; inspect it or run rollback")
    require(not lexists(HOLD), "Hold destination already exists")
    check_source(manifest)
    check_remote(manifest, proof["remote_entries"])
    check_source(manifest)
    HOLD.mkdir(parents=True, mode=0o700)
    require(HOLD.resolve() == HOLD and HOLD.stat().st_dev == SOURCE.stat().st_dev,
            "Hold must be on the original local filesystem")
    journal = {"phase": "switching", "started_ns": time.time_ns(), "completed_units": [],
               "manifest_sha256": digest(STATE / "manifest.json"), "hold": str(HOLD)}
    journal_write(journal)
    try:
        for unit in manifest["units"]:
            check_mount()
            assert_inventory(subset(manifest, unit), inventory(SOURCE, [unit]),
                             f"Source unit changed before switch: {unit}")
            check_remote(manifest, proof["remote_entries"], [unit])
            source, held = SOURCE / unit, HOLD / unit
            held.parent.mkdir(parents=True, exist_ok=True)
            os.rename(source, held)
            try:
                os.symlink(TARGET / unit, source, target_is_directory=held.is_dir())
            except BaseException:
                os.rename(held, source)
                raise
            journal["completed_units"].append(unit)
            journal_write(journal)
        check_source(manifest, HOLD)
        check_remote(manifest, proof["remote_entries"])
        require(all(expected_link(u) for u in manifest["units"]), "A switched source link changed")
        journal.update(phase="switched", finished_ns=time.time_ns())
        journal_write(journal)
    except BaseException:
        rollback_units(manifest, journal)
        raise
    print(f"Switched {len(manifest['units'])} units; all original bytes remain at {HOLD}")


def load_journal(manifest):
    path = STATE / "switch.json"
    require(path.is_file(), "No switch journal exists")
    journal = json.loads(path.read_text(encoding="utf-8"))
    require(journal.get("manifest_sha256") == digest(STATE / "manifest.json")
            and journal.get("hold") == str(HOLD), "Switch journal does not match manifest/hold")
    require(HOLD.resolve() == HOLD and HOLD.is_dir(), "Hold is missing or contains a symlink")
    return journal


def rollback(_args):
    manifest = load_manifest()
    journal = load_journal(manifest)
    require(journal["phase"] in {"switching", "switched", "rollback_failed", "rolled_back"},
            "Rollback is unavailable after cleanup started")
    rollback_units(manifest, journal)
    check_source(manifest)
    print("Original local data restored; remote copy retained")


def cleanup(args):
    require(args.delete_verified_hold, "cleanup requires --delete-verified-hold")
    manifest = load_manifest()
    journal = load_journal(manifest)
    require(journal["phase"] == "switched", "Only a completely switched hold can be deleted")
    proof = require_verification(manifest)
    require(all(expected_link(u) for u in manifest["units"]), "A source link changed; hold is preserved")
    check_source(manifest, HOLD)
    check_remote(manifest, proof["remote_entries"])
    check_source(manifest, HOLD)
    journal.update(phase="cleaning", deleted_units=[])
    journal_write(journal)
    try:
        for unit in manifest["units"]:
            check_mount()
            require(expected_link(unit), f"Source link changed: {unit}")
            assert_inventory(subset(manifest, unit), inventory(HOLD, [unit]),
                             f"Held source changed: {unit}")
            check_remote(manifest, proof["remote_entries"], [unit])
            held = HOLD / unit
            if held.is_dir() and not held.is_symlink():
                shutil.rmtree(held)
            else:
                held.unlink()
            journal["deleted_units"].append(unit)
            journal_write(journal)
    except BaseException:
        journal["phase"] = "cleanup_failed"
        journal_write(journal)
        raise
    for path in sorted(HOLD.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        require(path.is_dir() and not path.is_symlink(), "Unexpected file remains in hold")
        path.rmdir()
    HOLD.rmdir()
    journal.update(phase="cleaned", cleaned_ns=time.time_ns())
    journal_write(journal)
    print("Verified original hold removed; source links and SMB data retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scan")
    p = commands.add_parser("verify")
    p.add_argument("--download-report", required=True)
    p.add_argument("--download-exit-code", type=int, required=True)
    commands.add_parser("switch")
    commands.add_parser("rollback")
    p = commands.add_parser("cleanup")
    p.add_argument("--delete-verified-hold", action="store_true")
    args = parser.parse_args()
    try:
        with locked():
            globals()[args.command](args)
    except (MigrationError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
