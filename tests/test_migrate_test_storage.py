"""Safety checks use disposable data, never the real source or SMB mount."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "migrate_test_storage", Path(__file__).parents[1] / "scripts" / "migrate_test_storage.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class MigrationSafetyTests(unittest.TestCase):
    def setUp(self):
        scratch = Path.home() / ".local/state/ai-movie-storage"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="safety-test-", dir=scratch)
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.source, self.target, self.state = base / "source", base / "remote/AI-Movie", base / "state"
        self.hold = self.source / ".migration-hold/2026-10-04"
        self.source.mkdir()
        self.target.mkdir(parents=True)
        self.state.mkdir()
        for name in M.BASE_UNITS:
            path = self.source / name
            if name.endswith(".json"):
                path.write_text("project")
            else:
                path.mkdir()
                (path / "data.bin").write_bytes(b"original bytes")
        for name in ("_cache", "_web", "one", "two"):
            path = self.source / "workspace" / name
            path.mkdir(parents=True)
            (path / "data.bin").write_bytes(b"original bytes")
        (self.source / "workspace/web.log").write_text("active")
        (self.source / "inputs/link").symlink_to("data.bin")
        for name, value in {"SOURCE": self.source, "TARGET": self.target,
                            "STATE": self.state, "HOLD": self.hold}.items():
            p = patch.object(M, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(M, "check_mount")
        p.start()
        self.addCleanup(p.stop)
        self.silence = contextlib.redirect_stdout(io.StringIO())
        self.silence.__enter__()
        self.addCleanup(self.silence.__exit__, None, None, None)

    def prepared(self, remote_mtime_delta_ns=0):
        M.scan(None)
        manifest = M.load_manifest()
        for unit in manifest["units"]:
            source, target = self.source / unit, self.target / unit
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target, symlinks=True)
            else:
                shutil.copy2(source, target)
        if remote_mtime_delta_ns:
            for path, row in M.inventory(self.target, manifest["units"]).items():
                if row["type"] == "file":
                    stamp = row["mtime_ns"] + remote_mtime_delta_ns
                    os.utime(self.target / path, ns=(stamp, stamp))
        self.report = self.state / "download-check.combined"
        names = [path + (".rclonelink" if row["type"] == "symlink" else "")
                 for path, row in manifest["entries"].items() if row["type"] != "directory"]
        self.report.write_text("".join("= " + name + "\n" for name in names))
        M.verify(argparse.Namespace(download_report=str(self.report), download_exit_code=0))
        return manifest

    def test_scan_keeps_runtime_and_does_not_follow_links(self):
        M.scan(None)
        manifest = M.load_manifest()
        self.assertTrue({"workspace/_cache", "workspace/_web", "workspace/web.log"}.isdisjoint(manifest["units"]))
        row = manifest["entries"]["inputs/link"]
        self.assertEqual(row["type"], "symlink")
        self.assertEqual(row["target"], "data.bin")

    def test_download_report_must_cover_every_file_without_failure(self):
        manifest = self.prepared()
        lines = self.report.read_text().splitlines()
        self.report.write_text("\n".join(lines[:-1]) + "\n")
        with self.assertRaises(M.MigrationError):
            M.check_report(manifest, self.report)
        self.report.write_text("\n".join(lines) + "\n! inputs/data.bin\n")
        with self.assertRaises(M.MigrationError):
            M.check_report(manifest, self.report)

    def test_source_change_blocks_switch_without_moving_anything(self):
        self.prepared()
        (self.source / "inputs/data.bin").write_bytes(b"changed")
        with self.assertRaises(M.MigrationError):
            M.switch(None)
        self.assertFalse(self.hold.exists())
        self.assertFalse((self.source / "inputs").is_symlink())

    def test_destination_change_blocks_switch(self):
        self.prepared()
        (self.target / "inputs/link").unlink()
        (self.target / "inputs/link").symlink_to("different.bin")
        with self.assertRaises(M.MigrationError):
            M.switch(None)
        self.assertFalse(self.hold.exists())

    def test_same_size_remote_rewrite_blocks_switch(self):
        self.prepared()
        remote = self.target / "inputs/data.bin"
        old_stamp = remote.stat().st_mtime_ns
        remote.write_bytes(b"modified bytes")
        os.utime(remote, ns=(old_stamp + 1, old_stamp + 1))
        with self.assertRaisesRegex(M.MigrationError, "changed since download verification"):
            M.switch(None)
        self.assertFalse(self.hold.exists())

    def test_same_size_remote_rewrite_preserves_original_hold(self):
        self.prepared()
        M.switch(None)
        remote = self.target / "inputs/data.bin"
        old_stamp = remote.stat().st_mtime_ns
        remote.write_bytes(b"modified bytes")
        os.utime(remote, ns=(old_stamp + 1, old_stamp + 1))
        with self.assertRaisesRegex(M.MigrationError, "changed since download verification"):
            M.cleanup(argparse.Namespace(delete_verified_hold=True))
        self.assertEqual((self.hold / "inputs/data.bin").read_bytes(), b"original bytes")

    def test_verify_rejects_source_time_mismatch_and_newer_than_report(self):
        with self.assertRaisesRegex(M.MigrationError, "no longer matches copied source"):
            self.prepared(remote_mtime_delta_ns=2_000_000_000)
        self.assertFalse((self.state / "verified.json").exists())
        manifest = M.load_manifest()
        for path, row in manifest["entries"].items():
            if row["type"] == "file":
                os.utime(self.target / path, ns=(row["mtime_ns"], row["mtime_ns"]))
        stamp = self.report.stat().st_mtime_ns + 1
        os.utime(self.target / "inputs/data.bin", ns=(stamp, stamp))
        with self.assertRaisesRegex(M.MigrationError, "newer than its download-check report"):
            M.verify(argparse.Namespace(download_report=str(self.report), download_exit_code=0))

    def test_remote_timestamp_precision_is_compared_to_verified_remote(self):
        self.prepared(remote_mtime_delta_ns=-500_000_000)
        M.switch(None)
        M.cleanup(argparse.Namespace(delete_verified_hold=True))
        self.assertFalse(self.hold.exists())

    def test_successful_switch_and_rollback_preserve_original_paths(self):
        manifest = self.prepared()
        M.switch(None)
        self.assertTrue((self.source / "inputs").is_symlink())
        M.check_source(manifest, self.hold)
        self.assertEqual((self.source / "inputs/data.bin").read_bytes(), b"original bytes")
        M.rollback(None)
        M.check_source(manifest)
        self.assertFalse((self.source / "inputs").is_symlink())

    def test_partial_switch_failure_rolls_back(self):
        manifest = self.prepared()
        real_symlink = os.symlink
        count = 0

        def fail_second(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("simulated failure")
            return real_symlink(*args, **kwargs)

        with patch.object(M.os, "symlink", side_effect=fail_second):
            with self.assertRaises(OSError):
                M.switch(None)
        M.check_source(manifest)
        self.assertEqual(json.loads((self.state / "switch.json").read_text())["phase"], "rolled_back")

    def test_cleanup_requires_explicit_flag_and_unchanged_hold(self):
        self.prepared()
        M.switch(None)
        with self.assertRaises(M.MigrationError):
            M.cleanup(argparse.Namespace(delete_verified_hold=False))
        (self.hold / "inputs/data.bin").write_bytes(b"new local data")
        with self.assertRaises(M.MigrationError):
            M.cleanup(argparse.Namespace(delete_verified_hold=True))
        self.assertEqual((self.hold / "inputs/data.bin").read_bytes(), b"new local data")

    def test_cleanup_releases_only_verified_originals(self):
        manifest = self.prepared()
        M.switch(None)
        M.cleanup(argparse.Namespace(delete_verified_hold=True))
        self.assertFalse(self.hold.exists())
        M.check_remote(manifest)
        self.assertTrue((self.source / "inputs").is_symlink())
        self.assertTrue((self.source / "workspace/_cache/data.bin").exists())


if __name__ == "__main__":
    unittest.main()
