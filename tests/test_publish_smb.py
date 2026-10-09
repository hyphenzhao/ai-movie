"""LAN release guards and regression failure propagation (stdlib only, no media).

    .venv/bin/python tests/test_publish_smb.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("publish_smb", ROOT / "scripts" / "publish_smb.py")
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="smb_publish_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project"
        self.mount = Path(self.temp.name) / "share"
        self.root_patch = mock.patch.object(pub, "ROOT", self.root)
        self.mount_patch = mock.patch.object(pub, "MOUNT", self.mount)
        self.root_patch.start()
        self.mount_patch.start()
        self.flush_patch = mock.patch.object(pub, "wait_for_uploads")
        self.flush = self.flush_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.mount_patch.stop)
        self.addCleanup(self.flush_patch.stop)

    def accepted_fixture(self, *, extra_gates=()):
        deliver = self.root / "deliver"
        (deliver / "test_1_dubbed").mkdir(parents=True)
        (deliver / "test_1_dubbed.mp4").write_bytes(b"synthetic final")
        (deliver / "test_1_dubbed" / "ACCEPTANCE.md").write_text("accepted")
        report = self.root / "workspace" / "_release" / "v3.4" / "ACCEPTANCE.json"
        report.parent.mkdir(parents=True)
        report.write_text(json.dumps({
            "version": "v3.4", "passed": True,
            "gates": [{"key": "H1[test_1]", "ok": True}, *extra_gates],
        }))
        return report

    def test_unmounted_share_cannot_create_destination(self):
        with mock.patch.object(pub, "require_mount", side_effect=RuntimeError("unmounted")):
            with self.assertRaisesRegex(RuntimeError, "unmounted"):
                pub.publish("v3.4", ("test_1",))
        self.assertFalse(self.mount.exists())

    def test_mount_requires_exact_source_type_and_writable_target(self):
        def entry(source=pub.MOUNT_SOURCE, fstype="fuse.rclone", mode="rw", target=None):
            return f"123 1 0:50 / {target or self.mount} {mode},nosuid - {fstype} {source} rw\n"

        with mock.patch.object(Path, "read_text", return_value=entry()):
            self.assertEqual(pub.require_mount(), self.mount)
        for bad in (entry(source="unrelated:PrivateFiles"), entry(fstype="ext4"),
                    entry(mode="ro"), entry(target=self.mount / "child"), ""):
            with self.subTest(mountinfo=bad), mock.patch.object(Path, "read_text", return_value=bad):
                with self.assertRaises(RuntimeError):
                    pub.require_mount()

    def test_path_traversal_rejected_before_mount_or_write(self):
        for bad in ("../v3.4", "/tmp/release", "..", ".hidden", "v3.4/child", "v3.4\\child", ""):
            with self.subTest(version=bad), mock.patch.object(pub, "require_mount") as mount:
                with self.assertRaises(ValueError):
                    pub.publish(bad, ("test_1",))
                mount.assert_not_called()
        self.assertFalse(self.mount.exists())

    def test_failed_long_gate_overrides_legacy_passed_flag(self):
        self.accepted_fixture(extra_gates=[{"key": "L[SONE-846] L1", "ok": False}])
        with mock.patch.object(pub, "require_mount", return_value=self.mount):
            with self.assertRaisesRegex(RuntimeError, "failed acceptance gates"):
                pub.publish("v3.4", ("test_1",), "SONE-846")
        self.assertFalse(self.mount.exists())

    def test_acceptance_must_cover_selected_film_and_version(self):
        report = self.accepted_fixture()
        with mock.patch.object(pub, "require_mount", return_value=self.mount):
            with self.assertRaisesRegex(RuntimeError, "cover all requested"):
                pub.publish("v3.4", ("test_2",))
            data = json.loads(report.read_text())
            data["version"] = "v3.3"
            report.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, "no successful acceptance"):
                pub.publish("v3.4", ("test_1",))
        self.assertFalse(self.mount.exists())

    def test_changed_final_requires_new_acceptance(self):
        report = self.accepted_fixture()
        final = self.root / "deliver" / "test_1_dubbed.mp4"
        newer = report.stat().st_mtime_ns + 2_000_000_000
        os.utime(final, ns=(newer, newer))
        with mock.patch.object(pub, "require_mount", return_value=self.mount):
            with self.assertRaisesRegex(RuntimeError, "newer than"):
                pub.publish("v3.4", ("test_1",))
        self.assertFalse(self.mount.exists())

    def test_verified_release_and_no_overwrite(self):
        self.accepted_fixture()
        with mock.patch.object(pub, "require_mount", return_value=self.mount):
            target = pub.publish("v3.4", ("test_1",))
            manifest = json.loads((target / "MANIFEST.json").read_text())
            self.assertEqual(manifest["version"], "v3.4")
            self.assertEqual(len(manifest["files"]), 3)
            for file in manifest["files"]:
                self.assertEqual(pub._sha256(target / file["path"]), file["sha256"])
            self.assertEqual(self.flush.call_count, 2)
            with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                pub.publish("v3.4", ("test_1",))

    def test_copy_failure_does_not_create_a_finished_release(self):
        self.accepted_fixture()
        with mock.patch.object(pub, "require_mount", return_value=self.mount), \
                mock.patch.object(pub.shutil, "copyfile", side_effect=OSError("connection lost")):
            with self.assertRaisesRegex(OSError, "connection lost"):
                pub.publish("v3.4", ("test_1",))
        releases = self.mount / "AI-Movie" / "releases"
        self.assertFalse((releases / "v3.4").exists())
        self.assertTrue(list(releases.glob(".v3.4.partial-*")))


class UploadQueueTests(unittest.TestCase):
    def setUp(self):
        self.mount_patch = mock.patch.object(pub, "require_mount")
        self.mount_patch.start()
        self.addCleanup(self.mount_patch.stop)

    @staticmethod
    def stats(**changes):
        return {"fs": pub.MOUNT_SOURCE, "diskCache": {
            "uploadsInProgress": 0, "uploadsQueued": 0, "erroredFiles": 0,
            "outOfSpace": False, **changes,
        }}

    def test_waits_for_pending_and_errored_uploads(self):
        for field in ("uploadsInProgress", "uploadsQueued", "erroredFiles"):
            with self.subTest(field=field), \
                    mock.patch.object(pub, "_vfs_stats", side_effect=[self.stats(**{field: 1}), self.stats()]), \
                    mock.patch.object(pub.time, "sleep") as sleep:
                pub.wait_for_uploads(10)
                sleep.assert_called_once()

    def test_wrong_filesystem_or_missing_cache_or_out_of_space_fails(self):
        for stats in ({"fs": "wrong", "diskCache": self.stats()["diskCache"]},
                      {"fs": pub.MOUNT_SOURCE}, self.stats(outOfSpace=True)):
            with self.subTest(stats=stats), mock.patch.object(pub, "_vfs_stats", return_value=stats):
                with self.assertRaises(RuntimeError):
                    pub.wait_for_uploads(10)

    def test_pending_upload_timeout_is_an_error(self):
        with mock.patch.object(pub, "_vfs_stats", return_value=self.stats(uploadsQueued=1)), \
                mock.patch.object(pub.time, "monotonic", side_effect=[0, 20]):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                pub.wait_for_uploads(10)


class RegressionRunnerTests(unittest.TestCase):
    def test_failed_stage_never_reaches_publish_or_done(self):
        for failure in ("short", "long", "eval", "accept", "publish", "none"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(prefix="regression_runner_") as td:
                root = Path(td)
                (root / "scripts").mkdir()
                (root / ".venv" / "bin").mkdir(parents=True)
                (root / "workspace" / "_archive_v3.3").mkdir(parents=True)
                (root / "bin").mkdir()
                shutil.copyfile(ROOT / "scripts" / "run_v34_regression.sh",
                                root / "scripts" / "run_v34_regression.sh")

                def executable(path, body):
                    path.write_text("#!/bin/bash\n" + body)
                    path.chmod(0o755)

                executable(root / "bin" / "pgrep", "exit 1\n")
                executable(root / "bin" / "git", "echo v3.4-dev\n")
                executable(root / "scripts" / "run_v3.sh",
                           'echo short >> events\n[ "$FAIL_STAGE" != short ]\n')
                executable(root / "scripts" / "run_long.sh",
                           'echo long >> events\n[ "$FAIL_STAGE" != long ]\n')
                executable(root / ".venv" / "bin" / "python", '''
case "$1" in
  scripts/publish_smb.py)
    [ "$2" = --check-mount ] && exit 0
    echo publish >> events; [ "$FAIL_STAGE" != publish ] ;;
  scripts/eval_long.py) echo eval >> events; [ "$FAIL_STAGE" != eval ] ;;
  scripts/accept_release.py) echo accept >> events; [ "$FAIL_STAGE" != accept ] ;;
  *) exit 90 ;;
esac
''')
                result = subprocess.run(["bash", "scripts/run_v34_regression.sh"], cwd=root,
                                        env={**os.environ, "FAIL_STAGE": failure,
                                             "PATH": str(root / "bin") + ":" + os.environ["PATH"]},
                                        text=True, capture_output=True, timeout=15)
                events = (root / "events").read_text().splitlines()
                if failure == "none":
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(events[-1], "publish")
                    self.assertIn("V34_REGRESSION_DONE", result.stdout)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(events[-1], failure)
                    self.assertNotIn("V34_REGRESSION_DONE", result.stdout)
                    if failure != "publish":
                        self.assertNotIn("publish", events)


if __name__ == "__main__":
    unittest.main()
