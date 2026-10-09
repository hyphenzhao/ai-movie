"""Media boundary checks with synthetic local/NAS paths; no real mount or data."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ai_movie.web import projects as P


class MediaPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="media-path-test-")
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root, self.mount = base / "project", base / "share"
        self.workspace = self.root / "workspace"
        self.nas_workspace = self.mount / "AI-Movie/workspace"
        self.external = base / "external"
        for path in (self.workspace, self.root / "inputs", self.root / "deliver",
                     self.nas_workspace, self.external):
            path.mkdir(parents=True)
        self.media = self.nas_workspace / "test_1/output/final.mp4"
        self.media.parent.mkdir(parents=True)
        self.media.write_bytes(b"test video")
        (self.workspace / "test_1").symlink_to(self.nas_workspace / "test_1", target_is_directory=True)
        self.secret = self.external / "secret.txt"
        self.secret.write_text("private")
        values = {"ROOT": self.root, "WORKSPACE": self.workspace,
                  "INPUTS": self.root / "inputs", "DELIVER": self.root / "deliver",
                  "MEDIA_ROOTS": (self.workspace, self.root / "inputs", self.root / "deliver"),
                  "LAN_MOUNT": self.mount, "LAN_WORKSPACE": self.nas_workspace}
        for name, value in values.items():
            p = patch.object(P, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.mountinfo = f"123 1 0:50 / {self.mount} rw,nosuid - fuse.rclone {P.LAN_MOUNT_SOURCE} rw\n"
        original_read = Path.read_text

        def read(path, *args, **kwargs):
            if str(path) == "/proc/self/mountinfo":
                return self.mountinfo
            return original_read(path, *args, **kwargs)

        p = patch.object(Path, "read_text", autospec=True, side_effect=read)
        p.start()
        self.addCleanup(p.stop)

    def test_migrated_workspace_media_accepts_original_and_canonical_paths(self):
        for path in ("workspace/test_1/output/final.mp4",
                     str(self.workspace / "test_1/output/final.mp4"), str(self.media)):
            with self.subTest(path=path):
                self.assertEqual(P.safe_media_path(path), self.media)
        self.assertFalse(self.workspace.is_symlink())

    def test_existing_local_media_remains_available(self):
        local = self.workspace / "local/output.wav"
        local.parent.mkdir()
        local.write_bytes(b"audio")
        self.mountinfo = ""
        self.assertEqual(P.safe_media_path(str(local)), local)

    def test_unmounted_or_wrong_mount_cannot_expose_remote_directory(self):
        good = self.mountinfo
        for wrong in ("", good.replace(P.LAN_MOUNT_SOURCE, "wrong:PrivateFiles"),
                      good.replace("fuse.rclone", "ext4")):
            with self.subTest(mountinfo=wrong):
                self.mountinfo = wrong
                self.assertIsNone(P.safe_media_path(str(self.media)))
                self.assertIsNone(P.safe_media_path("workspace/test_1/output/final.mp4"))

    def test_external_symlinks_and_parent_traversal_remain_blocked(self):
        (self.workspace / "external").symlink_to(self.external, target_is_directory=True)
        (self.nas_workspace / "test_1/leak.txt").symlink_to(self.secret)
        sibling = self.mount / "AI-Movie/private.txt"
        sibling.write_text("private")
        for path in (str(self.secret), "workspace/external/secret.txt",
                     "workspace/test_1/leak.txt", str(sibling),
                     str(self.nas_workspace / "../private.txt")):
            with self.subTest(path=path):
                self.assertIsNone(P.safe_media_path(path))

    def test_remote_workspace_root_cannot_itself_redirect_elsewhere(self):
        alternate = self.mount / "elsewhere"
        self.nas_workspace.rename(alternate)
        self.nas_workspace.symlink_to(alternate, target_is_directory=True)
        self.assertIsNone(P.safe_media_path("workspace/test_1/output/final.mp4"))
        self.assertIsNone(P._mounted_lan_workspace())

    def test_missing_targets_and_unsupported_extensions_stay_blocked(self):
        (self.nas_workspace / "test_1/missing.mp4").symlink_to("absent.mp4")
        code = self.nas_workspace / "test_1/private.py"
        code.write_text("pass")
        self.assertIsNone(P.safe_media_path("workspace/test_1/missing.mp4"))
        self.assertIsNone(P.safe_media_path(str(code)))

    def test_whole_inputs_and_deliver_root_links_still_work(self):
        for name in ("inputs", "deliver"):
            local, remote = self.root / name, self.mount / "AI-Movie" / name
            remote.mkdir()
            media = remote / "test.mp4"
            media.write_bytes(b"video")
            local.rmdir()
            local.symlink_to(remote, target_is_directory=True)
            self.assertEqual(P.safe_media_path(str(local / "test.mp4")), media)


if __name__ == "__main__":
    unittest.main()
