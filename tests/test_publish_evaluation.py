"""Candidate delivery evidence and failure propagation; no models or full pipeline."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("publish_evaluation", ROOT / "scripts/publish_evaluation.py")
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="evaluation_publish_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project"
        self.mount = Path(self.temp.name) / "share"
        self.run_dir = self.root / "workspace/_release/v3.4/runs/current"
        self.run_dir.mkdir(parents=True)
        for name, value in [("ROOT", self.root), ("require_mount", mock.Mock(return_value=self.mount)),
                            ("wait_for_uploads", mock.Mock()),
                            ("probe_video", mock.Mock(return_value={"duration_seconds": 10, "streams": []}))]:
            patch = mock.patch.object(pub, name, value)
            patch.start()
            self.addCleanup(patch.stop)

    def write(self, p, data):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def record(self, p):
        return {"path": str(p), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
                "sha256": pub.sha256(p)}

    def film(self, name):
        w = self.root / "workspace" / name
        w.mkdir(parents=True, exist_ok=True)
        video = w / "current.mp4"
        video.write_bytes(b"mock video " + name.encode())
        # Completed cached stages may legitimately predate RUN.started_at_ns.
        os.utime(video, ns=(1, 1))
        state = self.write(w / "state.json", {"vc": {"video": str(video)}})
        rec = {"status": "completed", "run_id": "current", "code_hash": "code",
               "completed_at_ns": 200, "state": self.record(state), "video": self.record(video), "profiles": None}
        self.write(self.run_dir / "receipts" / f"{name}.json", rec)
        return video

    def fixture(self, *, passed=False, long=False):
        self.write(self.run_dir / "RUN.json", {"version": "v3.4", "run_id": "current", "started_at_ns": 100,
                   "code_hash": "code", "films": ["test_1"], "long": "FILM" if long else "SONE-846"})
        self.film("test_1")
        if long:
            self.film("FILM_p01")
            plan = self.write(self.root / "workspace/FILM/_split/plan.json", {"chunks": [{"index": 1}]})
            video = self.root / "deliver/FILM_dubbed_full.mp4"
            video.parent.mkdir();video.write_bytes(b"mock full video")
            chunk_video = self.root / "workspace/FILM_p01/current.mp4"
            concat = self.write(plan.parent / "concat_manifest.json", {
                "version": 1, "film": "FILM", "min_speech_min": .03, "plan": self.record(plan),
                "parts": [{"name": "FILM_p01", "mode": "dub", "source": self.record(chunk_video),
                           "processed": self.record(chunk_video), "transport": self.record(chunk_video)}],
                "video": self.record(video)})
            self.write(self.run_dir / "receipts/FILM.json", {"status": "completed", "run_id": "current",
                       "code_hash": "code", "completed_at_ns": 300, "state": None, "profiles": None,
                       "video": self.record(video), "concat": self.record(concat),
                       "plan": self.record(plan), "min_speech_min": .03, "passthrough": {},
                       "chunks": {"FILM_p01": pub.sha256(self.run_dir / "receipts/FILM_p01.json")}})
        gates = [{"key": "H1[test_1]", "ok": passed, "desc": "video quality", "detail": "measured"}]
        if long:
            gates.append({"key": "L[FILM] L2b", "ok": passed, "desc": "ASR", "detail": "0.72"})
        report = self.write(self.run_dir / "reports" / ("ACCEPTANCE.json" if long else "shorts-ACCEPTANCE.json"),
                            {"version": "v3.4", "passed": passed, "gates": gates})
        report.with_suffix(".md").write_text("quality report")
        self.write(self.run_dir / "evaluation.json", {"run_id": "current", "exit_code": 0 if passed else 1,
                   "report": self.record(report), "films": ["test_1"], "long": "FILM" if long else None})

    def publish(self, *, long=False):
        return pub.publish("v3.4", "current", ("test_1",), "FILM" if long else None, shorts_only=not long)

    def test_unmounted_or_invalid_component_creates_nothing(self):
        with mock.patch.object(pub, "require_mount", side_effect=RuntimeError("unmounted")):
            with self.assertRaisesRegex(RuntimeError, "unmounted"):
                self.publish()
        for bad in ("../oops", "/tmp/oops", "a/b", "", ".hidden"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pub.publish("v3.4", bad, shorts_only=True)
        self.assertFalse(self.mount.exists())

    def test_quality_fail_delivers_only_evaluation_and_unambiguous_pointers(self):
        self.fixture()
        target = self.publish()
        self.assertEqual(target.name, "current-shorts")
        manifest = pub.load(target / "MANIFEST.json")
        self.assertIs(manifest["quality_passed"], False)
        self.assertIs(manifest["formal_release_accepted"], False)
        self.assertFalse((self.mount / "AI-Movie/releases").exists())
        self.assertIn("EVALUATION / NOT ACCEPTED", (target / "README.md").read_text())
        self.assertIn("质量门未通过", (target / "00_从这里开始.html").read_text())
        latest = pub.load(target.parent / "LATEST_EVALUATION.json")
        self.assertEqual(latest["films"][0]["path"], "current-shorts/videos/test_1.mp4")
        for f in manifest["files"]:
            self.assertEqual(pub.sha256(target / f["path"]), f["sha256"])
        self.assertEqual(pub.wait_for_uploads.call_count, 5)
        with self.assertRaisesRegex(RuntimeError, "refusing overwrite"):
            self.publish()

    def test_quality_pass_is_still_not_formal_release(self):
        self.fixture(passed=True)
        manifest = pub.load(self.publish() / "MANIFEST.json")
        self.assertTrue(manifest["quality_passed"])
        self.assertFalse(manifest["formal_release_accepted"])

    def test_failed_stale_or_different_code_receipt_stops_before_destination(self):
        self.fixture()
        p = self.run_dir / "receipts/test_1.json"
        original = pub.load(p)
        for key, value in [("status", "failed"), ("run_id", "old"), ("code_hash", "other"), ("completed_at_ns", 99)]:
            with self.subTest(key=key):
                self.write(p, {**original, key: value})
                with self.assertRaisesRegex(RuntimeError, "no successful completion evidence"):
                    self.publish()
        self.assertFalse(self.mount.exists())

    def test_same_size_same_time_changed_video_is_rejected(self):
        self.fixture()
        video = self.root / "workspace/test_1/current.mp4"
        old = video.stat();video.write_bytes(b"x" * old.st_size);os.utime(video, ns=(old.st_atime_ns, old.st_mtime_ns))
        with self.assertRaisesRegex(RuntimeError, "content changed"):
            self.publish()
        self.assertFalse(self.mount.exists())

    def test_smb_timestamp_normalization_preserves_content_identity(self):
        self.fixture()
        video = self.root / "workspace/test_1/current.mp4"
        os.utime(video, ns=(100, 100))
        self.assertTrue((self.publish() / "videos/test_1.mp4").exists())

    def test_receipt_video_must_be_current_state_video(self):
        self.fixture()
        state = self.root / "workspace/test_1/state.json"
        other = state.parent / "old.mp4";other.write_bytes(b"old video")
        self.write(state, {"vc": {"video": str(other)}})
        rp = self.run_dir / "receipts/test_1.json"
        rec = pub.load(rp);rec["state"] = self.record(state);self.write(rp, rec)
        with self.assertRaisesRegex(RuntimeError, "different file"):
            self.publish()

    def test_evaluation_error_wrong_run_or_missing_film_stops(self):
        self.fixture()
        p = self.run_dir / "evaluation.json";original = pub.load(p)
        for key, value in [("exit_code", 2), ("run_id", "old"), ("films", []), ("long", "other")]:
            with self.subTest(key=key):
                self.write(p, {**original, key: value})
                with self.assertRaisesRegex(RuntimeError, "evaluation evidence"):
                    self.publish()
        self.assertFalse(self.mount.exists())

    def test_changed_acceptance_cannot_reuse_old_evaluation_receipt(self):
        self.fixture()
        p = self.run_dir / "reports/shorts-ACCEPTANCE.json"
        p.write_text('{"version":"v3.3"}')
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.publish()

    def test_old_report_with_new_receipt_is_rejected(self):
        self.fixture()
        rp = self.run_dir / "receipts/test_1.json"
        rec = pub.load(rp);rec["completed_at_ns"] = 5_000_000_000;self.write(rp, rec)
        report = self.run_dir / "reports/shorts-ACCEPTANCE.json"
        os.utime(report, ns=(1, 1))
        ep = self.run_dir / "evaluation.json"
        ev = pub.load(ep);ev["report"] = self.record(report);self.write(ep, ev)
        with self.assertRaisesRegex(RuntimeError, "predates"):
            self.publish()

    def test_intermediate_symlink_cannot_redirect_to_local_storage(self):
        outside = Path(self.temp.name) / "local-only"
        outside.mkdir()
        (self.mount / "AI-Movie").mkdir(parents=True)
        (self.mount / "AI-Movie/evaluations").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "not a real directory"):
            self.publish()
        self.assertEqual(list(outside.iterdir()), [])

    def test_long_requires_every_chunk_and_matching_receipt_hash(self):
        self.fixture(long=True)
        rp = self.run_dir / "receipts/FILM.json";original = pub.load(rp)
        for chunks in ({}, {"FILM_p01": "bad"}):
            self.write(rp, {**original, "chunks": chunks})
            with self.assertRaisesRegex(RuntimeError, "chunk"):
                self.publish(long=True)
        self.assertFalse(self.mount.exists())

    def test_full_delivery_contains_actual_long_and_short_and_chunk_state(self):
        self.fixture(long=True)
        target = self.publish(long=True)
        films = pub.load(target / "FINAL_FILMS.json")["films"]
        self.assertEqual([f["name"] for f in films], ["test_1", "FILM"])
        self.assertEqual(target.name, "current")
        self.assertTrue((target / "receipts/states/FILM_p01.json").exists())
        self.assertEqual((target / "videos/FILM_dubbed_full.mp4").read_bytes(), b"mock full video")

    def test_concat_cannot_substitute_another_source_even_with_fresh_manifest_hash(self):
        self.fixture(long=True)
        cp = self.root / "workspace/FILM/_split/concat_manifest.json"
        other = cp.parent / "older.mp4";other.write_bytes(b"old but complete")
        concat = pub.load(cp);concat["parts"][0]["source"] = self.record(other);self.write(cp, concat)
        rp = self.run_dir / "receipts/FILM.json"
        rec = pub.load(rp);rec["concat"] = self.record(cp);self.write(rp, rec)
        with self.assertRaisesRegex(RuntimeError, "concat source differs"):
            self.publish(long=True)

    def test_upload_failure_leaves_partial_and_no_new_final_pointer(self):
        self.fixture()
        with mock.patch.object(pub, "wait_for_uploads", side_effect=[None, RuntimeError("upload pending")]):
            with self.assertRaisesRegex(RuntimeError, "upload pending"):
                self.publish()
        parent = self.mount / "AI-Movie/evaluations/v3.4"
        self.assertFalse((parent / "current-shorts").exists())
        self.assertFalse((parent / "LATEST_EVALUATION.json").exists())
        self.assertTrue(list(parent.glob(".current-shorts.partial-*")))

    def test_failure_confirming_rename_does_not_update_latest_pointer(self):
        self.fixture()
        with mock.patch.object(pub, "wait_for_uploads", side_effect=[None, None, RuntimeError("rename unconfirmed")]):
            with self.assertRaisesRegex(RuntimeError, "rename unconfirmed"):
                self.publish()
        self.assertFalse((self.mount / "AI-Movie/evaluations/v3.4/LATEST_EVALUATION.json").exists())

    def test_long_marks_passthrough_blocks_without_fake_completion_receipts(self):
        self.fixture(long=True)
        pp = self.root / "workspace/FILM/_split/plan.json"
        original = pp.parent / "original.mp4";original.write_bytes(b"original chunk")
        self.write(pp, {"chunks": [{"index": 1, "speech_minutes": 1},
                                   {"index": 2, "speech_minutes": .01, "file": str(original)}]})
        rp = self.run_dir / "receipts/FILM.json"
        rec = pub.load(rp);rec["plan"] = self.record(pp)
        rec["min_speech_min"] = .03;rec["passthrough"] = {"FILM_p02": self.record(original)}
        cp = pp.parent / "concat_manifest.json";concat = pub.load(cp);concat["plan"] = self.record(pp)
        concat["parts"].append({"name": "FILM_p02", "mode": "pass", "source": self.record(original),
                               "processed": self.record(original), "transport": self.record(original)})
        self.write(cp, concat);rec["concat"] = self.record(cp);self.write(rp, rec)
        target = self.publish(long=True)
        long = pub.load(target / "FINAL_FILMS.json")["films"][1]
        self.assertEqual(long["dubbed_chunks"], ["FILM_p01"])
        self.assertEqual(long["passthrough_chunks"][0]["name"], "FILM_p02")
        self.assertIn("未宣称重新配音", (target / "README.md").read_text())

    def test_source_changes_during_upload_prevent_completed_pointer(self):
        self.fixture()
        calls = 0
        def mutate(_):
            nonlocal calls
            calls += 1
            if calls == 2:
                (self.run_dir / "receipts/test_1.json").write_text('{"status":"failed"}')
        with mock.patch.object(pub, "wait_for_uploads", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "evidence changed"):
                self.publish()
        parent = self.mount / "AI-Movie/evaluations/v3.4"
        self.assertFalse((parent / "current-shorts").exists())
        self.assertFalse((parent / "LATEST_EVALUATION.json").exists())

    def test_report_text_is_html_escaped(self):
        _, page = pub.entry_pages("v3.4", "r", [], {"passed": False, "gates": [
            {"ok": False, "key": "<script>bad()</script>", "desc": "<img>", "detail": "&"}]}, "reports/r.json", shorts_only=True)
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)


class VideoProbeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe unavailable")
    def test_real_small_av_file_and_invalid_file(self):
        with tempfile.TemporaryDirectory(prefix="evaluation_probe_") as td:
            video = Path(td) / "test.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=32x32:rate=5",
                            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "0.4",
                            "-c:v", "mpeg4", "-c:a", "aac", str(video)], check=True)
            self.assertGreater(pub.probe_video(video)["duration_seconds"], 0)
            video.write_bytes(b"not a film")
            with self.assertRaisesRegex(RuntimeError, "ffprobe|valid video"):
                pub.probe_video(video)


if __name__ == "__main__":
    unittest.main()
