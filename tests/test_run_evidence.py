"""Content-bound run receipts, profiles, and full-film assembly (no real media/models)."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("run_evidence_tests", ROOT / "scripts/run_evidence.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "run"
        for path in (self.run, self.root / "scripts", self.root / "ai_movie", self.root / "deliver"):
            path.mkdir()
        (self.root / "scripts/run.py").write_text("# code 1\n")
        self.addCleanup(patch.stopall)
        patch.object(evidence, "ROOT", self.root).start()
        env = dict(os.environ)
        for k in ("PROFILES", "MIN_SPEECH_MIN", "V34_RUN_DIR", "V34_CODE_HASH"):
            env.pop(k, None)
        env.update(V34_RUN_DIR=str(self.run), V34_CODE_HASH=evidence.source_hash())
        patch.dict(os.environ, env, clear=True).start()
        (self.run / "RUN.json").write_text(json.dumps({"run_id": "test-run", "code_hash": evidence.source_hash()}))
        probe = {"streams": [{"codec_type": "video"}, {"codec_type": "audio"}], "format": {"duration": "3"}}
        patch.object(evidence.subprocess, "check_output", return_value=json.dumps(probe).encode()).start()

    def clip(self, name="short"):
        work = self.root / "workspace" / name
        (work / "output").mkdir(parents=True)
        video = work / "output/current.mp4"
        video.write_bytes(b"current video bytes")
        state = work / "state.json"
        state.write_text(json.dumps({"vc": {"video": str(video)}}))
        with redirect_stdout(io.StringIO()):
            evidence.record(name)
        return state, video

    def full(self):
        film = self.root / "workspace/film"
        split = film / "_split"
        split.mkdir(parents=True)
        profile = film / "profiles.json"
        profile.write_text('{"profiles": {"P0": {"gender": "female"}}}')
        os.environ["PROFILES"] = str(profile)
        state, video = self.clip("film_p01")
        os.environ.pop("PROFILES")
        source1, source2 = split / "source1.mp4", split / "source2.mp4"
        source1.write_bytes(b"original dubbed chunk")
        source2.write_bytes(b"original no speech")
        plan = {"chunks": [{"index": 1, "file": str(source1), "speech_minutes": 1},
                           {"index": 2, "file": str(source2), "speech_minutes": 0}]}
        plan_path = split / "plan.json"
        plan_path.write_text(json.dumps(plan))
        full = self.root / "deliver/film_dubbed_full.mp4"
        full.write_bytes(b"full assembled video")
        parts = []
        for index, source in ((1, video), (2, source2)):
            processed, transport = split / f"processed{index}.mp4", split / f"p{index:02d}.ts"
            processed.write_bytes(b"processed " + str(index).encode())
            transport.write_bytes(b"transport " + str(index).encode())
            parts.append({"name": f"film_p{index:02d}", "mode": "dub" if index == 1 else "pass",
                          "source": evidence.file_record(source), "processed": evidence.file_record(processed),
                          "transport": evidence.file_record(transport), "lead_seconds": 0})
        manifest = {"version": 1, "film": "film", "min_speech_min": 0.03,
                    "plan": evidence.file_record(plan_path), "parts": parts,
                    "video": evidence.file_record(full)}
        manifest_path = split / "concat_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        with redirect_stdout(io.StringIO()):
            evidence.record("film", long=True)
        return {"state": state, "video": video, "profile": profile, "pass": source2,
                "plan": plan_path, "manifest": manifest_path, "transport": split / "p01.ts", "full": full}

    def test_video_state_and_profile_modification_reject_receipt(self):
        profile = self.root / "profiles.json"
        profile.write_text('{"version":1}')
        os.environ["PROFILES"] = str(profile)
        state, video = self.clip()
        self.assertTrue(evidence.check("short"))
        for path in (state, video, profile):
            with self.subTest(path=path.name):
                before = path.read_bytes()
                path.write_bytes(b"X" + before[1:])  # same size still fails content proof
                self.assertFalse(evidence.check("short"))
                path.write_bytes(before)
                self.assertTrue(evidence.check("short"))

    def test_profile_mode_must_match_for_independent_clip(self):
        profile = self.root / "profiles.json"
        profile.write_text('{"version":1}')
        os.environ["PROFILES"] = str(profile)
        self.clip()
        self.assertTrue(evidence.check("short"))
        os.environ.pop("PROFILES")
        self.assertFalse(evidence.check("short"))
        self.clip("no_profile")
        self.assertTrue(evidence.check("no_profile"))
        os.environ["PROFILES"] = str(profile)
        self.assertFalse(evidence.check("no_profile"))

    def test_changed_source_code_rejects_receipt(self):
        self.clip()
        self.assertTrue(evidence.check("short"))
        (self.root / "scripts/run.py").write_text("# code 2\n")
        self.assertFalse(evidence.check("short"))

    def test_missing_vc_registration_never_uses_legacy_filename(self):
        state, video = self.clip()
        (video.parent / "v2_cloned_dubbed.mp4").write_bytes(b"stale clone")
        state.write_text(json.dumps({"compose": {"video": str(video)}}))
        self.assertFalse(evidence.check("short"))
        with self.assertRaisesRegex(RuntimeError, "no vc.video"):
            evidence.output("short")
        with self.assertRaisesRegex(RuntimeError, "no vc.video"):
            evidence.record("short")

    def test_full_film_checks_profiles_without_inherited_environment(self):
        paths = self.full()
        self.assertNotIn("PROFILES", os.environ)
        self.assertTrue(evidence.check("film"))
        receipt = json.loads((self.run / "receipts/film.json").read_text())
        self.assertEqual(receipt["min_speech_min"], 0.03)
        self.assertEqual(set(receipt["chunks"]), {"film_p01"})
        self.assertEqual(set(receipt["passthrough"]), {"film_p02"})
        self.assertEqual(Path(receipt["concat"]["path"]), paths["manifest"])
        self.assertEqual(Path(receipt["plan"]["path"]), paths["plan"])
        for key in ("state", "video", "profile", "pass", "plan", "transport", "full"):
            with self.subTest(key=key):
                original = paths[key].read_bytes()
                paths[key].write_bytes(b"X" + original[1:])
                self.assertFalse(evidence.check("film"))
                paths[key].write_bytes(original)
                self.assertTrue(evidence.check("film"))

    def test_full_rejects_missing_chunks_wrong_order_or_changed_threshold(self):
        paths = self.full()
        rec_path = self.run / "receipts/film.json"
        original_receipt = rec_path.read_bytes()
        rec = json.loads(original_receipt)
        rec["chunks"] = {}
        rec_path.write_text(json.dumps(rec))
        self.assertFalse(evidence.check("film"))
        rec_path.write_bytes(original_receipt)
        manifest = json.loads(paths["manifest"].read_text())
        manifest["parts"].reverse()
        paths["manifest"].write_text(json.dumps(manifest))
        rec = json.loads(original_receipt)
        rec["concat"] = evidence.file_record(paths["manifest"])
        rec_path.write_text(json.dumps(rec))
        self.assertFalse(evidence.check("film"))
        manifest["parts"].reverse()
        paths["manifest"].write_text(json.dumps(manifest))
        rec["concat"] = evidence.file_record(paths["manifest"])
        rec_path.write_text(json.dumps(rec))
        self.assertTrue(evidence.check("film"))
        os.environ["MIN_SPEECH_MIN"] = "0.5"
        self.assertFalse(evidence.check("film"))

    def test_full_source_must_match_chunk_receipt_not_just_valid_hash(self):
        paths = self.full()
        manifest = json.loads(paths["manifest"].read_text())
        manifest["parts"][0]["source"] = evidence.file_record(paths["pass"])
        paths["manifest"].write_text(json.dumps(manifest))
        receipt_path = self.run / "receipts/film.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["concat"] = evidence.file_record(paths["manifest"])
        receipt_path.write_text(json.dumps(receipt))
        self.assertFalse(evidence.check("film"))
        with self.assertRaisesRegex(RuntimeError, "Concat manifest"):
            evidence.record("film", long=True)

    def test_smb_timestamp_rounding_does_not_change_content_identity(self):
        paths = self.full()
        manifest = json.loads(paths["manifest"].read_text())
        manifest["video"]["mtime_ns"] -= 99
        manifest["plan"]["mtime_ns"] -= 99
        for part in manifest["parts"]:
            part["source"]["mtime_ns"] -= 99
        paths["manifest"].write_text(json.dumps(manifest))
        receipt_path = self.run / "receipts/film.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["concat"] = evidence.file_record(paths["manifest"])
        receipt_path.write_text(json.dumps(receipt))
        self.assertTrue(evidence.check("film"))


if __name__ == "__main__":
    unittest.main()
