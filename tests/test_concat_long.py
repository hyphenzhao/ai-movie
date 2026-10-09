"""Reassembly must follow current state and invalidate stale trim/pass encodes."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("concat_long_tests", ROOT / "scripts/concat_long.py")
concat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(concat)
e_spec = importlib.util.spec_from_file_location("concat_run_evidence_tests", ROOT / "scripts/run_evidence.py")
evidence = importlib.util.module_from_spec(e_spec)
e_spec.loader.exec_module(evidence)


class ConcatCacheTests(unittest.TestCase):
    def test_existing_unproven_trim_rebuilt_then_resumed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src, dst = root / "source.mp4", root / "trim.mp4"
            src.write_bytes(b"new clip")
            dst.write_bytes(b"v3.3 old trim")
            def encode(cmd):
                Path(cmd[-1]).write_bytes(src.read_bytes())
            with patch.object(concat, "run", side_effect=encode) as run:
                concat.encode_cached(src, dst, {"lead": 1}, ["ffmpeg", "-i", str(src)])
                self.assertEqual(dst.read_bytes(), b"new clip")
                concat.encode_cached(src, dst, {"lead": 1}, ["ffmpeg", "-i", str(src)])
                self.assertEqual(run.call_count, 1)

    def test_source_settings_and_completed_output_changes_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src, dst = root / "source.mp4", root / "pass.mp4"
            src.write_bytes(b"clip one")
            def encode(cmd):
                Path(cmd[-1]).write_bytes(src.read_bytes())
            command = ["ffmpeg", "-i", str(src)]
            with patch.object(concat, "run", side_effect=encode) as run:
                concat.encode_cached(src, dst, {"gain_db": 1}, command)
                src.write_bytes(b"clip two")  # same length, different content
                concat.encode_cached(src, dst, {"gain_db": 1}, command)
                self.assertEqual(dst.read_bytes(), b"clip two")
                concat.encode_cached(src, dst, {"gain_db": 2}, command)
                dst.write_bytes(b"external")
                concat.encode_cached(src, dst, {"gain_db": 2}, command)
                self.assertEqual(run.call_count, 4)
                self.assertEqual(dst.read_bytes(), b"clip two")

    def test_failed_reencode_preserves_last_completed_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src, dst = root / "source.mp4", root / "trim.mp4"
            src.write_bytes(b"new clip")
            dst.write_bytes(b"previous completed trim")
            def fail(cmd):
                Path(cmd[-1]).write_bytes(b"unfinished")
                raise RuntimeError("interrupted encode")
            with patch.object(concat, "run", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    concat.encode_cached(src, dst, {}, ["ffmpeg"])
            self.assertEqual(dst.read_bytes(), b"previous completed trim")
            self.assertFalse(dst.with_suffix(".part.mp4").exists())
            self.assertFalse(dst.with_suffix(".mp4.cache.json").exists())

    def test_registered_builtin_fallback_beats_old_cloned_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work = root / "workspace/film_p01"
            (work / "output").mkdir(parents=True)
            old = work / "output/v2_cloned_dubbed.mp4"
            current = work / "output/film_p01_dubbed.mp4"
            old.write_bytes(b"old clone")
            current.write_bytes(b"new builtin")
            state = {"vc": {"video": str(current), "converted": 0}, "compose": {"video": str(current)}}
            self.assertEqual(concat.dubbed_source(state, work, root), current)
            current.unlink()
            with self.assertRaises(FileNotFoundError):
                concat.dubbed_source(state, work, root)

    def test_strict_source_requires_registered_vc_despite_old_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work = root / "workspace/film_p01"
            (work / "output").mkdir(parents=True)
            (work / "output/v2_cloned_dubbed.mp4").write_bytes(b"old clone")
            with self.assertRaisesRegex(RuntimeError, "no vc.video"):
                concat.dubbed_source({}, work, root, require_vc=True)

    def test_manifest_proves_actual_concat_inputs_and_no_speech_ignores_old_dub(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            split = root / "workspace/film/_split"
            split.mkdir(parents=True)
            parts = []
            for i, speech in ((1, 1), (2, 0)):
                source = split / f"source{i}.mp4"
                source.write_bytes(f"original-{i}".encode())
                work = root / "workspace" / f"film_p{i:02d}"
                work.mkdir()
                video = work / "current.mp4"
                video.write_bytes(f"dub-{i}".encode())
                (work / "state.json").write_text(json.dumps({"vc": {"video": str(video)}}))
                parts.append({"index": i, "start": i - 1, "end": i, "file": str(source),
                              "speech_minutes": speech})
            plan = {"video": parts[0]["file"], "duration": 2, "chunks": parts}
            (split / "plan.json").write_text(json.dumps(plan))
            def probe(path, entries, stream):
                if entries == "stream=r_frame_rate": return "25/1"
                if entries == "stream=width,height": return "16,16"
                return "2" if "dubbed_full" in str(path) else "1"
            def fake_ffmpeg(command):
                source = command[command.index("-i") + 1]
                if source.startswith("concat:"):
                    data = b"".join(Path(x).read_bytes() for x in source[7:].split("|"))
                else:
                    data = Path(source).read_bytes()
                Path(command[-1]).write_bytes(data)
            with patch.object(sys, "argv", ["concat_long.py", "film", str(split)]), \
                 patch.dict(sys.modules, {"run_evidence": evidence}), \
                 patch.dict(os.environ, {"STRICT_RUN": "1", "MIN_SPEECH_MIN": "0.03"}), \
                 patch.object(concat, "probe", side_effect=probe), \
                 patch.object(concat, "run", side_effect=fake_ffmpeg):
                self.assertEqual(concat.main(), 0)
            out = root / "deliver/film_dubbed_full.mp4"
            self.assertEqual(out.read_bytes(), b"dub-1original-2")
            manifest = json.loads((split / "concat_manifest.json").read_text())
            self.assertEqual([(p["name"], p["mode"]) for p in manifest["parts"]],
                             [("film_p01", "dub"), ("film_p02", "pass")])
            self.assertEqual(Path(manifest["parts"][1]["source"]["path"]), Path(parts[1]["file"]))
            self.assertEqual(manifest["video"], evidence.file_record(out))
            for part in manifest["parts"]:
                for key in ("source", "processed", "transport"):
                    self.assertTrue(evidence.validate_record(part[key]))


if __name__ == "__main__":
    unittest.main()
