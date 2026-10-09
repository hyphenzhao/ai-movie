"""A failed rerun must not concat stale chunk video into a strict release."""
from __future__ import annotations

import os
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent


class StrictLongRunTests(unittest.TestCase):
    def test_failed_chunk_with_old_video_stops_strict_concat(self):
        for strict in ("1", "0"):
            with self.subTest(strict=strict), tempfile.TemporaryDirectory(prefix="strict_long_") as td:
                root = Path(td)
                for rel in ("scripts", ".venv/bin", "bin", "inputs", "workspace/film",
                            "workspace/film_p01/output"):
                    (root / rel).mkdir(parents=True, exist_ok=True)
                (root / "workspace/film/profiles.json").write_text("{}")
                (root / "workspace/film_p01/output/v2_cloned_dubbed.mp4").write_bytes(b"old release")
                (root / "workspace/film_p01/run_v3.log").write_text("DONE in old run\n")
                (root / "inputs/original.mp4").write_bytes(b"source")
                shutil.copyfile(ROOT / "scripts/run_long.sh", root / "scripts/run_long.sh")

                def executable(rel, body):
                    path = root / rel
                    path.write_text("#!/bin/bash\n" + body)
                    path.chmod(0o755)

                executable("bin/sleep", "/bin/sleep 0.01\n")
                executable("scripts/run_v3.sh", 'echo attempted >> attempts\nexit 7\n')
                executable(".venv/bin/python", '''
case "$1" in
  -m) exit 0 ;;
  -)
    if [ "$#" -gt 2 ]; then printf '01\\t%s/inputs/original.mp4\\tdub\\t1\\n' "$PWD"; fi
    exit 0 ;;
  scripts/smart_split.py) exit 0 ;;
  scripts/concat_long.py) touch concatenated; exit 0 ;;
  -u) exit 0 ;;
  *) exit 90 ;;
esac
''')
                result = subprocess.run(
                    ["bash", "scripts/run_long.sh", "inputs/original.mp4", "film"], cwd=root,
                    env={**os.environ, "STRICT_RUN": strict, "RERUN": "v3.4",
                         "PATH": str(root / "bin") + ":" + os.environ["PATH"]},
                    text=True, capture_output=True, timeout=10)
                self.assertEqual(len((root / "attempts").read_text().splitlines()), 3)
                if strict == "1":
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("STRICT_RUN=1", result.stdout)
                    self.assertFalse((root / "concatenated").exists())
                    self.assertNotIn("LONG_DONE", result.stdout)
                else:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertTrue((root / "concatenated").exists())

    def make_resume_fixture(self, root):
        for rel in ("scripts", ".venv/bin", "bin", "inputs", "ai_movie", "run", "receipts",
                    "workspace/film/_split/chunks", "workspace/film_p01/output"):
            (root / rel).mkdir(parents=True, exist_ok=True)
        (root / "ai_movie/__init__.py").write_text("")
        (root / "ai_movie/config.py").write_text("PROFILE_ENROL_DENSITY=0.1\n")
        (root / "workspace/film/profiles.json").write_text('{"version":1}')
        (root / "workspace/film_p01/output/v2_cloned_dubbed.mp4").write_bytes(b"old release")
        (root / "workspace/film_p01/run_v3.log").write_text("DONE in old run\n")
        source = root / "workspace/film/_split/chunks/film_p01.mp4"
        source.write_bytes(b"source")
        (root / "inputs/original.mp4").write_bytes(b"source")
        (root / "workspace/film/_split/plan.json").write_text(json.dumps({"chunks": [
            {"index": 1, "file": str(source), "minutes": 2, "speech_minutes": 1}]}))
        shutil.copyfile(ROOT / "scripts/run_long.sh", root / "scripts/run_long.sh")

        def executable(rel, body):
            path = root / rel
            path.write_text("#!/bin/bash\n" + body)
            path.chmod(0o755)

        executable("bin/sleep", "/bin/sleep 0.01\n")
        real_python = shlex.quote(sys.executable)
        executable(".venv/bin/python", '''
case "$1" in
  -m) exit 0 ;;
  -) exec REAL_PY "$@" ;;
  scripts/smart_split.py) exit 0 ;;
  scripts/run_evidence.py) exec REAL_PY "$@" ;;
  scripts/concat_long.py) touch concatenated; exit 0 ;;
  -u)
    case "$2" in
      scripts/run_pipeline.py) echo enrol >> stages; exit "${FAKE_ENROL_RC:-0}" ;;
      scripts/build_profiles.py)
        echo build >> stages
        [ "${FAKE_BUILD_RC:-0}" = 0 ] || exit "$FAKE_BUILD_RC"
        printf '{"version":2}' > workspace/film/profiles.json
        exit 0 ;;
      scripts/voice_consistency.py) exit "${FAKE_CONSISTENCY_RC:-0}" ;;
    esac ;;
esac
exit 90
'''.replace("REAL_PY", real_python))
        executable("scripts/run_v3.sh", '''
echo attempted >> attempts
[ "${FAKE_CHUNK_RC:-0}" = 0 ] || exit "$FAKE_CHUNK_RC"
[ "${FAKE_NO_OUTPUT:-0}" = 0 ] || exit 0
REAL_PY scripts/fake_finish.py "$1"
'''.replace("REAL_PY", real_python))
        (root / "scripts/fake_finish.py").write_text('''
import hashlib,json,os,sys
from pathlib import Path
name=sys.argv[1]
w=Path('workspace')/name
v=(w/'output/current_builtin.mp4').absolute()
v.write_bytes(b'new builtin')
(w/'state.json').write_text(json.dumps({'vc':{'video':str(v)}}))
if os.environ.get('V34_RUN_DIR'):
    Path('receipts',name).write_text(hashlib.sha256(v.read_bytes()).hexdigest())
''')
        (root / "scripts/run_evidence.py").write_text('''
import hashlib,json,sys
from pathlib import Path
action,name=sys.argv[1:]
try:
    d=json.loads((Path('workspace')/name/'state.json').read_text())
    v=Path(d['vc']['video'])
    assert v.is_file() and v.stat().st_size
    if action=='output':print(str(v.absolute()))
    elif action=='check':
        assert Path('receipts',name).read_text()==hashlib.sha256(v.read_bytes()).hexdigest()
    else:raise ValueError(action)
except (OSError,ValueError,KeyError,AssertionError):raise SystemExit(1)
''')

    def run_fixture(self, root, **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("V34_", "FAKE_"))}
        env.update({"STRICT_RUN": "1", "RERUN": "v3.4", "REBUILD_PROFILES": "1",
                    "V34_RUN_DIR": str(root / "run"), "V34_CODE_HASH": "code-1",
                    "PATH": str(root / "bin") + ":" + os.environ["PATH"]})
        env.update(extra)
        return subprocess.run(["bash", "scripts/run_long.sh", "inputs/original.mp4", "film"],
                              cwd=root, env=env, text=True, capture_output=True, timeout=15)

    def test_resume_keeps_profile_checkpoint_and_skips_only_verified_chunk(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.make_resume_fixture(root)
            first = self.run_fixture(root)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertIn("current_builtin.mp4", first.stdout)
            self.assertEqual((root / "stages").read_text().splitlines(), ["enrol", "build"])
            self.assertTrue((root / "run/profile_checkpoints/film.json").exists())
            second = self.run_fixture(root)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn("verified completed chunk", second.stdout)
            self.assertEqual((root / "stages").read_text().splitlines(), ["enrol", "build"])
            self.assertEqual((root / "attempts").read_text().splitlines(), ["attempted"])

    def test_profile_or_code_change_stops_resume_before_any_model(self):
        for change in ("profile", "code"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                self.make_resume_fixture(root)
                first = self.run_fixture(root)
                self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
                env = {}
                if change == "profile":
                    (root / "workspace/film/profiles.json").write_text('{"version":99}')
                else:
                    env["V34_CODE_HASH"] = "code-2"
                (root / "concatenated").unlink()
                second = self.run_fixture(root, **env)
                self.assertNotEqual(second.returncode, 0)
                self.assertIn("profile checkpoint mismatch", second.stdout + second.stderr)
                self.assertEqual((root / "stages").read_text().splitlines(), ["enrol", "build"])
                self.assertEqual((root / "attempts").read_text().splitlines(), ["attempted"])
                self.assertFalse((root / "concatenated").exists())

    def test_old_tag_without_receipt_does_not_skip_chunk(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.make_resume_fixture(root)
            (root / "workspace/film/_split/status.txt").write_text("01 dub ok rerun=v3.4\n")
            result = self.run_fixture(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / "attempts").read_text().splitlines(), ["attempted"])

    def test_zero_exit_with_only_stale_v2_is_not_success(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.make_resume_fixture(root)
            result = self.run_fixture(root, FAKE_NO_OUTPUT="1")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no valid output", result.stdout)
            self.assertFalse((root / "concatenated").exists())
            self.assertEqual(len((root / "attempts").read_text().splitlines()), 3)

    def test_strict_enrolment_and_profile_failure_stop_before_chunks(self):
        for failure in ("FAKE_ENROL_RC", "FAKE_BUILD_RC"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                self.make_resume_fixture(root)
                result = self.run_fixture(root, **{failure: "7"})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("STRICT_RUN=1", result.stdout + result.stderr)
                self.assertFalse((root / "attempts").exists())
                self.assertFalse((root / "run/profile_checkpoints/film.json").exists())
                self.assertFalse((root / "concatenated").exists())

    def test_strict_film_consistency_failure_removes_only_old_reports_and_stops(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.make_resume_fixture(root)
            out = root / "deliver/film_full"
            out.mkdir(parents=True)
            old = [out / "VOICE_CONSISTENCY.json", out / "VOICE_CONSISTENCY.md"]
            for path in old:
                path.write_text("previous run")
            unrelated = out / "keep.txt"
            unrelated.write_text("keep")
            result = self.run_fixture(root, FAKE_CONSISTENCY_RC="2")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("film voice consistency execution failed", result.stdout + result.stderr)
            self.assertNotIn("LONG_DONE", result.stdout)
            self.assertTrue((root / "concatenated").exists())
            self.assertTrue(unrelated.exists())
            for path in old:
                self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
