"""Driver/CLI/exit-status contracts in a temporary fake project; no pipelines or models."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
R = importlib.import_module("run_v34_regression")
E = importlib.import_module("run_evidence")
PUB = importlib.import_module("publish_evaluation")


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v34_driver_test_")
        self.addCleanup(self.temp.cleanup)
        self.old_cwd = Path.cwd()
        self.addCleanup(os.chdir, self.old_cwd)
        self.root = Path(self.temp.name) / "project"
        self.mount = Path(self.temp.name) / "share"
        for p in (self.root / "scripts", self.root / "ai_movie", self.root / "workspace/_archive_v3.3", self.mount):
            p.mkdir(parents=True)
        self.code = self.root / "scripts/fixture.py"
        self.code.write_text("# frozen source\n")
        self.run_id = "test-current"
        self.run_dir = self.root / "workspace/_release/v3.4/runs" / self.run_id
        self.completed = set()
        self.commands, self.publications = [], []
        self.accept_rc = 1
        self.accept_report = "valid"
        self.long_rc = 1
        self.child_rc = 0
        self.mutate_final_publish = False
        self.addCleanup(patch.stopall)
        env = dict(os.environ)
        for key in ("V34_RUN_DIR", "V34_CODE_HASH", "PROFILES", "MIN_SPEECH_MIN"):
            env.pop(key, None)
        env["V34_RUN_ID"] = self.run_id
        patch.dict(os.environ, env, clear=True).start()
        patch.object(R, "ROOT", self.root).start()
        patch.object(E, "ROOT", self.root).start()
        patch.object(R, "require_mount", return_value=self.mount).start()
        patch.object(R, "wait_for_uploads").start()
        patch.object(R, "check", side_effect=lambda name: name in self.completed).start()
        patch.object(R, "record", side_effect=lambda name, **kw: self.completed.add(name)).start()
        patch.object(R.subprocess, "check_output", side_effect=lambda args, **kw:
                     "v3.4-dev\n" if "branch" in args else "frozen-git-head\n").start()
        patch.object(R.subprocess, "run", side_effect=self.command).start()

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def package(self, *, shorts_only):
        target = self.mount / "AI-Movie/evaluations/v3.4" / (self.run_id + ("-shorts" if shorts_only else ""))
        target.mkdir(parents=True, exist_ok=True)
        films = []
        inventory = []
        for name in R.FILMS + ([] if shorts_only else [R.LONG]):
            rel = f"videos/{name}.mp4" if name in R.FILMS else f"videos/{name}_dubbed_full.mp4"
            p = target / rel
            p.parent.mkdir(exist_ok=True)
            p.write_bytes(b"fixture film " + name.encode())
            row = dict(path=rel, size=p.stat().st_size, sha256=E.sha256(p))
            inventory.append(row)
            films.append(dict(name=name, **row))
        pointers = dict(status="EVALUATION / NOT ACCEPTED", version="v3.4", run_id=self.run_id,
                        scope="shorts-only" if shorts_only else "full", entry="00_从这里开始.html", films=films)
        for name, content in (("FINAL_FILMS.json", json.dumps(pointers)), ("00_从这里开始.html", "<p>fixture</p>")):
            p = target / name
            p.write_text(content)
            inventory.append(dict(path=name, size=p.stat().st_size, sha256=E.sha256(p)))
        self.write(target / "MANIFEST.json", dict(**pointers, formal_release_accepted=False,
                   code_hash=json.loads((self.run_dir / "RUN.json").read_text())["code_hash"], files=inventory))
        return target

    def command(self, args, **kwargs):
        self.commands.append(args)
        if "scripts/run_v3.sh" in args:
            if self.child_rc == 0:
                self.completed.add(args[-1])
            return subprocess.CompletedProcess(args, self.child_rc)
        if "scripts/run_long.sh" in args:
            self.assertEqual(kwargs["env"]["STRICT_RUN"], "1")
            self.assertEqual(kwargs["env"]["RERUN"], self.run_id)
            return subprocess.CompletedProcess(args, 0)
        if "scripts/accept_release.py" in args:
            release = self.root / "workspace/_release/v3.4"
            if self.accept_report == "valid":
                gates = [{"key": f"H1[{film}]", "ok": self.accept_rc == 0} for film in R.FILMS]
                if "--long" in args:
                    gates.append({"key": f"L[{R.LONG}] L1", "ok": self.accept_rc == 0})
                self.write(release / "ACCEPTANCE.json", {"version": "v3.4", "gates": gates,
                                                          "passed": self.accept_rc == 0})
                (release / "ACCEPTANCE.md").write_text("fresh report")
            elif self.accept_report == "malformed":
                (release / "ACCEPTANCE.json").write_text("{partial report")
            return subprocess.CompletedProcess(args, self.accept_rc)
        if "scripts/eval_long.py" in args:
            self.write(self.root / "deliver" / f"{R.LONG}_full/ACCEPTANCE_FULL.json",
                       {"film": R.LONG, "passed": self.long_rc == 0, "checks": [{"id": "L1", "ok": self.long_rc == 0}]})
            return subprocess.CompletedProcess(args, self.long_rc)
        if "scripts/publish_evaluation.py" in args:
            def fake_publish(version, run_id, shorts, long, *, shorts_only, upload_timeout):
                # The real CLI parses these: mismatched flags cannot be hidden by this fake.
                self.publications.append(dict(version=version, run_id=run_id, shorts=shorts,
                                              long=long, shorts_only=shorts_only))
                target = self.package(shorts_only=shorts_only)
                if self.mutate_final_publish and not shorts_only:
                    self.code.write_text("# source changed during final publication\n")
                return target
            i = args.index("scripts/publish_evaluation.py")
            with patch.object(sys, "argv", args[i:]), patch.object(PUB, "publish", side_effect=fake_publish):
                rc = PUB.main()
            return subprocess.CompletedProcess(args, rc)
        if "scripts/publish_smb.py" in args:
            return subprocess.CompletedProcess(args, 0)
        raise AssertionError(f"unexpected child process: {args}")

    def run_driver(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return R.main()

    def test_quality_failures_still_publish_short_then_full_candidates(self):
        self.assertEqual(self.run_driver(), 0)
        self.assertEqual([p["shorts_only"] for p in self.publications], [True, False])
        self.assertEqual(self.publications[0]["shorts"], tuple(R.FILMS))
        self.assertEqual(self.publications[1]["long"], R.LONG)
        self.assertFalse(any("scripts/publish_smb.py" in c for c in self.commands))
        self.assertNotIn("PROFILES", os.environ)
        progress = json.loads((self.run_dir / "progress.json").read_text())
        self.assertEqual(progress["stage"], "completed")
        self.assertIs(progress["accepted"], False)

    def test_execution_failure_stops_before_acceptance_or_publication(self):
        self.child_rc = 1
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(self.publications, [])
        self.assertEqual(json.loads((self.run_dir / "progress.json").read_text())["stage"], "failed")

    def test_exit_one_without_fresh_report_cannot_reuse_old_acceptance(self):
        self.write(self.root / "workspace/_release/v3.4/ACCEPTANCE.json",
                   {"version": "v3.4", "passed": False, "gates": [{"ok": False}]})
        self.accept_report = "missing"
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(self.publications, [])

    def test_partial_report_on_evaluator_exception_is_not_quality_failure(self):
        self.accept_report = "malformed"
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(self.publications, [])

    def test_exit_two_is_execution_failure_even_with_report(self):
        self.accept_rc = 2
        self.assertEqual(self.run_driver(), 1)
        self.assertEqual(self.publications, [])

    def test_existing_unverified_directories_must_not_count_as_completed_delivery(self):
        for suffix in ("", "-shorts"):
            (self.mount / "AI-Movie/evaluations/v3.4" / (self.run_id + suffix)).mkdir(parents=True)
        self.assertEqual(self.run_driver(), 1)
        self.assertNotEqual(json.loads((self.run_dir / "progress.json").read_text())["stage"], "completed")

    def test_source_change_during_last_command_must_not_be_marked_completed(self):
        self.mutate_final_publish = True
        self.assertEqual(self.run_driver(), 1)
        self.assertNotEqual(json.loads((self.run_dir / "progress.json").read_text())["stage"], "completed")

    def test_confirm_delivery_repairs_pointer_and_never_downgrades_existing_full(self):
        self.assertEqual(self.run_driver(), 0)
        parent = self.mount / "AI-Movie/evaluations/v3.4"
        latest = parent / "LATEST_EVALUATION.json"
        full = latest.read_bytes()
        R.confirm_delivery(self.run_id, shorts_only=True)
        self.assertEqual(latest.read_bytes(), full)
        latest.unlink()
        R.confirm_delivery(self.run_id, shorts_only=False)
        self.assertEqual(json.loads(latest.read_text())["directory"], self.run_id)
        self.assertEqual(json.loads(latest.read_text())["manifest_sha256"], E.sha256(parent / self.run_id / "MANIFEST.json"))

    def test_existing_package_hash_and_scope_must_match(self):
        self.assertEqual(self.run_driver(), 0)
        target = self.mount / "AI-Movie/evaluations/v3.4" / self.run_id
        video = target / "videos/test_1.mp4"
        original = video.read_bytes()
        video.write_bytes(b"x" * len(original))
        with self.assertRaisesRegex(RuntimeError, "file is missing or changed"):
            R.confirm_delivery(self.run_id, shorts_only=False)
        video.write_bytes(original)
        manifest = json.loads((target / "MANIFEST.json").read_text())
        manifest["scope"] = "shorts-only"
        self.write(target / "MANIFEST.json", manifest)
        with self.assertRaisesRegex(RuntimeError, "different run, code or scope"):
            R.confirm_delivery(self.run_id, shorts_only=False)


class VoiceMetricExitTests(unittest.TestCase):
    def test_failed_quality_gate_is_zero_but_uncomputed_metric_is_two(self):
        module = importlib.import_module("voice_consistency")
        with patch.object(sys, "argv", ["voice_consistency.py", "fake-state.json", "--nice", "0"]), \
                patch.object(module, "chunk_report", return_value={"rc": 0, "gates": [
                    {"gate": "V1", "group": "female", "ok": False, "value": 0.5, "limit": 0.1, "why": "quality"}], "notes": []}), \
                patch.object(module, "_print_gates"), redirect_stdout(io.StringIO()):
            self.assertEqual(module.main(), 0)
        with patch.object(sys, "argv", ["voice_consistency.py", "fake-state.json", "--nice", "0"]), \
                patch.object(module, "chunk_report", return_value={"rc": 2, "why": "encoder unavailable"}), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(module.main(), 2)


class SingleFilmReportTests(unittest.TestCase):
    def run_fake_film(self, report, rc):
        with tempfile.TemporaryDirectory(prefix="v34_film_shell_") as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / ".venv/bin").mkdir(parents=True)
            shutil.copyfile(ROOT / "scripts/run_v3.sh", root / "scripts/run_v3.sh")
            executor = root / ".venv/bin/python"
            executor.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
from pathlib import Path
args = [a for a in sys.argv[1:] if a != "-u"]
with open(os.environ["MOCK_CALLS"], "a") as stream:
    stream.write(json.dumps(args) + "\\n")
if args[0] == "-":
    os.execv(sys.executable, [sys.executable, *args])
if args[0] == "scripts/eval_pipeline.py":
    Path(args[args.index("--json") + 1]).write_text(os.environ["MOCK_REPORT"])
    raise SystemExit(int(os.environ["MOCK_EVAL_RC"]))
''')
            executor.chmod(0o755)
            calls = root / "calls.jsonl"
            env = dict(os.environ, MOCK_CALLS=str(calls), MOCK_REPORT=report, MOCK_EVAL_RC=str(rc),
                       PROFILES="", PIPE_ARGS="", V34_RUN_DIR=str(root / "run"))
            proc = subprocess.run(["bash", str(root / "scripts/run_v3.sh"), "fixture"],
                                  cwd=root, env=env, capture_output=True, text=True, timeout=20)
            commands = [json.loads(line) for line in calls.read_text().splitlines()]
            return proc.returncode, [c[0] for c in commands], proc.stdout + proc.stderr

    def test_complete_quality_failure_still_delivers_and_records(self):
        rc, calls, output = self.run_fake_film('[{"key":"C1","ok":false},{"key":"note","ok":null}]', 1)
        self.assertEqual(rc, 0, output)
        self.assertIn("scripts/deliver.py", calls)
        self.assertIn("scripts/run_evidence.py", calls)

    def test_partial_malformed_or_exit_disagreement_stops_before_delivery(self):
        for report, code in (("{partial", 1), ('{"ok":false}', 1), ('[{"key":"C1","ok":1}]', 0),
                             ('[{"key":"note","ok":null}]', 0), ('[{"key":"C1","ok":false}]', 0),
                             ('[{"key":"C1","ok":true}]', 1)):
            with self.subTest(report=report, code=code):
                rc, calls, output = self.run_fake_film(report, code)
                self.assertNotEqual(rc, 0, output)
                self.assertNotIn("scripts/deliver.py", calls)
                self.assertNotIn("scripts/run_evidence.py", calls)


if __name__ == "__main__":
    unittest.main()
