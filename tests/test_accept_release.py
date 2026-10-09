"""Release reports must include long-film failures and reject missing evaluations."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("accept_release_tests", ROOT / "scripts/accept_release.py")
accept = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accept)


class ReleaseAcceptanceTests(unittest.TestCase):
    def test_long_failure_controls_status_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "scripts").mkdir()
            (root / "scripts/eval_long.py").write_text(
                "def evaluate(film):\n"
                " return {'checks':[{'id':'L2','desc':'long quality','ok':False,'value':'bad'}],"
                "'notes':['long note']}\n")
            args = ["accept_release.py", "v3.4", "--films", "", "--long", "film",
                    "--baseline", str(root / "baseline")]
            with patch.object(accept, "ROOT", root), patch.object(sys, "argv", args), redirect_stdout(io.StringIO()):
                self.assertEqual(accept.main(), 1)
            out = root / "workspace/_release/v3.4"
            report = json.loads((out / "ACCEPTANCE.json").read_text())
            self.assertFalse(report["passed"])
            self.assertFalse(report["gates"][0]["ok"])
            md = (out / "ACCEPTANCE.md").read_text()
            self.assertIn("未通过", md)
            self.assertIn("L[film] L2", md)
            self.assertIn("film: long note", md)

    def test_evaluator_crash_cannot_reuse_previous_report(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            report = out / "film.eval.json"
            report.write_text('[{"key":"A1","ok":true}]')
            result = subprocess.CompletedProcess([], 1, stdout="", stderr="Traceback")
            with patch.object(accept.subprocess, "run", return_value=result):
                with self.assertRaisesRegex(RuntimeError, "new report"):
                    accept.run_eval_pipeline("film", out)
            self.assertFalse(report.exists())

    def test_quality_failure_with_fresh_report_is_a_valid_evaluation(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            def evaluate(*args, **kwargs):
                (out / "film.eval.json").write_text('[{"key":"A1","ok":false}]')
                return subprocess.CompletedProcess([], 1, stdout="", stderr="")
            with patch.object(accept.subprocess, "run", side_effect=evaluate):
                self.assertEqual(accept.run_eval_pipeline("film", out), [{"key": "A1", "ok": False}])

    def test_missing_previously_passing_check_fails_h2(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work = root / "workspace/film"
            base = root / "baseline/film/baseline"
            (work / "output").mkdir(parents=True)
            base.mkdir(parents=True)
            video = work / "output/film_dubbed.mp4"
            video.write_bytes(b"video")
            (work / "state.json").write_text(json.dumps({"vc": {"video": str(video)}}))
            (base / "eval.json").write_text('[{"key":"A1","ok":true}]')
            summary = {k: 0 for k in ["cues", "kana_segments", "n_units", "multi_units", "pronoun_units",
                                     "fragmented_units"]}
            args = ["accept_release.py", "v3.4", "--films", "film", "--baseline", str(root / "baseline")]
            with patch.object(accept, "ROOT", root), patch.object(sys, "argv", args), \
                 patch.object(accept, "run_eval_pipeline", return_value=[{"key": "B1", "ok": True}]), \
                 patch.object(accept, "subs_summary", return_value=(summary, [])), \
                 patch.object(accept, "duration", return_value=1), \
                 patch.object(accept, "ffprobe", return_value=["44100,2"]), redirect_stdout(io.StringIO()):
                self.assertEqual(accept.main(), 1)
            report = json.loads((root / "workspace/_release/v3.4/ACCEPTANCE.json").read_text())
            h2 = next(g for g in report["gates"] if g["key"] == "H2[film]")
            self.assertFalse(h2["ok"])
            self.assertIn("A1", h2["detail"])


if __name__ == "__main__":
    unittest.main()
